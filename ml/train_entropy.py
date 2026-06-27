"""Training script for EntropyEncoder using self-supervised contrastive learning (SimCLR style)."""

import argparse
import json
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psycopg

from typing import Tuple, List
from ml.entropy_encoder import EntropyEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_entropy")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")


class MinMaxFeatureScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, features: np.ndarray):
        self.min = np.min(features, axis=0)
        self.max = np.max(features, axis=0)

    def transform(self, features: np.ndarray) -> np.ndarray:
        denom = self.max - self.min
        # Use 1e-9 fallback where min == max to avoid div-by-zero
        denom = np.where(denom == 0.0, 1e-9, denom)
        return (features - self.min) / denom

    def to_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "min": self.min.tolist(),
                "max": self.max.tolist()
            }, f)

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        scaler = cls()
        scaler.min = np.array(data["min"])
        scaler.max = np.array(data["max"])
        return scaler


class EntropyDataset(Dataset):
    def __init__(self, features: np.ndarray):
        # features shape: (N, 8)
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        
        # Positive pair augmentation: Add Gaussian noise (std=0.05) and clip to [0, 1]
        noise1 = torch.randn_like(x) * 0.05
        noise2 = torch.randn_like(x) * 0.05
        
        x1 = torch.clamp(x + noise1, 0.0, 1.0)
        x2 = torch.clamp(x + noise2, 0.0, 1.0)
        
        return x1, x2


def nt_xent_loss(embeddings1: torch.Tensor, embeddings2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # embeddings1: (N, D), embeddings2: (N, D)
    N = embeddings1.size(0)
    
    # L2 normalize
    emb1 = F.normalize(embeddings1, p=2, dim=1)
    emb2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Concatenate: shape (2N, D)
    representations = torch.cat([emb1, emb2], dim=0)
    
    # Similarity matrix: shape (2N, 2N)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    # Labels for cross entropy: index of matching view in concatenated representations
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(embeddings1.device)
    
    # Scale by temperature
    logits = similarity_matrix / temperature
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * N, device=embeddings1.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -9e15)
    
    loss = F.cross_entropy(logits, labels)
    return loss


def fetch_entropy_features(db_url: str) -> Tuple[List[str], np.ndarray]:
    logger.info("Fetching entropy features from account_fingerprints table...")
    try:
        conn = psycopg.connect(db_url)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        # Retrieve columns
        cur.execute(
            """
            SELECT 
                account_id, platform, char_entropy_mean, char_entropy_std, 
                word_entropy_mean, word_entropy_std, entropy_sample_count, 
                event_count, mean_interval_sec, coefficient_of_variation
            FROM account_fingerprints;
            """
        )
        rows = cur.fetchall()
        
    conn.close()
    
    if not rows:
        return [], np.empty((0, 8))
        
    account_ids = []
    features_list = []
    
    for r in rows:
        account_id = r[0]
        platform = r[1]
        char_entropy_mean = r[2] if r[2] is not None else 0.0
        char_entropy_std = r[3] if r[3] is not None else 0.0
        word_entropy_mean = r[4] if r[4] is not None else 0.0
        word_entropy_std = r[5] if r[5] is not None else 0.0
        entropy_sample_count = r[6] if r[6] is not None else 0.0
        event_count = r[7] if r[7] is not None else 0.0
        mean_interval_sec = r[8] if r[8] is not None else 0.0
        cv = r[9] if r[9] is not None else 0.0
        
        # Feature vector assembly
        vec = [
            char_entropy_mean,
            char_entropy_std,
            word_entropy_mean,
            word_entropy_std,
            np.log1p(entropy_sample_count),
            np.log1p(event_count),
            np.log1p(mean_interval_sec),
            cv
        ]
        
        account_ids.append(f"{platform}:{account_id}")
        features_list.append(vec)
        
    return account_ids, np.array(features_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Entropy Contrastive Encoder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    # 1. Fetch features
    account_ids, raw_features = fetch_entropy_features(DATABASE_URL)
    if len(raw_features) < 2:
        logger.error("Need at least 2 accounts in database to train contrastive encoder.")
        sys.exit(1)
        
    # 2. Fit and save scaler
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler_path = os.path.join(checkpoint_dir, "entropy_scaler.json")
    
    scaler = MinMaxFeatureScaler()
    scaler.fit(raw_features)
    scaler.to_json(scaler_path)
    logger.info(f"Feature scaler fitted and saved to {scaler_path}")
    
    scaled_features = scaler.transform(raw_features)
    
    # 3. Train/val split: 80/20 at account level
    np.random.seed(42)
    indices = np.arange(len(scaled_features))
    np.random.shuffle(indices)
    split_idx = int(len(scaled_features) * 0.8)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Handle small datasets fallback
    if len(val_indices) < 2:
        logger.warning("Validation set size < 2. Falling back to using training set for validation.")
        train_indices = indices
        val_indices = indices
        
    train_features = scaled_features[train_indices]
    val_features = scaled_features[val_indices]
    
    train_dataset = EntropyDataset(train_features)
    val_dataset = EntropyDataset(val_features)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Initialize model
    device = torch.device(args.device)
    model = EntropyEncoder(input_dim=8, output_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    model_path = os.path.join(checkpoint_dir, "entropy_encoder_v1.pt")
    
    # 5. Training loop
    logger.info(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x1, x2 in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            optimizer.zero_grad()
            emb1 = model(x1)
            emb2 = model(x2)
            
            loss = nt_xent_loss(emb1, emb2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x1.size(0)
            
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2 in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                emb1 = model(x1)
                emb2 = model(x2)
                loss = nt_xent_loss(emb1, emb2)
                val_loss += loss.item() * x1.size(0)
                
        epoch_val_loss = val_loss / len(val_dataset)
        
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"--> Saved best model checkpoint to {model_path}")
            
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
