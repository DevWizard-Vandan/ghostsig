"""Training script for FusionEncoder using self-supervised reconstruction task."""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psycopg

from ml.fusion_encoder import FusionEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_fusion")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")


class FusionDataset(Dataset):
    def __init__(self, temporal_embs: np.ndarray, entropy_embs: np.ndarray):
        # temporal_embs: (N, 128), entropy_embs: (N, 64)
        self.temporal_embs = torch.tensor(temporal_embs, dtype=torch.float32)
        self.entropy_embs = torch.tensor(entropy_embs, dtype=torch.float32)

    def __len__(self):
        return len(self.temporal_embs)

    def __getitem__(self, idx):
        return self.temporal_embs[idx], self.entropy_embs[idx]


class FusionPretrainWrapper(nn.Module):
    def __init__(self, encoder: FusionEncoder):
        super().__init__()
        self.encoder = encoder
        # Decoders to project fused representation back to input modal spaces
        self.recon_temporal = nn.Linear(256, 128)
        self.recon_entropy = nn.Linear(256, 64)

    def forward(self, temporal_emb: torch.Tensor, entropy_emb: torch.Tensor):
        fused = self.encoder(temporal_emb, entropy_emb)
        pred_temporal = self.recon_temporal(fused)
        pred_entropy = self.recon_entropy(fused)
        return pred_temporal, pred_entropy


def fetch_modal_embeddings(db_url: str):
    logger.info("Fetching temporal and entropy embeddings from Postgres...")
    try:
        conn = psycopg.connect(db_url)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT account_id, temporal_embedding, entropy_embedding 
            FROM account_fingerprints;
            """
        )
        rows = cur.fetchall()
        
    conn.close()
    
    temporal_list = []
    entropy_list = []
    account_ids = []
    
    for r in rows:
        account_id = r[0]
        t_val = r[1]
        e_val = r[2]
        
        if t_val is None or e_val is None:
            continue
            
        # Parse pgvector string representation: e.g. "[0.1,0.2,...]"
        if isinstance(t_val, str):
            t_vec = np.array([float(x) for x in t_val[1:-1].split(",")], dtype=np.float32)
        else:
            t_vec = np.array(t_val, dtype=np.float32)
            
        if isinstance(e_val, str):
            e_vec = np.array([float(x) for x in e_val[1:-1].split(",")], dtype=np.float32)
        else:
            e_vec = np.array(e_val, dtype=np.float32)
            
        # Skip cold-start or all-zero embeddings
        if np.all(t_vec == 0.0) or np.all(e_vec == 0.0):
            continue
            
        account_ids.append(account_id)
        temporal_list.append(t_vec)
        entropy_list.append(e_vec)
        
    if not temporal_list:
        return [], np.empty((0, 128)), np.empty((0, 64))
        
    return account_ids, np.array(temporal_list, dtype=np.float32), np.array(entropy_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Encoder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    # 1. Fetch data
    account_ids, temporal_embs, entropy_embs = fetch_modal_embeddings(DATABASE_URL)
    if len(temporal_embs) < 2:
        logger.error("Need at least 2 accounts with valid embeddings to train fusion model.")
        sys.exit(1)
        
    # 2. Train/val split: 80/20 at account level
    np.random.seed(42)
    indices = np.arange(len(temporal_embs))
    np.random.shuffle(indices)
    split_idx = int(len(temporal_embs) * 0.8)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Small dataset fallback
    if len(val_indices) < 1:
        train_indices = indices
        val_indices = indices
        
    train_temporal = temporal_embs[train_indices]
    train_entropy = entropy_embs[train_indices]
    val_temporal = temporal_embs[val_indices]
    val_entropy = entropy_embs[val_indices]
    
    train_dataset = FusionDataset(train_temporal, train_entropy)
    val_dataset = FusionDataset(val_temporal, val_entropy)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Setup model
    device = torch.device(args.device)
    encoder = FusionEncoder().to(device)
    wrapper = FusionPretrainWrapper(encoder).to(device)
    
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "fusion_encoder_v1.pt")
    
    best_val_loss = float("inf")
    
    # 4. Training loop
    logger.info(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        train_loss = 0.0
        
        for t_batch, e_batch in train_loader:
            t_batch = t_batch.to(device)
            e_batch = e_batch.to(device)
            
            optimizer.zero_grad()
            pred_t, pred_e = wrapper(t_batch, e_batch)
            
            loss = criterion(pred_t, t_batch) + criterion(pred_e, e_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * t_batch.size(0)
            
        epoch_train_loss = train_loss / len(train_dataset)
        
        # Validation
        wrapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_batch, e_batch in val_loader:
                t_batch = t_batch.to(device)
                e_batch = e_batch.to(device)
                pred_t, pred_e = wrapper(t_batch, e_batch)
                loss = criterion(pred_t, t_batch) + criterion(pred_e, e_batch)
                val_loss += loss.item() * t_batch.size(0)
                
        epoch_val_loss = val_loss / len(val_dataset)
        
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )
        
        # Save best checkpoint of the encoder weights only
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the underlying FusionEncoder weights only
            torch.save(encoder.state_dict(), model_path)
            logger.info(f"--> Saved best model checkpoint to {model_path}")
            
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
