"""Training script for TemporalEncoder using self-supervised masked interval prediction."""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import psycopg

from ml.temporal_encoder import TemporalEncoder
from features.temporal import compute_inter_event_intervals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_temporal")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")


class TemporalDataset(Dataset):
    def __init__(self, accounts, db_url):
        self.data = []
        logger.info(f"Loading raw event timestamps for {len(accounts)} accounts from Postgres...")
        
        try:
            conn = psycopg.connect(db_url)
        except Exception as e:
            logger.critical(f"Failed to connect to Postgres for dataset load: {e}")
            raise e
            
        with conn.cursor() as cur:
            for account_id, platform in accounts:
                cur.execute(
                    """
                    SELECT event_ts 
                    FROM raw_events 
                    WHERE account_id = %s AND platform = %s 
                    ORDER BY event_ts ASC
                    """,
                    (account_id, platform)
                )
                rows = cur.fetchall()
                if len(rows) < 2:
                    continue  # need at least 2 events to compute 1 interval
                
                timestamps = [r[0].timestamp() for r in rows]
                intervals = compute_inter_event_intervals(timestamps)
                
                # Normalize: log1p
                normalized = np.log1p(intervals).astype(np.float32)
                
                # Pad/truncate to max_len=256
                padded = np.zeros(256, dtype=np.float32)
                mask = np.zeros(256, dtype=np.float32)
                
                seq_len = min(len(normalized), 256)
                padded[:seq_len] = normalized[:seq_len]
                mask[:seq_len] = 1.0  # 1 = real, 0 = padding
                
                self.data.append((padded, mask))
                
        conn.close()
        logger.info(f"Loaded dataset with {len(self.data)} valid accounts.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        padded, mask = self.data[idx]
        return torch.tensor(padded), torch.tensor(mask)


class TemporalPretrainWrapper(nn.Module):
    def __init__(self, encoder: TemporalEncoder):
        super().__init__()
        self.encoder = encoder
        self.pred_head = nn.Linear(encoder.input_proj.out_features, 1)

    def forward(self, intervals: torch.Tensor, mask: torch.Tensor = None):
        # mask: (B, S) where 1=real, 0=padding
        # Convert to src_key_padding_mask: True to ignore/ignore padding
        padding_mask = (mask == 0) if mask is not None else None
        
        x = intervals.unsqueeze(-1)                                # (B, S, 1)
        x = self.encoder.input_proj(x)                            # (B, S, d_model)
        x = self.encoder.transformer(x, src_key_padding_mask=padding_mask)  # (B, S, d_model)
        preds = self.pred_head(x).squeeze(-1)                     # (B, S)
        return preds


def collate_fn(batch):
    intervals = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    
    targets = intervals.clone()
    masked_positions = torch.zeros_like(intervals, dtype=torch.bool)
    
    for i in range(intervals.size(0)):
        real_indices = torch.nonzero(masks[i] == 1.0).squeeze(-1)
        if len(real_indices) > 0:
            num_mask = max(1, int(len(real_indices) * 0.15))
            perm = torch.randperm(len(real_indices))[:num_mask]
            mask_idx = real_indices[perm]
            masked_positions[i, mask_idx] = True
            
    masked_intervals = intervals.clone()
    masked_intervals[masked_positions] = 0.0
    
    return masked_intervals, targets, masks, masked_positions


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Encoder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to train on")
    args = parser.parse_args()
    
    # 1. Load accounts from Parquet
    logger.info("Scanning processed Parquet files for training accounts...")
    dataset_dirs = glob.glob("data/processed/features_*.parquet")
    if not dataset_dirs:
        logger.error("No processed Parquet directories found in data/processed/. Cannot run training.")
        sys.exit(1)
        
    dfs = []
    for d in dataset_dirs:
        try:
            dfs.append(pd.read_parquet(d))
        except Exception as e:
            logger.warning(f"Failed to read parquet dataset {d}: {e}")
            
    if not dfs:
        logger.error("No valid Parquet files were read.")
        sys.exit(1)
        
    df = pd.concat(dfs, ignore_index=True)
    if "account_id" not in df.columns or "platform" not in df.columns:
        logger.error("Parquet files lack account_id or platform columns.")
        sys.exit(1)
        
    unique_accounts = df[["account_id", "platform"]].drop_duplicates().values.tolist()
    logger.info(f"Found {len(unique_accounts)} unique accounts in Parquet files.")
    
    # 2. Train/val split: 80/20 on account level
    np.random.seed(42)
    indices = np.arange(len(unique_accounts))
    np.random.shuffle(indices)
    split_idx = int(len(unique_accounts) * 0.8)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_accounts = [unique_accounts[i] for i in train_indices]
    val_accounts = [unique_accounts[i] for i in val_indices]
    
    logger.info(f"Split: {len(train_accounts)} training accounts, {len(val_accounts)} validation accounts.")
    
    # 3. Create datasets and loaders
    train_dataset = TemporalDataset(train_accounts, DATABASE_URL)
    val_dataset = TemporalDataset(val_accounts, DATABASE_URL)
    
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Check if raw_events table has timestamps for these accounts.")
        sys.exit(1)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 4. Initialize model
    device = torch.device(args.device)
    encoder = TemporalEncoder(max_seq_len=256)
    model = TemporalPretrainWrapper(encoder).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss(reduction="none")
    
    best_val_loss = float("inf")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "temporal_encoder_v1.pt")
    
    # 5. Training loop
    logger.info(f"Starting training on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for masked_intervals, targets, masks, masked_positions in train_loader:
            masked_intervals = masked_intervals.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            masked_positions = masked_positions.to(device)
            
            optimizer.zero_grad()
            preds = model(masked_intervals, mask=masks)
            
            # Loss on masked positions only
            loss_elements = criterion(preds, targets)
            loss = torch.sum(loss_elements * masked_positions) / (torch.sum(masked_positions) + 1e-9)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * torch.sum(masked_positions).item()
            train_count += torch.sum(masked_positions).item()
            
        scheduler.step()
        epoch_train_loss = train_loss / (train_count + 1e-9)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for masked_intervals, targets, masks, masked_positions in val_loader:
                masked_intervals = masked_intervals.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                masked_positions = masked_positions.to(device)
                
                preds = model(masked_intervals, mask=masks)
                loss_elements = criterion(preds, targets)
                loss = torch.sum(loss_elements * masked_positions) / (torch.sum(masked_positions) + 1e-9)
                
                val_loss += loss.item() * torch.sum(masked_positions).item()
                val_count += torch.sum(masked_positions).item()
                
        epoch_val_loss = val_loss / (val_count + 1e-9)
        
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )
        
        # Save best checkpoint (based on encoder weights only)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.encoder.state_dict(), checkpoint_path)
            logger.info(f"--> Saved new best checkpoint to {checkpoint_path}")
            
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
