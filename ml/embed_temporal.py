"""Embedding script to generate 128-d temporal embeddings for all accounts using the trained model."""

import logging
import os
import random
import sys
import numpy as np
import torch
import psycopg

from ml.temporal_encoder import TemporalEncoder
from features.temporal import compute_inter_event_intervals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("embed_temporal")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")
CHECKPOINT_PATH = "checkpoints/temporal_encoder_v1.pt"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    # 1. Connect to database and ensure pgvector column exists
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)
        
    with conn.cursor() as cur:
        logger.info("Ensuring pgvector extension and column exist...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            ALTER TABLE account_fingerprints 
            ADD COLUMN IF NOT EXISTS temporal_embedding vector(128);
            """
        )
        conn.commit()

    # 2. Load the trained checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Run training first.")
        conn.close()
        sys.exit(1)
        
    logger.info(f"Loading TemporalEncoder model from {CHECKPOINT_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalEncoder(max_seq_len=256).to(device)
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        conn.close()
        sys.exit(1)

    # 3. Retrieve all accounts in the fingerprints table
    with conn.cursor() as cur:
        cur.execute("SELECT account_id, platform FROM account_fingerprints;")
        accounts = cur.fetchall()
        
    logger.info(f"Found {len(accounts)} accounts to embed.")
    
    if not accounts:
        logger.warning("No accounts found in account_fingerprints table.")
        conn.close()
        sys.exit(0)

    # 4. Prepare data for encoding
    records = []
    
    # We load all data sequentially and prepare inputs
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
            
            # If less than 2 events, we cannot calculate interval.
            # We assign a dummy 128-d zero vector later to keep it NOT NULL.
            if len(rows) < 2:
                records.append({
                    "account_id": account_id,
                    "has_intervals": False,
                    "padded": np.zeros(256, dtype=np.float32),
                    "mask": np.zeros(256, dtype=np.float32)
                })
                continue
                
            timestamps = [r[0].timestamp() for r in rows]
            intervals = compute_inter_event_intervals(timestamps)
            
            # log1p normalization
            normalized = np.log1p(intervals).astype(np.float32)
            
            padded = np.zeros(256, dtype=np.float32)
            mask = np.zeros(256, dtype=np.float32)
            
            seq_len = min(len(normalized), 256)
            padded[:seq_len] = normalized[:seq_len]
            mask[:seq_len] = 1.0
            
            records.append({
                "account_id": account_id,
                "has_intervals": True,
                "padded": padded,
                "mask": mask
            })

    # 5. Process in batches of 256
    batch_size = 256
    embeddings_map = {}
    
    for idx in range(0, len(records), batch_size):
        batch = records[idx:idx + batch_size]
        
        # Split into valid (having intervals) and invalid
        valid_batch = [r for r in batch if r["has_intervals"]]
        
        if valid_batch:
            # Stack inputs
            padded_t = torch.stack([torch.tensor(r["padded"]) for r in valid_batch]).to(device)
            mask_t = torch.stack([torch.tensor(r["mask"]) for r in valid_batch]).to(device)
            
            # Convert mask: 0=padding, 1=real -> True where padding (0)
            padding_mask = (mask_t == 0)
            
            with torch.no_grad():
                # Forward pass through encoder
                outputs = model(padded_t, mask=padding_mask)
                outputs_np = outputs.cpu().numpy()
                
            for r, emb in zip(valid_batch, outputs_np):
                embeddings_map[r["account_id"]] = emb
                
        # For invalid ones, assign zero vector
        for r in batch:
            if not r["has_intervals"]:
                embeddings_map[r["account_id"]] = np.zeros(128, dtype=np.float32)

    # 6. Write back to postgres
    logger.info("Writing embeddings back to Postgres...")
    embedded_count = 0
    with conn.cursor() as cur:
        for account_id, emb in embeddings_map.items():
            # Format as string representation for pgvector: [val, val, ...]
            vector_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            cur.execute(
                """
                UPDATE account_fingerprints 
                SET temporal_embedding = %s::vector 
                WHERE account_id = %s;
                """,
                (vector_str, account_id)
            )
            embedded_count += 1
        conn.commit()

    logger.info(f"Successfully embedded {embedded_count} accounts.")
    
    # 7. Print cosine similarity sample
    embedded_accounts = list(embeddings_map.keys())
    if len(embedded_accounts) >= 2:
        logger.info("Calculating sample cosine similarity between random pairs...")
        num_samples = min(5, len(embedded_accounts) * (len(embedded_accounts) - 1) // 2)
        pairs_seen = set()
        
        tries = 0
        while len(pairs_seen) < num_samples and tries < 100:
            tries += 1
            a_id = random.choice(embedded_accounts)
            b_id = random.choice(embedded_accounts)
            if a_id == b_id:
                continue
            pair = tuple(sorted([a_id, b_id]))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)
            
            sim = cosine_similarity(embeddings_map[a_id], embeddings_map[b_id])
            print(f"Similarity({a_id[:20]}, {b_id[:20]}): {sim:.4f}")
            
    conn.close()


if __name__ == "__main__":
    main()
