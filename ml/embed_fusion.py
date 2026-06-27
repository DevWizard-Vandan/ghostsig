"""Embedding script to generate 256-d fused embeddings for accounts using the pretrained model checkpoint."""

import logging
import os
import sys
import torch
import psycopg

from ml.fusion_encoder import FusionEncoder
from ml.train_fusion import fetch_modal_embeddings, DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("embed_fusion")

CHECKPOINT_PATH = "checkpoints/fusion_encoder_v1.pt"


def main():
    # 1. Connect to database
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    # 2. Load model
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint file {CHECKPOINT_PATH} missing. Run train_fusion.py first.")
        conn.close()
        sys.exit(1)

    logger.info(f"Loading FusionEncoder model from {CHECKPOINT_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionEncoder().to(device)
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        conn.close()
        sys.exit(1)

    # 3. Fetch modal embeddings
    account_ids, temporal_embs, entropy_embs = fetch_modal_embeddings(DATABASE_URL)
    if not account_ids:
        logger.warning("No valid accounts with temporal+entropy embeddings to fuse.")
        conn.close()
        sys.exit(0)

    # 4. Batch encoding
    batch_size = 256
    embeddings_map = {}
    
    for idx in range(0, len(account_ids), batch_size):
        batch_ids = account_ids[idx:idx + batch_size]
        batch_t = temporal_embs[idx:idx + batch_size]
        batch_e = entropy_embs[idx:idx + batch_size]
        
        t_tensor = torch.tensor(batch_t, dtype=torch.float32).to(device)
        e_tensor = torch.tensor(batch_e, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(t_tensor, e_tensor)
            outputs_np = outputs.cpu().numpy()
            
        for a_id, emb in zip(batch_ids, outputs_np):
            embeddings_map[a_id] = emb

    # 5. Write back to Postgres
    logger.info("Writing fused embeddings back to Postgres...")
    fused_count = 0
    with conn.cursor() as cur:
        for account_id, emb in embeddings_map.items():
            vector_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            cur.execute(
                """
                UPDATE account_fingerprints 
                SET fused_embedding = %s::vector 
                WHERE account_id = %s;
                """,
                (vector_str, account_id)
            )
            fused_count += 1
        conn.commit()

    logger.info(f"Successfully fused and stored embeddings for {fused_count} accounts.")
    conn.close()


if __name__ == "__main__":
    main()
