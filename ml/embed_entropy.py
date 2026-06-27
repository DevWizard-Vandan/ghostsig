"""Embedding script to generate 64-d linguistic entropy embeddings for all accounts using the trained model."""

import logging
import os
import sys
import numpy as np
import torch
import psycopg

from ml.entropy_encoder import EntropyEncoder
from ml.train_entropy import MinMaxFeatureScaler, fetch_entropy_features, DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("embed_entropy")

CHECKPOINT_PATH = "checkpoints/entropy_encoder_v1.pt"
SCALER_PATH = "checkpoints/entropy_scaler.json"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    # 1. Connect to database and ensure column exists
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
        logger.info("Ensuring pgvector extension and entropy_embedding column exist...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            ALTER TABLE account_fingerprints 
            ADD COLUMN IF NOT EXISTS entropy_embedding vector(64);
            """
        )
        conn.commit()

    # 2. Load checkpoints and config
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(SCALER_PATH):
        logger.error("Checkpoint or scaler files missing. Run train_entropy.py first.")
        conn.close()
        sys.exit(1)

    logger.info(f"Loading feature scaler from {SCALER_PATH}...")
    try:
        scaler = MinMaxFeatureScaler.from_json(SCALER_PATH)
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        conn.close()
        sys.exit(1)

    logger.info(f"Loading EntropyEncoder model from {CHECKPOINT_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntropyEncoder(input_dim=8, output_dim=64).to(device)
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        conn.close()
        sys.exit(1)

    # 3. Fetch all entropy features
    # Note: account_ids contains strings "platform:account_id"
    account_ids, raw_features = fetch_entropy_features(DATABASE_URL)
    if not account_ids:
        logger.warning("No accounts found in account_fingerprints table.")
        conn.close()
        sys.exit(0)

    # 4. Scale features
    scaled_features = scaler.transform(raw_features)

    # 5. Embed in batches of 256
    batch_size = 256
    embeddings_map = {}
    
    for idx in range(0, len(account_ids), batch_size):
        batch_ids = account_ids[idx:idx + batch_size]
        batch_feats = scaled_features[idx:idx + batch_size]
        
        feats_t = torch.tensor(batch_feats, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(feats_t)
            outputs_np = outputs.cpu().numpy()
            
        for a_id, emb in zip(batch_ids, outputs_np):
            embeddings_map[a_id] = emb

    # 6. Update database
    logger.info("Writing embeddings back to Postgres...")
    embedded_count = 0
    with conn.cursor() as cur:
        for namespaced_id, emb in embeddings_map.items():
            # Parse namespaced platform:account_id back
            parts = namespaced_id.split(":", 1)
            platform = parts[0]
            account_id = parts[1]
            
            # Format as string representation for pgvector: [val, val, ...]
            vector_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
            cur.execute(
                """
                UPDATE account_fingerprints 
                SET entropy_embedding = %s::vector 
                WHERE account_id = %s AND platform = %s;
                """,
                (vector_str, account_id, platform)
            )
            embedded_count += 1
        conn.commit()

    logger.info(f"Successfully embedded {embedded_count} accounts.")

    # 7. Print pairwise similarities (Top-5 and Bottom-5)
    embedded_keys = list(embeddings_map.keys())
    if len(embedded_keys) >= 2:
        logger.info("Calculating pairwise similarity report...")
        pairs = []
        for i in range(len(embedded_keys)):
            for j in range(i + 1, len(embedded_keys)):
                key_a = embedded_keys[i]
                key_b = embedded_keys[j]
                sim = cosine_similarity(embeddings_map[key_a], embeddings_map[key_b])
                pairs.append((sim, key_a, key_b))
                
        # Sort by similarity
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        print("\n=== TOP 5 MOST SIMILAR PAIRS ===")
        for sim, a, b in pairs[:5]:
            print(f"Similarity: {sim:.4f} | Pair: ({a}, {b})")
            
        print("\n=== BOTTOM 5 LEAST SIMILAR PAIRS ===")
        for sim, a, b in pairs[-5:]:
            print(f"Similarity: {sim:.4f} | Pair: ({a}, {b})")
            
    conn.close()


if __name__ == "__main__":
    main()
