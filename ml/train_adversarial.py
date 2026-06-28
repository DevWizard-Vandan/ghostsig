"""Training script for XGBoost adversarial classifier distinguishing bot vs organic accounts."""

import argparse
import logging
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_adversarial")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")
CHECKPOINT_PATH = "checkpoints/adversarial_clf_v1.json"


def fetch_synthetic_dataset(db_url: str):
    logger.info("Fetching synthetic fused embeddings from Postgres...")
    try:
        conn = psycopg.connect(db_url)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT account_id, fused_embedding 
            FROM account_fingerprints 
            WHERE account_id LIKE 'synth_%' AND fused_embedding IS NOT NULL;
            """
        )
        rows = cur.fetchall()
        
    conn.close()
    
    X = []
    y = []
    
    for r in rows:
        account_id = r[0]
        emb_val = r[1]
        
        # Parse pgvector string representation: e.g. "[0.1,0.2,...]"
        if isinstance(emb_val, str):
            emb = np.array([float(x) for x in emb_val[1:-1].split(",")], dtype=np.float32)
        else:
            emb = np.array(emb_val, dtype=np.float32)
            
        X.append(emb)
        
        # Label: synth_bot = 1, synth_organic = 0
        if "synth_bot:" in account_id:
            y.append(1)
        elif "synth_organic:" in account_id:
            y.append(0)
            
    if not X:
        return np.empty((0, 256)), np.empty((0,))
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Adversarial Classifier")
    parser.add_argument("--epochs", type=int, default=200, help="Number of trees (n_estimators)")
    parser.add_argument("--batch-size", type=int, default=32, help="Placeholder for signature matching")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    # 1. Fetch synthetic data
    X, y = fetch_synthetic_dataset(DATABASE_URL)
    if len(X) < 10:
        logger.error(f"Insufficient synthetic dataset (only {len(X)} accounts). Run synthetic.bot_generator first.")
        sys.exit(1)
        
    logger.info(f"Loaded dataset: {len(X)} synthetic accounts.")
    
    # 2. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Train XGBoost classifier
    # n_estimators mapped from CLI --epochs
    n_estimators = args.epochs
    
    logger.info("Training XGBoost binary classifier...")
    
    # Setup device parameter
    device_param = "cpu"
    if args.device == "cuda":
        device_param = "cuda"
        
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        device=device_param,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 4. Evaluate classifier
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("\n" + "=" * 50)
    print("         ADVERSARIAL CLASSIFIER METRICS       ")
    print("=" * 50)
    print(f"Accuracy:                {acc:.4%}")
    print(f"ROC-AUC:                 {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Organic):  {cm[0, 0]}")
    print(f"  False Positives:          {cm[0, 1]}")
    print(f"  False Negatives:          {cm[1, 0]}")
    print(f"  True Positives (Bots):    {cm[1, 1]}")
    print("=" * 50 + "\n")
    
    # Feature Importances (top 10 dimensions of fused pgvector)
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("=== TOP 10 FEATURE IMPORTANCE DIMS ===")
    for rank, idx in enumerate(top_indices, 1):
        print(f"Rank {rank:02d} | Dim {idx:3d} | Importance: {importances[idx]:.6f}")
    print("=" * 50 + "\n")
    
    # 5. Save model checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    clf.save_model(CHECKPOINT_PATH)
    logger.info(f"Adversarial classifier checkpoint successfully saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
