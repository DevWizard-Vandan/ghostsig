"""Script to score campaigns and determine validation confidence tiers using the trained adversarial classifier."""

import logging
import os
import sys
import numpy as np
import xgboost as xgb
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("score_campaigns")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")
CHECKPOINT_PATH = "checkpoints/adversarial_clf_v1.json"


def main():
    # 1. Connect to Postgres
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    # 2. Load model
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Adversarial classifier checkpoint missing at {CHECKPOINT_PATH}. Run train_adversarial.py first.")
        conn.close()
        sys.exit(1)

    logger.info(f"Loading trained XGBoost classifier from {CHECKPOINT_PATH}...")
    clf = xgb.XGBClassifier()
    try:
        clf.load_model(CHECKPOINT_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        conn.close()
        sys.exit(1)

    # 3. Retrieve campaigns
    with conn.cursor() as cur:
        cur.execute("SELECT campaign_id, label FROM campaigns;")
        campaigns = cur.fetchall()

    if not campaigns:
        logger.warning("No campaigns found in campaigns table. Run clustering first.")
        conn.close()
        sys.exit(0)

    logger.info(f"Scoring {len(campaigns)} campaigns...")
    report_rows = []

    # 4. Score each campaign
    with conn.cursor() as cur:
        for campaign_id, label in campaigns:
            cur.execute(
                """
                SELECT f.fused_embedding, ca.account_id 
                FROM campaign_accounts ca 
                JOIN account_fingerprints f ON ca.account_id = f.account_id 
                WHERE ca.campaign_id = %s AND f.fused_embedding IS NOT NULL;
                """,
                (campaign_id,)
            )
            member_rows = cur.fetchall()
            
            if not member_rows:
                confidence = 0.0
                member_count = 0
            else:
                member_count = len(member_rows)
                embs = []
                for emb_val, _ in member_rows:
                    if isinstance(emb_val, str):
                        emb = np.array([float(x) for x in emb_val[1:-1].split(",")], dtype=np.float32)
                    else:
                        emb = np.array(emb_val, dtype=np.float32)
                    embs.append(emb)
                    
                embs = np.array(embs)
                # Compute bot probability P(bot) for each member
                probs = clf.predict_proba(embs)[:, 1]
                confidence = float(np.mean(probs))
                
            # Determine Tier
            if confidence > 0.7:
                tier = "HIGH"
            elif confidence >= 0.4:
                tier = "REVIEW"
            else:
                tier = "LIKELY_FP"
                
            # Update database
            cur.execute(
                """
                UPDATE campaigns 
                SET confidence = %s 
                WHERE campaign_id = %s;
                """,
                (confidence, campaign_id)
            )
            
            report_rows.append({
                "label": label,
                "member_count": member_count,
                "confidence": confidence,
                "tier": tier
            })
            
        conn.commit()

    conn.close()
    logger.info("Successfully updated campaigns confidence scores in PostgreSQL.")

    # 5. Print Tabular Report
    # Sort campaigns by confidence descending
    report_rows.sort(key=lambda x: x["confidence"], reverse=True)
    
    print("\n" + "=" * 70)
    print("                    CAMPAIGN CONFIDENCE REPORT                    ")
    print("=" * 70)
    print(f"{'Campaign Label':<20} | {'Members':<8} | {'Confidence':<10} | {'Tier':<12}")
    print("-" * 70)
    for row in report_rows:
        print(f"{row['label']:<20} | {row['member_count']:<8d} | {row['confidence']:.4f}     | {row['tier']:<12}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
