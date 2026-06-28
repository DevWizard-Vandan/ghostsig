"""Unit and integration tests for the synthetic dataset simulators, adversarial classifier, and campaign scoring."""

import os
import sys
import numpy as np
import pytest
import xgboost as xgb
import psycopg

from synthetic.bot_generator import BotSimulator, OrganicSimulator
from ml.train_adversarial import fetch_synthetic_dataset, DATABASE_URL
from ml.score_campaigns import main as run_scoring


def test_simulators_entropy():
    # BotSimulator generates accounts with char_entropy_mean < 3.5
    # OrganicSimulator generates accounts with char_entropy_mean > 4.0
    bot = BotSimulator(9999)
    organic = OrganicSimulator(9999)
    
    # Run simulation base-time setup
    from datetime import datetime, timezone
    base_time = datetime.now(timezone.utc)
    
    fp_bot, _ = bot.simulate(base_time)
    fp_org, _ = organic.simulate(base_time)
    
    assert fp_bot["char_entropy_mean"] < 3.5
    assert fp_org["char_entropy_mean"] > 4.0
    
    # Confirm formatting
    assert fp_bot["account_id"].startswith("synth_bot:")
    assert fp_org["account_id"].startswith("synth_organic:")
    assert fp_bot["event_count"] >= 20
    assert fp_org["event_count"] <= 50


def test_adversarial_classifier_roc_auc():
    # Classifier ROC-AUC > 0.75 on 100 synthetic test accounts
    # Load dataset from database
    X, y = fetch_synthetic_dataset(DATABASE_URL)
    
    # If database doesn't have enough data (e.g. truncated during parallel test suite runs)
    if len(X) < 100:
        from synthetic.bot_generator import generate_dataset
        from ml.embed_temporal import main as run_embed_temporal
        from ml.embed_entropy import main as run_embed_entropy
        from ml.embed_fusion import main as run_embed_fusion
        generate_dataset(n_bots=50, n_organic=50)
        run_embed_temporal()
        run_embed_entropy()
        run_embed_fusion()
        X, y = fetch_synthetic_dataset(DATABASE_URL)
        
    assert len(X) >= 100
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Score
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    assert auc > 0.75, f"Adversarial classifier ROC-AUC is too low: {auc:.4f}"


def test_score_campaigns_execution():
    # score_campaigns runs without error and updates confidence column
    # 1. Run scoring script
    run_scoring()
    
    # 2. Query campaigns to verify confidence column is NOT NULL
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT confidence FROM campaigns;")
            rows = cur.fetchall()
            
            assert len(rows) > 0, "No campaigns found to verify scoring."
            for r in rows:
                confidence = r[0]
                assert confidence is not None
                assert 0.0 <= confidence <= 1.0
