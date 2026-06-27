"""Unit and integration tests for the EntropyEncoder pretraining and embedding pipeline."""

import json
import os
import shutil
from datetime import datetime, timezone, timedelta

import pytest
import torch
import numpy as np
import psycopg

from ml.entropy_encoder import EntropyEncoder
from ml.train_entropy import MinMaxFeatureScaler, nt_xent_loss, DATABASE_URL
from consumers.raw_to_postgres import process_message
from pipeline.normalize import run_pipeline, KAFKA_BOOTSTRAP_SERVERS
from kafka import KafkaProducer


def test_model_forward():
    # Forward pass: input (8, 8) -> output shape (8, 64)
    model = EntropyEncoder(input_dim=8, output_dim=64)
    dummy_input = torch.rand(8, 8)
    out = model(dummy_input)
    assert out.shape == (8, 64)


def test_contrastive_similarity():
    # Contrastive loss: augmented pair of same account has higher similarity than random pair
    model = EntropyEncoder(input_dim=8, output_dim=64)
    model.eval()
    
    # Generate a random base feature vector
    x = torch.rand(1, 8)
    # Add small noise (augmented pair)
    noise1 = torch.randn(1, 8) * 0.05
    noise2 = torch.randn(1, 8) * 0.05
    x1 = torch.clamp(x + noise1, 0.0, 1.0)
    x2 = torch.clamp(x + noise2, 0.0, 1.0)
    
    # A completely random negative vector representing a different account
    y = torch.rand(1, 8)
    while torch.dist(x, y).item() < 0.2:
        y = torch.rand(1, 8)
        
    with torch.no_grad():
        emb_x1 = torch.nn.functional.normalize(model(x1), p=2, dim=1)
        emb_x2 = torch.nn.functional.normalize(model(x2), p=2, dim=1)
        emb_y = torch.nn.functional.normalize(model(y), p=2, dim=1)
        
    pos_sim = torch.dot(emb_x1.squeeze(), emb_x2.squeeze()).item()
    neg_sim = torch.dot(emb_x1.squeeze(), emb_y.squeeze()).item()
    
    # The augmented pair should map closer in representation space than a random vector
    assert pos_sim > neg_sim


def test_scaler():
    # Scaler: fit on 20 synthetic accounts, verify all features in [0,1] after transform
    features = np.random.randn(20, 8) * 10.0  # Wide range of features
    scaler = MinMaxFeatureScaler()
    scaler.fit(features)
    transformed = scaler.transform(features)
    
    # Verify shape
    assert transformed.shape == (20, 8)
    # Verify min/max values
    assert np.all(transformed >= 0.0)
    assert np.all(transformed <= 1.0)
    
    # Check that at least one value reaches 0 and 1 per column (min-max property)
    assert np.all(np.min(transformed, axis=0) == pytest.approx(0.0))
    assert np.all(np.max(transformed, axis=0) == pytest.approx(1.0))


def test_embed_pipeline():
    # Embed pipeline: 10 synthetic accounts -> entropy_embedding NOT NULL
    # 1. Clean up database
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM raw_events;")
            cur.execute("DELETE FROM account_fingerprints;")
            conn.commit()
            
    # Clean up processed directory
    processed_dir = os.path.join("data", "processed")
    if os.path.exists(processed_dir):
        try:
            shutil.rmtree(processed_dir)
        except Exception:
            pass

    # 2. Recreate Redpanda topic
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import UnknownTopicOrPartitionError
    import time
    
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    try:
        admin.delete_topics(["raw.events"])
        time.sleep(1.0)
    except UnknownTopicOrPartitionError:
        pass
        
    admin.create_topics([NewTopic(name="raw.events", num_partitions=6, replication_factor=1)])
    admin.close()
    time.sleep(1.0)

    # 3. Create 10 synthetic accounts with different text contents
    events = []
    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    
    # We populate some text for all of them so we get non-null entropy values
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A complete sentence with diverse vocabulary and unique characters",
        "Repeated words word word word word word word",
        "Short post here",
        "Linguistic entropy represents character distributions",
        "This is another test message for text processing",
        "We want to extract distinct entropy scores per account",
        "Highly redundant text redundancy redundant redundancy redundant",
        "Unpredictable sequences of letters abc xyz rst lmn opq",
        "Final account post test with custom strings"
    ]
    
    for i in range(10):
        account_id = f"entropy_test_user_{i}"
        text_content = texts[i % len(texts)]
        
        # 5 posts each to establish stats
        for j in range(5):
            event_ts = (base_time + timedelta(seconds=j * 200)).isoformat()
            events.append({
                "account_id": f"reddit:{account_id}",
                "platform": "reddit",
                "event_type": "post",
                "event_ts": event_ts,
                "metadata": {
                    "title": f"Title {j}",
                    "selftext": text_content
                }
            })
            
    # 4. Ingest into Postgres raw_events
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for ev in events:
                process_message(ev, cur)
            conn.commit()
            
    # Send events to Redpanda
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for ev in events:
        producer.send("raw.events", ev)
    producer.flush()
    producer.close()
    
    # Run pipeline to populate account_fingerprints and Parquet
    run_pipeline(once=True)
    
    # Verify account_fingerprints table has 10 rows
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM account_fingerprints;")
            count = cur.fetchone()[0]
            assert count == 10
            
    # 5. Run train script for 2 epochs
    from ml.train_entropy import main as run_train
    import sys
    
    orig_argv = sys.argv
    sys.argv = ["train_entropy.py", "--epochs", "2", "--device", "cpu", "--batch-size", "8"]
    try:
        run_train()
    finally:
        sys.argv = orig_argv
        
    assert os.path.exists("checkpoints/entropy_encoder_v1.pt")
    assert os.path.exists("checkpoints/entropy_scaler.json")
    
    # 6. Run embed script
    from ml.embed_entropy import main as run_embed
    run_embed()
    
    # 7. Assert that all 10 accounts in account_fingerprints have entropy_embedding populated
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT account_id, entropy_embedding FROM account_fingerprints;")
            rows = cur.fetchall()
            assert len(rows) == 10
            for account_id, embedding in rows:
                assert embedding is not None
                assert embedding.startswith("[")
                assert embedding.endswith("]")
                coords = [float(x) for x in embedding[1:-1].split(",")]
                assert len(coords) == 64
