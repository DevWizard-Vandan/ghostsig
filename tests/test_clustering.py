"""Unit and integration tests for FusionEncoder, HDBSCAN clustering, and the full pipeline."""

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta

import torch
import numpy as np
import hdbscan
import psycopg
import subprocess

from ml.fusion_encoder import FusionEncoder
from ml.train_fusion import DATABASE_URL
from consumers.raw_to_postgres import process_message
from pipeline.normalize import run_pipeline, KAFKA_BOOTSTRAP_SERVERS
from kafka import KafkaProducer


def test_fusion_forward():
    # Fusion forward pass: temporal (8,128) + entropy (8,64) -> fused (8,256)
    model = FusionEncoder()
    t_emb = torch.rand(8, 128)
    e_emb = torch.rand(8, 64)
    fused = model(t_emb, e_emb)
    assert fused.shape == (8, 256)


def test_hdbscan_synthetic_clusters():
    # HDBSCAN on 50 synthetic embeddings with 3 planted clusters -> detects >= 2 clusters
    np.random.seed(42)
    
    # Generate 3 tight clusters in 32-d space
    # (Since UMAP is Cosine, we make them distinct coordinates and normalized)
    c1 = np.random.normal(loc=0.0, scale=0.02, size=(15, 32))
    c2 = np.random.normal(loc=5.0, scale=0.02, size=(15, 32))
    c3 = np.random.normal(loc=-5.0, scale=0.02, size=(20, 32))
    
    data = np.vstack([c1, c2, c3])
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
    labels = clusterer.fit_predict(data)
    
    unique_labels = set(labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    
    # Verify that HDBSCAN successfully discovers the planted structures
    assert n_clusters >= 2


def test_full_pipeline_once_skip_training():
    # Setup database with valid temporal and entropy embeddings first
    # 1. Clean up database
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM raw_events;")
            cur.execute("DELETE FROM account_fingerprints;")
            cur.execute("DELETE FROM campaign_accounts;")
            cur.execute("DELETE FROM campaigns;")
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
    
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    try:
        admin.delete_topics(["raw.events"])
        time.sleep(1.0)
    except UnknownTopicOrPartitionError:
        pass
        
    admin.create_topics([NewTopic(name="raw.events", num_partitions=6, replication_factor=1)])
    admin.close()
    time.sleep(1.0)

    # 3. Create 10 synthetic accounts with different text and temporal intervals
    events = []
    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    
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
        account_id = f"pipeline_test_user_{i}"
        text_content = texts[i % len(texts)]
        
        # 15 events each to establish temporal intervals (non-zero embeddings)
        for j in range(15):
            event_ts = (base_time + timedelta(seconds=j * 120)).isoformat()
            events.append({
                "account_id": f"twitter:{account_id}",
                "platform": "twitter",
                "event_type": "tweet",
                "event_ts": event_ts,
                "metadata": {
                    "text": f"tweet content {j} - {text_content}"
                }
            })
            
    # 4. Ingest into Postgres
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for ev in events:
                process_message(ev, cur)
            conn.commit()
            
    # Send to Redpanda
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for ev in events:
        producer.send("raw.events", ev)
    producer.flush()
    producer.close()
    
    # Run pipeline normalize to write Parquet and update fingerprints
    run_pipeline(once=True)
    
    # 5. Run training + embedding for temporal and entropy models to create valid inputs
    from ml.train_temporal import main as train_temporal
    from ml.embed_temporal import main as embed_temporal
    from ml.train_entropy import main as train_entropy
    from ml.embed_entropy import main as embed_entropy
    from ml.train_fusion import main as train_fusion
    
    orig_argv = sys.argv
    try:
        sys.argv = ["train_temporal.py", "--epochs", "2", "--device", "cpu", "--batch-size", "8"]
        train_temporal()
        embed_temporal()
        
        sys.argv = ["train_entropy.py", "--epochs", "2", "--device", "cpu", "--batch-size", "8"]
        train_entropy()
        embed_entropy()
        
        sys.argv = ["train_fusion.py", "--epochs", "2", "--device", "cpu", "--batch-size", "8"]
        train_fusion()
    finally:
        sys.argv = orig_argv
        
    # Check that all checkpoints exist
    assert os.path.exists("checkpoints/temporal_encoder_v1.pt")
    assert os.path.exists("checkpoints/entropy_encoder_v1.pt")
    assert os.path.exists("checkpoints/entropy_scaler.json")
    assert os.path.exists("checkpoints/fusion_encoder_v1.pt")

    # 6. Execute full pipeline in skip training mode
    cmd = [sys.executable, "-m", "pipeline.full_pipeline", "--once", "--skip-training"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check execution status
    assert res.returncode == 0, f"Full pipeline failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    
    # Verify that campaigns and campaign_accounts have records
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM campaigns;")
            n_campaigns = cur.fetchone()[0]
            
            cur.execute("SELECT count(*) FROM campaign_accounts;")
            n_members = cur.fetchone()[0]
            
            cur.execute("SELECT count(*) FROM account_fingerprints WHERE fused_embedding IS NOT NULL;")
            n_fused = cur.fetchone()[0]
            
    # There should be at least some campaigns, memberships and fused embeddings populated
    assert n_fused == 10
    assert n_campaigns >= 1
    assert n_members >= 3
