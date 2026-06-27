"""Unit and integration tests for the TemporalEncoder training and embedding pipeline."""

import json
import os
import random
import shutil
from datetime import datetime, timezone, timedelta

import pytest
import torch
import torch.nn as nn
import numpy as np
import psycopg

from ml.temporal_encoder import TemporalEncoder
from ml.train_temporal import TemporalPretrainWrapper, collate_fn, DATABASE_URL
from kafka import KafkaProducer
from consumers.raw_to_postgres import process_message
from pipeline.normalize import run_pipeline, KAFKA_BOOTSTRAP_SERVERS


def test_model_forward():
    # Model forward pass: input (8, 64) -> output shape (8, 128)
    model = TemporalEncoder(max_seq_len=64)
    dummy_input = torch.rand(8, 64)
    out = model(dummy_input)
    assert out.shape == (8, 128)


def test_masked_prediction():
    # Set seed for deterministic test execution
    torch.manual_seed(42)
    
    # Masked prediction: masked positions have higher loss than unmasked
    encoder = TemporalEncoder(max_seq_len=64)
    model = TemporalPretrainWrapper(encoder)
    
    # 8 sequences of length 64
    intervals = torch.rand(8, 64)
    masks = torch.ones(8, 64)  # All positions are real
    batch = [(intervals[i], masks[i]) for i in range(8)]
    
    # Generate a fixed masked batch for the brief training
    masked_intervals, targets, masks_col, masked_positions = collate_fn(batch)
    
    # Train the model briefly to learn to copy the unmasked inputs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss(reduction="none")
    
    # Run in eval mode to disable dropout, allowing stable convergence of identity copy mapping
    model.eval()
    for _ in range(100):
        optimizer.zero_grad()
        preds = model(masked_intervals, mask=masks_col)
        
        # Loss on UNMASKED positions to force the model to copy them
        unmasked_positions = (~masked_positions) & (masks_col == 1.0)
        loss = torch.sum(criterion(preds, targets) * unmasked_positions) / (torch.sum(unmasked_positions) + 1e-9)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        preds = model(masked_intervals, mask=masks_col)
        loss_elements = criterion(preds, targets)
        
        # Unmasked positions
        unmasked_positions = (~masked_positions) & (masks_col == 1.0)
        
        masked_loss = torch.sum(loss_elements * masked_positions) / (torch.sum(masked_positions) + 1e-9)
        unmasked_loss = torch.sum(loss_elements * unmasked_positions) / (torch.sum(unmasked_positions) + 1e-9)
        
        # Unmasked positions should have significantly lower loss since the model learned to reconstruct them
        assert masked_loss.item() > unmasked_loss.item()


def test_embedding_pipeline():
    # Embed script: given 10 synthetic accounts, temporal_embedding column populated
    # 1. Clean up database
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM raw_events;")
            cur.execute("DELETE FROM account_fingerprints;")
            conn.commit()
            
    # Clean up processed folder
    processed_dir = os.path.join("data", "processed")
    if os.path.exists(processed_dir):
        try:
            shutil.rmtree(processed_dir)
        except Exception:
            pass

    # 2. Generate 10 synthetic accounts
    events = []
    base_time = datetime.now(timezone.utc) - timedelta(days=2)
    
    for i in range(10):
        account_id = f"embed_test_user_{i}"
        # Even accounts have intervals, odd accounts have only 1 event (no intervals)
        num_events = 15 if i % 2 == 0 else 1
        
        for j in range(num_events):
            event_ts = (base_time + timedelta(seconds=j * 120)).isoformat()
            events.append({
                "account_id": f"twitter:{account_id}",
                "platform": "twitter",
                "event_type": "tweet",
                "event_ts": event_ts,
                "metadata": {"text": f"tweet content {j}"}
            })
            
    # 3. Ingest into Postgres raw_events
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for ev in events:
                process_message(ev, cur)
            conn.commit()
            
    # 4. Ingest into Postgres account_fingerprints by running normalizer
    # We must also produce these events to Kafka so the normalizer picks them up
    # Wait! In tests, we can just run normalizer batch by producing to Kafka.
    # Recreate raw.events topic
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
    
    # Send events
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for ev in events:
        producer.send("raw.events", ev)
    producer.flush()
    producer.close()
    
    # Run pipeline to populate account_fingerprints and create Parquet files
    run_pipeline(once=True)
    
    # Verify account_fingerprints table has 10 rows
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM account_fingerprints;")
            count = cur.fetchone()[0]
            assert count == 10
            
    # 5. Run train script for 2 epochs to create a model checkpoint
    from ml.train_temporal import main as run_train
    import sys
    
    # Save original sys.argv
    orig_argv = sys.argv
    sys.argv = ["train_temporal.py", "--epochs", "2", "--device", "cpu", "--batch-size", "8"]
    try:
        run_train()
    finally:
        sys.argv = orig_argv
        
    # Check that model checkpoint was saved
    assert os.path.exists("checkpoints/temporal_encoder_v1.pt")
    
    # 6. Run embed script programmatically
    from ml.embed_temporal import main as run_embed
    run_embed()
    
    # 7. Assert that all 10 accounts in account_fingerprints have temporal_embedding populated
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT account_id, temporal_embedding FROM account_fingerprints;")
            rows = cur.fetchall()
            assert len(rows) == 10
            for account_id, embedding in rows:
                assert embedding is not None
                # Verify pgvector string representation: [val, val, ...]
                assert embedding.startswith("[")
                assert embedding.endswith("]")
                coords = [float(x) for x in embedding[1:-1].split(",")]
                assert len(coords) == 128
                
                # Check that for accounts with 1 event, the embedding is all zeros
                if "embed_test_user_1" in account_id or "embed_test_user_3" in account_id:
                    assert all(c == 0.0 for c in coords)
