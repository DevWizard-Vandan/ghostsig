"""Integration and unit tests for the normalization pipeline."""

import json
import os
import shutil
from datetime import datetime, timezone, timedelta

import pytest
from kafka import KafkaProducer
import psycopg

from pipeline.normalize import run_pipeline, DATABASE_URL, KAFKA_BOOTSTRAP_SERVERS
from consumers.raw_to_postgres import process_message


@pytest.fixture
def synthetic_events():
    events = []
    # 5 accounts, 20 events each = 100 events total
    platforms = ["reddit", "twitter", "gdelt"]
    for i in range(5):
        account_id = f"test_user_{i}"
        platform = platforms[i % len(platforms)]
        base_time = datetime.now(timezone.utc) - timedelta(days=1)
        
        for j in range(20):
            # posts spaced exactly 60 seconds apart
            event_ts = (base_time + timedelta(seconds=j * 60)).isoformat()
            
            # For reddit, add some text for entropy validation
            metadata = {}
            if platform == "reddit":
                metadata = {
                    "title": "Hello world this is a test post",
                    "selftext": "Repeated word repeated word repeated word"
                }
            else:
                metadata = {
                    "field": f"value_{j}"
                }
                
            events.append({
                "account_id": f"{platform}:{account_id}",
                "platform": platform,
                "event_type": "post" if platform == "reddit" else "tweet" if platform == "twitter" else "mention",
                "event_ts": event_ts,
                "metadata": metadata
            })
    return events


def test_normalization_pipeline(synthetic_events):
    # 1. Clean up database tables to avoid interference from prior runs
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM raw_events;")
            cur.execute("DELETE FROM account_fingerprints;")
            conn.commit()
            
    # Clean up Redpanda topic
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
            
    # Clean up processed data folder
    processed_dir = os.path.join("data", "processed")
    if os.path.exists(processed_dir):
        try:
            shutil.rmtree(processed_dir)
        except Exception:
            pass
        
    # 2. Produce synthetic events to Redpanda raw.events topic
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    for ev in synthetic_events:
        producer.send("raw.events", ev)
    producer.flush()
    producer.close()
    
    # 3. Use database consumer logic to ingest them into Postgres
    # (Since normalizer queries the database, we must ensure these raw events are in the database)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            for ev in synthetic_events:
                process_message(ev, cur)
            conn.commit()
            
    # Verify that raw_events has 100 rows
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_events;")
            count = cur.fetchone()[0]
            assert count == 100
            
    # 4. Run the normalization pipeline (once mode)
    # This will consume the 100 messages from Redpanda, group them, query Postgres, compute features,
    # save Parquet, and upsert to Postgres.
    run_pipeline(once=True)
    
    # 5. Verify database upserts inside account_fingerprints
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM account_fingerprints;")
            fingerprint_count = cur.fetchone()[0]
            assert fingerprint_count == 5  # 5 unique accounts
            
            cur.execute("SELECT account_id, platform, mean_interval_sec, event_count, char_entropy_mean FROM account_fingerprints;")
            rows = cur.fetchall()
            for row in rows:
                account_id, platform, mean_interval, event_count, char_entropy_mean = row
                assert event_count == 20
                assert mean_interval == pytest.approx(60.0) # posts spaced 60 seconds apart
                
                # Reddit should have char_entropy_mean populated, others might be None
                if platform == "reddit":
                    assert char_entropy_mean is not None
                    assert char_entropy_mean > 0.0
                    
    # 6. Verify Parquet files are generated
    assert os.path.exists(processed_dir)
    # Check if there are any parquet files recursively
    parquet_files = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
                
    assert len(parquet_files) > 0
