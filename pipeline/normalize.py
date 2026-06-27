"""Event Normalization Pipeline — consumes raw events, computes features, and updates database."""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from kafka import KafkaConsumer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
from dateutil.parser import parse

from features.temporal import temporal_stats
from features.entropy import entropy_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("normalize")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")


def extract_text(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    text = metadata.get("text")
    if text:
        return str(text)
    title = metadata.get("title", "")
    selftext = metadata.get("selftext", "")
    return f"{title} {selftext}".strip()


def fetch_account_events(cur: psycopg.Cursor, account_id: str, platform: str) -> List[Tuple[datetime, Dict[str, Any]]]:
    cur.execute(
        """
        SELECT event_ts, metadata 
        FROM raw_events 
        WHERE account_id = %s AND platform = %s 
        ORDER BY event_ts ASC
        """,
        (account_id, platform)
    )
    return cur.fetchall()


def merge_and_deduplicate(
    db_events: List[Tuple[datetime, Dict[str, Any]]], 
    batch_events: List[Dict[str, Any]]
) -> List[Tuple[datetime, Dict[str, Any]]]:
    merged = []
    seen = set()

    for ts, meta in db_events:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
            
        meta_str = json.dumps(meta, sort_keys=True)
        key = (ts.timestamp(), meta_str)
        if key not in seen:
            seen.add(key)
            merged.append((ts, meta))

    for ev in batch_events:
        ts = ev.get("event_ts")
        if isinstance(ts, str):
            try:
                ts = parse(ts)
            except Exception:
                ts = datetime.now(timezone.utc)
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        meta = ev.get("metadata", {}) or {}
        meta_str = json.dumps(meta, sort_keys=True)
        key = (ts.timestamp(), meta_str)
        if key not in seen:
            seen.add(key)
            merged.append((ts, meta))

    merged.sort(key=lambda x: x[0])
    return merged


def process_account_features(events: List[Tuple[datetime, Dict[str, Any]]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    timestamps = [ts.timestamp() for ts, _ in events]
    texts = [extract_text(meta) for _, meta in events]
    texts = [t for t in texts if t]
    
    t_stats = temporal_stats(timestamps)
    e_profile = entropy_profile(texts)
    
    return t_stats, e_profile


def upsert_fingerprint(
    cur: psycopg.Cursor, 
    account_id: str, 
    platform: str, 
    t_stats: Dict[str, Any], 
    e_profile: Dict[str, Any]
):
    cur.execute(
        """
        INSERT INTO account_fingerprints (
            account_id, platform, mean_interval_sec, std_interval_sec,
            coefficient_of_variation, burst_freqs_hz, event_count,
            char_entropy_mean, char_entropy_std, word_entropy_mean,
            word_entropy_std, entropy_sample_count, last_updated
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (account_id) DO UPDATE SET
            platform = EXCLUDED.platform,
            mean_interval_sec = EXCLUDED.mean_interval_sec,
            std_interval_sec = EXCLUDED.std_interval_sec,
            coefficient_of_variation = EXCLUDED.coefficient_of_variation,
            burst_freqs_hz = EXCLUDED.burst_freqs_hz,
            event_count = EXCLUDED.event_count,
            char_entropy_mean = EXCLUDED.char_entropy_mean,
            char_entropy_std = EXCLUDED.char_entropy_std,
            word_entropy_mean = EXCLUDED.word_entropy_mean,
            word_entropy_std = EXCLUDED.word_entropy_std,
            entropy_sample_count = EXCLUDED.entropy_sample_count,
            last_updated = NOW()
        """,
        (
            account_id,
            platform,
            t_stats.get("mean_interval_sec"),
            t_stats.get("std_interval_sec"),
            t_stats.get("coefficient_of_variation"),
            t_stats.get("burst_freqs_hz", []),
            t_stats.get("event_count", 0),
            e_profile.get("char_entropy", {}).get("mean"),
            e_profile.get("char_entropy", {}).get("std"),
            e_profile.get("word_entropy", {}).get("mean"),
            e_profile.get("word_entropy", {}).get("std"),
            e_profile.get("sample_count", 0)
        )
    )


def process_batch(records: List[Dict[str, Any]], conn: psycopg.Connection):
    # Group batch events by account_id and platform
    grouped_events = defaultdict(list)
    for rec in records:
        account_id = rec.get("account_id")
        platform = rec.get("platform")
        if account_id and platform:
            grouped_events[(account_id, platform)].append(rec)
            
    if not grouped_events:
        return
        
    parquet_rows = []
    
    with conn.cursor() as cur:
        for (account_id, platform), batch_evs in grouped_events.items():
            # 1. Fetch historical from Postgres
            db_evs = fetch_account_events(cur, account_id, platform)
            
            # 2. Merge and deduplicate
            all_evs = merge_and_deduplicate(db_evs, batch_evs)
            
            # 3. Calculate features
            t_stats, e_profile = process_account_features(all_evs)
            
            # 4. Upsert into account_fingerprints
            upsert_fingerprint(cur, account_id, platform, t_stats, e_profile)
            
            # 5. Build parquet row
            parquet_rows.append({
                "account_id": account_id,
                "platform": platform,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "mean_interval_sec": t_stats.get("mean_interval_sec"),
                "std_interval_sec": t_stats.get("std_interval_sec"),
                "coefficient_of_variation": t_stats.get("coefficient_of_variation"),
                "burst_freqs_hz": t_stats.get("burst_freqs_hz", []),
                "event_count": t_stats.get("event_count", 0),
                "char_entropy_mean": e_profile.get("char_entropy", {}).get("mean"),
                "char_entropy_std": e_profile.get("char_entropy", {}).get("std"),
                "word_entropy_mean": e_profile.get("word_entropy", {}).get("mean"),
                "word_entropy_std": e_profile.get("word_entropy", {}).get("std"),
                "entropy_sample_count": e_profile.get("sample_count", 0)
            })
            
        conn.commit()
        
    # Write to Parquet file
    if parquet_rows:
        df = pd.DataFrame(parquet_rows)
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        parquet_dir = os.path.join("data", "processed", f"features_{datetime.now(timezone.utc).strftime('%Y%m%d')}.parquet")
        
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=parquet_dir,
            partition_cols=['platform', 'date']
        )
        logger.info(f"Successfully processed {len(parquet_rows)} accounts. Features written to Parquet.")


def run_pipeline(once: bool = False):
    logger.info("Initializing database connection...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to Postgres database: {e}")
        sys.exit(1)

    logger.info(f"Connecting to Redpanda at {KAFKA_BOOTSTRAP_SERVERS}...")
    try:
        consumer = KafkaConsumer(
            "raw.events",
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="normalizer",
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            consumer_timeout_ms=5000 if once else -1
        )
    except Exception as e:
        logger.critical(f"Failed to connect to Redpanda: {e}")
        conn.close()
        sys.exit(1)

    try:
        if once:
            logger.info("Processing currently queued events and exiting...")
            empty_polls = 0
            max_empty_polls = 2
            
            while True:
                records_dict = consumer.poll(timeout_ms=2000)
                if not records_dict:
                    empty_polls += 1
                    if empty_polls >= max_empty_polls:
                        logger.info("No more events to consume. Exiting.")
                        break
                    continue
                
                empty_polls = 0
                batch_records = []
                for tp, messages in records_dict.items():
                    for msg in messages:
                        batch_records.append(msg.value)
                        
                if batch_records:
                    process_batch(batch_records, conn)
        else:
            logger.info("Starting normalizer pipeline continuous loop...")
            # We can aggregate in micro-batches
            while True:
                records_dict = consumer.poll(timeout_ms=5000)
                if records_dict:
                    batch_records = []
                    for tp, messages in records_dict.items():
                        for msg in messages:
                            batch_records.append(msg.value)
                    if batch_records:
                        try:
                            process_batch(batch_records, conn)
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user.")
    finally:
        consumer.close()
        conn.close()
        logger.info("Pipeline connections closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GhostSig Normalization Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="Process current queued events and exit")
    group.add_argument("--continuous", action="store_true", help="Run normalization pipeline continuously")
    args = parser.parse_args()
    
    run_pipeline(once=args.once)
