"""Raw Events Consumer — reads raw.events from Redpanda and inserts into PostgreSQL."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

from kafka import KafkaConsumer
import psycopg
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("raw_to_postgres")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")

# Schema Validation using Pydantic
class RawEventSchema(BaseModel):
    account_id: str
    platform: str
    event_type: str = "unknown"
    event_ts: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

def process_message(val: Dict[str, Any], cur: psycopg.Cursor) -> bool:
    try:
        # Validate schema
        event = RawEventSchema(**val)
        
        # Ensure event_ts is timezone-aware
        ts = event.event_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
            
        cur.execute(
            """
            INSERT INTO raw_events (account_id, platform, event_type, event_ts, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (platform, account_id, event_ts) DO NOTHING
            """,
            (
                event.account_id,
                event.platform,
                event.event_type,
                ts,
                json.dumps(event.metadata)
            )
        )
        return True
    except ValidationError as ve:
        logger.warning(f"Validation failed for event: {val}. Errors: {ve}")
        return False
    except Exception as e:
        logger.error(f"Error inserting event into database: {e}")
        raise e

def run_consumer(once: bool = False):
    logger.info(f"Connecting to database at: {DATABASE_URL.split('@')[-1]}")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)
        
    logger.info(f"Connecting to Redpanda at: {KAFKA_BOOTSTRAP_SERVERS}")
    try:
        consumer = KafkaConsumer(
            "raw.events",
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="raw-events-db-consumer",
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            consumer_timeout_ms=5000 if once else -1
        )
    except Exception as e:
        logger.critical(f"Failed to connect to Redpanda: {e}")
        conn.close()
        sys.exit(1)

    try:
        if once:
            logger.info("Consuming currently queued messages and exiting...")
            empty_polls = 0
            max_empty_polls = 2
            inserted_count = 0
            
            while True:
                # Poll with 2-second timeout
                records = consumer.poll(timeout_ms=2000)
                if not records:
                    empty_polls += 1
                    if empty_polls >= max_empty_polls:
                        logger.info("No more messages found in Redpanda raw.events.")
                        break
                    continue
                
                empty_polls = 0
                with conn.cursor() as cur:
                    for tp, messages in records.items():
                        for msg in messages:
                            if process_message(msg.value, cur):
                                inserted_count += 1
                    conn.commit()
            
            logger.info(f"Successfully processed batch ingestion. Approx events inserted/ignored: {inserted_count}")
        else:
            logger.info("Starting raw.events consumer loop...")
            for msg in consumer:
                try:
                    with conn.cursor() as cur:
                        if process_message(msg.value, cur):
                            conn.commit()
                except Exception as e:
                    logger.error(f"Message processing error, rolling back: {e}")
                    conn.rollback()
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    finally:
        consumer.close()
        conn.close()
        logger.info("Closed database and Redpanda connections.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume raw events from Redpanda and ingest to Postgres")
    parser.add_argument("--once", action="store_true", help="Consume all currently queued events and exit")
    args = parser.parse_args()
    
    run_consumer(once=args.once)
