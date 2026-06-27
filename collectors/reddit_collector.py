"""Reddit Collector — uses Pushshift archive (no auth, no 403)."""

import asyncio
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx
from kafka import KafkaProducer

logger = logging.getLogger(__name__)

PUSHSHIFT_URL = "https://api.pushshift.io/reddit/submission/search"
DEFAULT_SUBREDDITS = ["worldnews", "politics", "news", "technology", "india", "europe", "geopolitics"]

HEADERS = {"User-Agent": "ghostsig-research-bot/0.1 (academic OSINT research)"}

async def fetch_subreddit_posts(subreddit: str, limit: int = 100) -> AsyncGenerator[dict, None]:
    params = {
        "subreddit": subreddit,
        "size": min(limit, 100),
        "sort": "desc",
        "sort_type": "created_utc",
        "fields": "id,author,created_utc,score,num_comments,is_self,domain,title,selftext",
    }
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        r = await client.get(PUSHSHIFT_URL, params=params)
        r.raise_for_status()
        data = r.json()

    for post in data.get("data", []):
        text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
        yield {
            "account_id": f"reddit:{post.get('author', 'unknown')}",
            "platform": "reddit",
            "event_type": "post",
            "event_ts": datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc).isoformat(),
            "metadata": {
                "subreddit": subreddit,
                "post_id": post.get("id"),
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "is_self": post.get("is_self"),
                "domain": post.get("domain"),
                "text": text[:5000],  # truncate for storage
            },
        }

async def run_collector(subreddits: list[str] = DEFAULT_SUBREDDITS, kafka_producer=None, once: bool = False):
    while True:
        for sub in subreddits:
            try:
                async for event in fetch_subreddit_posts(sub):
                    if kafka_producer:
                        kafka_producer.send("raw.events", event)
                    else:
                        print(event)
            except Exception as e:
                logger.error(f"Reddit collector error ({sub}): {e}")
            await asyncio.sleep(1)
            
            if once:
                break
                
        if once:
            logger.info("Run once specified. Exiting Reddit collector.")
            break
        await asyncio.sleep(300)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        logger.info(f"Connected to Redpanda at {bootstrap_servers}")
    except Exception as e:
        logger.warning(f"Could not connect to Redpanda ({e}). Printing events instead.")
        producer = None
        
    asyncio.run(run_collector(kafka_producer=producer, once=args.once))
    
    if producer:
        producer.flush()
        producer.close()
