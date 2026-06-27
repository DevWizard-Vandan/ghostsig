"""Twitter/X Public Collector — uses only public, unauthenticated endpoints.

Note: Twitter v2 search/recent requires a Bearer token (free tier).
This collector uses the FREE tier Academic/Basic app bearer token.
NO user PII is collected — only public post metadata and timestamps.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN", "")
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

DEFAULT_QUERIES = [
    "election lang:en -is:retweet",
    "disinformation lang:en -is:retweet",
    "breaking news lang:en -is:retweet",
]


async def fetch_recent_tweets(query: str, max_results: int = 100) -> AsyncGenerator[dict, None]:
    if not TWITTER_BEARER:
        logger.warning("TWITTER_BEARER_TOKEN not set — skipping Twitter collector")
        return

    headers = {"Authorization": f"Bearer {TWITTER_BEARER}"}
    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,author_id,public_metrics,lang,source",
        "expansions": "author_id",
        "user.fields": "public_metrics,created_at",
    }

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        r = await client.get(SEARCH_URL, params=params)
        r.raise_for_status()
        data = r.json()

    for tweet in data.get("data", []):
        yield {
            "account_id": f"twitter:{tweet.get('author_id')}",
            "platform": "twitter",
            "event_type": "tweet",
            "event_ts": tweet.get("created_at", datetime.now(timezone.utc).isoformat()),
            "metadata": {
                "tweet_id": tweet.get("id"),
                "lang": tweet.get("lang"),
                "source": tweet.get("source"),
                "like_count": tweet.get("public_metrics", {}).get("like_count"),
                "retweet_count": tweet.get("public_metrics", {}).get("retweet_count"),
            },
        }


async def run_collector(queries: list[str] = DEFAULT_QUERIES, kafka_producer=None):
    while True:
        for q in queries:
            try:
                async for event in fetch_recent_tweets(q):
                    if kafka_producer:
                        kafka_producer.send("raw.events", event)
                    else:
                        print(event)
            except Exception as e:
                logger.error(f"Twitter collector error: {e}")
            await asyncio.sleep(2)
        await asyncio.sleep(300)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_collector())
