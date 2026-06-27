"""Reddit Public Collector — polls subreddit .json endpoints (no auth needed)."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = [
    "worldnews", "politics", "news", "technology",
    "india", "europe", "geopolitics"
]

HEADERS = {"User-Agent": "ghostsig-research-bot/0.1 (academic OSINT research)"}


async def fetch_subreddit_posts(subreddit: str, limit: int = 100) -> AsyncGenerator[dict, None]:
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    posts = data.get("data", {}).get("children", [])
    for post in posts:
        p = post.get("data", {})
        yield {
            "account_id": f"reddit:{p.get('author', 'unknown')}",
            "platform": "reddit",
            "event_type": "post",
            "event_ts": datetime.fromtimestamp(
                p.get("created_utc", 0), tz=timezone.utc
            ).isoformat(),
            "metadata": {
                "subreddit": subreddit,
                "post_id": p.get("id"),
                "score": p.get("score"),
                "num_comments": p.get("num_comments"),
                "is_self": p.get("is_self"),
                "domain": p.get("domain"),
            },
        }


async def run_collector(subreddits: list[str] = DEFAULT_SUBREDDITS, kafka_producer=None):
    """Poll all target subreddits every 5 minutes."""
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
            await asyncio.sleep(1)  # polite rate limiting
        await asyncio.sleep(300)  # 5-minute polling cycle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_collector())
