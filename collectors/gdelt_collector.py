"""GDELT 2.0 Collector — streams global event metadata every 15 minutes."""

import asyncio
import csv
import io
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"


async def fetch_latest_gdelt_urls() -> list[str]:
    """Fetch the 3 latest GDELT update file URLs."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(GDELT_LASTUPDATE_URL)
        r.raise_for_status()
    urls = [line.split()[-1] for line in r.text.strip().splitlines() if line]
    return urls


async def stream_gdelt_events(url: str) -> AsyncGenerator[dict, None]:
    """Download and parse a GDELT CSV export, yielding normalized event dicts."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()

    # GDELT export files may be gzipped CSV
    content = r.content
    if url.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            content = zf.read(name)

    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")

    for row in reader:
        if len(row) < 10:
            continue
        try:
            yield {
                "account_id": f"gdelt:{row[5]}",   # Actor1Code
                "platform": "gdelt",
                "event_type": "mention",
                "event_ts": _parse_gdelt_ts(row[1]),
                "metadata": {
                    "event_code": row[26] if len(row) > 26 else None,
                    "source_url": row[57] if len(row) > 57 else None,
                    "actor1": row[5] if len(row) > 5 else None,
                    "actor2": row[15] if len(row) > 15 else None,
                },
            }
        except Exception as e:
            logger.debug(f"Skipping GDELT row: {e}")
            continue


def _parse_gdelt_ts(raw: str) -> str:
    """Convert GDELT YYYYMMDDHHMMSS to ISO 8601 UTC string."""
    try:
        dt = datetime.strptime(raw.strip()[:14], "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


async def run_collector(kafka_producer=None):
    """Main loop: poll GDELT every 15 minutes and push to Kafka or stdout."""
    while True:
        try:
            urls = await fetch_latest_gdelt_urls()
            for url in urls[:1]:  # events CSV only (first entry)
                async for event in stream_gdelt_events(url):
                    if kafka_producer:
                        kafka_producer.send("raw.events", event)
                    else:
                        print(event)
        except Exception as e:
            logger.error(f"GDELT collector error: {e}")
        await asyncio.sleep(900)  # 15 minutes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_collector())
