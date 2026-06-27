"""GDELT 2.0 Collector — corrected column mapping per official spec."""

import asyncio
import argparse
import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx
from kafka import KafkaProducer

logger = logging.getLogger(__name__)

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT 2.0 Event Export columns (0-indexed):
# 0:GLOBALEVENTID 1:SQLDATE 2:MonthYear 3:Year 4:FractionDate
# 5:Actor1Code 6:Actor1Name 7:Actor1CountryCode 8:Actor1KnownGroupCode 9:Actor1EthnicCode 10:Actor1Religion1Code 11:Actor1Religion2Code 12:Actor1Type1Code 13:Actor1Type2Code 14:Actor1Type3Code
# 15:Actor2Code 16:Actor2Name 17:Actor2CountryCode 18:Actor2KnownGroupCode 19:Actor2EthnicCode 20:Actor2Religion1Code 21:Actor2Religion2Code 22:Actor2Type1Code 23:Actor2Type2Code 24:Actor2Type3Code
# 25:IsRootEvent 26:EventCode 27:EventBaseCode 28:EventRootCode 29:QuadClass 30:GoldsteinScale
# 31:NumMentions 32:NumSources 33:NumArticles 34:AvgTone
# 35:Actor1Geo_Type 36:Actor1Geo_Fullname 37:Actor1Geo_CountryCode 38:Actor1Geo_ADM1Code 39:Actor1Geo_Lat 40:Actor1Geo_Long 41:Actor1Geo_FeatureID
# 42:Actor2Geo_Type 43:Actor2Geo_Fullname 44:Actor2Geo_CountryCode 45:Actor2Geo_ADM1Code 46:Actor2Geo_Lat 47:Actor2Geo_Long 48:Actor2Geo_FeatureID
# 49:ActionGeo_Type 50:ActionGeo_Fullname 51:ActionGeo_CountryCode 52:ActionGeo_ADM1Code 53:ActionGeo_Lat 54:ActionGeo_Long 55:ActionGeo_FeatureID
# 56:DATEADDED 57:SOURCEURL

async def fetch_latest_gdelt_urls() -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(GDELT_LASTUPDATE_URL)
        r.raise_for_status()
    urls = [line.split()[-1] for line in r.text.strip().splitlines() if line]
    return urls

async def stream_gdelt_events(url: str) -> AsyncGenerator[dict, None]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
        r.raise_for_status()
    content = r.content
    if url.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = zf.namelist()[0]
            content = zf.read(name)
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 58:
            continue
        try:
            yield {
                "account_id": f"gdelt:{row[5]}",   # Actor1Code
                "platform": "gdelt",
                "event_type": "mention",
                "event_ts": _parse_gdelt_ts(row[1]),  # SQLDATE
                "metadata": {
                    "event_code": row[26],           # EventCode (CAMEO)
                    "source_url": row[57],           # SOURCEURL
                    "actor1": row[5],                # Actor1Code
                    "actor2": row[15],               # Actor2Code
                },
            }
        except Exception as e:
            logger.debug(f"Skipping GDELT row: {e}")
            continue

def _parse_gdelt_ts(raw: str) -> str:
    try:
        dt = datetime.strptime(raw.strip()[:14], "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

async def run_collector(kafka_producer=None, once: bool = False):
    while True:
        try:
            urls = await fetch_latest_gdelt_urls()
            for url in urls[:1]:
                async for event in stream_gdelt_events(url):
                    if kafka_producer:
                        kafka_producer.send("raw.events", event)
                    else:
                        print(event)
        except Exception as e:
            logger.error(f"GDELT collector error: {e}")
        
        if once:
            logger.info("Run once specified. Exiting GDELT collector.")
            break
        await asyncio.sleep(900)

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
