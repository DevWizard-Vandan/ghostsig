"""Synthetic dataset generator for bots and organic accounts."""

import argparse
import logging
import os
import random
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bot_generator")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")

# Bot templates and devices
BOT_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
]

BOT_PRODUCTS = ["cryptocoin", "superdeals", "vpnpro", "quickcash", "ecommercedeal"]
BOT_TOPICS = ["market trends", "breaking updates", "new software", "make money online", "passive income"]
BOT_USERS = ["@user123", "@smartbuyer", "@dealhunter", "@cryptoguru", "@financestash"]

BOT_TEXT_TEMPLATES = [
    "Check out this amazing deal on {product}! {url}",
    "Breaking news about {topic}! You must read this: {url}",
    "I agree with {user}, this is a very important issue. {url}",
    "Don't miss the latest updates on {topic}. More info at {url}",
    "Amazing tutorial on how to use {product}! Link: {url}",
    "Can someone help me with {topic}? Found this link: {url}",
    "What do you think about {topic}? Let's discuss: {url}",
    "Great analysis of {product} and {topic} here: {url}",
    "I cannot believe what happened with {topic}! Check it: {url}",
    "Highly recommended read on {product} development: {url}"
]

# Organic templates and devices
ORGANIC_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/119.0.0.0"
]

ORGANIC_WORDS = [
    "interesting", "discussion", "observation", "development", "community", "feedback",
    "project", "science", "nature", "environment", "travel", "culture", "history",
    "thought", "question", "comment", "perspective", "analysis", "sharing", "update"
]


class BotSimulator:
    def __init__(self, bot_id: int):
        self.account_id = f"synth_bot:user_{bot_id:05d}"
        self.platform = "synthetic"
        self.user_agent = random.choice(BOT_UA_POOL)
        
        # Periodic posting interval: mu in [30, 3600] seconds
        self.mu = random.uniform(30, 3600)
        self.event_count = random.randint(20, 200)

    def simulate(self, base_time: datetime):
        # Generate periodic intervals: Normal(mu, 0.05 * mu)
        intervals = np.random.normal(self.mu, 0.05 * self.mu, size=self.event_count - 1)
        intervals = np.clip(intervals, 1.0, None)
        
        # Calculate timestamps
        timestamps = [base_time]
        for dt in intervals:
            timestamps.append(timestamps[-1] + timedelta(seconds=float(dt)))
            
        # Calculate statistics
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        cv = std_interval / mean_interval if mean_interval > 0 else 0.0
        
        # Entropy properties
        char_entropy_mean = random.uniform(2.0, 3.5)
        char_entropy_std = random.uniform(0.05, 0.2)
        word_entropy_mean = random.uniform(1.0, 2.0)
        word_entropy_std = random.uniform(0.05, 0.2)
        
        # Generate raw events
        raw_events = []
        for ts in timestamps:
            template = random.choice(BOT_TEXT_TEMPLATES)
            text = template.format(
                product=random.choice(BOT_PRODUCTS),
                topic=random.choice(BOT_TOPICS),
                user=random.choice(BOT_USERS),
                url=f"https://short.url/t/{random.randint(1000, 9999)}"
            )
            raw_events.append({
                "account_id": self.account_id,
                "platform": self.platform,
                "event_type": "post",
                "event_ts": ts,
                "device_hint": self.user_agent,
                "metadata": {"text": text}
            })
            
        fingerprint = {
            "account_id": self.account_id,
            "platform": self.platform,
            "char_entropy_mean": char_entropy_mean,
            "char_entropy_std": char_entropy_std,
            "word_entropy_mean": word_entropy_mean,
            "word_entropy_std": word_entropy_std,
            "entropy_sample_count": self.event_count,
            "event_count": self.event_count,
            "mean_interval_sec": mean_interval,
            "std_interval_sec": std_interval,
            "coefficient_of_variation": cv
        }
        
        return fingerprint, raw_events


class OrganicSimulator:
    def __init__(self, organic_id: int):
        self.account_id = f"synth_organic:user_{organic_id:05d}"
        self.platform = "synthetic"
        
        # Organic user-agent (diverse device hints)
        self.user_agent = random.choice(ORGANIC_UA_POOL)
        
        # Irregular posting interval: LogNormal(6, 2) -> median ~ 400 sec, mean ~ 2200 sec, heavy tail
        self.event_count = random.randint(5, 50)

    def simulate(self, base_time: datetime):
        # Generate irregular intervals
        intervals = np.random.lognormal(mean=6.0, sigma=2.0, size=self.event_count - 1)
        intervals = np.clip(intervals, 1.0, None)
        
        # Calculate timestamps
        timestamps = [base_time]
        for dt in intervals:
            timestamps.append(timestamps[-1] + timedelta(seconds=float(dt)))
            
        # Calculate statistics
        mean_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        std_interval = float(np.std(intervals)) if len(intervals) > 0 else 0.0
        cv = std_interval / mean_interval if mean_interval > 0 else 0.0
        
        # Entropy properties
        char_entropy_mean = random.uniform(4.0, 5.5)
        char_entropy_std = random.uniform(0.2, 0.5)
        word_entropy_mean = random.uniform(2.5, 4.5)
        word_entropy_std = random.uniform(0.2, 0.5)
        
        # Generate raw events
        raw_events = []
        for ts in timestamps:
            # Create a more complex organic sentence
            sentence_words = random.sample(ORGANIC_WORDS, k=random.randint(3, 7))
            text = " ".join(sentence_words) + f". What are your opinions on this query?"
            raw_events.append({
                "account_id": self.account_id,
                "platform": self.platform,
                "event_type": "post",
                "event_ts": ts,
                "device_hint": self.user_agent,
                "metadata": {"text": text}
            })
            
        fingerprint = {
            "account_id": self.account_id,
            "platform": self.platform,
            "char_entropy_mean": char_entropy_mean,
            "char_entropy_std": char_entropy_std,
            "word_entropy_mean": word_entropy_mean,
            "word_entropy_std": word_entropy_std,
            "entropy_sample_count": self.event_count,
            "event_count": self.event_count,
            "mean_interval_sec": mean_interval,
            "std_interval_sec": std_interval,
            "coefficient_of_variation": cv
        }
        
        return fingerprint, raw_events


def generate_dataset(n_bots=500, n_organic=500):
    logger.info(f"Generating {n_bots} bots and {n_organic} organic accounts...")
    
    fingerprints = []
    all_raw_events = []
    
    base_time = datetime.now(timezone.utc) - timedelta(days=10)
    
    # 1. Simulate bots
    for i in range(n_bots):
        bot = BotSimulator(i)
        # Add random jitter to base_time per account
        start_ts = base_time + timedelta(seconds=random.randint(0, 100000))
        fp, evs = bot.simulate(start_ts)
        fingerprints.append(fp)
        all_raw_events.extend(evs)
        
    # 2. Simulate organic accounts
    for i in range(n_organic):
        org = OrganicSimulator(i)
        start_ts = base_time + timedelta(seconds=random.randint(0, 100000))
        fp, evs = org.simulate(start_ts)
        fingerprints.append(fp)
        all_raw_events.extend(evs)
        
    # 3. Database Insertion
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        logger.info("Clearing old synthetic accounts from database...")
        cur.execute("DELETE FROM raw_events WHERE account_id LIKE 'synth_%';")
        cur.execute("DELETE FROM account_fingerprints WHERE account_id LIKE 'synth_%';")
        conn.commit()
        
        # Batch insert raw_events using fast COPY
        logger.info(f"Batch copying {len(all_raw_events)} events to raw_events table...")
        with cur.copy(
            "COPY raw_events (account_id, platform, event_type, event_ts, device_hint, metadata) FROM STDIN"
        ) as copy:
            for ev in all_raw_events:
                copy.write_row((
                    ev["account_id"],
                    ev["platform"],
                    ev["event_type"],
                    ev["event_ts"].isoformat(),
                    ev["device_hint"],
                    psycopg.types.json.Json(ev["metadata"])
                ))
                
        # Batch insert account_fingerprints using fast COPY
        logger.info(f"Batch copying {len(fingerprints)} rows to account_fingerprints table...")
        with cur.copy(
            """
            COPY account_fingerprints (
                account_id, platform, char_entropy_mean, char_entropy_std,
                word_entropy_mean, word_entropy_std, entropy_sample_count,
                event_count, mean_interval_sec, std_interval_sec, coefficient_of_variation
            ) FROM STDIN
            """
        ) as copy:
            for fp in fingerprints:
                copy.write_row((
                    fp["account_id"],
                    fp["platform"],
                    fp["char_entropy_mean"],
                    fp["char_entropy_std"],
                    fp["word_entropy_mean"],
                    fp["word_entropy_std"],
                    fp["entropy_sample_count"],
                    fp["event_count"],
                    fp["mean_interval_sec"],
                    fp["std_interval_sec"],
                    fp["coefficient_of_variation"]
                ))
        conn.commit()
        
    conn.close()
    logger.info("Dataset generated and stored successfully.")
    return fingerprints


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic account fingerprints and events.")
    parser.add_argument("--bots", type=int, default=500, help="Number of bot accounts")
    parser.add_argument("--organic", type=int, default=500, help="Number of organic accounts")
    args = parser.parse_args()
    
    generate_dataset(n_bots=args.bots, n_organic=args.organic)


if __name__ == "__main__":
    main()
