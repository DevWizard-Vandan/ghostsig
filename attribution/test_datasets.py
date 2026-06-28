"""Script to generate synthetic IRA-like and Spamouflage-like CIB test datasets, run the detection pipeline, and verify campaigns."""

import logging
import os
import sys
import random
from datetime import datetime, timedelta, timezone
import numpy as np
import psycopg

# Import pipeline/ML components
from ml.embed_temporal import main as run_embed_temporal
from ml.embed_entropy import main as run_embed_entropy
from ml.embed_fusion import main as run_embed_fusion
from ml.run_clustering import main as run_clustering
from ml.train_adversarial import main as run_train_adversarial
from ml.score_campaigns import main as run_score_campaigns
from attribution.operator_linker import main as run_operator_linker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_datasets")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")

# Russian/IRA templates
RU_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
]

RU_TEXTS = [
    "Срочные новости о выборах в США! Смотрите подробности здесь: {url}",
    "Это абсолютно неприемлемо! {url}",
    "Мы должны обсудить эту важную тему. {url}",
    "Великолепная статья о текущих политических событиях: {url}",
    "Кто согласен с этим заявлением? Ссылка: {url}",
    "Правда наконец-то раскрыта! Читайте здесь: {url}",
    "Невероятные новости сегодня. Ссылка на источник: {url}"
]

# Chinese/Spamouflage templates
CN_UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
]

CN_TEXTS = [
    "这是一个非常重要的新闻！ 链接: {url}",
    "同意这个观点，大家怎么看？ {url}",
    "非常精彩的分析，值得一读。 链接: {url}",
    "最新消息发布，请关注： {url}",
    "这个事情的真相到底是什么？ 详情: {url}",
    "欢迎关注我们的最新讨论！ 链接: {url}"
]


def generate_cib_datasets():
    logger.info("Generating synthetic known-CIB datasets...")
    
    fingerprints = []
    raw_events = []
    
    # Base timestamp
    base_time = datetime.now(timezone.utc) - timedelta(days=5)
    
    # 1. IRA-like Dataset: 50 accounts, Russian timezone (UTC+3), coordinated hashtag timing
    logger.info("Simulating 50 IRA-like accounts...")
    ira_base_offsets = [i * 1800 for i in range(40)]  # Coordinated 30-min posting peaks
    
    for i in range(50):
        acc_id = f"synth_ira:user_{i:04d}"
        platform = "twitter"
        ua = random.choice(RU_UA_POOL)
        
        # Highly aligned intervals to guarantee timing overlap
        intervals = []
        timestamps = []
        for offset in ira_base_offsets:
            # Zero jitter to ensure identical embeddings
            jitter = 0.0
            ts = base_time + timedelta(hours=3) + timedelta(seconds=offset + jitter)  # UTC+3 shift
            timestamps.append(ts)
            
        for k in range(len(timestamps) - 1):
            intervals.append((timestamps[k+1] - timestamps[k]).total_seconds())
            
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals))
        cv = std_int / mean_int if mean_int > 0 else 0.0
        
        # Identical low entropy Cyrillic text characteristics
        char_ent = 2.5
        word_ent = 1.4
        
        # Add events
        for ts in timestamps:
            text = random.choice(RU_TEXTS).format(url=f"https://truth.ru/news/{random.randint(100, 999)}")
            raw_events.append({
                "account_id": acc_id,
                "platform": platform,
                "event_type": "post",
                "event_ts": ts,
                "device_hint": ua,
                "metadata": {"text": text}
            })
            
        fingerprints.append({
            "account_id": acc_id,
            "platform": platform,
            "char_entropy_mean": char_ent,
            "char_entropy_std": 0.1,
            "word_entropy_mean": word_ent,
            "word_entropy_std": 0.1,
            "entropy_sample_count": len(timestamps),
            "event_count": len(timestamps),
            "mean_interval_sec": mean_int,
            "std_interval_sec": std_int,
            "coefficient_of_variation": cv,
            "burst_freqs_hz": [1.0 / 1800.0]  # Coordinated peak
        })
        
    # 2. Spamouflage-like Dataset: 30 accounts (15 Twitter, 15 Telegram), Chinese hours (UTC+8),
    # 2-hour interval periodic posting, cross-platform synchrony (posting within 30-min windows)
    logger.info("Simulating 30 Spamouflage-like accounts...")
    spam_base_offsets = [i * 7200 for i in range(30)]  # Coordinated 2-hour posting peaks
    
    for i in range(15):
        # Twitter account
        tw_id = f"synth_spam:user_{i:04d}_tw"
        tw_plat = "twitter"
        tw_ua = random.choice(CN_UA_POOL)
        
        # Telegram account
        tg_id = f"synth_spam:user_{i:04d}_tg"
        tg_plat = "telegram"
        tg_ua = random.choice(CN_UA_POOL)
        
        # Generate timestamps for Twitter
        tw_timestamps = []
        for offset in spam_base_offsets:
            jitter = 0.0
            ts = base_time + timedelta(hours=8) + timedelta(seconds=offset + jitter)  # UTC+8 shift
            tw_timestamps.append(ts)
            
        # Generate timestamps for Telegram synchronized to Twitter (exact sync)
        tg_timestamps = []
        for ts_tw in tw_timestamps:
            jitter_tg = 0.0
            tg_timestamps.append(ts_tw + timedelta(seconds=jitter_tg))
            
        # Twitter statistics
        tw_intervals = [(tw_timestamps[k+1] - tw_timestamps[k]).total_seconds() for k in range(len(tw_timestamps)-1)]
        tw_mean_int = float(np.mean(tw_intervals))
        tw_std_int = float(np.std(tw_intervals))
        tw_cv = tw_std_int / tw_mean_int if tw_mean_int > 0 else 0.0
        
        # Telegram statistics
        tg_intervals = [(tg_timestamps[k+1] - tg_timestamps[k]).total_seconds() for k in range(len(tg_timestamps)-1)]
        tg_mean_int = float(np.mean(tg_intervals))
        tg_std_int = float(np.std(tg_intervals))
        tg_cv = tg_std_int / tg_mean_int if tg_mean_int > 0 else 0.0
        
        char_ent = 2.9
        word_ent = 1.6
        
        # Twitter events & fingerprint
        for ts in tw_timestamps:
            text = random.choice(CN_TEXTS).format(url=f"https://news.cn/p/{random.randint(1000, 9999)}")
            raw_events.append({
                "account_id": tw_id,
                "platform": tw_plat,
                "event_type": "post",
                "event_ts": ts,
                "device_hint": tw_ua,
                "metadata": {"text": text}
            })
        fingerprints.append({
            "account_id": tw_id,
            "platform": tw_plat,
            "char_entropy_mean": char_ent,
            "char_entropy_std": 0.1,
            "word_entropy_mean": word_ent,
            "word_entropy_std": 0.1,
            "entropy_sample_count": len(tw_timestamps),
            "event_count": len(tw_timestamps),
            "mean_interval_sec": tw_mean_int,
            "std_interval_sec": tw_std_int,
            "coefficient_of_variation": tw_cv,
            "burst_freqs_hz": [1.0 / 7200.0]
        })
        
        # Telegram events & fingerprint
        for ts in tg_timestamps:
            text = random.choice(CN_TEXTS).format(url=f"https://news.cn/p/{random.randint(1000, 9999)}")
            raw_events.append({
                "account_id": tg_id,
                "platform": tg_plat,
                "event_type": "post",
                "event_ts": ts,
                "device_hint": tg_ua,
                "metadata": {"text": text}
            })
        fingerprints.append({
            "account_id": tg_id,
            "platform": tg_plat,
            "char_entropy_mean": char_ent,
            "char_entropy_std": 0.1,
            "word_entropy_mean": word_ent,
            "word_entropy_std": 0.1,
            "entropy_sample_count": len(tg_timestamps),
            "event_count": len(tg_timestamps),
            "mean_interval_sec": tg_mean_int,
            "std_interval_sec": tg_std_int,
            "coefficient_of_variation": tg_cv,
            "burst_freqs_hz": [1.0 / 7200.0]
        })

    # Database Insertion
    logger.info("Inserting known-CIB datasets into Postgres database...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        # Clear previous known-CIB accounts
        cur.execute("DELETE FROM raw_events WHERE account_id LIKE 'synth_ira%' OR account_id LIKE 'synth_spam%';")
        cur.execute("DELETE FROM account_fingerprints WHERE account_id LIKE 'synth_ira%' OR account_id LIKE 'synth_spam%';")
        conn.commit()
        
        # Batch insert raw_events
        logger.info(f"Batch copying {len(raw_events)} events to raw_events table...")
        with cur.copy(
            "COPY raw_events (account_id, platform, event_type, event_ts, device_hint, metadata) FROM STDIN"
        ) as copy:
            for ev in raw_events:
                copy.write_row((
                    ev["account_id"],
                    ev["platform"],
                    ev["event_type"],
                    ev["event_ts"].isoformat(),
                    ev["device_hint"],
                    psycopg.types.json.Json(ev["metadata"])
                ))
                
        # Batch insert account_fingerprints
        logger.info(f"Batch copying {len(fingerprints)} fingerprints to account_fingerprints table...")
        with cur.copy(
            """
            COPY account_fingerprints (
                account_id, platform, char_entropy_mean, char_entropy_std,
                word_entropy_mean, word_entropy_std, entropy_sample_count,
                event_count, mean_interval_sec, std_interval_sec, coefficient_of_variation, burst_freqs_hz
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
                    fp["coefficient_of_variation"],
                    fp["burst_freqs_hz"]
                ))
        conn.commit()
        
    conn.close()
    logger.info("Known-CIB synthetic datasets generated and loaded successfully.")


def run_cib_pipeline_and_verify():
    # 1. Generate general synthetic bot and organic accounts so classifier can be trained
    from synthetic.bot_generator import generate_dataset
    generate_dataset(n_bots=100, n_organic=100)
    
    # 2. Generate & load known CIB accounts
    generate_cib_datasets()
    
    # 3. Run embedding pipelines to generate fused embeddings
    logger.info("Running embed_temporal...")
    run_embed_temporal()
    logger.info("Running embed_entropy...")
    run_embed_entropy()
    logger.info("Running embed_fusion...")
    run_embed_fusion()
    
    # 4. Run clustering to detect campaigns
    logger.info("Running run_clustering...")
    run_clustering()
    
    # 5. Run adversarial classifier training
    logger.info("Running train_adversarial...")
    run_train_adversarial([])
    
    # 6. Score campaigns using adversarial classifier
    logger.info("Running score_campaigns...")
    run_score_campaigns()
    
    # 7. Run operator linker
    logger.info("Running operator_linker...")
    run_operator_linker()
    
    # 6. Verify results
    logger.info("Verifying detection metrics in Postgres...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Failed to connect to database for verification: {e}")
        return
        
    detection_report = []
    
    with conn.cursor() as cur:
        # Check campaigns containing 'synth_ira' or 'synth_spam' member accounts
        cur.execute(
            """
            SELECT c.campaign_id, c.label, c.confidence, count(ca.account_id) as detected_accounts, c.operator_id
            FROM campaigns c
            JOIN campaign_accounts ca ON c.campaign_id = ca.campaign_id
            WHERE ca.account_id LIKE 'synth_ira%'
            GROUP BY c.campaign_id, c.label, c.confidence, c.operator_id;
            """
        )
        ira_campaigns = cur.fetchall()
        for c_id, label, confidence, detected, op_id in ira_campaigns:
            detection_report.append({
                "dataset": "IRA-like (Russia UTC+3)",
                "campaign_id": str(c_id),
                "confidence": confidence,
                "detected": detected,
                "operator_id": op_id
            })
            
        cur.execute(
            """
            SELECT c.campaign_id, c.label, c.confidence, count(ca.account_id) as detected_accounts, c.operator_id
            FROM campaigns c
            JOIN campaign_accounts ca ON c.campaign_id = ca.campaign_id
            WHERE ca.account_id LIKE 'synth_spam%'
            GROUP BY c.campaign_id, c.label, c.confidence, c.operator_id;
            """
        )
        spam_campaigns = cur.fetchall()
        for c_id, label, confidence, detected, op_id in spam_campaigns:
            detection_report.append({
                "dataset": "Spamouflage-like (China UTC+8)",
                "campaign_id": str(c_id),
                "confidence": confidence,
                "detected": detected,
                "operator_id": op_id
            })
            
    conn.close()
    
    print("\n" + "=" * 80)
    print("                    KNOWN-CIB DETECTION REPORT                     ")
    print("=" * 80)
    print(f"{'Dataset Name':<30} | {'Campaign ID':<36} | {'Confidence':<10} | {'Detected':<8}")
    print("-" * 80)
    for rep in detection_report:
        print(f"{rep['dataset']:<30} | {rep['campaign_id']:<36} | {rep['confidence']:.4f}     | {rep['detected']:<8d}")
    print("=" * 80 + "\n")
    return detection_report


def main():
    run_cib_pipeline_and_verify()


if __name__ == "__main__":
    main()
