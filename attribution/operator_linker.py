"""Script to link campaigns to unique operators using shared behavioral fingerprint markers."""

import hashlib
import logging
import os
import sys
import numpy as np
import psycopg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("operator_linker")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_timing_overlap(bursts_a, bursts_b) -> float:
    if not bursts_a or not bursts_b:
        return 0.0
    matches = 0
    for f_a in bursts_a:
        for f_b in bursts_b:
            denom = max(f_a, f_b)
            if denom > 0 and abs(f_a - f_b) / denom <= 0.05:
                matches += 1
                break
    return matches / max(1, len(bursts_a), len(bursts_b))


def run_attribution(db_url: str):
    logger.info("Connecting to Postgres database for operator linking...")
    try:
        conn = psycopg.connect(db_url)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e
        
    with conn.cursor() as cur:
        # Ensure operator_id column exists on campaigns table
        cur.execute("ALTER TABLE campaigns ADD COLUMN IF NOT EXISTS operator_id TEXT;")
        conn.commit()

        # Query campaigns with confidence > 0.4
        cur.execute("SELECT campaign_id, label, confidence FROM campaigns WHERE confidence > 0.4;")
        campaigns = cur.fetchall()
        
    logger.info(f"Analyzing {len(campaigns)} campaigns with confidence > 0.4...")
    
    unique_operators = set()
    campaigns_processed = 0

    for campaign_id, label, confidence in campaigns:
        with conn.cursor() as cur:
            # Get member accounts
            cur.execute(
                """
                SELECT f.account_id, f.burst_freqs_hz, f.entropy_embedding, f.platform
                FROM campaign_accounts ca
                JOIN account_fingerprints f ON ca.account_id = f.account_id
                WHERE ca.campaign_id = %s;
                """,
                (campaign_id,)
            )
            members = cur.fetchall()
            
        if not members:
            logger.warning(f"Campaign {label} has no member accounts.")
            continue
            
        num_accounts = len(members)
        account_ids = [m[0] for m in members]
        
        # 1. Timing Overlap Percent
        # Compare pairwise, take mean overlap %
        timing_overlaps = []
        if num_accounts <= 1:
            timing_overlap_pct = 1.0
        else:
            for i in range(num_accounts):
                for j in range(i + 1, num_accounts):
                    bursts_i = members[i][1]
                    bursts_j = members[j][1]
                    timing_overlaps.append(compute_timing_overlap(bursts_i, bursts_j))
            timing_overlap_pct = float(np.mean(timing_overlaps)) if timing_overlaps else 1.0

        # 2. Entropy Overlap Percent
        # % of pairs with cosine(entropy_embedding_i, entropy_embedding_j) > 0.85
        entropy_overlaps = []
        if num_accounts <= 1:
            entropy_overlap_pct = 1.0
        else:
            for i in range(num_accounts):
                for j in range(i + 1, num_accounts):
                    emb_val_i = members[i][2]
                    emb_val_j = members[j][2]
                    
                    if emb_val_i is None or emb_val_j is None:
                        entropy_overlaps.append(0.0)
                        continue
                        
                    if isinstance(emb_val_i, str):
                        emb_i = np.array([float(x) for x in emb_val_i[1:-1].split(",")], dtype=np.float32)
                    else:
                        emb_i = np.array(emb_val_i, dtype=np.float32)
                        
                    if isinstance(emb_val_j, str):
                        emb_j = np.array([float(x) for x in emb_val_j[1:-1].split(",")], dtype=np.float32)
                    else:
                        emb_j = np.array(emb_val_j, dtype=np.float32)
                        
                    sim = cosine_similarity(emb_i, emb_j)
                    entropy_overlaps.append(1.0 if sim > 0.85 else 0.0)
            entropy_overlap_pct = float(np.mean(entropy_overlaps)) if entropy_overlaps else 1.0

        # 3. Device Echo Markers
        # count of shared device_hint values across >= 20% of accounts
        # Load raw_events for member accounts
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT account_id, platform, event_ts, device_hint
                FROM raw_events
                WHERE account_id = ANY(%s);
                """,
                (account_ids,)
            )
            events = cur.fetchall()
            
        # Group by account and collect user agents
        account_devices = {}
        for acc_id, _, _, hint in events:
            if acc_id not in account_devices:
                account_devices[acc_id] = set()
            if hint:
                account_devices[acc_id].add(hint)
                
        # Count accounts per device_hint
        device_counts = {}
        for hints in account_devices.values():
            for h in hints:
                device_counts[h] = device_counts.get(h, 0) + 1
                
        device_echo_markers = 0
        threshold_20_pct = 0.20 * num_accounts
        for hint, count in device_counts.items():
            if count >= threshold_20_pct:
                device_echo_markers += 1

        # 4. Cross Platform Sync
        # Count of account pairs active on 2+ platforms within 30-min windows
        # Group timestamps by account_id and platform
        account_platform_events = {}
        for acc_id, plat, ts, _ in events:
            if acc_id not in account_platform_events:
                account_platform_events[acc_id] = []
            account_platform_events[acc_id].append((plat, ts))
            
        cross_platform_sync = 0
        if num_accounts > 1:
            for i in range(num_accounts):
                acc_i = members[i][0]
                evs_i = account_platform_events.get(acc_i, [])
                for j in range(i + 1, num_accounts):
                    acc_j = members[j][0]
                    evs_j = account_platform_events.get(acc_j, [])
                    
                    # Check overlap
                    synced = False
                    for plat_i, ts_i in evs_i:
                        for plat_j, ts_j in evs_j:
                            if plat_i != plat_j:
                                if abs((ts_i - ts_j).total_seconds()) <= 1800:
                                    synced = True
                                    break
                        if synced:
                            break
                    if synced:
                        cross_platform_sync += 1

        # 5. Operator Fingerprint Hash
        # Bins: 0-33%=low, 34-66%=med, 67-100%=high
        def bin_pct(val):
            if val < 0.34:
                return "low"
            elif val < 0.67:
                return "med"
            return "high"
            
        def bin_count(val):
            if val == 0:
                return "low"
            elif val == 1:
                return "med"
            return "high"
            
        timing_bin = bin_pct(timing_overlap_pct)
        entropy_bin = bin_pct(entropy_overlap_pct)
        device_bin = bin_count(device_echo_markers)
        
        stable_str = f"timing:{timing_bin}|entropy:{entropy_bin}|device:{device_bin}"
        hash_val = hashlib.sha256(stable_str.encode("utf-8")).hexdigest()
        operator_id = f"operator_{hash_val[:12]}"
        unique_operators.add(operator_id)
        
        # Save metrics to evidence_json and update operator_id
        evidence_json = {
            "timing_overlap_pct": timing_overlap_pct,
            "entropy_overlap_pct": entropy_overlap_pct,
            "device_echo_markers": device_echo_markers,
            "cross_platform_sync": cross_platform_sync
        }
        
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE campaigns
                SET operator_id = %s, evidence_json = %s
                WHERE campaign_id = %s;
                """,
                (operator_id, psycopg.types.json.Json(evidence_json), campaign_id)
            )
            # Propagate to member accounts in account_fingerprints table
            cur.execute(
                """
                UPDATE account_fingerprints
                SET operator_id = %s
                WHERE account_id = ANY(%s);
                """,
                (operator_id, account_ids)
            )
        conn.commit()
        campaigns_processed += 1
        
    conn.close()
    
    print("\n" + "=" * 50)
    print("               OPERATOR ATTRIBUTION REPORT        ")
    print("=" * 50)
    print(f"Campaigns Processed:      {campaigns_processed}")
    print(f"Unique Operators Identified: {len(unique_operators)}")
    print("=" * 50 + "\n")
    return campaigns_processed, len(unique_operators)


def main():
    run_attribution(DATABASE_URL)


if __name__ == "__main__":
    main()
