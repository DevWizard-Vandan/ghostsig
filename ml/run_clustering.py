"""Clustering script to group fused account embeddings into campaigns using UMAP and HDBSCAN."""

import logging
import os
import sys
import uuid
import numpy as np
import pandas as pd
import psycopg
import umap
import hdbscan
from sklearn.metrics import silhouette_score
import plotly.express as px

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_clustering")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ghostsig:ghostsig_dev@localhost:5432/ghostsig")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def fetch_fused_embeddings(db_url: str):
    logger.info("Fetching fused embeddings from Postgres...")
    try:
        conn = psycopg.connect(db_url)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        raise e

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT account_id, platform, fused_embedding 
            FROM account_fingerprints 
            WHERE fused_embedding IS NOT NULL;
            """
        )
        rows = cur.fetchall()
        
    conn.close()
    
    account_ids = []
    platforms = []
    embeddings = []
    
    for r in rows:
        account_id = r[0]
        platform = r[1]
        emb_val = r[2]
        
        # Parse pgvector string representation: e.g. "[0.1,0.2,...]"
        if isinstance(emb_val, str):
            emb = np.array([float(x) for x in emb_val[1:-1].split(",")], dtype=np.float32)
        else:
            emb = np.array(emb_val, dtype=np.float32)
            
        # Skip if all zeros
        if np.all(emb == 0.0):
            continue
            
        account_ids.append(account_id)
        platforms.append(platform)
        embeddings.append(emb)
        
    if not embeddings:
        return [], [], np.empty((0, 256))
        
    return account_ids, platforms, np.array(embeddings, dtype=np.float32)


def main():
    # 1. Fetch fused embeddings
    account_ids, platforms, embeddings = fetch_fused_embeddings(DATABASE_URL)
    if len(embeddings) < 3:
        logger.error("Not enough fused embeddings in database to run clustering.")
        sys.exit(1)

    logger.info(f"Loaded {len(embeddings)} fused embeddings.")

    # 2. UMAP dimensionality reduction
    n_comp = 32
    init_mode = "spectral"
    if len(embeddings) <= 32:
        n_comp = max(2, len(embeddings) - 2)
        init_mode = "random"
        logger.info(f"Small dataset detected. Setting UMAP n_components={n_comp}, init={init_mode}")
    else:
        logger.info("Running UMAP (n_components=32, metric=cosine)...")
        
    reducer_32 = umap.UMAP(
        n_components=n_comp,
        metric="cosine",
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        init=init_mode,
        random_state=42
    )
    umap_embeddings = reducer_32.fit_transform(embeddings)

    # 3. HDBSCAN clustering
    logger.info("Running HDBSCAN (min_cluster_size=3, min_samples=2)...")
    # min_cluster_size must be <= size of dataset
    min_cluster_size = min(3, len(embeddings))
    min_samples = min(2, len(embeddings))
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True
    )
    labels = clusterer.fit_predict(umap_embeddings)

    # 4. Compute metrics
    unique_labels = set(labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    n_noise = list(labels).count(-1)
    noise_fraction = float(n_noise) / len(labels)
    
    non_noise_mask = labels != -1
    sil_score = 0.0
    if n_clusters >= 2 and np.sum(non_noise_mask) >= 2:
        sil_score = float(silhouette_score(umap_embeddings[non_noise_mask], labels[non_noise_mask], metric="cosine"))

    print("\n=== CLUSTERING METRICS ===")
    print(f"Total accounts clustered: {len(embeddings)}")
    print(f"Number of clusters found: {n_clusters}")
    print(f"Noise fraction: {noise_fraction:.2%}")
    print(f"Silhouette score (non-noise): {sil_score:.4f}")

    # Cluster sizes report
    cluster_counts = pd.Series(labels).value_counts()
    for cid, count in cluster_counts.items():
        if cid == -1:
            print(f"Noise (-1): {count} accounts")
        else:
            print(f"Cluster {cid}: {count} accounts")

    # 5. Populate campaigns and campaign_accounts in Postgres
    logger.info("Connecting to database to write campaign results...")
    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:
        logger.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
        logger.info("Clearing old campaigns and campaign_accounts...")
        cur.execute("DELETE FROM campaign_accounts;")
        cur.execute("DELETE FROM campaigns;")
        conn.commit()

        # Iterate over non-noise clusters
        for cid in unique_labels:
            if cid == -1:
                continue
                
            cluster_indices = np.where(labels == cid)[0]
            cluster_accounts = [account_ids[idx] for idx in cluster_indices]
            cluster_platforms = [platforms[idx] for idx in cluster_indices]
            cluster_embs = embeddings[cluster_indices]
            
            # Compute centroid (mean embedding of the cluster)
            centroid = np.mean(cluster_embs, axis=0)
            
            # Compute similarities to centroid
            similarities = [cosine_similarity(emb, centroid) for emb in cluster_embs]
            
            # Platforms list and count
            unique_platforms = list(set(cluster_platforms))
            
            # Fetch first_seen and last_seen timestamps from raw_events
            cur.execute(
                """
                SELECT MIN(event_ts), MAX(event_ts) 
                FROM raw_events 
                WHERE account_id = ANY(%s);
                """,
                (cluster_accounts,)
            )
            first_seen, last_seen = cur.fetchone()
            
            # Campaign info
            campaign_id = str(uuid.uuid4())
            label = f"Campaign {cid}"
            confidence = 0.85
            
            evidence_json = {
                "centroid_stability": float(np.mean(similarities)),
                "platforms": unique_platforms,
                "created_by": "hdbscan_pipeline"
            }
            
            # Insert campaign
            cur.execute(
                """
                INSERT INTO campaigns (
                    campaign_id, label, confidence, account_count, 
                    platform_count, platform_list, first_seen, last_seen, evidence_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    campaign_id, label, confidence, len(cluster_accounts),
                    len(unique_platforms), unique_platforms, first_seen, last_seen,
                    psycopg.types.json.Json(evidence_json)
                )
            )
            
            # Insert memberships
            for acc_id, sim in zip(cluster_accounts, similarities):
                cur.execute(
                    """
                    INSERT INTO campaign_accounts (campaign_id, account_id, similarity) 
                    VALUES (%s, %s, %s);
                    """,
                    (campaign_id, acc_id, sim)
                )
                
        conn.commit()
    conn.close()
    logger.info("Successfully updated campaigns and campaign_accounts tables.")

    # 6. Save UMAP 2D Plot for visualization only
    logger.info("Running UMAP (n_components=2) for visualization...")
    reducer_2 = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        random_state=42
    )
    vis_embeddings = reducer_2.fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "UMAP Dimension 1": vis_embeddings[:, 0],
        "UMAP Dimension 2": vis_embeddings[:, 1],
        "Cluster ID": [f"Cluster {lbl}" if lbl != -1 else "Noise" for lbl in labels],
        "Account ID": account_ids,
        "Platform": platforms
    })

    fig = px.scatter(
        df_plot,
        x="UMAP Dimension 1",
        y="UMAP Dimension 2",
        color="Cluster ID",
        hover_data=["Account ID", "Platform"],
        title="GhostSig Fused Embeddings Cluster Visualization (UMAP + HDBSCAN)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.D3
    )

    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    html_path = os.path.join(docs_dir, "umap_clusters.html")
    fig.write_html(html_path)
    logger.info(f"UMAP visualization saved to {html_path}")


if __name__ == "__main__":
    main()
