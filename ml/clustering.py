"""Clustering & Attribution Layer — HDBSCAN on behavioral fingerprint embeddings."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import hdbscan
    import umap
except ImportError:
    hdbscan = None
    umap = None


@dataclass
class CampaignCluster:
    cluster_id: int
    account_ids: list[str]
    confidence: float
    size: int
    centroid: Optional[np.ndarray] = None


def cluster_fingerprints(
    embeddings: np.ndarray,
    account_ids: list[str],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[CampaignCluster]:
    """Run HDBSCAN on fingerprint embeddings. Returns list of CampaignCluster."""
    if hdbscan is None:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    # UMAP dimensionality reduction before clustering (optional, improves quality)
    if umap and embeddings.shape[1] > 32:
        reducer = umap.UMAP(n_components=32, metric="cosine", random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        reduced = embeddings

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    probabilities = clusterer.probabilities_

    clusters = []
    for cid in set(labels):
        if cid == -1:  # noise
            continue
        mask = labels == cid
        cluster_accounts = [account_ids[i] for i, m in enumerate(mask) if m]
        confidence = float(np.mean(probabilities[mask]))
        centroid = reduced[mask].mean(axis=0)
        clusters.append(CampaignCluster(
            cluster_id=int(cid),
            account_ids=cluster_accounts,
            confidence=confidence,
            size=len(cluster_accounts),
            centroid=centroid,
        ))

    # Sort by size descending
    return sorted(clusters, key=lambda c: c.size, reverse=True)
