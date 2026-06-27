"""Temporal feature extraction — inter-event intervals, burst detection, periodicity."""

import numpy as np
from datetime import datetime


def compute_inter_event_intervals(timestamps: list[float]) -> np.ndarray:
    """Given sorted Unix timestamps, return inter-event intervals in seconds."""
    ts = np.array(sorted(timestamps))
    if len(ts) < 2:
        return np.array([])
    return np.diff(ts)


def burst_periodicity(intervals: np.ndarray, top_k: int = 3) -> list[float]:
    """FFT-based burst periodicity: return top-k dominant frequencies (Hz)."""
    if len(intervals) < 4:
        return []
    fft = np.abs(np.fft.rfft(intervals))
    freqs = np.fft.rfftfreq(len(intervals))
    top_indices = np.argsort(fft)[-top_k:][::-1]
    return [float(freqs[i]) for i in top_indices if freqs[i] > 0]


def temporal_stats(timestamps: list[float]) -> dict:
    """Compute statistical summary of posting behavior for an account."""
    intervals = compute_inter_event_intervals(timestamps)
    if len(intervals) == 0:
        return {
            "mean_interval_sec": None,
            "std_interval_sec": None,
            "coefficient_of_variation": None,
            "burst_freqs_hz": [],
            "event_count": len(timestamps),
        }
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    cv = std / mean if mean > 0 else 0.0  # coefficient of variation
    return {
        "mean_interval_sec": mean,
        "std_interval_sec": std,
        "coefficient_of_variation": cv,
        "burst_freqs_hz": burst_periodicity(intervals),
        "event_count": len(timestamps),
    }
