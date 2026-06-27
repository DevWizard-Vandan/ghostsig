"""Linguistic entropy feature extraction — char/word n-gram entropy."""

import math
from collections import Counter
from typing import list


def char_ngram_entropy(text: str, n: int = 3) -> float:
    """Compute Shannon entropy of character n-gram distribution."""
    if not text or len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def word_entropy(text: str) -> float:
    """Compute Shannon entropy of word unigram distribution."""
    words = text.lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def entropy_profile(texts: list[str]) -> dict:
    """Aggregate entropy profile across all public posts for an account."""
    char_entropies = [char_ngram_entropy(t) for t in texts if t]
    word_entropies = [word_entropy(t) for t in texts if t]
    def _stats(arr):
        if not arr:
            return {"mean": None, "std": None}
        import statistics
        return {
            "mean": statistics.mean(arr),
            "std": statistics.stdev(arr) if len(arr) > 1 else 0.0
        }
    return {
        "char_entropy": _stats(char_entropies),
        "word_entropy": _stats(word_entropies),
        "sample_count": len(texts),
    }
