"""Unit tests for feature extraction modules."""

import pytest
from features.temporal import compute_inter_event_intervals, temporal_stats
from features.entropy import char_ngram_entropy, word_entropy, entropy_profile


def test_inter_event_intervals_basic():
    ts = [1000.0, 1010.0, 1025.0, 1040.0]
    intervals = compute_inter_event_intervals(ts)
    assert list(intervals) == [10.0, 15.0, 15.0]


def test_inter_event_empty():
    assert len(compute_inter_event_intervals([1000.0])) == 0


def test_temporal_stats():
    ts = [float(i * 60) for i in range(20)]  # posts every 60 seconds
    stats = temporal_stats(ts)
    assert stats["mean_interval_sec"] == pytest.approx(60.0)
    assert stats["std_interval_sec"] == pytest.approx(0.0, abs=1e-9)
    assert stats["coefficient_of_variation"] == pytest.approx(0.0, abs=1e-9)


def test_char_ngram_entropy_uniform():
    # Uniform character distribution should have high entropy
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    entropy = char_ngram_entropy(text, n=2)
    assert entropy > 3.0


def test_char_ngram_entropy_repetitive():
    # Repetitive text should have low entropy
    text = "aaaaaaaaaa" * 10
    entropy = char_ngram_entropy(text, n=2)
    assert entropy == pytest.approx(0.0)


def test_word_entropy():
    text = "hello world hello world"
    e = word_entropy(text)
    assert e == pytest.approx(1.0)  # 2 words, equal probability → H=1 bit


def test_entropy_profile():
    texts = ["the quick brown fox", "hello world test abc"]
    profile = entropy_profile(texts)
    assert profile["sample_count"] == 2
    assert profile["char_entropy"]["mean"] is not None
