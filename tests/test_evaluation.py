"""Tests for src/evaluation/metrics.py."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from src.evaluation import metrics as m
from src.features.preprocessing import STAT_COLS
from src.models.clustering import fit_kmeans


@pytest.fixture
def three_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    return np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in centers])


def test_shuffle_keeps_each_column_but_breaks_pairing():
    rng = np.random.default_rng(0)
    stats = pd.DataFrame({c: np.arange(100) + i for i, c in enumerate(STAT_COLS)})
    shuffled = m.shuffle_columns(stats, rng)
    for c in STAT_COLS:
        assert sorted(shuffled[c]) == sorted(stats[c])        # same values per column
    assert not (shuffled["hp"].to_numpy() == stats["hp"].to_numpy()).all()  # order changed
    # originally hp and attack move together perfectly; after shuffling they don't
    assert abs(np.corrcoef(shuffled["hp"], shuffled["attack"])[0, 1]) < 0.5


def test_stability_is_high_for_real_groups(three_blobs):
    assert m.bootstrap_stability(three_blobs, k=3, n_boot=10).mean() > 0.95


def test_stability_is_lower_for_wrong_k(three_blobs):
    right = m.bootstrap_stability(three_blobs, k=3, n_boot=10).mean()
    wrong = m.bootstrap_stability(three_blobs, k=5, n_boot=10).mean()
    assert wrong < right


def test_real_structure_beats_shuffled_baseline():
    # stats strongly tied together in 3 groups -> shuffling should destroy that
    rng = np.random.default_rng(0)
    groups = np.repeat([50, 100, 150], 40)
    df = pd.DataFrame({c: groups + rng.normal(0, 5, 120) for c in STAT_COLS})
    X = (df - df.mean()) / df.std(ddof=0)
    real = silhouette_score(X, fit_kmeans(X, 3).labels_)
    baseline = m.shuffled_silhouettes(df, "standardized", k=3, n_shuffles=5)
    assert real > baseline.max()
