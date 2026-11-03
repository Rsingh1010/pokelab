"""Tests for src/models/anomaly_detection.py."""
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import STAT_COLS
from src.models import anomaly_detection as ad


@pytest.fixture
def stats_with_planted_outlier():
    """200 ordinary Pokémon around 80 in every stat, plus one Shuckle-like oddball (row 0)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(80, 10, size=(200, 6)), columns=STAT_COLS)
    df.iloc[0] = [20, 10, 230, 10, 230, 5]
    return df


def test_planted_outlier_ranks_first(stats_with_planted_outlier):
    scores = ad.anomaly_scores(stats_with_planted_outlier)
    assert scores.argmax() == 0           # higher = more anomalous


def test_scaling_does_not_change_ranking(stats_with_planted_outlier):
    X = stats_with_planted_outlier
    standardized = (X - X.mean()) / X.std()
    a = pd.Series(ad.anomaly_scores(X))
    b = pd.Series(ad.anomaly_scores(standardized))
    assert a.corr(b, method="spearman") > 0.99


def test_combination_outlier_missed_by_baseline_but_caught_by_forest():
    # every stat of row 0 is ordinary on its own, but the PAIRING is impossible
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 300)
    df = pd.DataFrame({c: base + rng.normal(0, 0.1, 300) for c in STAT_COLS})  # stats move together
    df.iloc[0] = [1.5, -1.5, 1.5, -1.5, 1.5, -1.5]                            # each value normal, mix is not
    assert ad.max_abs_z(df).rank(ascending=False)[0] > 20                      # baseline doesn't see it
    assert pd.Series(ad.anomaly_scores(df)).rank(ascending=False)[0] <= 10     # forest does (top ~3%)


def test_baseline_and_driver():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(0, 1, size=(50, 6)), columns=STAT_COLS)
    X.loc[3, "speed"] = 10.0
    assert ad.max_abs_z(X).idxmax() == 3
    assert ad.most_extreme_stat(X)[3] == "speed+"


def test_seed_stability_high_on_clear_data(stats_with_planted_outlier):
    assert ad.seed_stability(stats_with_planted_outlier, seeds=range(3)) > 0.8
