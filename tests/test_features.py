"""Tests for src/features/preprocessing.py."""
import numpy as np
import pandas as pd
import pytest

from src.features import preprocessing as pp


@pytest.fixture
def toy():
    return pd.DataFrame({
        "hp":         [100, 50, 80],
        "attack":     [100, 150, 80],
        "defense":    [100, 50, 80],
        "sp_attack":  [100, 50, 80],
        "sp_defense": [100, 50, 80],
        "speed":      [100, 200, 80],
        "is_fully_evolved": [True, True, False],
    })


def test_relative_rows_sum_to_one(toy):
    rel = pp.relative_stats(toy)
    assert np.allclose(rel.sum(axis=1), 1.0)


def test_relative_ignores_overall_strength(toy):
    # Rows 0 and 2 have the same shape (all stats equal), different strength
    rel = pp.relative_stats(toy)
    assert np.allclose(rel.iloc[0], rel.iloc[2])
    assert np.allclose(rel.iloc[0], 1 / 6)


def test_standardized_has_mean_0_std_1(toy):
    X, scaler = pp.make_features(toy, "standardized")
    assert np.allclose(X.mean(), 0)
    assert np.allclose(X.std(ddof=0), 1)
    assert scaler is not None


def test_standardized_keeps_strength_relative_does_not(toy):
    Xs, _ = pp.make_features(toy, "standardized")
    Xr, _ = pp.make_features(toy, "relative")
    assert not np.allclose(Xs.iloc[0], Xs.iloc[2])   # 100s vs 80s differ
    assert np.allclose(Xr.iloc[0], Xr.iloc[2])       # same shape, so identical


def test_subset_filters_and_rejects_unknown(toy):
    assert len(pp.select_subset(toy, "all")) == 3
    assert len(pp.select_subset(toy, "fully_evolved")) == 2
    with pytest.raises(ValueError):
        pp.select_subset(toy, "legendaries")
    with pytest.raises(ValueError):
        pp.make_features(toy, "log")


@pytest.mark.parametrize("subset", pp.SUBSETS)
@pytest.mark.parametrize("representation", pp.REPRESENTATIONS)
def test_real_features_are_clean(subset, representation):
    df = pp.select_subset(pp.load_pokemon(), subset)
    X, _ = pp.make_features(df, representation)
    assert X.shape == (len(df), 6)
    assert np.isfinite(X.to_numpy()).all()       # no NaN or inf
    assert X.index.equals(df.index)              # rows still line up with names
