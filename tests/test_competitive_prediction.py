"""Tests for src/models/competitive_prediction.py."""
import numpy as np
import pandas as pd
import pytest

from src.models import competitive_prediction as cp


def test_target_handles_zero_usage():
    y = cp.target(pd.Series([0.0, 0.01, 0.2]))
    assert np.isfinite(y).all() and y[0] < y[1] < y[2]


def test_type_flags_mark_both_types():
    t = pd.DataFrame({"hp": [80], "attack": [80], "defense": [80], "sp_attack": [80], "sp_defense": [80],
                      "speed": [80], "bst": [480], "type_1": ["fire"], "type_2": ["flying"]})
    full = cp.feature_sets(t)["full"]
    assert full.filter(like="type_").sum(axis=1).item() == 2
    assert full["type_fire"].item() == 1 and full["type_water"].item() == 0
    assert full["hp_share"].item() == pytest.approx(1 / 6)


def test_cv_rewards_real_signal_not_noise():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    X_sets = {"signal": pd.DataFrame({"x": signal}), "noise": pd.DataFrame({"x": rng.normal(size=n)})}
    y = signal + rng.normal(scale=0.5, size=n)
    specs = {"bst_ridge": ("signal", lambda: cp.make_pipeline(cp.StandardScaler(), cp.Ridge())),
             "noise_model": ("noise", lambda: cp.make_pipeline(cp.StandardScaler(), cp.Ridge()))}
    scores = cp.cv_scores(X_sets, y, specs, n_splits=5, n_repeats=2)
    assert len(scores) == 10
    s = cp.summarize(scores)
    assert s["bst_ridge"]["mean_heldout_spearman"] > 0.8
    assert abs(s["noise_model"]["mean_heldout_spearman"]) < 0.2
    assert s["noise_model"]["share_of_folds_beating_bst"] == 0
