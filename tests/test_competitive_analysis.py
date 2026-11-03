"""Tests for src/models/competitive_analysis.py helpers."""
import numpy as np
import pandas as pd

from src.models import competitive_analysis as ca


def test_spearman_ci_detects_clear_and_no_link():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    strong = ca.spearman_ci(x, x + rng.normal(scale=0.3, size=300), n_boot=200)
    none = ca.spearman_ci(x, rng.normal(size=300), n_boot=200)
    assert strong["clear_link"] and strong["rho"] > 0.8
    assert not none["clear_link"]
    assert none["ci95"][0] < 0 < none["ci95"][1]


def test_cluster_labels_come_from_data():
    X = pd.DataFrame({"hp": [0.3, 0.3, 0.1, 0.1], "speed": [0.1, 0.1, 0.3, 0.3]})
    labels = ca.cluster_labels(X, [0, 0, 1, 1])
    assert labels[0] == "hp+ / speed-" and labels[1] == "speed+ / hp-"
