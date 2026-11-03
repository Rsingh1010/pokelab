"""Tests for src/models/clustering.py."""
import numpy as np
import pytest

from src.models import clustering as cl


@pytest.fixture
def three_blobs():
    """Three clearly separated groups of 30 points. K-Means must find them."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    return np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in centers])


def test_finds_known_groups(three_blobs):
    labels = cl.fit_kmeans(three_blobs, k=3).labels_
    # each true group of 30 must land in exactly one cluster
    for start in (0, 30, 60):
        assert len(set(labels[start:start + 30])) == 1
    assert len(set(labels)) == 3


def test_same_seed_same_result(three_blobs):
    a = cl.fit_kmeans(three_blobs, k=4, seed=1).labels_
    b = cl.fit_kmeans(three_blobs, k=4, seed=1).labels_
    assert (a == b).all()


def test_sweep_shape_and_metric_ranges(three_blobs):
    table = cl.sweep_k(three_blobs, ks=range(2, 6))
    assert table["k"].tolist() == [2, 3, 4, 5]
    assert table["silhouette"].between(-1, 1).all()
    assert (table["davies_bouldin"] >= 0).all()
    assert table["inertia"].is_monotonic_decreasing      # more clusters, never worse fit
    assert (table["smallest_cluster"] >= 1).all()


def test_metrics_prefer_true_k(three_blobs):
    table = cl.sweep_k(three_blobs, ks=range(2, 6)).set_index("k")
    assert table["silhouette"].idxmax() == 3
    assert table["davies_bouldin"].idxmin() == 3


def test_profiles_and_margins(three_blobs):
    import pandas as pd
    X = pd.DataFrame(three_blobs, columns=["a", "b"])
    data = pd.DataFrame({"name": [f"p{i}" for i in range(len(X))], "bst": range(len(X))})
    model = cl.fit_kmeans(X, k=3)
    prof = cl.cluster_profiles(data, X, model, n_examples=3)
    assert prof["size"].sum() == len(X)
    assert abs(prof["share"].sum() - 1) < 1e-9
    for _, row in prof.iterrows():  # typical members really belong to that cluster
        names = row["typical_members"].split(", ")
        idx = data.index[data["name"].isin(names)]
        assert set(model.labels_[idx]) == {row["cluster"]}
    margins = cl.assignment_margins(X, model)
    assert margins["margin"].between(0, 1).all()
    assert not margins["borderline"].any()   # well-separated blobs: nobody is borderline
