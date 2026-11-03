"""Tests for src/models/gmm.py."""
import numpy as np
import pytest

from src.evaluation.metrics import bootstrap_stability
from src.models import gmm


@pytest.fixture
def three_blobs_3d():
    """First column is pure noise (gets dropped), columns 2-3 hold 3 clear groups."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    blobs = np.vstack([c + rng.normal(scale=0.5, size=(40, 2)) for c in centers])
    return np.column_stack([rng.normal(size=len(blobs)), blobs])


def test_drop_first_column():
    X = np.arange(12).reshape(2, 6)
    assert gmm.drop_first_column(X).shape == (2, 5)
    assert (gmm.drop_first_column(X) == X[:, 1:]).all()


def test_finds_known_groups(three_blobs_3d):
    labels = gmm.fit_gmm(three_blobs_3d, 3).predict(three_blobs_3d)
    for start in (0, 40, 80):
        assert len(set(labels[start:start + 40])) == 1
    assert len(set(labels)) == 3


def test_bic_prefers_true_k(three_blobs_3d):
    scores = {k: gmm.bic(gmm.fit_gmm(three_blobs_3d, k), three_blobs_3d) for k in range(2, 6)}
    assert min(scores, key=scores.get) == 3


def test_soft_membership_valid(three_blobs_3d):
    model = gmm.fit_gmm(three_blobs_3d, 3)
    proba = model.predict_proba(three_blobs_3d)
    assert np.allclose(proba.sum(axis=1), 1)
    summary = gmm.soft_summary(model, three_blobs_3d)
    assert summary["mean_top_probability"] > 0.95       # clear groups -> confident
    assert summary["uncertain_share"] < 0.05


def test_gmm_works_with_shared_stability_check(three_blobs_3d):
    assert bootstrap_stability(three_blobs_3d, 3, n_boot=5, fit=gmm.fit_gmm).mean() > 0.95


def test_handles_real_relative_features_without_error():
    # the sum-to-1 problem: this would fail or warn without dropping a column
    from src.features.preprocessing import load_pokemon, make_features, select_subset
    X, _ = make_features(select_subset(load_pokemon(), "fully_evolved"), "relative")
    model = gmm.fit_gmm(X, 5)
    assert np.isfinite(gmm.bic(model, X))
