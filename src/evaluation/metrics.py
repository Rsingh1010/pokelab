"""Checks that tell us whether a clustering reflects real structure."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import RANDOM_SEED
from src.features.preprocessing import STAT_COLS, make_features
from src.models.clustering import fit_kmeans


def shuffle_columns(stats: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each stat column on its own.

    Every column keeps exactly the same values (same distribution), but which
    HP goes with which Speed becomes random. Any structure that depends on how
    stats combine is destroyed. This is our 'no real structure' baseline.
    """
    return stats.apply(lambda col: rng.permutation(col.to_numpy()))


def shuffled_silhouettes(df: pd.DataFrame, representation: str, k: int,
                         n_shuffles: int = 10, seed: int = RANDOM_SEED, fit=fit_kmeans) -> np.ndarray:
    """Silhouette scores a model gets on shuffled (structureless) versions of the data.

    fit: any function (X, k, seed) -> fitted model with .predict(X). Default K-Means.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_shuffles):
        X, _ = make_features(shuffle_columns(df[STAT_COLS], rng), representation)
        scores.append(silhouette_score(X, fit(X, k, seed).predict(X)))
    return np.array(scores)


def bootstrap_stability(X, k: int, n_boot: int = 20, seed: int = RANDOM_SEED, fit=fit_kmeans) -> np.ndarray:
    """How much do clusters change if the data changes a little?

    1. Cluster the full data once: the reference labels.
    2. Many times: resample the rows with replacement, cluster the resample,
       then use that model to label ALL original rows.
    3. Compare each new labeling to the reference with Adjusted Rand Index (ARI):
       1 = identical grouping, ~0 = no better than random.
    ARI ignores label names, so 'cluster 0' vs 'cluster 3' for the same group is fine.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    reference = fit(X, k, seed).predict(X)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        model = fit(X[idx], k, seed + b + 1)
        scores.append(adjusted_rand_score(reference, model.predict(X)))
    return np.array(scores)
