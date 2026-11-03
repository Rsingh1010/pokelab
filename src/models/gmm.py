"""Gaussian Mixture Models (EXP-03): soft clustering for overlapping groups.

Run: python -m src.models.gmm
Applies the SAME pre-registered rule as K-Means (k_selection.py) to the main
setup (fully evolved / relative), plus BIC to choose among eligible k.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import RANDOM_SEED, RESULTS_DIR
from src.evaluation.metrics import bootstrap_stability, shuffled_silhouettes
from src.features.preprocessing import load_pokemon, make_features, select_subset
from src.models.clustering import K_RANGE
from src.models.k_selection import (
    CANDIDATE_K, MAIN_SETUP, MIN_CLUSTER_SHARE, MIN_STABILITY_ARI,
)

N_INIT = 5
UNCERTAIN_BELOW = 0.6  # a Pokémon is "mixed" if its top group probability is under 60%


def drop_first_column(X):
    """Relative stats sum to 1, so the 6th column is fully determined by the other 5.

    A full-covariance Gaussian on all 6 would have a covariance matrix with no
    spread in one direction (singular). Dropping any one column loses no
    information and fixes that. We drop the first (HP share).
    """
    return np.asarray(X)[:, 1:]


def fit_gmm(X, k: int, seed: int = RANDOM_SEED):
    """GMM with full covariance: each group can be stretched and tilted, unlike K-Means' round groups."""
    model = make_pipeline(
        FunctionTransformer(drop_first_column),
        GaussianMixture(n_components=k, covariance_type="full", n_init=N_INIT, random_state=seed),
    )
    return model.fit(X)


def bic(model, X) -> float:
    """Bayesian Information Criterion: fit quality minus a penalty for extra parameters. Lower = better."""
    return model[-1].bic(drop_first_column(X))


def soft_summary(model, X) -> dict:
    proba = model.predict_proba(X)
    top = proba.max(axis=1)
    return {"mean_top_probability": top.mean(), "uncertain_share": (top < UNCERTAIN_BELOW).mean()}


def run_selection() -> pd.DataFrame:
    subset, representation = MAIN_SETUP
    data = select_subset(load_pokemon(), subset)
    X, _ = make_features(data, representation)
    rows = []
    for k in K_RANGE:
        model = fit_gmm(X, k)
        labels = model.predict(X)
        baseline = shuffled_silhouettes(data, representation, k, fit=fit_gmm)
        stability = bootstrap_stability(X, k, fit=fit_gmm)
        rows.append({
            "k": k,
            "bic": bic(model, X),
            "silhouette": silhouette_score(X, labels),  # scored on all 6 features, same as K-Means
            "baseline_p95": np.percentile(baseline, 95),
            "stability_ari": stability.mean(),
            "stability_ari_std": stability.std(),
            "smallest_share": pd.Series(labels).value_counts().min() / len(X),
            **soft_summary(model, X),
        })
        print(f"  k={k} done")
    t = pd.DataFrame(rows)
    t["beats_baseline"] = t["silhouette"] > t["baseline_p95"]
    t["stable"] = t["stability_ari"] >= MIN_STABILITY_ARI
    t["big_enough"] = t["smallest_share"] >= MIN_CLUSTER_SHARE
    t["eligible"] = t["k"].isin(CANDIDATE_K) & t["beats_baseline"] & t["stable"] & t["big_enough"]
    return t


if __name__ == "__main__":
    print(f"GMM selection on {MAIN_SETUP} (takes a few minutes)")
    results = run_selection()
    results.to_csv(RESULTS_DIR / "gmm_selection.csv", index=False)
    cols = ["k", "bic", "silhouette", "baseline_p95", "stability_ari", "smallest_share",
            "mean_top_probability", "uncertain_share", "eligible"]
    print(results[cols].round(3).to_string(index=False))
    eligible = results[results["eligible"]]
    best = int(eligible.loc[eligible["bic"].idxmin(), "k"]) if len(eligible) else None
    print(f"\nEligible k: {eligible['k'].tolist() or 'NONE'}"
          + (f" | lowest BIC among eligible: k={best}" if best else ""))
