"""Run the pre-agreed rule for choosing k (decided BEFORE looking at cluster profiles).

Rule:
- Main setup: fully_evolved / relative. Candidates: k = 3..8.
- A k is eligible only if ALL hold:
    1. silhouette beats the shuffled baseline (above its 95th percentile)
    2. bootstrap stability: mean ARI >= 0.6
    3. smallest cluster >= 5% of Pokémon
- Among eligible k: pick most distinct centroids (judged with profiles next);
  if close, pick the smaller k. If none eligible, report that honestly.

All four setups get the same checks so EXP-01 / EXP-05 can compare them.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import RESULTS_DIR
from src.evaluation.metrics import bootstrap_stability, shuffled_silhouettes
from src.features.preprocessing import (
    REPRESENTATIONS, SUBSETS, load_pokemon, make_features, select_subset,
)
from src.models.clustering import K_RANGE, fit_kmeans

MAIN_SETUP = ("fully_evolved", "relative")
CANDIDATE_K = range(3, 9)
MIN_STABILITY_ARI = 0.6
MIN_CLUSTER_SHARE = 0.05


def check_setup(df: pd.DataFrame, subset: str, representation: str) -> pd.DataFrame:
    data = select_subset(df, subset)
    X, _ = make_features(data, representation)
    rows = []
    for k in K_RANGE:
        labels = fit_kmeans(X, k).labels_
        baseline = shuffled_silhouettes(data, representation, k)
        stability = bootstrap_stability(X, k)
        rows.append({
            "subset": subset, "representation": representation, "k": k,
            "silhouette": silhouette_score(X, labels),
            "baseline_mean": baseline.mean(),
            "baseline_p95": np.percentile(baseline, 95),
            "stability_ari": stability.mean(),
            "stability_ari_std": stability.std(),
            "smallest_share": pd.Series(labels).value_counts().min() / len(X),
        })
        print(f"  k={k} done")
    table = pd.DataFrame(rows)
    table["beats_baseline"] = table["silhouette"] > table["baseline_p95"]
    table["stable"] = table["stability_ari"] >= MIN_STABILITY_ARI
    table["big_enough"] = table["smallest_share"] >= MIN_CLUSTER_SHARE
    return table


def evolution_check(df: pd.DataFrame) -> pd.DataFrame:
    """Is the best raw-stat split (all / standardized, k=2) just evolved vs not?"""
    X, _ = make_features(df, "standardized")
    labels = fit_kmeans(X, 2).labels_
    table = pd.crosstab(pd.Series(labels, name="cluster"), df["is_fully_evolved"])
    ari = adjusted_rand_score(df["is_fully_evolved"], labels)
    agreement = max((labels == df["is_fully_evolved"]).mean(), (labels != df["is_fully_evolved"]).mean())
    print(table, f"\nARI vs is_fully_evolved: {ari:.3f} | same-group agreement: {agreement:.1%}")
    return table


if __name__ == "__main__":
    df = load_pokemon()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("1) Evolution check (all / standardized, k=2)")
    evolution_check(df).to_csv(RESULTS_DIR / "evolution_crosstab_k2.csv")

    print("\n2-3) Baseline + stability for every setup (takes a few minutes)")
    tables = []
    for subset in SUBSETS:
        for representation in REPRESENTATIONS:
            print(f"{subset} / {representation}")
            tables.append(check_setup(df, subset, representation))
    results = pd.concat(tables, ignore_index=True)

    main = (results["subset"] == MAIN_SETUP[0]) & (results["representation"] == MAIN_SETUP[1])
    results["eligible"] = (
        main & results["k"].isin(CANDIDATE_K)
        & results["beats_baseline"] & results["stable"] & results["big_enough"]
    )
    results.to_csv(RESULTS_DIR / "k_selection.csv", index=False)

    cols = ["subset", "representation", "k", "silhouette", "baseline_p95",
            "stability_ari", "smallest_share", "eligible"]
    print(results[cols].round(3).to_string(index=False))
    eligible = results.loc[results["eligible"], "k"].tolist()
    print(f"\nEligible k for main setup {MAIN_SETUP}: {eligible or 'NONE'}")
