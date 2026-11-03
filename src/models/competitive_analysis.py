"""EXP-06: are stat profiles associated with competitive usage?

Run: python -m src.models.competitive_analysis   (after src.data.fetch_competitive)

Population: fully evolved Pokémon LEGAL in Gen 9 National Dex (banned ones removed).
Usage = mean weighted usage over the pinned months, for that exact form.
Usage is popularity, not win rate. Moves, abilities and items drive it too, and
we do not model them, so weak associations are the expected result.

All tests are exploratory (no correction for running several), reported with
bootstrap 95% intervals so the uncertainty is visible.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from src.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from src.features.preprocessing import STAT_COLS, load_pokemon, make_features
from src.models.clustering import fit_kmeans
from src.optimization.team_optimizer import add_team_features

HIGH_USAGE = 0.0452   # Smogon's own cutoff for a Pokémon to count as part of a tier's metagame
N_BOOT = 1000


def load_analysis_table() -> pd.DataFrame:
    usage = pd.read_csv(PROCESSED_DIR / "usage_gen9natdex.csv")
    fe = add_team_features(load_pokemon())                    # role axes on all fully evolved
    X, _ = make_features(fe, "relative")
    fe["cluster"] = fit_kmeans(X, 5).labels_                  # descriptive k=5 (not validated)
    fe["cluster_label"] = fe["cluster"].map(cluster_labels(X, fe["cluster"]))
    anomaly = pd.read_csv(RESULTS_DIR / "anomaly_scores.csv")[["dex_id", "fe_relative_score"]]
    t = (usage[usage["legal"]][["dex_id", "usage_mean"]]
         .merge(fe, on="dex_id").merge(anomaly, on="dex_id"))
    t["high_usage"] = t["usage_mean"] >= HIGH_USAGE
    return t


def cluster_labels(X: pd.DataFrame, labels) -> dict:
    """Name each descriptive cluster by its most over- and under-weighted stat (from the data, not by hand)."""
    diff = X.groupby(np.asarray(labels)).mean() - X.mean()
    return {c: f"{row.idxmax()}+ / {row.idxmin()}-" for c, row in diff.iterrows()}


def spearman_ci(x, y, n_boot=N_BOOT, seed=RANDOM_SEED) -> dict:
    """Rank correlation with a bootstrap 95% interval. If the interval crosses 0, treat as 'no clear link'."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    rho = spearmanr(x, y).statistic
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": round(float(rho), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "clear_link": bool(lo > 0 or hi < 0)}


def similar_stats_different_usage(t: pd.DataFrame, n=10) -> pd.DataFrame:
    """Pairs with near-identical stats but very different usage: evidence that non-stat factors matter."""
    Z, _ = make_features(t, "standardized")
    Z = Z.to_numpy()
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    j = d.argmin(axis=1)
    pairs = pd.DataFrame({"a": t["name"].to_numpy(), "b": t["name"].to_numpy()[j],
                          "stat_distance": d[np.arange(len(t)), j],
                          "usage_a": t["usage_mean"].to_numpy(), "usage_b": t["usage_mean"].to_numpy()[j]})
    pairs = pairs[pairs["a"] < pairs["b"]]                                   # each pair once
    close = pairs[pairs["stat_distance"] <= pairs["stat_distance"].quantile(0.25)]
    close = close.assign(usage_gap=(close["usage_a"] - close["usage_b"]).abs())
    return close.nlargest(n, "usage_gap")


def run() -> tuple[dict, pd.DataFrame]:
    t = load_analysis_table()
    u = t["usage_mean"]
    by_cluster = t.groupby("cluster_label").agg(
        n=("name", "size"), median_usage=("usage_mean", "median"),
        high_usage_share=("high_usage", "mean"), mean_bst=("bst", "mean"))
    by_cluster["share_of_high_usage"] = t[t["high_usage"]].groupby("cluster_label").size() / t["high_usage"].sum()
    by_cluster["share_of_population"] = by_cluster["n"] / len(t)

    types = pd.concat([t[["type_1", "usage_mean", "high_usage"]].rename(columns={"type_1": "type"}),
                       t[["type_2", "usage_mean", "high_usage"]].rename(columns={"type_2": "type"})]).dropna()
    by_type = types.groupby("type").agg(n=("usage_mean", "size"), median_usage=("usage_mean", "median"),
                                        high_usage_share=("high_usage", "mean"))

    summary = {
        "population": int(len(t)), "with_any_usage": int((u > 0).sum()),
        "high_usage_cutoff": HIGH_USAGE, "n_high_usage": int(t["high_usage"].sum()),
        "spearman_vs_usage": {
            "bst": spearman_ci(t["bst"], u),
            "speed_axis (fast+ / bulky-)": spearman_ci(t["speed_axis"], u),
            "physical_axis (physical+ / special-)": spearman_ci(t["physical_axis"], u),
            "anomaly_score (fe_relative)": spearman_ci(t["fe_relative_score"], u),
            **{f"{s} (raw)": spearman_ci(t[s], u) for s in STAT_COLS},
        },
        "kruskal_usage_by_cluster_p": float(kruskal(*[g["usage_mean"] for _, g in t.groupby("cluster")]).pvalue),
        "usage_by_legendary": t.groupby(t["is_legendary"] | t["is_mythical"])["usage_mean"].median().round(4).to_dict(),
    }
    summary["usage_by_legendary"] = {("legendary/mythical" if k else "regular"): v
                                     for k, v in summary["usage_by_legendary"].items()}
    return summary, {"by_cluster": by_cluster, "by_type": by_type,
                     "pairs": similar_stats_different_usage(t), "table": t}


if __name__ == "__main__":
    summary, tables = run()
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print("\nBy descriptive cluster:\n", tables["by_cluster"].round(3).to_string())
    print("\nBy type (top/bottom by median usage):\n",
          tables["by_type"].sort_values("median_usage", ascending=False).round(4).to_string())
    print("\nSimilar stats, very different usage:\n", tables["pairs"].round(3).to_string(index=False))
    (RESULTS_DIR / "competitive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k in ("by_cluster", "by_type", "pairs"):
        tables[k].to_csv(RESULTS_DIR / f"competitive_{k}.csv")
