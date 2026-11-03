"""Milestone 5: which Pokémon have unusual stat profiles? (Isolation Forest)

Run: python -m src.models.anomaly_detection

Two views:
- raw:      the six stats as-is. Unusual = extreme or odd combinations of values.
            (Isolation Forest only compares values within one stat at a time, so
            standardizing would give the same result. We test that below.)
- relative: stat / BST. Unusual = odd way of spending the stat budget, regardless of strength.

No ground truth exists, so there is no accuracy. We check instead:
- seed stability: do different random forests rank Pokémon the same way?
- a simple baseline: the single most extreme z-score per Pokémon. If Isolation
  Forest just repeats it, it adds little; where they differ, it is catching
  unusual COMBINATIONS that no single stat shows.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RANDOM_SEED, RESULTS_DIR
from src.features.preprocessing import STAT_COLS, load_pokemon, relative_stats

N_TREES = 500
TOP_SHARE = 0.05   # "flagged" = top 5% of scores. A reporting cutoff, not a truth threshold.
SEEDS = range(5)


def anomaly_scores(X, seed: int = RANDOM_SEED) -> np.ndarray:
    """Higher = more anomalous. (sklearn's score_samples is the opposite, so we flip the sign.)"""
    model = IsolationForest(n_estimators=N_TREES, random_state=seed).fit(X)
    return -model.score_samples(X)


def max_abs_z(X: pd.DataFrame) -> pd.Series:
    """Baseline: how extreme is this Pokémon's MOST extreme single stat?"""
    z = (X - X.mean()) / X.std(ddof=0)
    return z.abs().max(axis=1)


def most_extreme_stat(X: pd.DataFrame) -> pd.Series:
    """Which stat drives the baseline score, with direction, e.g. 'defense+' or 'speed-'."""
    z = (X - X.mean()) / X.std(ddof=0)
    col = z.abs().idxmax(axis=1)
    sign = np.where(z.to_numpy()[np.arange(len(z)), z.columns.get_indexer(col)] > 0, "+", "-")
    return col + sign


def seed_stability(X, seeds=SEEDS) -> float:
    """Average Spearman rank correlation of scores between every pair of seeds (1 = identical ranking)."""
    runs = [pd.Series(anomaly_scores(X, s)) for s in seeds]
    return float(np.mean([a.corr(b, method="spearman") for a, b in combinations(runs, 2)]))


def score_view(df: pd.DataFrame, X: pd.DataFrame, view: str) -> pd.DataFrame:
    scores = pd.Series(anomaly_scores(X), index=df.index)
    return pd.DataFrame({
        f"{view}_score": scores,
        f"{view}_rank": scores.rank(ascending=False).astype(int),
        f"{view}_flagged": scores.rank(ascending=False) <= round(TOP_SHARE * len(df)),
        f"{view}_max_abs_z": max_abs_z(X),
        f"{view}_driver": most_extreme_stat(X),
    })


def views(df: pd.DataFrame) -> dict:
    return {"raw": df[STAT_COLS], "relative": relative_stats(df)}


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = ["dex_id", "name", "generation", "type_1", "type_2", "bst",
            "is_fully_evolved", "is_legendary", "is_mythical"]
    parts, summary = [df[info]], {}
    for view, X in views(df).items():
        table = score_view(df, X, view)
        parts.append(table)
        summary[view] = {
            "seed_stability_spearman": round(seed_stability(X), 3),
            "spearman_vs_max_abs_z": round(table[f"{view}_score"].corr(table[f"{view}_max_abs_z"], method="spearman"), 3),
        }
    return pd.concat(parts, axis=1), summary


def form_override_check(with_overrides: pd.DataFrame, top_n: int = 20) -> dict:
    """Did the Wishiwashi / Terapagos form choice change the anomaly results?"""
    from src.data.build_dataset import build
    without = build(use_overrides=False)
    out = {}
    for view in ("raw", "relative"):
        a = with_overrides.set_index("name")[f"{view}_rank"]
        b = run(without)[0].set_index("name")[f"{view}_rank"]
        out[view] = {
            "top_overlap": len(set(a.nsmallest(top_n).index) & set(b.nsmallest(top_n).index)),
            "top_n": top_n,
            "rank_with_overrides": {n: int(a[n]) for n in ("Wishiwashi", "Terapagos")},
            "rank_default_forms": {n: int(b[n]) for n in ("Wishiwashi", "Terapagos")},
        }
    return out


if __name__ == "__main__":
    import json

    df = load_pokemon()
    table, summary = run(df)
    table.to_csv(RESULTS_DIR / "anomaly_scores.csv", index=False)
    print("Checks:", json.dumps(summary, indent=2))

    for view in ("raw", "relative"):
        top = table.nsmallest(15, f"{view}_rank")
        print(f"\nTop 15 [{view}]")
        print(top[["name", "bst", f"{view}_score", f"{view}_max_abs_z", f"{view}_driver",
                   "is_fully_evolved", "is_legendary"]].round(3).to_string(index=False))

    print("\nFlagged (top 5%) vs everyone: share fully evolved / legendary or mythical")
    for view in ("raw", "relative"):
        f = table[table[f"{view}_flagged"]]
        print(f"  {view:8s}: fully evolved {f['is_fully_evolved'].mean():.0%} (all: {df['is_fully_evolved'].mean():.0%}), "
              f"legendary/mythical {(f['is_legendary'] | f['is_mythical']).mean():.0%} "
              f"(all: {(df['is_legendary'] | df['is_mythical']).mean():.0%})")

    print("\nForm override check:", json.dumps(form_override_check(table), indent=2))
    with open(RESULTS_DIR / "anomaly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
