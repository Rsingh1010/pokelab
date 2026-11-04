"""Milestone 5: which Pokémon have unusual stat profiles? (Isolation Forest)

Run: python -m src.models.anomaly_detection

Two views:
- raw:      the six stats as-is. Unusual = extreme or odd combinations of values.
            (Isolation Forest only compares values within one stat at a time, so
            standardizing would give the same result. We test that below.)
- relative: stat / BST. Unusual = odd way of spending the stat budget, regardless of strength.
- fe_relative: relative, fully evolved Pokémon only (same setup as the main clustering).
            Unevolved Pokémon get no score in this view (blank, not zero).

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


def score_view(X: pd.DataFrame, view: str) -> pd.DataFrame:
    scores = pd.Series(anomaly_scores(X), index=X.index)
    return pd.DataFrame({
        f"{view}_score": scores,
        f"{view}_rank": scores.rank(ascending=False).astype(int),
        f"{view}_flagged": scores.rank(ascending=False) <= round(TOP_SHARE * len(X)),
        f"{view}_max_abs_z": max_abs_z(X),
        f"{view}_driver": most_extreme_stat(X),
    })


VIEWS = ("raw", "relative", "fe_relative")


def views(df: pd.DataFrame) -> dict:
    return {
        "raw": df[STAT_COLS],
        "relative": relative_stats(df),
        "fe_relative": relative_stats(df[df["is_fully_evolved"]]),
    }


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = ["dex_id", "name", "generation", "type_1", "type_2", "bst",
            "is_fully_evolved", "is_legendary", "is_mythical"]
    parts, summary = [df[info]], {}
    for view, X in views(df).items():
        table = score_view(X, view)
        parts.append(table)
        summary[view] = {
            "n_scored": len(X),
            "seed_stability_spearman": round(seed_stability(X), 3),
            "spearman_vs_max_abs_z": round(table[f"{view}_score"].corr(table[f"{view}_max_abs_z"], method="spearman"), 3),
        }
    return pd.concat(parts, axis=1), summary


def form_override_check(with_overrides: pd.DataFrame, top_n: int = 20) -> dict:
    """Did the Wishiwashi / Terapagos form choice change the anomaly results?"""
    from src.data.build_dataset import build
    without = build(use_overrides=False)
    out = {}
    for view in VIEWS:
        a = with_overrides.set_index("name")[f"{view}_rank"]
        b = run(without)[0].set_index("name")[f"{view}_rank"]
        out[view] = {
            "top_overlap": len(set(a.nsmallest(top_n).index) & set(b.nsmallest(top_n).index)),
            "top_n": top_n,
            "rank_with_overrides": {n: int(a[n]) for n in ("Wishiwashi", "Terapagos")},
            "rank_default_forms": {n: int(b[n]) for n in ("Wishiwashi", "Terapagos")},
        }
    return out


def flagged_composition(table: pd.DataFrame) -> dict:
    """Are flagged Pokémon different from the Pokémon scored in that view?"""
    out = {}
    for view in VIEWS:
        scored = table[table[f"{view}_score"].notna()]
        flagged = scored[scored[f"{view}_flagged"].eq(True)]
        legend = lambda d: float((d["is_legendary"] | d["is_mythical"]).mean())
        out[view] = {
            "n_flagged": len(flagged),
            "fully_evolved_flagged": round(float(flagged["is_fully_evolved"].mean()), 3),
            "fully_evolved_all": round(float(scored["is_fully_evolved"].mean()), 3),
            "legendary_or_mythical_flagged": round(legend(flagged), 3),
            "legendary_or_mythical_all": round(legend(scored), 3),
        }
    return out


if __name__ == "__main__":
    import json

    df = load_pokemon()
    table, summary = run(df)
    table.to_csv(RESULTS_DIR / "anomaly_scores.csv", index=False)
    print("Checks:", json.dumps(summary, indent=2))

    for view in VIEWS:
        top = table.nsmallest(15, f"{view}_rank")
        print(f"\nTop 15 [{view}]")
        print(top[["name", "bst", f"{view}_score", f"{view}_max_abs_z", f"{view}_driver",
                   "is_fully_evolved", "is_legendary"]].round(3).to_string(index=False))

    summary["flagged_composition"] = flagged_composition(table)
    print("\nFlagged vs all scored:", json.dumps(summary["flagged_composition"], indent=2))
    try:
        summary["form_override_check"] = form_override_check(table)
        print("\nForm override check:", json.dumps(summary["form_override_check"], indent=2))
    except FileNotFoundError:
        # needs the raw PokéAPI cache (not committed); keep the previous result if there is one
        old = RESULTS_DIR / "anomaly_summary.json"
        prev = json.loads(old.read_text(encoding="utf-8")).get("form_override_check") if old.exists() else None
        summary["form_override_check"] = prev
        print("\nForm override check skipped (raw cache missing); kept previous result.")
    with open(RESULTS_DIR / "anomaly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
