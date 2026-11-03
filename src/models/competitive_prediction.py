"""EXP-06 (prediction part): does anything beyond BST help predict usage?

Run: python -m src.models.competitive_prediction   (after competitive_analysis)

Target: log10(usage + 0.0001). Usage is very skewed (most Pokémon near 0), the log
spreads it out; the small constant keeps 0% usage finite.
Score: Spearman between predicted and actual usage on HELD-OUT Pokémon only.

Models (all hyperparameters fixed in advance, no tuning, so nothing peeks at test folds):
  bst_ridge      - BST only, linear. The baseline.
  full_ridge     - 6 raw stats + 6 stat shares + 18 type flags, linear.
  full_forest    - same features, random forest (can use interactions, e.g. type x stat).

No leakage: no tiers, no usage-derived inputs; scaling is fit inside each training fold;
every feature is computed per Pokémon without looking at other Pokémon.
Validation: 5-fold cross-validation repeated 10 times with the same splits for every
model, so they can be compared fold by fold.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED, RESULTS_DIR
from src.features.preprocessing import STAT_COLS, relative_stats
from src.optimization.type_chart import TYPES

N_SPLITS, N_REPEATS = 5, 10


def target(usage: pd.Series) -> np.ndarray:
    return np.log10(usage.to_numpy() + 1e-4)


def feature_sets(t: pd.DataFrame) -> dict:
    types = pd.DataFrame({f"type_{x}": ((t["type_1"] == x) | (t["type_2"] == x)).astype(int) for x in TYPES},
                         index=t.index)
    full = pd.concat([t[STAT_COLS], relative_stats(t).add_suffix("_share"), types], axis=1)
    return {"bst": t[["bst"]], "full": full}


def models() -> dict:
    return {
        "bst_ridge": ("bst", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "full_ridge": ("full", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "full_forest": ("full", lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)),
    }


def cv_scores(X_sets: dict, y: np.ndarray, model_specs: dict, n_splits=N_SPLITS, n_repeats=N_REPEATS) -> pd.DataFrame:
    """Held-out Spearman per fold, per model. Same folds for every model."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    rows = []
    n = len(y)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(n))):
        row = {"fold": fold}
        for name, (feat, make) in model_specs.items():
            X = X_sets[feat].to_numpy()
            pred = make().fit(X[tr], y[tr]).predict(X[te])
            row[name] = spearmanr(y[te], pred).statistic
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame, baseline="bst_ridge") -> dict:
    out = {}
    for m in scores.columns.drop("fold"):
        s = {"mean_heldout_spearman": round(float(scores[m].mean()), 3),
             "std": round(float(scores[m].std()), 3)}
        if m != baseline:
            diff = scores[m] - scores[baseline]
            s["mean_gain_over_bst"] = round(float(diff.mean()), 3)
            s["share_of_folds_beating_bst"] = round(float((diff > 0).mean()), 2)
        out[m] = s
    return out


if __name__ == "__main__":
    from src.models.competitive_analysis import load_analysis_table

    t = load_analysis_table()
    y = target(t["usage_mean"])
    scores = cv_scores(feature_sets(t), y, models())
    summary = {"population": len(t), "cv": f"{N_SPLITS}-fold x {N_REPEATS} repeats",
               "target": "log10(usage + 0.0001)", "models": summarize(scores)}
    print(json.dumps(summary, indent=2))
    scores.to_csv(RESULTS_DIR / "competitive_prediction_folds.csv", index=False)
    (RESULTS_DIR / "competitive_prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
