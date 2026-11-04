"""Precompute everything the app shows, so the app never fits a model on page load.

Run: python -m src.features.build_app_data
Writes results/app/pokemon_app.csv (one row per species) and results/app/meta.json.

Re-uses the exact functions the analysis used (same seeds), so the app shows the
same clusters, axes and scores as the notebooks and experiment records.
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR
from src.features.preprocessing import STAT_COLS, load_pokemon, make_features
from src.models.clustering import fit_kmeans
from src.models.competitive_analysis import cluster_labels
from src.optimization.team_optimizer import ROLES, add_team_features

APP_DIR = RESULTS_DIR / "app"
N_SIMILAR = 5


def strength_map(df: pd.DataFrame):
    """PCA of standardized stats for ALL Pokémon. PC1 is oriented so higher = stronger."""
    X, _ = make_features(df, "standardized")
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(X)
    coords = pca.transform(X)
    if np.corrcoef(coords[:, 0], df["bst"])[0, 1] < 0:
        coords[:, 0] *= -1
    return pd.DataFrame(coords, index=df.index, columns=["strength_pc1", "strength_pc2"]), pca


def similar_pokemon(df: pd.DataFrame, n=N_SIMILAR) -> pd.Series:
    """Nearest neighbours on standardized stats (all Pokémon). Names, closest first."""
    X, _ = make_features(df, "standardized")
    nn = NearestNeighbors(n_neighbors=n + 1).fit(X)
    _, idx = nn.kneighbors(X)
    names = df["name"].to_numpy()
    return pd.Series(["|".join(names[row[1:]]) for row in idx], index=df.index)


def build() -> tuple[pd.DataFrame, dict]:
    df = load_pokemon()
    out = df.copy()

    # role map (fully evolved, relative stats) + descriptive k=5 clusters + role tags
    fe = add_team_features(df)
    X_rel, _ = make_features(fe, "relative")
    labels = fit_kmeans(X_rel, 5).labels_
    names = cluster_labels(X_rel, labels)
    fe["cluster"] = labels
    fe["cluster_label"] = fe["cluster"].map(names)
    keep = ["speed_axis", "physical_axis", "cluster", "cluster_label", *ROLES]
    out = out.join(fe[keep])
    role_pca = PCA(random_state=RANDOM_SEED).fit(X_rel)

    smap, strength_pca = strength_map(df)
    out = out.join(smap)
    out["similar"] = similar_pokemon(df)

    anomaly = pd.read_csv(RESULTS_DIR / "anomaly_scores.csv")
    a_cols = [c for c in anomaly.columns if c.startswith(("raw_", "fe_relative_"))]
    out = out.merge(anomaly[["dex_id", *a_cols]], on="dex_id", how="left")

    usage_path = PROCESSED_DIR / "usage_gen9natdex.csv"
    if usage_path.exists():
        usage = pd.read_csv(usage_path)[["dex_id", "natdex_tier", "legal", "usage_mean"]]
        out = out.merge(usage, on="dex_id", how="left")

    # What each map axis means: correlation of each stat with the axis (sign already oriented)
    shares = out[STAT_COLS].div(out[STAT_COLS].sum(axis=1), axis=0)
    axis_meaning = {
        "role": {ax: shares.corrwith(out[ax]).round(2).to_dict() for ax in ("speed_axis", "physical_axis")},
        "strength": {ax: out[STAT_COLS].corrwith(out[ax]).round(2).to_dict() for ax in ("strength_pc1", "strength_pc2")},
    }

    meta = {
        "axis_meaning": axis_meaning,
        "role_map_explained_variance": [round(float(v), 3) for v in role_pca.explained_variance_ratio_[:2]],
        "strength_map_explained_variance": [round(float(v), 3) for v in strength_pca.explained_variance_ratio_],
        "cluster_labels": {int(k): v for k, v in names.items()},
        "n_similar": N_SIMILAR,
        "stat_cols": STAT_COLS,
    }
    return out, meta


if __name__ == "__main__":
    APP_DIR.mkdir(parents=True, exist_ok=True)
    table, meta = build()
    table.to_csv(APP_DIR / "pokemon_app.csv", index=False)
    (APP_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(table)} rows x {table.shape[1]} columns to {APP_DIR / 'pokemon_app.csv'}")
    print(json.dumps(meta, indent=2))
