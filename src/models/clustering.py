"""K-Means clustering and the k-sweep used to choose the number of clusters."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import RANDOM_SEED, RESULTS_DIR
from src.features.preprocessing import (
    REPRESENTATIONS,
    SUBSETS,
    load_pokemon,
    make_features,
    select_subset,
)

K_RANGE = range(2, 11)
N_INIT = 10  # K-Means restarts from 10 different starting centroids and keeps the best


def fit_kmeans(X, k: int, seed: int = RANDOM_SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)


def score_clustering(X, labels) -> dict:
    """Quality numbers for one clustering. Each answers a different question."""
    sizes = pd.Series(labels).value_counts()
    return {
        "silhouette": silhouette_score(X, labels),          # -1..1, higher = better separated
        "davies_bouldin": davies_bouldin_score(X, labels),  # 0..inf, lower = better separated
        "smallest_cluster": int(sizes.min()),
        "largest_cluster": int(sizes.max()),
    }


def sweep_k(X, ks=K_RANGE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rows = []
    for k in ks:
        model = fit_kmeans(X, k, seed)
        rows.append({"k": k, "inertia": model.inertia_, **score_clustering(X, model.labels_)})
    return pd.DataFrame(rows)


def run_all_sweeps() -> pd.DataFrame:
    """Sweep k for every subset x representation and save one tidy table."""
    df = load_pokemon()
    tables = []
    for subset in SUBSETS:
        for representation in REPRESENTATIONS:
            X, _ = make_features(select_subset(df, subset), representation)
            print(f"Sweeping k for {subset} / {representation} ({len(X)} Pokémon)")
            table = sweep_k(X)
            table.insert(0, "representation", representation)
            table.insert(0, "subset", subset)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


if __name__ == "__main__":
    results = run_all_sweeps()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "kmeans_k_sweep.csv"
    results.to_csv(out, index=False)
    print(f"Saved {out}")
    print(results.round(3).to_string(index=False))
