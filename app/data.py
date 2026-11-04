"""All app data loading. Reads only saved files: nothing is trained or fitted here."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

APP_DIR = RESULTS_DIR / "app"


class MissingResults(Exception):
    pass


@st.cache_data
def pokemon() -> pd.DataFrame:
    path = APP_DIR / "pokemon_app.csv"
    if not path.exists():
        raise MissingResults("Run: python -m src.features.build_app_data")
    return pd.read_csv(path)


@st.cache_data
def meta() -> dict:
    return json.loads((APP_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_data
def experiments() -> dict:
    """{id: record} for every saved experiment record, in a fixed order."""
    out = {}
    for p in sorted((RESULTS_DIR / "experiments").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[rec["id"]] = rec
    order = ["EXP-01", "EXP-02", "EXP-03", "EXP-04", "EXP-05", "EXP-06", "ANOM-01", "OPT-01"]
    return {k: out[k] for k in order if k in out} | {k: v for k, v in out.items() if k not in order}


@st.cache_data
def result_csv(name: str, index_col=None) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name, index_col=index_col)


@st.cache_data
def result_json(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def type_chart() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "type_chart.csv", index_col=0)


@st.cache_data
def default_team() -> dict:
    """The optimizer's default team (same settings as OPT-01). Solving takes well under a second."""
    from src.features.preprocessing import load_pokemon
    from src.optimization.team_optimizer import optimize
    from src.optimization.type_chart import load_type_chart

    team, scores, _, _ = optimize(load_pokemon(), load_type_chart())
    return {"names": team["name"].tolist(), "scores": scores}


@st.cache_data
def evaluation_numbers() -> dict:
    """Real headline metrics for the main clustering setup (fully evolved, relative, k=5)."""
    sweep = result_csv("kmeans_k_sweep.csv")
    sel = result_csv("k_selection.csv")
    pick = lambda t: t[(t["subset"] == "fully_evolved") & (t["representation"] == "relative") & (t["k"] == 5)].iloc[0]
    s, k = pick(sweep), pick(sel)
    return {
        "silhouette": float(s["silhouette"]), "davies_bouldin": float(s["davies_bouldin"]),
        "stability": float(k["stability_ari"]), "baseline_p95": float(k["baseline_p95"]),
        "explained_2d": float(sum(meta()["role_map_explained_variance"])),
        "eligible_k": sel.loc[sel["eligible"], "k"].tolist(),
    }


@st.cache_data(show_spinner="Solving the integer program...")
def solve_team(weights: tuple, min_bst: int, allow_legendary: bool, max_weak: int,
               max_per_type: int, locked: tuple) -> dict:
    """Run the optimizer with the user's settings. Cached per settings, so repeat clicks are instant."""
    from src.features.preprocessing import load_pokemon
    from src.optimization.explain import explain_members, explain_team
    from src.optimization.team_optimizer import TERMS, optimize
    from src.optimization.type_chart import load_type_chart

    w = dict(zip(TERMS, weights))
    team, scores, pool, M = optimize(load_pokemon(), load_type_chart(), weights=w, min_bst=min_bst,
                                     allow_legendary=allow_legendary, max_weak=max_weak,
                                     max_per_type=max_per_type, locked=list(locked))
    idx = [M.names.index(n) for n in team["name"]]
    return {"names": team["name"].tolist(), "scores": scores, "members": explain_members(M, idx, w),
            "summary": explain_team(M, idx, w), "M": M, "idx": idx, "weights": w, "pool_size": len(pool)}


@st.cache_data(show_spinner="Scoring baseline teams...")
def live_benchmark(_M, key: tuple, weights: tuple, max_weak: int, max_per_type: int, n_random: int = 3000) -> dict:
    """Random / top-BST / greedy teams for the CURRENT settings. `key` identifies _M for caching."""
    from src.optimization.benchmark import evaluate, greedy_team, random_teams, top_bst_team
    from src.optimization.team_optimizer import TERMS

    w = dict(zip(TERMS, weights))
    rand = pd.DataFrame([evaluate(_M, t, w, max_weak, max_per_type) for t in random_teams(_M, n_random)])
    g = greedy_team(_M, w, max_weak, max_per_type)
    return {"random": rand,
            "top_bst": evaluate(_M, top_bst_team(_M), w, max_weak, max_per_type),
            "greedy": None if g is None else evaluate(_M, g, w, max_weak, max_per_type)}
