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
