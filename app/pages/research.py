"""Research Lab: every experiment record, and the evidence behind each decision. Reads saved results only."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data
from app.components import ui
from app.theme import COLORS

st.title("Research Lab")
st.markdown('<p class="pl-muted">Each experiment was defined before running it. Selection rules were written '
            'down before looking at cluster profiles. Everything here is read from <code>results/</code>.</p>',
            unsafe_allow_html=True)

try:
    exps = data.experiments()
    sweep, sel = data.result_csv("kmeans_k_sweep.csv"), data.result_csv("k_selection.csv")
    gmm = data.result_csv("gmm_selection.csv")
except FileNotFoundError as e:
    st.error(f"Missing result file: {e}")
    st.stop()

SETUPS = {"Fully evolved · stat shares (main)": ("fully_evolved", "relative"),
          "Fully evolved · standardized": ("fully_evolved", "standardized"),
          "All · stat shares": ("all", "relative"),
          "All · standardized": ("all", "standardized")}
MIN_ARI, MIN_SHARE = 0.6, 0.05


def line(fig, x, y, name, color, dash=None, mode="lines+markers"):
    fig.add_scatter(x=x, y=y, name=name, mode=mode, line=dict(color=color, dash=dash))


tab_exp, tab_k, tab_gmm, tab_evo, tab_anom = st.tabs(
    ["Experiments", "Choosing k", "K-Means vs GMM", "Strength & evolution", "Anomalies"])

# ---------------------------------------------------------------- experiments
with tab_exp:
    done = sum(r["status"].startswith("completed") for r in exps.values())
    st.caption(f"{done} of {len(exps)} records completed")
    for exp_id, rec in exps.items():
        with st.expander(f"{exp_id} · {rec['title']}", expanded=(exp_id == "EXP-04")):
            st.markdown(ui.status(rec["status"]), unsafe_allow_html=True)
            for field, label in (("question", "Question"), ("hypothesis", "Hypothesis"),
                                 ("methodology", "Method"), ("interpretation", "Result")):
                if rec.get(field):
                    st.markdown(f"**{label}.** {rec[field]}")
            if rec.get("limitations"):
                ui.note(f"<b>Limitations.</b> {rec['limitations']}")
            if rec.get("metrics"):
                with st.popover("Raw metrics"):
                    st.code(json.dumps(rec["metrics"], indent=2), language="json")
            if rec.get("data_files"):
                st.caption("Source files: " + ", ".join(f"`{f}`" for f in rec["data_files"]))

# ---------------------------------------------------------------- choosing k
with tab_k:
    label = st.selectbox("Feature setup", list(SETUPS))
    subset, rep = SETUPS[label]
    s = sweep[(sweep["subset"] == subset) & (sweep["representation"] == rep)]
    k = sel[(sel["subset"] == subset) & (sel["representation"] == rep)]

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        line(fig, s["k"], s["inertia"], "inertia", COLORS["blue"])
        fig.update_layout(height=320, title="Elbow: inertia (within-cluster spread)", xaxis_title="k",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Always falls as k grows. Look for a bend; a smooth curve means no natural k.")
    with b:
        fig = go.Figure()
        line(fig, k["k"], k["silhouette"], "real data", COLORS["blue"])
        line(fig, k["k"], k["baseline_p95"], "shuffled data (95th pct)", COLORS["muted"], dash="dash")
        fig.update_layout(height=320, title="Silhouette vs a no-structure baseline", xaxis_title="k",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shuffling each stat column destroys how stats combine. Real data must beat it.")

    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_scatter(x=k["k"], y=k["stability_ari"], mode="lines+markers", line=dict(color=COLORS["blue"]),
                        error_y=dict(type="data", array=k["stability_ari_std"], color=COLORS["border"]),
                        name="bootstrap ARI")
        fig.add_hline(y=MIN_ARI, line_dash="dash", line_color=COLORS["red"], annotation_text="required 0.60")
        fig.update_layout(height=320, title="Stability across 20 bootstrap resamples (ARI)", xaxis_title="k",
                          yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        line(fig, s["k"], s["davies_bouldin"], "Davies-Bouldin", COLORS["amber"])
        fig.update_layout(height=320, title="Davies-Bouldin (lower = better separated)", xaxis_title="k",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pre-registered rule** (main setup, k = 3 to 8): beats baseline, ARI ≥ 0.60, "
                "smallest cluster ≥ 5%.")
    table = k[["k", "silhouette", "baseline_p95", "stability_ari", "smallest_share",
               "beats_baseline", "stable", "big_enough"]].copy()
    if (subset, rep) == ("fully_evolved", "relative"):
        table["eligible"] = k["eligible"]
    table["smallest_share"] = table["smallest_share"] * 100
    st.dataframe(table.round(3), hide_index=True, use_container_width=True,
                 column_config={"smallest_share": st.column_config.NumberColumn("smallest cluster %", format="%.1f")})

# ---------------------------------------------------------------- K-Means vs GMM
with tab_gmm:
    km = sel[(sel["subset"] == "fully_evolved") & (sel["representation"] == "relative")]
    st.caption("Main setup only: fully evolved Pokémon, stat shares.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        line(fig, km["k"], km["stability_ari"], "K-Means", COLORS["blue"])
        line(fig, gmm["k"], gmm["stability_ari"], "GMM", COLORS["red"])
        fig.add_hline(y=MIN_ARI, line_dash="dash", line_color=COLORS["muted"])
        fig.update_layout(height=340, title="Stability (bootstrap ARI)", xaxis_title="k", yaxis_range=[0, 1],
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        line(fig, gmm["k"], gmm["bic"], "BIC", COLORS["amber"])
        best = gmm.loc[gmm["bic"].idxmin()]
        fig.add_annotation(x=best["k"], y=best["bic"], text=f"lowest BIC: k = {int(best['k'])}", showarrow=True)
        fig.update_layout(height=340, title="GMM BIC (lower = better fit for its complexity)", xaxis_title="k",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    fig.add_scatter(x=gmm["mean_top_probability"], y=gmm["stability_ari"], mode="markers+text",
                    text=[f"k={int(v)}" for v in gmm["k"]], textposition="top center",
                    marker=dict(size=11, color=COLORS["red"]))
    fig.update_layout(height=360, title="GMM: confident, yet unstable",
                      xaxis_title="mean top probability (how sure GMM is per Pokémon)",
                      yaxis_title="stability across resamples", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    lo, hi = gmm.loc[gmm["k"].between(3, 8), "mean_top_probability"].agg(["min", "max"])
    ui.note(f"A model's probabilities assume the model is right. For k = 3 to 8, GMM is on average "
            f"{lo:.0%}-{hi:.0%} sure of each assignment, yet its groups change between resamples. "
            "Confidence is not stability.")

# ---------------------------------------------------------------- strength & evolution
with tab_evo:
    xt = data.result_csv("evolution_crosstab_k2.csv", index_col=0)
    a, b = st.columns([2, 3])
    with a:
        fig = go.Figure(go.Heatmap(z=xt.values, x=["not fully evolved", "fully evolved"],
                                   y=[f"cluster {c}" for c in xt.index], text=xt.values, texttemplate="%{text}",
                                   colorscale=[[0, COLORS["bg2"]], [1, COLORS["blue"]]], showscale=False))
        fig.update_layout(height=320, title="All Pokémon, standardized stats, k = 2")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        rec = exps.get("EXP-01", {})
        st.markdown(f"**EXP-01.** {rec.get('interpretation', 'Pending.')}")
        agreement = max(np.trace(xt.values), np.trace(xt.values[::-1])) / xt.values.sum()
        ui.kpi(f"{agreement:.0%}", "of Pokémon: k = 2 cluster matches evolution stage")

# ---------------------------------------------------------------- anomalies
with tab_anom:
    scores = data.result_csv("anomaly_scores.csv")
    summ = data.result_json("anomaly_summary.json")
    views = {"Fully evolved · stat shares": "fe_relative", "All · stat shares": "relative", "All · raw stats": "raw"}
    v = views[st.radio("View", list(views), horizontal=True)]
    c = st.columns(3)
    with c[0]:
        ui.kpi(f"{summ[v]['seed_stability_spearman']:.3f}", "rank agreement across seeds")
    with c[1]:
        ui.kpi(f"{summ[v]['spearman_vs_max_abs_z']:.2f}", "agreement with single-stat rule")
    with c[2]:
        ui.kpi(f"{summ[v]['n_scored']}", "Pokémon scored")
    d = scores.dropna(subset=[f"{v}_score"])
    a, b = st.columns([3, 2])
    with a:
        fit = np.polyfit(d[f"{v}_max_abs_z"], d[f"{v}_score"], 1)
        resid = d[f"{v}_score"] - np.polyval(fit, d[f"{v}_max_abs_z"])
        fig = go.Figure()
        fig.add_scatter(x=d[f"{v}_max_abs_z"], y=d[f"{v}_score"], mode="markers", text=d["name"],
                        marker=dict(size=6, color=COLORS["blue"], opacity=.5),
                        hovertemplate="<b>%{text}</b><extra></extra>", showlegend=False)
        top = d.loc[resid.nlargest(6).index]
        fig.add_scatter(x=top[f"{v}_max_abs_z"], y=top[f"{v}_score"], mode="markers+text", text=top["name"],
                        textposition="top center", marker=dict(size=9, color=COLORS["red"]), showlegend=False)
        fig.update_layout(height=420, title="Isolation Forest vs the most extreme single stat",
                          xaxis_title="most extreme single stat (|z|)", yaxis_title="anomaly score")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red: scored much higher than their single most extreme stat predicts, "
                   "i.e. unusual combinations.")
    with b:
        top15 = d.nsmallest(15, f"{v}_rank")[["name", f"{v}_score", f"{v}_driver"]]
        top15.columns = ["Pokémon", "score", "most extreme stat"]
        st.dataframe(top15.round(3), hide_index=True, use_container_width=True, height=420)
    ui.note("An anomaly means rare in this dataset, not wrong, broken or strong. "
            "'Flagged' (top 5%) is a reporting cutoff, not a truth threshold.")
