"""Competitive Usage (EXP-06): how stat profiles relate to Smogon usage. Popularity, not win rate."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data
from app.components import ui
from app.theme import COLORS

HIGH_USAGE = 0.0452   # same cutoff as src/models/competitive_analysis.py (Smogon's tier threshold)

st.title("Competitive Usage")
try:
    df = data.pokemon()
    summ = data.result_json("competitive_summary.json")
    pred = data.result_json("competitive_prediction.json")
    folds = data.result_csv("competitive_prediction_folds.csv")
    by_cluster = data.result_csv("competitive_by_cluster.csv", index_col=0)
    by_type = data.result_csv("competitive_by_type.csv", index_col=0)
    pairs = data.result_csv("competitive_pairs.csv", index_col=0)
    unmatched = data.result_csv("usage_unmatched.csv")
except (FileNotFoundError, data.MissingResults):
    st.info("Competitive results are not available. Run `python -m src.data.fetch_competitive`, then "
            "`python -m src.models.competitive_analysis` and `python -m src.models.competitive_prediction`. "
            "The rest of PokéLab works without them.")
    st.stop()

ui.note("<b>Usage = share of teams that use a Pokémon</b> (Smogon, Gen 9 National Dex, skill level 1630, "
        "mean of Jul-Sep 2026). It measures <b>popularity, not win rate or strength</b>. Moves, abilities and "
        "items drive it too, and PokéLab does not model them.")

legal = df[df["legal"].astype(str) == "True"].copy()
legal["log_usage"] = np.log10(legal["usage_mean"] + 1e-4)
bst = summ["spearman_vs_usage"]["bst"]

k = st.columns(4)
with k[0]:
    ui.kpi(f"{summ['population']}", "legal fully evolved")
with k[1]:
    ui.kpi(f"{summ['with_any_usage']}", "used at least once")
with k[2]:
    ui.kpi(f"{summ['n_high_usage']}", f"at or above {HIGH_USAGE:.2%}")
with k[3]:
    ui.kpi(f"{bst['rho']:.2f}", "BST vs usage (Spearman)")

# selected Pokémon in context
name = st.session_state.get("selected_name", "Charizard")
row = df[df["name"] == name].iloc[0]
if name in set(legal["name"]):
    rank = int((legal["usage_mean"] > row["usage_mean"]).sum()) + 1
    st.markdown(f"**{name}:** {row['usage_mean']:.2%} of teams · tier {row['natdex_tier']} · "
                f"usage rank #{rank} of {len(legal)}")
elif row["is_fully_evolved"]:
    st.markdown(f"**{name}** is banned in National Dex (tier {row['natdex_tier']}), so it is excluded here.")
else:
    st.markdown(f"**{name}** is not fully evolved, so it is outside this analysis.")

t_str, t_shape, t_groups, t_twins, t_pred, t_cov = st.tabs(
    ["Strength", "Stat shape", "Groups & types", "Stat twins", "Prediction", "Data coverage"])

with t_str:
    fig = go.Figure()
    fig.add_scatter(x=legal["bst"], y=legal["log_usage"], mode="markers", text=legal["name"],
                    marker=dict(size=7, color=COLORS["blue"], opacity=.55),
                    hovertemplate="<b>%{text}</b><br>BST %{x}<extra></extra>", showlegend=False)
    top = legal.nlargest(8, "usage_mean")
    fig.add_scatter(x=top["bst"], y=top["log_usage"], mode="text", text=top["name"], textposition="top center",
                    showlegend=False, hoverinfo="skip")
    if name in set(legal["name"]):
        fig.add_scatter(x=[row["bst"]], y=[np.log10(row["usage_mean"] + 1e-4)], mode="markers", showlegend=False,
                        marker=dict(size=15, color=COLORS["red"], line=dict(width=2, color=COLORS["text"])))
    fig.add_hline(y=np.log10(HIGH_USAGE + 1e-4), line_dash="dash", line_color=COLORS["amber"],
                  annotation_text="high-usage cutoff")
    fig.update_layout(height=480, xaxis_title="BST", yaxis_title="log10(usage + 0.0001)",
                      title=f"Spearman {bst['rho']} (95% CI {bst['ci95'][0]} to {bst['ci95'][1]})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Strength raises the odds of being used, but there is a wide spread at every BST level.")

with t_shape:
    sp = pd.DataFrame(summ["spearman_vs_usage"]).T
    sp[["lo", "hi"]] = pd.DataFrame(sp["ci95"].tolist(), index=sp.index)
    sp = sp.sort_values("rho")
    colors = [COLORS["blue"] if c else COLORS["muted"] for c in sp["clear_link"]]
    fig = go.Figure()
    for (label, r), color in zip(sp.iterrows(), colors):
        fig.add_scatter(x=[r["lo"], r["hi"]], y=[label, label], mode="lines", line=dict(color=color, width=3),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=sp["rho"], y=sp.index, mode="markers", marker=dict(size=10, color=colors),
                    showlegend=False, hovertemplate="%{y}: %{x}<extra></extra>")
    fig.add_vline(x=0, line_color=COLORS["text"], line_width=1)
    fig.update_layout(height=440, xaxis_title="Spearman rho with usage (95% bootstrap interval)",
                      title="Grey = interval crosses 0: no clear link")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Total strength matters. Stat shape (the role axes) and unusualness (anomaly score) show no clear "
               "link: the same pattern as archetype discovery.")

with t_groups:
    st.markdown(f"Kruskal-Wallis across descriptive clusters: **p = {summ['kruskal_usage_by_cluster_p']:.2f}** "
                f"(no difference). Only {summ['n_high_usage']} Pokémon are high-usage, so differences of a few "
                "Pokémon look large below. Treat them as hints.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        labels = [ui.describe_group(c) for c in by_cluster.index]
        fig.add_bar(x=labels, y=by_cluster["share_of_population"], name="share of all", marker_color=COLORS["border"])
        fig.add_bar(x=labels, y=by_cluster["share_of_high_usage"], name="share of high-usage", marker_color=COLORS["blue"])
        fig.update_layout(height=380, barmode="group", title="Descriptive clusters (not validated)",
                          yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        bt = by_type.sort_values("high_usage_share", ascending=False)
        counts = [f"{round(s * n)}/{int(n)}" for s, n in zip(bt["high_usage_share"], bt["n"])]
        fig = go.Figure(go.Bar(x=bt.index.str.capitalize(), y=bt["high_usage_share"], text=counts,
                               textposition="outside", marker_color=COLORS["blue"]))
        fig.update_layout(height=380, title="Share of each type that is high-usage", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with t_twins:
    st.markdown("Pairs among the closest 25% of nearest neighbours on standardized stats, with the largest usage gaps. "
                "Similar numbers, very different popularity. Only pairs where both are on at least 0.5% of teams: "
                "ratios of near-zero usage are noise.")
    pairs = pairs[pairs[["usage_a", "usage_b"]].min(axis=1) >= 0.005]
    info = df.set_index("name")
    for _, p in pairs.head(5).iterrows():
        cols = st.columns([1, 4, 1, 4, 2])
        for (img_col, txt_col), who, usage in (((cols[0], cols[1]), p["a"], p["usage_a"]),
                                               ((cols[2], cols[3]), p["b"], p["usage_b"])):
            r = info.loc[who]
            img_col.image(r["sprite_url"], width=56)
            txt_col.markdown(f"**{who}** {ui.type_badges(r['type_1'], r['type_2'])}<br>"
                             f"<span class='pl-mono'>{int(r['hp'])}/{int(r['attack'])}/{int(r['defense'])}/"
                             f"{int(r['sp_attack'])}/{int(r['sp_defense'])}/{int(r['speed'])}</span> · "
                             f"<b>{usage:.2%}</b>", unsafe_allow_html=True)
        lo, hi = sorted([p["usage_a"], p["usage_b"]])
        cols[4].markdown(f"<div class='pl-kpi' style='font-size:1.3rem'>{hi / lo:.0f}×</div>"
                         f"<div class='pl-kpi-label'>usage ratio</div>" if lo > 0 else
                         "<div class='pl-kpi-label'>one never used</div>", unsafe_allow_html=True)
        st.divider()
    st.caption("Stats shown as HP/Atk/Def/SpA/SpD/Spe. The gap comes from what PokéLab does not model: "
               "typing interactions, abilities, moves, items.")

with t_pred:
    m = pred["models"]
    names = {"bst_ridge": "BST only (baseline)", "full_ridge": "Stats + shares + types (linear)",
             "full_forest": "Stats + shares + types (random forest)"}
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for key, color in zip(names, (COLORS["muted"], COLORS["blue"], COLORS["red"])):
            fig.add_box(y=folds[key], name=names[key], marker_color=color)
        fig.update_layout(height=400, title="Held-out Spearman across 50 folds", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        for key, color in (("full_ridge", COLORS["blue"]), ("full_forest", COLORS["red"])):
            fig.add_histogram(x=folds[key] - folds["bst_ridge"], name=names[key], marker_color=color,
                              opacity=.6, nbinsx=20)
        fig.add_vline(x=0, line_color=COLORS["text"])
        fig.update_layout(height=400, barmode="overlay", title="Gain over BST-only, per fold",
                          xaxis_title="Spearman difference", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    fr, ff = m["full_ridge"], m["full_forest"]
    st.markdown(f"Linear: **{fr['mean_gain_over_bst']:+.3f}**, better than BST in {fr['share_of_folds_beating_bst']:.0%} "
                f"of folds. Forest: **{ff['mean_gain_over_bst']:+.3f}**, better in {ff['share_of_folds_beating_bst']:.0%}. "
                "Beyond total strength, stats and typing add little; the more flexible model overfits.")
    st.caption("5-fold cross-validation repeated 10 times, same folds for every model. No tiers or usage-derived "
               "inputs; scaling fit inside each training fold; hyperparameters fixed in advance.")

with t_cov:
    kind = unmatched["smogon_name"].map(
        lambda n: "Mega Evolutions" if "-Mega" in n else
        "Regional forms" if re.search(r"-(Alola|Galar|Hisui|Paldea)", n) else "Other forms / not fully evolved")
    mass = unmatched.groupby(kind)["usage"].sum()
    total = mass.sum() + legal["usage_mean"].sum()
    shares = pd.concat([pd.Series({"Matched to a PokéLab row": legal["usage_mean"].sum()}), mass]) / total
    fig = go.Figure(go.Bar(x=shares.values, y=shares.index, orientation="h", marker_color=COLORS["blue"],
                           text=[f"{v:.0%}" for v in shares.values], textposition="outside"))
    fig.update_layout(height=300, xaxis_tickformat=".0%", title="Where competitive usage goes")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("PokéLab has one default form per species, so usage of other forms is not credited to any row: "
                "Landorus-Therian's usage is not given to Landorus (Incarnate). Each row's usage is for **that "
                "exact stat profile**. Eviolite users such as Chansey and Porygon2 are excluded as not fully evolved.")
    st.dataframe(unmatched.head(20).assign(usage=lambda d: d["usage"] * 100), hide_index=True,
                 use_container_width=True,
                 column_config={"usage": st.column_config.NumberColumn("usage %", format="%.2f")})
