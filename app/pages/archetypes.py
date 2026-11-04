"""Archetype Explorer: where every Pokémon sits on the axes that held up, with optional descriptive groups."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data
from app.components import charts, ui
from app.theme import COLORS
from src.optimization.type_chart import TYPES

try:
    df, meta, exps = data.pokemon(), data.meta(), data.experiments()
except data.MissingResults as e:
    st.error(str(e))
    st.stop()

STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
selected = st.session_state.get("selected_name", "Charizard")

st.title("Archetype Explorer")
st.markdown('<p class="pl-muted">Unsupervised learning looked for stat archetypes. None were stable (EXP-04), '
            'so this map shows the continuous space the analysis did support.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- controls
c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
view_label = c1.segmented_control("View", ["Role map", "Strength map"], default="Role map")
view = "strength" if view_label == "Strength map" else "role"
color_by = c2.selectbox("Color by", list(charts.COLOR_OPTIONS),
                        index=list(charts.COLOR_OPTIONS).index("Descriptive groups (k = 5)") if view == "role" else 0)
types = c3.multiselect("Type filter", TYPES, format_func=str.capitalize)
legend_ok = c4.checkbox("Legendaries", value=True)

shown = df.copy()
if types:
    shown = shown[shown["type_1"].isin(types) | shown["type_2"].isin(types)]
if not legend_ok:
    shown = shown[~(shown["is_legendary"] | shown["is_mythical"])]
if view == "strength" and color_by == "Descriptive groups (k = 5)":
    st.info("Descriptive groups only exist for fully evolved Pokémon on the role map. Showing BST instead.")
    color_by = "BST"

if color_by == "Descriptive groups (k = 5)":
    ev = data.evaluation_numbers()
    ui.note(f"<b>Descriptive grouping, not validated.</b> K-Means with k = 5 reached bootstrap stability "
            f"{ev['stability']:.2f}; the pre-registered rule required 0.60. Silhouette {ev['silhouette']:.2f}. "
            "Group borders move when the data changes slightly.")

# ---------------------------------------------------------------- map (click a point to select it)
explained = sum(meta["role_map_explained_variance"]) if view == "role" else sum(meta["strength_map_explained_variance"])


def on_map_select():
    sel = st.session_state["explorer_map"].selection
    points = sel["points"] if "points" in sel else []
    if points and points[0].get("customdata"):
        ui.select(points[0]["customdata"][0])


fig = charts.stat_map(shown, view=view, color_by=color_by, highlight=selected, explained=explained, height=620)
st.plotly_chart(fig, use_container_width=True, key="explorer_map", on_select=on_map_select,
                selection_mode="points")
st.caption("Click a point to select that Pokémon across the app. Hover for names.")

# ---------------------------------------------------------------- what the axes mean + where the selection sits
a, b = st.columns(2, gap="large")
with a:
    meaning = pd.DataFrame(meta["axis_meaning"][view]).rename(index=ui.STAT_LABELS)
    meaning.columns = (["← bulky · fast →", "← special · physical →"] if view == "role"
                       else ["← weaker · stronger →", "second component"])
    fig_m = go.Figure()
    for i, col in enumerate(meaning.columns):
        fig_m.add_bar(x=meaning.index, y=meaning[col], name=col,
                      marker_color=[COLORS["blue"], COLORS["red"]][i])
    fig_m.update_layout(height=320, barmode="group", yaxis_title="correlation with axis",
                        title=dict(text="What the axes measure", font=dict(size=13)),
                        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_m, use_container_width=True)
    if view == "role":
        st.caption("Stat shares (stat ÷ total). Speed vs Defense drives the first axis; "
                   "Attack vs Sp. Atk/Sp. Def the second.")
    else:
        st.caption("Raw stats. Every stat leans the same way on the first axis: it is overall strength (EXP-01).")

with b:
    row = df[df["name"] == selected].iloc[0]
    x, y = ("speed_axis", "physical_axis") if view == "role" else ("strength_pc1", "strength_pc2")
    if pd.isna(row[x]):
        ui.card(f"{selected}", '<p class="pl-muted">Not fully evolved, so it is not on the role map. '
                               'Switch to the strength map to see it.</p>')
    else:
        pool = df[df[x].notna()]
        px, py = (pool[x] < row[x]).mean(), (pool[y] < row[y]).mean()
        if view == "role":
            body = (f'<p>Faster-leaning than <b>{px:.0%}</b> of fully evolved Pokémon '
                    f'(bulkier than {1 - px:.0%}).</p>'
                    f'<p>More physical-leaning than <b>{py:.0%}</b> (more special than {1 - py:.0%}).</p>'
                    f'<p class="pl-muted">Descriptive group: {ui.describe_group(row["cluster_label"])}</p>')
        else:
            body = (f'<p>Stronger on the main axis than <b>{px:.0%}</b> of all Pokémon.</p>'
                    f'<p class="pl-muted">BST {int(row["bst"])}</p>')
        ui.card(f"Where {selected} sits", ui.type_badges(row["type_1"], row.get("type_2")) + body)

# ---------------------------------------------------------------- group profiles (only when groups are shown)
if color_by == "Descriptive groups (k = 5)":
    st.subheader("Descriptive group profiles")
    fe = df[df["is_fully_evolved"]].copy()
    shares = fe[STATS].div(fe[STATS].sum(axis=1), axis=0)
    diff = (shares.groupby(fe["cluster_label"]).mean() - shares.mean()) * 100
    sizes = fe["cluster_label"].value_counts()
    fig_g = go.Figure()
    for i, stat in enumerate(STATS):
        fig_g.add_bar(x=[f"{ui.describe_group(g)}<br>(n = {sizes[g]})" for g in diff.index], y=diff[stat],
                      name=ui.STAT_LABELS[stat])
    fig_g.update_layout(height=420, barmode="group", yaxis_title="percentage points vs average share",
                        legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_g, use_container_width=True)
    st.caption("Bars above 0: the group puts more of its stat total into that stat than the average fully "
               "evolved Pokémon. Mean BST is similar across groups, so these differ in shape, not strength.")
