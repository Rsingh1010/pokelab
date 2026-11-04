"""Pokédex: browse, filter and inspect any Pokémon. Click a row to select it everywhere in the app."""
import streamlit as st

from app import data
from app.components import ui
from src.optimization.type_chart import TYPES, defensive_profile

try:
    df = data.pokemon()
except data.MissingResults as e:
    st.error(str(e))
    st.stop()

st.title("Pokédex")
n_fe = int(df["is_fully_evolved"].sum())

# ---------------------------------------------------------------- selected Pokémon
name = st.session_state.get("selected_name", "Charizard")
row = df[df["name"] == name].iloc[0]
left, right = st.columns([3, 2], gap="large")
with left:
    ui.profile_card(row, n_fe)
with right:
    chart = data.type_chart()
    ui.card("Defensive matchups", ui.matchup_badges(defensive_profile(chart, row["type_1"], row.get("type_2"))))
    st.markdown('<div class="pl-kpi-label" style="margin-bottom:.4rem">Most similar stats (all Pokémon)</div>',
                unsafe_allow_html=True)
    sims = row["similar"].split("|")
    sprite = df.set_index("name")["sprite_url"]
    cols = st.columns(len(sims))
    for col, s in zip(cols, sims):
        with col:
            st.image(sprite[s], width=56)
            st.button(s, key=f"sim_{s}", on_click=ui.select, args=(s,), use_container_width=True)
    st.caption("Nearest neighbours on standardized base stats. Same numbers does not mean same role in battle.")

st.divider()

# ---------------------------------------------------------------- browse
st.subheader("Browse")
f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
types = f1.multiselect("Type", TYPES, format_func=str.capitalize)
gens = f2.multiselect("Generation", list(range(1, 10)))
category = f3.selectbox("Category", ["All", "Regular", "Legendary", "Mythical"])
evolved = f4.selectbox("Evolution", ["All", "Fully evolved", "Not fully evolved"])

view = df.copy()
if types:
    view = view[view["type_1"].isin(types) | view["type_2"].isin(types)]
if gens:
    view = view[view["generation"].isin(gens)]
if category == "Regular":
    view = view[~(view["is_legendary"] | view["is_mythical"])]
elif category != "All":
    view = view[view[f"is_{category.lower()}"]]
if evolved != "All":
    view = view[view["is_fully_evolved"] == (evolved == "Fully evolved")]

view = view.assign(types=view["type_1"].str.capitalize() + view["type_2"].fillna("").map(lambda t: f" / {t.capitalize()}" if t else ""),
                   usage_pct=view["usage_mean"] * 100)
cols = ["sprite_url", "dex_id", "name", "types", "generation", "bst", *["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"], "usage_pct"]
table = view[cols].reset_index(drop=True)


def on_row_select():
    rows = st.session_state["dex_table"].selection.rows
    if rows and rows[0] < len(table):
        ui.select(table.loc[rows[0], "name"])


st.caption(f"{len(table)} Pokémon · click a row to select it")
st.dataframe(
    table, key="dex_table", on_select=on_row_select, selection_mode="single-row",
    hide_index=True, use_container_width=True, height=520,
    column_config={
        "sprite_url": st.column_config.ImageColumn("", width="small"),
        "dex_id": st.column_config.NumberColumn("#", format="%04d"),
        "name": "Name", "types": "Types", "generation": st.column_config.NumberColumn("Gen"),
        "bst": st.column_config.ProgressColumn("BST", min_value=175, max_value=720, format="%d"),
        "hp": "HP", "attack": "Atk", "defense": "Def", "sp_attack": "SpA", "sp_defense": "SpD", "speed": "Spe",
        "usage_pct": st.column_config.NumberColumn("Usage %", format="%.2f",
                                                   help="Gen 9 National Dex, Jul-Sep 2026. Blank = not fully evolved."),
    },
)
