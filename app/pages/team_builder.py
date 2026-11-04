"""Team Builder: the integer program, live. Set weights and rules, lock favourites, get a proven-best team."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import data
from app.components import ui
from app.theme import COLORS
from src.optimization.team_optimizer import (DEFAULT_MAX_PER_TYPE, DEFAULT_MAX_WEAK, DEFAULT_MIN_BST,
                                             DEFAULT_WEIGHTS, TERMS)
from src.optimization.type_chart import TYPES, defensive_profile

try:
    df = data.pokemon()
except data.MissingResults as e:
    st.error(str(e))
    st.stop()

fe_names = sorted(df.loc[df["is_fully_evolved"], "name"])
selected = st.session_state.get("selected_name", "Charizard")
st.session_state.setdefault("locked", [])

st.title("Team Builder")
st.markdown('<p class="pl-muted">An integer program picks the six Pokémon that maximize your weighted score '
            'under the rules below. The answer is proven optimal, not a guess. It measures type coverage and '
            'stat balance, <b>not battle outcomes</b>: abilities, moves and items are not modeled.</p>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------- lock the sidebar Pokémon (or its final forms)
def final_forms(name: str) -> list:
    """Fully evolved descendants of a Pokémon (Charmander -> Charizard, Eevee -> 8 options)."""
    slug = df.loc[df["name"] == name, "species_slug"].item()
    children = df.groupby("evolves_from")["species_slug"].apply(list).to_dict()
    out, todo = [], [slug]
    while todo:
        s = todo.pop()
        for c in children.get(s, []):
            (out if c not in children else todo).append(c)
    return df[df["species_slug"].isin(out)].sort_values("dex_id")["name"].tolist()


def lock(name: str):
    if name not in st.session_state["locked"] and len(st.session_state["locked"]) < 5:
        st.session_state["locked"] = st.session_state["locked"] + [name]


if selected in fe_names:
    st.button(f"Lock {selected} into the team", on_click=lock, args=(selected,), icon=":material/lock:")
else:
    options = final_forms(selected)
    if options:
        st.caption(f"{selected} is not fully evolved. Lock a final form instead:")
        cols = st.columns(min(len(options), 8))
        for col, o in zip(cols, options):
            col.button(o, key=f"final_{o}", on_click=lock, args=(o,), use_container_width=True)

# ---------------------------------------------------------------- settings (form: solve only on click)
with st.form("settings"):
    st.multiselect("Locked Pokémon (up to 5)", fe_names, key="locked", max_selections=5,
                   help="Locked Pokémon skip the BST minimum and legendary filter: you asked for them.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Objective weights** (rescaled to sum to 1)")
        labels = {"coverage": "Offensive coverage", "defense": "Defensive coverage",
                  "roles": "Role coverage", "balance": "Stat floor"}
        weights = tuple(st.slider(labels[t], 0, 100, int(DEFAULT_WEIGHTS[t] * 100), step=5, key=f"w_{t}")
                        for t in TERMS)
    with b:
        st.markdown("**Rules**")
        min_bst = st.slider("Minimum BST", 300, 600, DEFAULT_MIN_BST, step=10)
        max_weak = st.slider("Max weakness load per attacking type", 1, 4, DEFAULT_MAX_WEAK,
                             help="A 2× weakness counts 1, a 4× weakness counts 2.")
        max_per_type = st.slider("Max members sharing a type", 1, 3, DEFAULT_MAX_PER_TYPE)
        allow_legendary = st.toggle("Allow legendary & mythical", value=False)
    st.form_submit_button("Build team", type="primary", use_container_width=True)

if sum(weights) == 0:
    st.error("Set at least one weight above 0.")
    st.stop()

settings = (tuple(w / 100 for w in weights), min_bst, allow_legendary, max_weak, max_per_type,
            tuple(st.session_state["locked"]))
try:
    res = data.solve_team(*settings)
except ValueError as e:          # InfeasibleTeamError and friends
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------- team cards
s = res["scores"]
k = st.columns(5)
with k[0]:
    ui.kpi(f"{round(s['coverage'] * 36)}/36", "offensive coverage")
with k[1]:
    ui.kpi(f"{round(s['defense'] * 36)}/36", "defensive coverage")
with k[2]:
    ui.kpi(f"{round(s['roles'] * 4)}/4", "roles covered")
with k[3]:
    ui.kpi(f"{s['balance']:.0%}", "stat floor (percentile)")
with k[4]:
    ui.kpi(f"{s['total']:.3f}", f"total score · pool {res['pool_size']}")

info = df.set_index("name")
members = res["members"].set_index("name")
cols = st.columns(3)
for i, name in enumerate(res["names"]):
    r, m = info.loc[name], members.loc[name]
    locked_tag = ' <span class="pl-tag">locked</span>' if name in st.session_state["locked"] else ""
    roles = "".join(f'<span class="pl-tag">{x}</span>' for x in m["roles"]) or '<span class="pl-muted">no role tag</span>'

    def line(label, items, empty="none"):
        txt = ", ".join(items) if items else empty
        return f'<div style="margin:.2rem 0"><span class="pl-kpi-label">{label}</span> {html.escape(txt)}</div>'

    body = (f'<div style="display:flex;gap:.8rem;align-items:center">'
            f'<img src="{r["sprite_url"]}" width="72" style="image-rendering:pixelated">'
            f'<div>{ui.type_badges(r["type_1"], r["type_2"])}<div class="pl-mono">BST {int(r["bst"])} · '
            f'contribution {m["contribution"]:+.3f}</div>{roles}{locked_tag}</div></div>'
            + line("Only answer to", m["only_answer"])
            + line("Patches teammates vs", m["patches"])
            + line("Weak to", m["weak_to"]))
    with cols[i % 3]:
        ui.card(name, body)
st.caption("Contribution = team score minus the score of the other five alone. Near zero means the member "
           "mostly fills a slot within the rules.")

# ---------------------------------------------------------------- defensive chart
st.subheader("Defensive matchups")
chart = data.type_chart()
prof = pd.DataFrame({n: defensive_profile(chart, info.loc[n, "type_1"], info.loc[n, "type_2"]) for n in res["names"]}).T
labels = prof.map(lambda v: {0: "0", 0.25: "¼", 0.5: "½", 1: "", 2: "2", 4: "4"}[v])
fig = go.Figure(go.Heatmap(
    z=prof.map(lambda v: {0: -3, 0.25: -2, 0.5: -1, 1: 0, 2: 1, 4: 2}[v]).values,
    x=[t.capitalize() for t in TYPES], y=prof.index, text=labels.values, texttemplate="%{text}",
    colorscale=[[0, COLORS["green"]], [0.5, COLORS["bg2"]], [1, COLORS["red"]]], zmin=-3, zmax=3,
    showscale=False, hovertemplate="%{y} takes %{text}× from %{x}<extra></extra>"))
fig.update_layout(height=60 + 42 * len(prof), xaxis_side="top", margin=dict(t=60))
st.plotly_chart(fig, use_container_width=True)

summary = res["summary"]
g1, g2 = st.columns(2)
with g1:
    ui.card("Gaps", "".join([
        f'<div><span class="pl-kpi-label">No super-effective answer</span> {", ".join(summary["no_super_effective_answer"]) or "none"}</div>',
        f'<div><span class="pl-kpi-label">Only one answer</span> {", ".join(summary["only_one_answer"]) or "none"}</div>',
        f'<div><span class="pl-kpi-label">Nothing resists</span> {", ".join(summary["nothing_resists"]) or "none"}</div>',
        f'<div><span class="pl-kpi-label">Missing roles</span> {", ".join(summary["roles_missing"]) or "none"}</div>']))
with g2:
    load_txt = ", ".join(f"{t} ({n})" for t, n in summary["weakness_load_at_limit"].items()) or "none"
    quad_txt = "; ".join(n + ": " + ", ".join(t) for n, t in summary["quad_weaknesses"].items()) or "none"
    ui.card("Risk", "".join([
        f'<div><span class="pl-kpi-label">Weakness load at the limit</span> {load_txt}</div>',
        f'<div><span class="pl-kpi-label">4× weaknesses</span> {quad_txt}</div>',
        f'<div><span class="pl-kpi-label">Weakest team stat</span> {summary["weakest_stat"]} '
        f'({summary["weakest_stat_percentile"]:.0%} percentile)</div>']))

# ---------------------------------------------------------------- live benchmark
st.subheader("How does this compare? (your settings, live)")
bench = data.live_benchmark(res["M"], settings, settings[0], max_weak, max_per_type)
rand = bench["random"]
valid = rand[rand["follows_rules"]]
fig = go.Figure()
fig.add_histogram(x=valid["total"], nbinsx=40, marker_color=COLORS["border"], name="random teams that follow the rules")
fig.add_vline(x=s["total"], line_color=COLORS["red"], line_width=3, annotation_text="optimized")
if bench["greedy"]:
    fig.add_vline(x=bench["greedy"]["total"], line_color=COLORS["amber"], line_dash="dash", annotation_text="greedy",
                  annotation_position="bottom right")
tb = bench["top_bst"]
fig.add_vline(x=tb["total"], line_color=COLORS["blue"], line_dash="dot",
              annotation_text="top BST" + ("" if tb["follows_rules"] else " (breaks rules)"),
              annotation_position="top left")
fig.update_layout(height=340, xaxis_title="total score", yaxis_title="count", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"{len(rand):,} random teams from the same pool: {rand['follows_rules'].mean():.1%} follow the rules, "
           f"{(rand['total'] >= s['total'] - 1e-9).mean():.2%} reach the optimized score. "
           "The optimizer is judged on the score it maximizes, so the gap and rule-following are the result, not the win.")
