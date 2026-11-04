"""Overview dashboard: one Pokémon, the role map, the default team, findings and evaluation. All real data."""
import html

import streamlit as st

from app import data
from app.components import charts, ui

try:
    df, exps, meta = data.pokemon(), data.experiments(), data.meta()
except data.MissingResults as e:
    st.error(str(e))
    st.stop()

name = st.session_state.get("selected_name", "Charizard")
row = df[df["name"] == name].iloc[0]
n_fe = int(df["is_fully_evolved"].sum())

# ---------------------------------------------------------------- row 1: profile + map
left, right = st.columns([5, 7], gap="large")
with left:
    ui.profile_card(row, n_fe)
    st.page_link("pages/pokedex.py", label="Open in Pokédex", icon=":material/arrow_forward:")
with right:
    explained = sum(meta["role_map_explained_variance"])
    view = "role" if row["is_fully_evolved"] else "strength"
    fig = charts.stat_map(df, view=view, color_by="BST", highlight=name, explained=explained if view == "role" else None,
                          height=440)
    st.plotly_chart(fig, use_container_width=True)
    if view == "strength":
        st.caption(f"{name} is not fully evolved, so it is shown on the strength map (all Pokémon).")
    ui.note("No stable archetypes were found (EXP-04): stat profiles form a continuum. "
            "The map shows position on the two axes that did hold up.")
    st.page_link("pages/archetypes.py", label="Open Archetype Explorer", icon=":material/arrow_forward:")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------- row 2: team, findings, evaluation
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    team = data.default_team()
    sprites = df.set_index("name").loc[team["names"], "sprite_url"]
    imgs = "".join(f'<img src="{u}" title="{html.escape(n)}">' for n, u in sprites.items())
    s = team["scores"]
    checks = [f"Offensive coverage {round(s['coverage'] * 36)}/36",
              f"Defensive coverage {round(s['defense'] * 36)}/36",
              f"Roles covered {round(s['roles'] * 4)}/4",
              f"Stat floor: {s['balance']:.0%} percentile"]
    lines = "".join(f'<div><span class="pl-check">✓</span>{c}</div>' for c in checks)
    ui.card("Team Builder · default team",
            f'<div class="pl-team">{imgs}</div>{lines}'
            '<p class="pl-muted" style="margin-top:.6rem">Fully evolved, BST ≥ 500, no legendaries. '
            'Coverage counts up to 2 members per type.</p>')
    st.page_link("pages/team_builder.py", label="Build your own team", icon=":material/arrow_forward:")

with c2:
    rows = ""
    for exp_id in ("EXP-04", "EXP-03", "EXP-06", "OPT-01"):
        rec = exps.get(exp_id)
        if rec:
            first = rec["interpretation"].split(". ")[0].rstrip(".") + "."
            rows += (f'<div class="pl-row"><div><b>{html.escape(rec["title"])}</b>'
                     f'<div class="pl-muted">{html.escape(first)}</div></div>'
                     f'<span class="pl-mono">{exp_id}</span></div>')
    ui.card("Research findings", rows)
    st.page_link("pages/research.py", label="View all experiments", icon=":material/arrow_forward:")

with c3:
    ev = data.evaluation_numbers()
    tiles = (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.5rem">'
             f'<div><div class="pl-kpi" style="font-size:1.4rem">{ev["silhouette"]:.2f}</div>'
             f'<div class="pl-kpi-label">Silhouette</div></div>'
             f'<div><div class="pl-kpi" style="font-size:1.4rem">{ev["stability"]:.2f}</div>'
             f'<div class="pl-kpi-label">Stability ARI</div></div>'
             f'<div><div class="pl-kpi" style="font-size:1.4rem">{ev["explained_2d"]:.0%}</div>'
             f'<div class="pl-kpi-label">Map variance</div></div></div>')
    limits = ("<ul class='pl-muted' style='margin:.6rem 0 0 1rem;padding:0'>"
              f"<li>k = 5 shown: needed stability ≥ 0.60; eligible k: {ev['eligible_k'] or 'none'}.</li>"
              "<li>Silhouette near 0.2 = weak separation (baseline "
              f"{ev['baseline_p95']:.2f}).</li>"
              "<li>Usage is popularity, not win rate.</li>"
              "<li>Abilities, moves and items are not modeled.</li></ul>")
    ui.card("Model evaluation · fully evolved, stat shares, k = 5", tiles + limits)
