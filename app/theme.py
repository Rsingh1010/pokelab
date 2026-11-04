"""Design system: colors, CSS, and the Plotly chart template. Change the look here only."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORS = {
    "bg": "#101827", "bg2": "#172235", "card": "#1D2B42",
    "red": "#E63946", "blue": "#38BDF8",
    "text": "#F1F5F9", "muted": "#94A3B8", "border": "#334155",
    "green": "#22C55E", "amber": "#F59E0B",
}

# Widely used community colors for the 18 types
TYPE_COLORS = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0", "electric": "#F7D02C",
    "grass": "#7AC74C", "ice": "#96D9D6", "fighting": "#C22E28", "poison": "#A33EA1",
    "ground": "#E2BF65", "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC", "dark": "#705746",
    "steel": "#B7B7CE", "fairy": "#D685AD",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1300px; }}
section[data-testid="stSidebar"] {{ background: {COLORS['bg2']}; border-right: 1px solid {COLORS['border']}; }}
.pl-card {{ background: {COLORS['card']}; border: 1px solid {COLORS['border']}; border-radius: 14px;
           padding: 1.1rem 1.25rem; margin-bottom: 0.9rem; }}
.pl-card h4 {{ margin: 0 0 .4rem 0; font-size: 1rem; }}
.pl-muted {{ color: {COLORS['muted']}; font-size: .9rem; }}
.pl-mono {{ font-family: 'JetBrains Mono', monospace; color: {COLORS['muted']}; font-size: .85rem; white-space: nowrap; }}
.pl-kpi {{ font-size: 1.9rem; font-weight: 700; line-height: 1.1; }}
.pl-kpi-label {{ color: {COLORS['muted']}; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
.pl-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
            font-weight: 600; color: #101827; margin-right: 4px; text-transform: capitalize; }}
.pl-tag {{ display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: .75rem;
          border: 1px solid {COLORS['border']}; color: {COLORS['muted']}; margin: 0 4px 4px 0; }}
.pl-status-done {{ color: {COLORS['green']}; font-weight: 600; font-size: .8rem; }}
.pl-status-pending {{ color: {COLORS['amber']}; font-weight: 600; font-size: .8rem; }}
.pl-stat-row {{ display: flex; align-items: center; gap: .6rem; margin: 3px 0; }}
.pl-stat-name {{ width: 70px; color: {COLORS['muted']}; font-size: .8rem; }}
.pl-stat-val {{ width: 34px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: .85rem; }}
.pl-stat-bar {{ flex: 1; background: {COLORS['bg']}; border-radius: 4px; height: 8px; overflow: hidden; }}
.pl-stat-fill {{ height: 100%; border-radius: 4px; }}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: {COLORS['red']} !important; border-radius: 10px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: #fff !important; font-weight: 600; }}
.pl-team {{ display: flex; flex-wrap: nowrap; }}
.pl-team img {{ width: 40px; height: 40px; flex: 0 0 auto; background: {COLORS['bg']}; border: 1px solid {COLORS['border']};
              border-radius: 8px; margin: 0 4px 6px 0; image-rendering: pixelated; }}
.pl-check {{ color: {COLORS['green']}; margin-right: .35rem; }}
.pl-row {{ display: flex; justify-content: space-between; align-items: center; padding: .55rem 0;
          border-bottom: 1px solid {COLORS['border']}; }}
.pl-row:last-child {{ border-bottom: none; }}
.pl-hero h1 {{ margin-bottom: 0; }}
[data-testid="stLogo"] {{ height: 3rem; max-width: 100%; }}
.pl-hero .pl-muted {{ font-size: 1.05rem; }}
.pl-note {{ border-left: 3px solid {COLORS['amber']}; padding: .5rem .9rem; background: {COLORS['bg2']};
           border-radius: 0 8px 8px 0; color: {COLORS['muted']}; font-size: .88rem; margin: .5rem 0 1rem; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)
    pio.templates["pokelab"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["bg2"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        colorway=[COLORS["blue"], COLORS["red"], COLORS["amber"], COLORS["green"], "#A78BFA", "#F472B6"],
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    ))
    pio.templates.default = "pokelab"
