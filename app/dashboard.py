"""PokéLab app entry point.  Run from the repo root:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import theme  # noqa: E402

st.set_page_config(page_title="PokéLab", page_icon="🔬", layout="wide")
theme.apply()

pages = st.navigation([
    st.Page("pages/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/pokedex.py", title="Pokédex", icon=":material/search:"),
    st.Page("pages/archetypes.py", title="Archetype Explorer", icon=":material/scatter_plot:"),
    st.Page("pages/research.py", title="Research Lab", icon=":material/science:"),
    st.Page("pages/competitive.py", title="Competitive Usage", icon=":material/bar_chart:"),
    st.Page("pages/team_builder.py", title="Team Builder", icon=":material/groups:"),
])

from app import data  # noqa: E402
from app.components import ui  # noqa: E402

GITHUB_URL = "https://github.com/Rsingh1010/pokelab"

st.logo(str(ROOT / "app" / "assets" / "logo.svg"), size="large")   # sits above the page menu

with st.sidebar:
    try:
        names = data.pokemon().sort_values("dex_id")["name"].tolist()
        st.session_state.setdefault("selected_name", "Charizard")
        st.selectbox("Search a Pokémon", names, key="selected_name",
                     help="Your pick carries across every page.")
    except data.MissingResults as e:
        st.error(str(e))

pages.run()

st.sidebar.markdown("---")
st.sidebar.link_button("View on GitHub", GITHUB_URL, use_container_width=True)
st.sidebar.caption("Independent fan-made educational project. Not affiliated with Nintendo, "
                   "Game Freak, or The Pokémon Company. Data: PokéAPI, Smogon.")
