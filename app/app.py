import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scout_tab
import clusters_tab
import about_tab

st.set_page_config(page_title="PL Player Scouting", layout="wide")
st.title("Premier League Player Scouting & Similarity")

tab1, tab2, tab3 = st.tabs(["Player Scout", "Playstyle Clusters", "About / Methodology"])

with tab1:
    scout_tab.render()

with tab2:
    clusters_tab.render()

with tab3:
    about_tab.render()
