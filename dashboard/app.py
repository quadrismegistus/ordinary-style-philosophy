import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_HERE not in sys.path: sys.path.append(PATH_HERE)

import streamlit as st
from utils import *

st.set_page_config(page_title="Ordinary Style Philosophy", layout="wide")

st.title("Ordinary Style Philosophy Dashboard")

st.markdown("""
### Welcome to the Ordinary Style Philosophy Analysis Dashboard

This dashboard provides tools for analyzing the stylistic features of philosophical and literary texts.

#### Available Tools:

1. **[Predict Custom Input](/Predict_Custom_Input)**: Paste your own text to see how it compares to philosophical and literary styles.
2. **[Corpus Explorer](/Corpus_Explorer)**: Explore and visualize pre-existing slices from the research corpus.

---
""")

setup_sidebar()

st.info("Select a tool from the sidebar to get started.")
