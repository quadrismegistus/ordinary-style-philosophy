import sys, os

# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path:
    sys.path.append(PATH_DASHBOARD)

import streamlit as st
from utils import *

st.set_page_config(page_title="Predict Custom Input", layout="wide")
st.title("Predict Custom Input")

# Sidebar for visualization settings (same controls as Passages.py)
word_feat_type, color_column, view_mode = setup_sidebar()

# URL params (txt mode only; Passages.py also supports slice_id, but Custom is for ad-hoc text)
query_params = st.query_params
txt_input = query_params.get("txt")

if not txt_input:
    text_input = st.text_area(
        "Paste text here to analyze:",
        height=300,
        placeholder="Type or paste text here...",
        value=newtext,
    )
    # open URL on Ctrl+Enter or button click
    if st.button("Analyze"):
        st.query_params["txt"] = text_input
        st.rerun()
else:
    with st.spinner("Loading text..."):
        try:
            doc = get_nlp_doc(txt_input)
        except Exception as e:
            st.error(f"Error loading text: {e}")
            st.stop()

    # Reuse the same visualization component as Passages.py
    display_slice_analysis(
        doc,
        color_column,
        word_feat_type,
        view_mode=view_mode,
        cache_key=None,
    )

