import streamlit as st

st.set_page_config(page_title="Slice Visualization", layout="wide")

import sys, os

# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path:
    sys.path.append(PATH_DASHBOARD)

import stanza
from osp import *
from utils import *

st.title("Slice Visualization")

# Sidebar for visualization settings
word_feat_type, color_column, view_mode = setup_sidebar()

# Get parameters from URL
query_params = st.query_params
slice_id = query_params.get("slice_id")
txt_input = query_params.get("txt")

left_col, right_col = st.columns([1, 1])

if not txt_input and not slice_id:
    text_input = st.text_area("Paste text here to analyze:", height=300, placeholder="Type or paste text here...", value=newtext)
    # open URL on Ctrl+Enter or button click
    if st.button("Analyze"):
        st.query_params['txt'] = text_input
        st.rerun()
else:
    # if slice_id:
    #     st.markdown(f"### Visualizing Slice: `{slice_id}`")
    # else:
    #     st.markdown(f"### Visualizing Text")

    with st.spinner(f"Loading text..."):
        try:
            if slice_id:
                docstr = STASH_SLICES_NLP[slice_id]
                doc = stanza.Document.from_serialized(docstr)
            elif txt_input:
                doc = get_nlp_doc(txt_input)
        except Exception as e:
            st.error(f"Error loading slice: {e}")
    


    with left_col:
        display_slice_predictions(
            doc,
            color_column=color_column,
            word_feat_type=word_feat_type,
            view_mode=view_mode,
            cache_key=slice_id,
        )
    
    with right_col:
        plot_weight_distribution(doc, color_column=color_column)

    display_slice_analysis(
        doc,
        color_column,
        word_feat_type,
        view_mode=view_mode,
        cache_key=slice_id,
    )

