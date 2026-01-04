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

if slice_id:
    with left_col:
        st.markdown(f"### Visualizing Slice: `{slice_id}`")
    
    with right_col:
        try:
            with st.spinner(f"Loading slice {slice_id}..."):
                # Load the serialized doc
                if slice_id in STASH_SLICES_NLP:
                    docstr = STASH_SLICES_NLP[slice_id]
                    doc = stanza.Document.from_serialized(docstr)
                else:
                    st.error(f"Slice ID '{slice_id}' not found in stash.")
        except Exception as e:
            st.error(f"Error loading slice: {e}")

    
        display_slice_predictions(
            doc,
            color_column=color_column,
            word_feat_type=word_feat_type,
            view_mode=view_mode,
            cache_key=slice_id,
        )
    
    display_slice_analysis(
        doc,
        color_column,
        word_feat_type,
        view_mode=view_mode,
        cache_key=slice_id,
    )

elif txt_input:
    with left_col:
        st.markdown("### Visualizing Custom Text")
    try:
        with st.spinner("Processing text..."):
            doc = get_nlp_doc(txt_input)
            display_slice_analysis(
                doc,
                color_column,
                word_feat_type,
                view_mode=view_mode,
                cache_key=txt_input[:20],
            )
    except Exception as e:
        st.error(f"Error processing text: {e}")
else:
    st.info(
        "No content to visualize. Please provide a `slice_id` or `txt` parameter in the URL."
    )
