import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path: sys.path.append(PATH_DASHBOARD)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import stanza
from osp import *
from utils import *

st.set_page_config(page_title="Predictions", layout="wide")

st.title("Predictions")

# Sidebar for global settings
word_feat_type, color_column, view_mode = setup_sidebar()

with st.sidebar:
    pass
    # word_feat_type and color_column are already in sidebar via setup_sidebar()

@st.cache_data
def load_corpus_summary(by):
    """Loads a nice summary of predictions across the corpus with slice IDs for filtering."""
    return get_nice_df_preds2(by=list(by) if isinstance(by, tuple) else by, incl_slice_ids=True).reset_index()

@st.cache_data
def load_all_slice_preds_by_slice():
    """Loads all individual slice predictions via get_nice_df_preds2(by=None)."""
    return get_nice_df_preds2(by=None).reset_index()

# 1. Main Layout
st.markdown("### Corpus Overview")
# 2. Side-by-side Tables
col_left, col_right = st.columns([1, 1])

with col_left:
    grouping_cols = st.multiselect(
        "Group results by:",
        options=['author', 'journal', 'title', 'period'],
        default=['period','journal',]
    )
selected_by = grouping_cols if grouping_cols else "discipline"

# Pass as tuple for caching compatibility
df_nice = load_corpus_summary(tuple(selected_by) if isinstance(selected_by, list) else selected_by)



# Search/Filter
with col_right:
    search_term = st.text_input("Search overview:", placeholder="")
if search_term:
    # Filter across all columns in the summary
    mask = df_nice.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
    df_filtered = df_nice[mask].copy()
else:
    df_filtered = df_nice.copy()

# Sort by "n" (Slices) descending by default
# sort_by = 'P(Phil 1925-1950)'
# if sort_by in df_filtered.columns:
    # df_filtered = df_filtered.sort_values(sort_by, ascending=False)

# Column config for get_nice_df_preds2 output
col_cfg = {
    "text_id": None,
    "slice_ids": None, # Hidden column used for filtering
}

# Style the dataframe
def style_prediction_df(df):
    if df.empty:
        return df
    
    # Identify quantitative columns: those starting with 'P' or 'n'
    quant_cols = [c for c in df.columns if "P(" in c]
    
    styler = df.style
    for col in quant_cols:
        # if col.startswith("P"):
        #     # Probabilities/Diffs: use fixed range if possible for consistency
        #     vmin = -1.0 if '-' in col else 0.0
        #     vmax = 1.0
        #     styler = styler.background_gradient(cmap='RdBu', subset=[col], vmin=vmin, vmax=vmax)
        # else:
            # Other quantitative columns (like 'n'): use column min/max
        styler = styler.background_gradient(cmap='RdBu', subset=[col])
    
    # Format probabilities to 2 decimal places
    # prob_cols = [c for c in df.columns if c.startswith("P")]
    if quant_cols:
        styler = styler.format({c: "{:.3f}" for c in quant_cols})

    # Color-code discipline column
    if 'discipline' in df.columns:
        def color_discipline(val):
            if val == 'Literature':
                return 'background-color: #b2182b; color: white;'  # Red (from RdBu colormap)
            elif val == 'Philosophy':
                return 'background-color: #2166ac; color: white;'  # Blue (from RdBu colormap)
            return ''
        styler = styler.applymap(color_discipline, subset=['discipline'])
        
    return styler

styled_df_filtered = style_prediction_df(df_filtered)

selection = st.dataframe(
    styled_df_filtered,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config=col_cfg,
)

# 2. Handle Table Selection and Show Slices Table in Right Column
selected_slice_id = None

if selection and selection.selection.rows:
    selected_row_idx = selection.selection.rows[0]
    selected_row = df_filtered.iloc[selected_row_idx]
    
    # Get the specific slice IDs for this group
    selected_slice_ids_str = selected_row.get('slice_ids', '')
    selected_slice_ids = [s.strip() for s in selected_slice_ids_str.split(';') if s.strip()]
    
    # Build filter for slices based on grouping
    df_all_slices = load_all_slice_preds_by_slice()
    
    if selected_slice_ids:
        df_text_slices = df_all_slices[df_all_slices['id'].isin(selected_slice_ids)].copy()
    else:
        df_text_slices = pd.DataFrame()
    
    if df_text_slices.empty:
        st.warning("No slice-level prediction data found for this selection.")
    else:
        st.info(f"Showing {len(df_text_slices)} slices.")
        # Format the slice table
        df_text_slices = df_text_slices.sort_values("id")
        
        # Prioritize P(Phil) and specific period columns
        prob_cols = [c for c in df_text_slices.columns if c.startswith("P")]
        meta_cols_to_show = ['id'] + [c for c in DF_PREDS_METADATA_COLS if c not in selected_by]
        display_cols = meta_cols_to_show + prob_cols
        
        # Add a column for the link
        # We need to render this as a clickable link.
        df_text_slices['Visual Analysis'] = df_text_slices['id'].apply(
            lambda x: f"/Passages?slice_id={x}"
        )
        
        # Include the link column in the data to be styled
        display_cols_with_link = ['Visual Analysis'] + display_cols
        
        # Style and display the slice table
        styled_text_slices = style_prediction_df(df_text_slices[display_cols_with_link])

        slice_selection = st.dataframe(
            styled_text_slices,
            use_container_width=True,
            hide_index=True,
            # on_select="rerun", # No longer selecting rows for inline display
            # selection_mode="single-row",
            column_config={
                **col_cfg,
                "Visual Analysis": st.column_config.LinkColumn(
                    "Visualize", 
                    help="Click to open visualization in new tab",
                    display_text="Open ↗"
                )
            }
        )

        # Removed: Handle Slice Table Selection for Visual Analysis
        # if slice_selection and slice_selection.selection.rows:
        #     ...

else:
    st.info("Select a group from the left table to view its constituent slices.")

# 3. Visual Analysis (Full Width)
# st.divider()
# if selected_slice_id:
#    ...
