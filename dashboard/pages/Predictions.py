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
from dashboard.components import render_prediction_explorer

st.set_page_config(page_title="Predictions", layout="wide")

st.title("Predictions")

# Sidebar for global settings
word_feat_type, color_column, view_mode = setup_sidebar()

with st.sidebar:
    pass
    # word_feat_type and color_column are already in sidebar via setup_sidebar()

@st.cache_data
def load_all_slice_preds_by_slice():
    """Loads all individual slice predictions via get_nice_df_preds2(by=None)."""
    df = get_nice_df_preds2(by=None).reset_index()
    
    # Calculate target and accuracy
    if 'discipline' in df.columns:
        df['target'] = df['discipline']
        
        # Assume 'P(Phil)' column exists for probability of Philosophy
        if 'P(Phil)' in df.columns:
            # Correct if (Phil and P > 0.5) or (Lit and P < 0.5)
            # Or simplified: if target is Phil, prob should be > 0.5. If Lit, prob should be < 0.5
            # Note: P(Phil) is prob_Philosophy
            
            def check_correct(row):
                if row['target'] == 'Philosophy':
                    return row['P(Phil)'] > 0.5
                elif row['target'] == 'Literature':
                    return row['P(Phil)'] < 0.5
                return False # Unknown target?
                
            df['correct'] = df.apply(check_correct, axis=1)
            
    return df

st.markdown("### Corpus Overview")

# Load all predictions
df_slices = load_all_slice_preds_by_slice()

# Use the reusable component
render_prediction_explorer(df_slices)

