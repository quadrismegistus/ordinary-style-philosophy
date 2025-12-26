import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path: sys.path.append(PATH_DASHBOARD)

MIN_NUM = -2
MAX_NUM = 2

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import *

st.set_page_config(page_title="Feature Explorer", layout="wide")

st.title("Feature Explorer")

# Sidebar for global settings
word_feat_type, color_column, view_mode = setup_sidebar()

@st.cache_data
def load_df_feats_gui():
    return get_dashboard_df_feats().replace([np.inf, -np.inf], np.nan).dropna()

df_feats_gui = load_df_feats_gui()
column_config = {}
for col in df_feats_gui.columns:
    if col.startswith('vals_'):
        col_nums = [num for nums in df_feats_gui[col].values for num in nums]
        min_num = min(col_nums)
        max_num = max(col_nums)
        if min_num<MIN_NUM:
            min_num = MIN_NUM
        if max_num>MAX_NUM:
            max_num = MAX_NUM
        column_config[col] = st.column_config.LineChartColumn(
            col,
            help=COLS_FEAT_DESCS.get(col, ''),
            y_min=min_num,
            y_max=max_num,
            color='auto',
        )
    elif df_feats_gui[col].dtype in ['float64', 'int64']:
        column_config[col] = st.column_config.NumberColumn(
            col,
            help=COLS_FEAT_DESCS.get(col, ''),
        )
    else:
        column_config[col] = st.column_config.TextColumn(
            col,
            help=COLS_FEAT_DESCS.get(col, ''),
        )

st.dataframe(df_feats_gui[[c for c in COLS_FEATS if c in df_feats_gui.columns]], column_config=column_config, height=600)