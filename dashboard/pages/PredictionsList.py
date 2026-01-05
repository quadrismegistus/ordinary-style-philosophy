import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path: sys.path.append(PATH_DASHBOARD)

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode

from utils import setup_sidebar

st.set_page_config(page_title="Predictions List", layout="wide")

# Registry path
REGISTRY_PATH = Path(PATH_REPO) / "data" / "raw" / "stash" / "predictions_registry.json"

def load_registry():
    if not REGISTRY_PATH.exists():
        return []
    try:
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading registry: {e}")
        return []

st.title("Saved Predictions")
st.caption("History of classification runs.")

# Sidebar
setup_sidebar()

registry = load_registry()

if not registry:
    st.info("No saved predictions found. Run a classification on the 'Predict' page to see it here.")
else:
    # Convert to DataFrame
    df = pd.DataFrame(registry)
    
    # Process timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
    
    # Process params for link generation
    def get_link_url(row):
        if 'params' in row and isinstance(row['params'], dict):
            params = row['params']
            # Map params to URL params
            url_params = {
                'group1': params.get('group1_name', ''),
                'group2': params.get('group2_name', ''),
                'sample_size': params.get('sample_size', 1000),
                'num_runs': params.get('num_runs', 10),
                'cv': params.get('cv', 10),
                'balance': str(params.get('balance', True)).lower(),
                'replace': str(params.get('replace', False)).lower(),
                'normalize': str(params.get('normalize', True)).lower(),
            }
            return f"./Predict?{urlencode(url_params)}"
        return ""

    df['url'] = df.apply(get_link_url, axis=1)

    # Select and rename columns for display
    cols_map = {
        'timestamp': 'Date',
        'group1': 'Group 1',
        'group2': 'Group 2',
        'sample_size': 'N (Sample)',
        'accuracy': 'Accuracy (%)',
        'cv': 'CV Folds',
        'balance': 'Balanced',
        'url': 'Link'
    }
    
    display_cols = [c for c in cols_map.keys() if c in df.columns]
    df_display = df[display_cols].rename(columns=cols_map).copy()
    
    # Format Date
    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Format Accuracy
    if 'Accuracy (%)' in df_display.columns:
        df_display['Accuracy (%)'] = df_display['Accuracy (%)'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

    # Configure columns
    column_config = {
        "Link": st.column_config.LinkColumn(
            "View Results",
            help="Click to view full results",
            validate="^./Predict",
            display_text="View ↗"
        ),
        "Accuracy (%)": st.column_config.NumberColumn(
            "Accuracy (%)",
            format="%.1f%%"
        ),
        "Balanced": st.column_config.CheckboxColumn(
            "Balanced",
            default=True
        )
    }

    st.dataframe(
        df_display,
        width='stretch',
        hide_index=True,
        column_config=column_config
    )
    
    # Option to clear history
    with st.expander("Manage History"):
        if st.button("Clear History", type="secondary"):
            if REGISTRY_PATH.exists():
                os.remove(REGISTRY_PATH)
                st.rerun()

