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
import hashlib
import pickle
import subprocess
import time
import json
import random
from datetime import datetime
from urllib.parse import urlencode
from pathlib import Path
from osp import *
from utils import (
    setup_sidebar,
    log_progress,
    load_slice_ids,
    load_slice_feat_examples,
    load_comparison_stats,
)
from dashboard.components import (
    render_prediction_explorer,
    render_feature_summary,
    lookup_examples,
)

st.set_page_config(page_title="Predict", layout="wide")

# Directory for async job results
JOBS_DIR = Path(PATH_REPO) / "data" / "raw" / "stash" / "predict_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Registry file for saved predictions
REGISTRY_PATH = Path(PATH_REPO) / "data" / "raw" / "stash" / "predictions_registry.json"

# --- Helper Functions ---

def save_prediction_to_registry(job_id, params, accuracy):
    """Save prediction metadata to the registry."""
    try:
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
        
        # Check if job already exists
        for entry in registry:
            if entry.get('job_id') == job_id:
                # Update accuracy if it wasn't there
                if 'accuracy' not in entry or entry['accuracy'] != accuracy:
                    entry['accuracy'] = accuracy
                    with open(REGISTRY_PATH, 'w') as f:
                        json.dump(registry, f, indent=2)
                return

        # Create new entry
        entry = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'group1': params.get('group1_name'),
            'group2': params.get('group2_name'),
            'sample_size': params.get('sample_size'),
            'cv': params.get('cv'),
            'balance': params.get('balance'),
            'replace': params.get('replace'),
            'normalize': params.get('normalize'),
            'accuracy': accuracy,
            'params': params
        }
        
        registry.append(entry)
        
        with open(REGISTRY_PATH, 'w') as f:
            json.dump(registry, f, indent=2)
            
    except Exception as e:
        print(f"Error saving to registry: {e}")

def _load_saved_groups():
    """Load saved groups from the dashboard stash."""
    try:
        return {k: STASH_DASHBOARD_GROUPS[k] for k in STASH_DASHBOARD_GROUPS.keys()}
    except Exception:
        return {}


def _get_saved_group_names():
    """Returns list of saved group names."""
    return sorted(_load_saved_groups().keys())


def get_job_id(params: dict) -> str:
    """Generate a unique job ID from parameters."""
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def get_job_path(job_id: str) -> Path:
    """Get the path to a job's result file."""
    return JOBS_DIR / f"{job_id}.pkl"


def get_job_status_path(job_id: str) -> Path:
    """Get the path to a job's status file."""
    return JOBS_DIR / f"{job_id}.status"


def get_job_log_path(job_id: str) -> Path:
    """Get the path to a job's log file."""
    return JOBS_DIR / f"{job_id}.log"


def is_job_complete(job_id: str) -> bool:
    """Check if a job is complete."""
    return get_job_path(job_id).exists()


def is_job_running(job_id: str) -> bool:
    """Check if a job is currently running."""
    status_path = get_job_status_path(job_id)
    if not status_path.exists():
        return False
    try:
        with open(status_path, 'r') as f:
            status = f.read().strip()
        return status == 'running'
    except:
        return False


def load_job_results(job_id: str):
    """Load results from a completed job."""
    job_path = get_job_path(job_id)
    with open(job_path, 'rb') as f:
        return pickle.load(f)


def start_job(params: dict) -> str:
    """Start a classification job in a subprocess."""
    job_id = get_job_id(params)
    status_path = get_job_status_path(job_id)
    job_path = get_job_path(job_id)
    log_path = get_job_log_path(job_id)
    
    # If job already complete, return immediately
    if job_path.exists():
        return job_id
    
    # If already running, return
    if is_job_running(job_id):
        return job_id
    
    # Write status file
    with open(status_path, 'w') as f:
        f.write('running')
    
    # Create a Python script to run the job
    script = f'''
import sys
sys.path.insert(0, "{PATH_REPO}")

import pickle
import traceback
from pathlib import Path

try:
    from osp.classify import classify_then_predict_group
    
    groups_train = [
        ("{params['group1_name']}", {repr(params['group1_query'])}),
        ("{params['group2_name']}", {repr(params['group2_query'])}),
    ]
    
    print(f"Starting classification for {{groups_train}}")
    
    df_preds, df_feats = classify_then_predict_group(
        groups_train,
        target_col='discipline',
        balance={params['balance']},
        num_runs={params['num_runs']},
        verbose=True,
        return_models=False,
        normalize={params['normalize']},
        sample_size={params['sample_size']},
        cv={params['cv']},
        replace={params['replace']},
    )
    
    result = {{
        'df_preds': df_preds,
        'df_feats': df_feats,
        'params': {repr(params)},
    }}
    
    print("Saving results to {job_path}...")
    with open("{job_path}", 'wb') as f:
        pickle.dump(result, f)
    
    print("Updating status to complete...")
    with open("{status_path}", 'w') as f:
        f.write('complete')
        
except Exception as e:
    print(f"ERROR: {{e}}")
    traceback.print_exc()
    with open("{status_path}", 'w') as f:
        f.write(f'error: {{e}}')
    raise
'''
    
    # Run in subprocess (detached) with logging
    with open(log_path, 'w') as log_file:
        subprocess.Popen(
            [sys.executable, '-c', script],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    
    return job_id


def style_results_df(df, prob_cols=None, as_percentage=False):
    """Style a results dataframe with color gradients for probability columns.
    
    Args:
        df: DataFrame to style
        prob_cols: List of probability column names
        as_percentage: If True, assumes values are 0-100; if False, assumes 0-1
    """
    if df.empty:
        return df
    
    if prob_cols is None:
        prob_cols = [c for c in df.columns if c.startswith('prob_') or c.startswith('% ') or 'Accuracy' in c]
    
    # Filter to only columns that exist
    prob_cols = [c for c in prob_cols if c in df.columns]
    
    if not prob_cols:
        return df.style
    
    vmin, vmax = (0, 100) if as_percentage else (0, 1)
    fmt = "{:.1f}" if as_percentage else "{:.3f}"
    
    # Build format dict for all columns at once
    format_dict = {col: fmt for col in prob_cols}
    
    styler = df.style.format(format_dict)
    
    # Apply gradient to each column
    for col in prob_cols:
        styler = styler.background_gradient(cmap='RdBu', subset=[col], vmin=vmin, vmax=vmax)
    
    return styler


def style_feats_df(df):
    """Style a features dataframe with color gradients for weight columns."""
    if df.empty:
        return df
    
    weight_cols = [c for c in df.columns if 'weight' in c.lower() or 'mean_' in c]
    
    styler = df.style
    for col in weight_cols:
        if col in df.columns and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            styler = styler.background_gradient(cmap='RdBu', subset=[col])
            styler = styler.format({col: "{:.4f}"})
    
    return styler


# --- URL Parameter Handling ---

# Read URL parameters
query_params = st.query_params

# Check if we have classification parameters in URL
has_url_params = 'group1' in query_params and 'group2' in query_params

if has_url_params:
    # --- RESULTS MODE: Display results from URL parameters ---
    
    # Parse parameters
    url_group1 = query_params.get('group1', '')
    url_group2 = query_params.get('group2', '')
    url_sample_size = int(query_params.get('sample_size', 1000))
    url_num_runs = int(query_params.get('num_runs', 10))
    url_cv = int(query_params.get('cv', 10))
    url_balance = query_params.get('balance', 'true').lower() == 'true'
    url_replace = query_params.get('replace', 'false').lower() == 'true'
    url_normalize = query_params.get('normalize', 'true').lower() == 'true'
    url_force = query_params.get('force', 'false').lower() == 'true'
    
    # Load saved groups to get queries
    saved_groups = _load_saved_groups()
    
    if url_group1 not in saved_groups or url_group2 not in saved_groups:
        st.error(f"Group not found. Available groups: {list(saved_groups.keys())}")
        st.stop()
    
    g1 = saved_groups[url_group1]
    g2 = saved_groups[url_group2]
    query1 = g1.get('query_str', '1==1')
    query2 = g2.get('query_str', '1==1')
    
    st.title(f"Predict: {url_group1} vs {url_group2}")
    
    # Build job parameters
    job_params = {
        'group1_name': url_group1,
        'group1_query': query1,
        'group2_name': url_group2,
        'group2_query': query2,
        'sample_size': url_sample_size,
        'num_runs': url_num_runs,
        'cv': url_cv,
        'balance': url_balance,
        'replace': url_replace,
        'normalize': url_normalize,
    }
    
    # Show parameters
    with st.expander("Classification parameters", expanded=False):
        st.json(job_params)
    
    # Start or check job
    job_id = get_job_id(job_params)
    
    # Handle Force Regeneration
    if url_force:
        # Clear session state results
        if 'predict_df_preds' in st.session_state: del st.session_state['predict_df_preds']
        if 'predict_df_feats' in st.session_state: del st.session_state['predict_df_feats']
        
        # Delete job files
        try:
            get_job_path(job_id).unlink(missing_ok=True)
            get_job_status_path(job_id).unlink(missing_ok=True)
            get_job_log_path(job_id).unlink(missing_ok=True)
        except Exception:
            pass # Ignore errors if files don't exist
            
        # Remove force param to prevent loop
        new_params = query_params.to_dict()
        if 'force' in new_params: del new_params['force']
        st.query_params.from_dict(new_params)
        st.rerun()
    
    if is_job_complete(job_id):
        # Job is complete - load results
        try:
            result = load_job_results(job_id)
            
            df_preds = result['df_preds']
            df_feats = result['df_feats']
            
            # Store in session state for display
            st.session_state['predict_df_preds'] = df_preds
            st.session_state['predict_df_feats'] = df_feats
            st.session_state['predict_comparison'] = f"{url_group1} vs {url_group2}"
            st.session_state['predict_group_names'] = [url_group1, url_group2]
            st.session_state['predict_params'] = job_params
            
            st.success("Classification complete!")
            
        except Exception as e:
            st.error(f"Failed to load results: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    elif is_job_running(job_id):
        # Job is running - show progress and auto-refresh
        st.info("⏳ Classification is running in the background...")
        st.caption(f"Job ID: {job_id}")
        
        # Show log tail if available
        log_path = get_job_log_path(job_id)
        if log_path.exists():
            with st.expander("Logs", expanded=True):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        # Show last 20 lines
                        st.code(''.join(lines[-20:]))
                except Exception:
                    st.caption("Could not read log file.")
        
        st.caption("This page will automatically refresh when complete.")
        
        if st.button("Force Restart", help="Stop current job and restart."):
            st.query_params['force'] = 'true'
            st.rerun()
        
        # Auto-refresh every 3 seconds
        time.sleep(3)
        st.rerun()
    
    else:
        # Start new job
        st.info("🚀 Starting classification job...")
        start_job(job_params)
        st.caption(f"Job ID: {job_id}")
        
        # Refresh to show running status
        time.sleep(1)
        st.rerun()
    
    # Back link
    col_back, col_regen = st.columns([1, 1])
    with col_back:
        st.markdown("[← Back to configuration](./Predict)")
    with col_regen:
        if st.button("Regenerate Results", help="Force a new classification run with these parameters."):
            st.query_params['force'] = 'true'
            st.rerun()

else:
    # --- CONFIGURATION MODE: Show form to set up classification ---
    
    st.title("Predict")
    st.caption("Run classification predictions with customizable comparison settings.")
    
    # Sidebar for global settings
    word_feat_type, color_column, view_mode = setup_sidebar()

    # Get saved groups (inside else block for config mode)
    saved_groups = _load_saved_groups()
    saved_group_names = _get_saved_group_names()

    if len(saved_group_names) < 2:
        st.warning("Please create at least 2 groups in the Groups page to run a comparison.")
        st.stop()

    # Settings columns
    col_settings, col_run = st.columns([3, 1])

    with col_settings:
        st.markdown("### Select Groups to Compare")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            group1_name = st.selectbox(
                "Group 1:",
                options=saved_group_names,
                index=0,
                help="Select the first group for comparison."
            )
        
        with col_g2:
            # Filter out group1 from options for group2
            group2_options = [g for g in saved_group_names if g != group1_name]
            group2_name = st.selectbox(
                "Group 2:",
                options=group2_options,
                index=0 if group2_options else None,
                help="Select the second group for comparison."
            )
        
        st.markdown("### Classification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_size = st.number_input(
                "Sample size (per group):",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100,
                help="Number of slices to sample from each group."
            )
            
            num_runs = st.number_input(
                "Number of runs:",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help="Number of classification runs to average results over."
            )
        
        with col2:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=20,
                value=10,
                step=1,
                help="Number of folds for cross-validation."
            )
            
            balance = st.checkbox(
                "Balance classes",
                value=True,
                help="Ensure equal sample sizes for both groups."
            )
            
            replace = st.checkbox(
                "Sample with replacement",
                value=False,
                help="Allow the same slice to be sampled multiple times."
            )
            
            normalize = st.checkbox(
                "Normalize features",
                value=True,
                help="Z-score normalize features before classification."
            )

    # Build comparison from selected groups
    selected_comparison = None
    if group1_name and group2_name:
        g1 = saved_groups[group1_name]
        g2 = saved_groups[group2_name]
        query1 = g1.get('query_str', '1==1')
        query2 = g2.get('query_str', '1==1')
        name1, name2 = group1_name, group2_name
        selected_comparison = f"{name1} vs {name2}"
        
        # Count available slices for each group
        @st.cache_data
        def count_slices_for_query(query_str):
            """Count slices matching a query."""
            from osp.slices import get_text_id2slice_ids
            metadata = get_corpus_metadata()
            try:
                matching_texts = metadata.query(query_str).index.tolist() if query_str and query_str != '1==1' else metadata.index.tolist()
            except Exception:
                matching_texts = []
            text2slices = get_text_id2slice_ids()
            total_slices = sum(len(text2slices.get(tid, [])) for tid in matching_texts)
            return len(matching_texts), total_slices
        
        n_texts1, n_slices1 = count_slices_for_query(query1)
        n_texts2, n_slices2 = count_slices_for_query(query2)
        
        with st.expander("Comparison details", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"**Group 1:** {name1}")
                st.caption(f"{n_texts1} texts, {n_slices1} slices")
                st.code(query1, language="python")
            with col_d2:
                st.markdown(f"**Group 2:** {name2}")
                st.caption(f"{n_texts2} texts, {n_slices2} slices")
                st.code(query2, language="python")

    with col_run:
        st.markdown("### ")  # Spacer for alignment
        
        # Build URL with parameters
        if group1_name and group2_name:
            params = {
                'group1': group1_name,
                'group2': group2_name,
                'sample_size': sample_size,
                'num_runs': num_runs,
                'cv': cv_folds,
                'balance': str(balance).lower(),
                'replace': str(replace).lower(),
                'normalize': str(normalize).lower(),
            }
            run_url = f"./Predict?{urlencode(params)}"
            
            st.link_button(
                "Run Classification ↗",
                run_url,
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("Run Classification", type="primary", use_container_width=True, disabled=True)

    st.stop()  # Stop here in config mode - don't show results section

from dashboard.components import render_prediction_explorer

# --- Display Results ---

if 'predict_df_preds' in st.session_state and 'predict_df_feats' in st.session_state:
    df_preds = st.session_state['predict_df_preds']
    df_feats = st.session_state['predict_df_feats']
    params = st.session_state.get('predict_params', {})
    comparison = st.session_state.get('predict_comparison', '')
    
    st.markdown(f"### Results: {comparison}")
    st.caption(f"Parameters: {params}")
    
    # Debug: Show data summary
    st.caption(f"Predictions: {len(df_preds)} rows, {len(df_preds.columns)} cols | Features: {len(df_feats)} rows")
    
    # --- Prepare Data (Shared across tabs) ---
    # We'll create an enhanced dataframe with target, correct, and metadata info
    df_enhanced = df_preds.reset_index().copy()
    if 'id' not in df_enhanced.columns and 'index' in df_enhanced.columns:
        df_enhanced = df_enhanced.rename(columns={'index': 'id'})
    
    df_enhanced['text_id'] = df_enhanced['id'].astype(str).str.split('__').str[0]
    
    # Get probability columns
    prob_cols = [c for c in df_preds.columns if c.startswith('prob_')]
    group_names = [c.replace('prob_', '') for c in prob_cols]
    stored_group_names = st.session_state.get('predict_group_names', group_names)
    
    # Determine targets
    metadata = get_corpus_metadata()
    saved_groups = _load_saved_groups()
    text_id_to_target = {}
    unique_text_ids = df_enhanced['text_id'].unique()
    for tid in unique_text_ids: text_id_to_target[tid] = 'Unknown'

    for gname in stored_group_names:
        if gname in saved_groups:
            query = saved_groups[gname].get('query_str', '')
            if query and query != '1==1':
                try:
                    matching_texts = set(metadata.query(query).index)
                    for tid in unique_text_ids:
                        if tid in matching_texts:
                            text_id_to_target[tid] = gname
                except Exception: pass
        elif 'discipline' in metadata.columns:
             for tid in unique_text_ids:
                if tid in metadata.index and metadata.loc[tid, 'discipline'] == gname:
                    text_id_to_target[tid] = gname

    df_enhanced['target'] = df_enhanced['text_id'].map(text_id_to_target)
    
    # Calculate correctness
    def is_correct(row):
        target = row['target']
        prob_col = f'prob_{target}'
        if prob_col in row.index:
            return row[prob_col] > 0.5
        return False
    
    df_enhanced['correct'] = df_enhanced.apply(is_correct, axis=1)
    
    # Merge full metadata for explorer
    meta_to_merge = metadata[DF_PREDS_METADATA_COLS]
    df_enhanced = df_enhanced.merge(meta_to_merge, left_on='text_id', right_index=True, how='left')

    group1_name = params.get('group1_name', url_group1)
    group2_name = params.get('group2_name', url_group2)
    group1_query = params.get('group1_query', query1)
    group2_query = params.get('group2_query', query2)
    groups_train = [
        (group1_name, group1_query),
        (group2_name, group2_query),
    ]

    results_tab, feats_tab, feature_summary_tab, viz_tab, explorer_tab = st.tabs(
        ["Predictions", "Feature Weights", "Feature Summary", "Visualization", "Prediction Explorer"]
    )
    
    with results_tab:
        st.markdown("#### Prediction Results")
        
        # Debug info
        if df_preds.empty:
            st.warning("No prediction data available.")
            st.stop()
        
        if not prob_cols:
            st.warning("No probability columns found.")
            st.stop()
            
        # Use df_enhanced for display logic (it has target/correct)
        df_preds_display = df_enhanced.copy()
        
        # Check for predict_type column
        if 'predict_type' not in df_preds_display.columns:
            st.warning("'predict_type' column not found.")
            st.stop()
        
        # Average across runs to get one prediction per slice
        agg_by_slice = df_preds_display.groupby(['id', 'text_id', 'target', 'predict_type'])[prob_cols].mean().reset_index()
        
        # Calculate accuracy: correct if prob_{target} > 0.5 using averaged probabilities
        def is_correct_agg(row):
            target = row['target']
            prob_col = f'prob_{target}'
            if prob_col in row.index:
                return row[prob_col] > 0.5
            return False
        
        agg_by_slice['correct'] = agg_by_slice.apply(is_correct_agg, axis=1)
        
        # ... rest of results tab using agg_by_slice ...
        
        # Filter to valid targets for overall accuracy
        valid_targets_mask = agg_by_slice['target'].isin(group_names)
        valid_slices_df = agg_by_slice[valid_targets_mask]
        
        # Summary statistics by group
        st.markdown("##### Summary by Target Group")
        
        # Create columns: Total slices, Overall accuracy, then each group
        n_groups = len(group_names)
        metric_cols = st.columns(n_groups + 2)
        
        with metric_cols[0]:
            unique_slices = agg_by_slice['id'].nunique()
            st.metric("Unique Slices", unique_slices)
        
        with metric_cols[1]:
            # Calculate overall accuracy only on slices belonging to the target groups
            if not valid_slices_df.empty:
                overall_accuracy = valid_slices_df['correct'].mean() * 100
            else:
                overall_accuracy = 0.0
                
            st.metric(
                "Overall Accuracy",
                f"{overall_accuracy:.1f}%",
                help=f"% of slices correctly classified (prob > 50% for true target). Excludes {len(agg_by_slice) - len(valid_slices_df)} 'Unknown' slices."
            )

            # Auto-save to registry
            save_prediction_to_registry(
                job_id=hashlib.md5(str(sorted(params.items())).encode()).hexdigest()[:12],
                params=params,
                accuracy=round(overall_accuracy, 1)
            )
        
        for i, group in enumerate(group_names):
            with metric_cols[i + 2]:
                group_preds = agg_by_slice[agg_by_slice['target'] == group]
                if len(group_preds) > 0:
                    # Accuracy for this group
                    group_accuracy = group_preds['correct'].mean() * 100
                    n_slices = len(group_preds)
                    st.metric(
                        f"{group} Accuracy",
                        f"{group_accuracy:.1f}%",
                        help=f"% of {group} slices correctly classified (n={n_slices})"
                    )
        
        # Detailed breakdown by predict_type and target (using slice-averaged data)
        st.markdown("##### Aggregated by Prediction Type & Target")
        
        agg_cols = ['predict_type', 'target']
        df_agg = (
            agg_by_slice
            .groupby(agg_cols)
            .agg({
                'id': 'count',  # Count unique slices
                'correct': 'mean',  # Accuracy
                **{col: 'mean' for col in prob_cols}  # Mean probabilities
            })
            .reset_index()
        )
        
        # Rename columns
        df_agg = df_agg.rename(columns={'id': 'n', 'correct': 'accuracy'})
        
        # Keep only useful columns and sort by target then predict_type
        display_cols = ['predict_type', 'target', 'n', 'accuracy'] + prob_cols
        df_agg_display = df_agg[[c for c in display_cols if c in df_agg.columns]].query('predict_type=="cv"').copy()
        df_agg_display = df_agg_display.sort_values(['target', 'predict_type'])
        
        # Rename predict_type values
        predict_type_map = {
            'cv': 'Validation set (CV)',
            'unseen': 'Test set (unseen)'
        }
        df_agg_display['predict_type'] = df_agg_display['predict_type'].map(
            lambda x: predict_type_map.get(x, x)
        )
        
        # Convert probabilities and accuracy to percentages (multiply by 100)
        for col in prob_cols + ['accuracy']:
            if col in df_agg_display.columns:
                df_agg_display[col] = (df_agg_display[col] * 100).round(1)
        
        # Rename columns to human-friendly names
        col_rename = {
            'predict_type': 'Prediction Type',
            'target': 'Target',
            'n': 'Slices',
            'accuracy': 'Accuracy %',
        }
        # Rename prob_ columns to % format
        for col in prob_cols:
            group_name = col.replace('prob_', '')
            col_rename[col] = f'% {group_name}'
        
        df_agg_display = df_agg_display.rename(columns=col_rename)
        
        # Get renamed percentage columns for styling (including accuracy)
        pct_cols = ['Accuracy %'] + [f'% {col.replace("prob_", "")}' for col in prob_cols]
        
        try:
            styled_agg = style_results_df(df_agg_display, prob_cols=pct_cols, as_percentage=True)
            st.dataframe(styled_agg, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error displaying aggregated results table: {e}")
            st.dataframe(df_agg_display)
        
        # Show detailed predictions
        with st.expander("Detailed Predictions", expanded=False):
            try:
                # Prepare dataframe for detailed view
                df_detailed = df_preds.reset_index()
                
                # Add View link column
                df_detailed['View'] = df_detailed['id'].apply(
                    lambda x: f"/Passages?slice_id={x}"
                )
                
                # Style the dataframe
                styled_detailed = style_results_df(df_detailed.head(500), prob_cols)
                
                # Configure columns
                column_config = {
                    "View": st.column_config.LinkColumn(
                        "View",
                        help="View passage details",
                        display_text="Open ↗"
                    )
                }
                
                st.dataframe(
                    styled_detailed,
                    use_container_width=True,
                    height=400,
                    column_config=column_config,
                    column_order=['View', 'id'] + [c for c in df_detailed.columns if c not in ['View', 'id']]
                )
            except Exception as e:
                st.error(f"Error displaying detailed predictions: {e}")
                st.dataframe(df_preds.head(100))
    
    with feats_tab:
        st.markdown("#### Feature Weights")
        st.caption("Features ranked by their importance in distinguishing the two groups.")
        
        # Sort options
        sort_col = st.selectbox(
            "Sort by:",
            options=['weight'] + [c for c in df_feats.columns if c != 'feature' and c != 'weight'],
            index=0
        )
        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
        ascending = sort_order == "Ascending"
        
        df_feats_sorted = df_feats.sort_values(sort_col, ascending=ascending)
        
        # Display top features
        n_display = st.slider("Number of features to display:", 10, 100, 25)
        
        st.dataframe(
            style_feats_df(df_feats_sorted.head(n_display)),
            use_container_width=True,
            height=500
        )
        
        # Download button
        csv_feats = df_feats_sorted.to_csv(index=False)
        st.download_button(
            label="Download Feature Weights (CSV)",
            data=csv_feats,
            file_name=f"feature_weights_{comparison.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with feature_summary_tab:
        st.markdown("#### Feature Summary")
        try:
            df_smpl_feats = load_comparison_stats(tuple(groups_train))
            df_smpl_feats = df_smpl_feats.sample(frac=1)
            top_feats_g1 = (
                df_smpl_feats[df_smpl_feats["target"] == url_group1]
                .sort_values("feat_rank1")
                .head(NUM_DISTINCTIVE_FEATS)["feat"]
                .astype(str)
                .tolist()
            )
            top_feats_g2 = (
                df_smpl_feats[df_smpl_feats["target"] == url_group2]
                .sort_values("feat_rank2")
                .head(NUM_DISTINCTIVE_FEATS)["feat"]
                .astype(str)
                .tolist()
            )
            slice_ids_g1 = load_slice_ids(query1)
            slice_ids_g2 = load_slice_ids(query2)
            random.shuffle(slice_ids_g1)
            random.shuffle(slice_ids_g2)
            df_egs_g1 = load_slice_feat_examples(slice_ids_g1, top_feats_g1, num_egs=NUM_EG_PER_FEAT)
            df_egs_g2 = load_slice_feat_examples(slice_ids_g2, top_feats_g2, num_egs=NUM_EG_PER_FEAT)
            egs_g1 = lookup_examples(df_egs_g1)
            egs_g2 = lookup_examples(df_egs_g2)
        except Exception as e:
            st.error(f"Unable to load feature summary: {e}")
            df_smpl_feats = pd.DataFrame()
            egs_g1 = {}
            egs_g2 = {}

        render_feature_summary(df_smpl_feats, url_group1, url_group2, egs_g1, egs_g2)

    with viz_tab:
        st.markdown("#### Probability Distributions")
        if not prob_cols:
            st.warning("No probability columns found.")
        else:
            col_g1, col_g2 = st.columns(2)
            distribution_configs = [
                (col_g1, group1_name, f"prob_{group1_name}"),
                (col_g2, group2_name, f"prob_{group2_name}"),
            ]

            for col_widget, group_name, prob_col in distribution_configs:
                col_widget.markdown(f"##### {group_name} slices")
                if prob_col not in df_enhanced.columns:
                    col_widget.warning(f"{prob_col} missing from predictions.")
                    continue
                subset = df_enhanced[df_enhanced["target"] == group_name]
                if subset.empty:
                    col_widget.info("No slices mapped to this group.")
                    continue
                chart = (
                    alt.Chart(subset)
                    .mark_bar(opacity=0.75)
                    .encode(
                        x=alt.X(f"{prob_col}:Q", bin=alt.Bin(maxbins=35), title="Probability"),
                        y=alt.Y("count():Q", title="Slices"),
                        tooltip=[alt.Tooltip(f"{prob_col}:Q", format=".2f"), "count()"],
                    )
                    .properties(height=320, title=f"{group_name} probability")
                    .interactive()
                )
                col_widget.altair_chart(chart, use_container_width=True)

    with explorer_tab:
        # Render component with prepared enhanced dataframe
        render_prediction_explorer(df_enhanced, key_prefix="predict_explorer")

else:
    st.info("Configure settings above and click 'Run Classification' to see results.")

