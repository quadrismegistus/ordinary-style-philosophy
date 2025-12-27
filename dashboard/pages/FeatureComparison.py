import sys, os
import contextlib
import io
# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path:
    sys.path.append(PATH_DASHBOARD)
from osp import *

import streamlit as st
import pandas as pd
from collections import Counter
from streamlit_local_storage import LocalStorage
from utils import *
DEFAULT_COMPARISONS = []
for (name_a, query_a), (name_b, query_b) in COMPARISONS:
    DEFAULT_COMPARISONS.append({
        "label": f"{name_a} vs {name_b}",
        "group_a": {
            "name": name_a,
            "query_str": query_a
        },
        "group_b": {
            "name": name_b,
            "query_str": query_b
        }
    })
DEFAULT_COMPARISON = DEFAULT_COMPARISONS[0]

st.set_page_config(page_title="Feature Comparison", layout="wide")
st.title("Feature Comparison")

ls = LocalStorage()

@contextlib.contextmanager
def st_stdout(out_empty):
    """Context manager to redirect stdout to a streamlit empty element."""
    with io.StringIO() as buffer:
        with contextlib.redirect_stdout(buffer):
            yield
            out_empty.text(buffer.getvalue())

# 1. Define hardcoded default comparison
# Import COMPARISONS from osp.constants

# Build default comparisons from COMPARISONS constant

# 2. Retrieve saved comparison from localStorage
saved_data = ls.getItem("osp_comparison_groups")

# 3. Build options list
comparison_options = DEFAULT_COMPARISONS

if saved_data and isinstance(saved_data, dict):
    g1 = saved_data.get("group_a", {})
    g2 = saved_data.get("group_b", {})
    if g1 and g2:
        label = f"{g1.get('name', 'Group 1')} vs {g2.get('name', 'Group 2')} (Saved)"
        comparison_options.append({
            "label": label,
            "group_a": g1,
            "group_b": g2
        })

# 4. Display dropdown
selected_comparison = st.selectbox(
    "Select Comparison",
    options=comparison_options,
    format_func=lambda x: x["label"]
)

# Show the selected comparison details
if selected_comparison:
    st.write(f"### {selected_comparison['label']}")
    
    # Load sample and features once for both groups
    q_a = selected_comparison['group_a']['query_str']
    q_b = selected_comparison['group_b']['query_str']
    name_a = selected_comparison['group_a']['name']
    name_b = selected_comparison['group_b']['name']
    
    groups_train = [(name_a, q_a), (name_b, q_b)]
    
    with st.status("Calculating feature statistics...", expanded=True) as status:
        out_msg = st.empty()
        try:
            with st_stdout(out_msg):
                df_smpl_feats = get_balanced_slice_sample_feats(groups_train, with_diff_rows=True).reset_index()
            status.update(label="Feature statistics complete!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error calculating feature statistics: {e}")
            df_smpl_feats = pd.DataFrame()
            status.update(label="Error calculating feature statistics", state="error")

    col1, col2, col3 = st.columns(3)
    
    # helper to render metrics
    def render_feat_metrics(df, target, diff_target, title, sort_by='z', ascending=False, is_diff=False):
        st.markdown(f"#### {title}")
        if df.empty:
            st.write("No data available.")
            return
        
        df_target = df[df['target'] == target].sort_values(sort_by, ascending=ascending).head(10)
        if df_target.empty and ' - ' in target:
            # Try reverse difference if it's a difference target and not found
            t1, t2 = target.split(' - ')
            rev_target = f'{t2} - {t1}'
            df_target = df[df['target'] == rev_target].sort_values(sort_by, ascending=ascending).head(10)
            
        if df_target.empty:
            st.write("No distinctive features found.")
            return

        # Prepare difference data for badges
        df_diff = df[df['target'] == diff_target]
        if df_diff.empty and ' - ' in diff_target:
             t1, t2 = diff_target.split(' - ')
             rev_target = f'{t2} - {t1}'
             df_diff = df[df['target'] == rev_target]
        
        diff_lookup = df_diff.set_index('feat')

        # Prepare Group 1 and Group 2 lookups for badges
        lookup_g1 = df[df['target'] == name_a].set_index('feat')
        lookup_g2 = df[df['target'] == name_b].set_index('feat')

        # Display metrics in a grid or list
        metric_num = 0
        for _, row in df_target.iterrows():
            feat = row['feat']
            if feat in BAD_SLICE_FEATS:
                continue
            z = row['z']
            raw = row['raw']
            
            # Get values for both groups for the badge subtraction string
            try:
                row_g1 = lookup_g1.loc[feat]
                row_g2 = lookup_g2.loc[feat]
                if isinstance(row_g1, pd.DataFrame): row_g1 = row_g1.iloc[0]
                if isinstance(row_g2, pd.DataFrame): row_g2 = row_g2.iloc[0]
                
                raw_a, raw_b = row_g1['raw'], row_g2['raw']
                z_a, z_b = row_g1['z'], row_g2['z']
            except (KeyError, IndexError):
                raw_a = raw_b = z_a = z_b = 0

            # Show 2 columns inside each of the 3 main cols
            feat_desc = FEAT2DESC.get(feat, feat)
            metric_num += 1
            feat_type, feat_name = feat.split('_', 1)
            feat_hdr = f"{feat_name} ({feat_type})\n*{feat_desc}*"
            st.divider()
            st.markdown(f"##### {metric_num}. {feat_hdr}")
            c1, c2 = st.columns(2)
            
            if is_diff:
                # Difference column: Δ(G1 - G2)
                # main text: raw and z fields in delta form (+/-)
                # badge: raw_a - raw_b format
                delta_raw_str = f"{raw_a-raw_b:+.2f} = {raw_a:.2f} - {raw_b:.2f}"
                delta_z_str = f"{z_a-z_b:+.2f} = {z_a:+.2f} - {z_b:+.2f}"
                
                with c1:
                    st.metric(label=' ', value=f"{raw:+.2f}/Kw", delta=delta_raw_str)
                with c2:
                    st.metric(label=' ', value=f"{z:+.2f} z", delta=delta_z_str)
            elif target == name_a:
                # Group 1 column: G1 - G2
                delta_raw_str = f"{raw_a-raw_b:+.2f} = {raw_a:.2f} - {raw_b:.2f}"
                delta_z_str = f"{z_a-z_b:+.2f} = {z_a:+.2f} - {z_b:+.2f}"
                
                with c1:
                    st.metric(label=' ', value=f"{raw:.1f}/Kw", delta=delta_raw_str)
                with c2:
                    st.metric(label=' ', value=f"{z:+.1f} z", delta=delta_z_str)
            elif target == name_b:
                # Group 2 column: G2 - G1 (reversed)
                delta_raw_str = f"{raw_b-raw_a:+.2f} = {raw_b:.2f} - {raw_a:.2f}"
                delta_z_str = f"{z_b-z_a:+.2f} = {z_b:+.2f} - {z_a:+.2f}"
                
                with c1:
                    st.metric(label=' ', value=f"{raw:.1f}/Kw", delta=delta_raw_str)
                with c2:
                    st.metric(label=' ', value=f"{z:+.1f} z", delta=delta_z_str)
            
            # st.divider()

    with col1:
        st.info(f"**Group 1: {name_a}**")
        st.code(q_a, language="python")
        render_feat_metrics(df_smpl_feats, name_a, f"{name_a} - {name_b}", f"{name_a} (G1)")

    with col2:
        st.error(f"**Group 2: {name_b}**")
        st.code(q_b, language="python")
        render_feat_metrics(df_smpl_feats, name_b, f"{name_a} - {name_b}", f"{name_b} (G2)")
        
    with col3:
        st.info(f"**Difference: Δ(Group 1 - Group 2)**")
        st.code(f'({q_a}) - ({q_b})', language="python")
        
        # The key in the dataframe is f"{name_a} - {name_b}"
        actual_diff_target = f"{name_a} - {name_b}"
        display_diff_label = "Δ(G1 - G2)"
        render_feat_metrics(
            df_smpl_feats, 
            actual_diff_target, 
            actual_diff_target, 
            display_diff_label, 
            sort_by='feat_rank', 
            ascending=True, 
            is_diff=True
        )

    
