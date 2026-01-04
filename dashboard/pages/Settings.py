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
import pandas as pd
from datetime import datetime
from osp import (
    STASH_DASHBOARD_GROUPS,
    STASH_DASHBOARD_COMPARISONS,
    STASH_DASHBOARD_STATE,
    COMPARISONS,
    get_corpus_metadata,
)

st.set_page_config(page_title="Settings", layout="wide")

st.title("Settings")
st.caption("Manage dashboard state, stashes, and default data.")

# --- Helpers ---
def run_query(df: pd.DataFrame, query_str: str):
    q = query_str if query_str else "1==1"
    return df.query(q)

def _seed_comparisons_from_constants():
    """Seed default groups/comparisons from osp.constants.COMPARISONS."""
    df_meta = get_corpus_metadata()
    seeded = []

    def _save_group_payload(name: str, query_str: str):
        try:
            df_filtered = run_query(df_meta, query_str)
            n_texts = int(len(df_filtered))
        except Exception:
            n_texts = 0
        payload = {
            "name": name,
            "query_str": query_str or "1==1",
            "query_struct": {},
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "n_texts": n_texts,
        }
        STASH_DASHBOARD_GROUPS[name] = payload

    for g1, g2 in COMPARISONS:
        name_a, query_a = g1
        name_b, query_b = g2

        _save_group_payload(name_a, query_a)
        _save_group_payload(name_b, query_b)

        comp_name = f"{name_a} vs {name_b}"
        comp_payload = {
            "name": comp_name,
            "group_a": {"name": name_a, "query_str": query_a, "query_struct": {}},
            "group_b": {"name": name_b, "query_str": query_b, "query_struct": {}},
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        STASH_DASHBOARD_COMPARISONS[comp_name] = comp_payload
        seeded.append(comp_name)
    return seeded

# --- UI ---

st.header("Stash Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Groups Stash")
    n_groups = len(STASH_DASHBOARD_GROUPS)
    st.metric("Total saved groups", n_groups)
    
    if st.button("Clear groups stash", type="secondary", help="Delete all saved groups from HashStash."):
        for k in list(STASH_DASHBOARD_GROUPS.keys()):
            del STASH_DASHBOARD_GROUPS[k]
        st.success("Cleared groups stash.")
        st.rerun()

with col2:
    st.subheader("Comparisons Stash")
    n_comps = len(STASH_DASHBOARD_COMPARISONS)
    st.metric("Total saved comparisons", n_comps)
    
    if st.button("Clear comparisons stash", type="secondary", help="Delete all saved comparisons from HashStash."):
        for k in list(STASH_DASHBOARD_COMPARISONS.keys()):
            del STASH_DASHBOARD_COMPARISONS[k]
        st.success("Cleared comparisons stash.")
        st.rerun()

st.subheader("App State Stash")
n_state = len(STASH_DASHBOARD_STATE)
st.metric("Total state keys", n_state)
if st.button("Clear app state stash", type="secondary", help="Reset all persistent UI state (last viewed items, etc.)"):
    for k in list(STASH_DASHBOARD_STATE.keys()):
        del STASH_DASHBOARD_STATE[k]
    st.success("Cleared app state stash.")
    st.rerun()

st.divider()

st.header("Data Seeding")
st.write("Populate stashes with default comparisons defined in `osp.constants.COMPARISONS`.")

if st.button("Seed defaults", type="primary"):
    with st.spinner("Seeding default groups and comparisons..."):
        try:
            seeded = _seed_comparisons_from_constants()
            st.success(f"Successfully seeded {len(seeded)} comparisons and their associated groups.")
        except Exception as e:
            st.error(f"Error seeding defaults: {e}")

st.header("Cache Management")
if st.button("Clear all Streamlit caches", help="Clear st.cache_data and st.cache_resource"):
    st.cache_data.clear()
    st.cache_resource.clear()
    # Also clear the lru_cache in osp
    from osp import cache
    cache.cache_clear()
    st.success("All caches cleared.")
    st.rerun()

st.divider()

st.header("Session State")
if st.checkbox("Show raw session state"):
    st.write(st.session_state)

