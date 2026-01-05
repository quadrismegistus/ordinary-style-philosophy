import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *

import streamlit as st
from utils import *
from dashboard.components import render_feature_summary


def _load_saved_groups():
    try:
        return {k: STASH_DASHBOARD_GROUPS[k] for k in STASH_DASHBOARD_GROUPS.keys()}
    except Exception:
        return {}

st.set_page_config(page_title="Most Distinctive Features", layout="wide")

topcol1, topcol2 = st.columns(2, vertical_alignment="bottom")
with topcol1:
    st.title("Most Distinctive Features")

# Inject CSS for colored backgrounds in metrics
st.markdown(
    """
    <style>
    /* Target the metrics within the specific columns */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] {
        background-color: rgba(0, 104, 201, 0.1);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(0, 104, 201, 0.2);
    }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] {
        background-color: rgba(255, 43, 43, 0.1);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(255, 43, 43, 0.2);
    }
    div[data-testid="column"]:nth-of-type(5) div[data-testid="stMetric"] {
        background-color: rgba(125, 53, 171, 0.1);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(125, 53, 171, 0.2);
    }
    </style>
""",
    unsafe_allow_html=True,
)

last_group_a = get_current_group_a()
last_group_b = get_current_group_b()

saved_groups = _load_saved_groups()
if not saved_groups:
    st.warning(
        "No saved groups found. Create one on the Groups page before comparing features."
    )
    st.stop()

group_names = sorted(saved_groups.keys())

def _default_index(last_selected: str | None, fallback: int) -> int:
    if last_selected and last_selected in group_names:
        return group_names.index(last_selected)
    fallback_idx = fallback if fallback is not None else 0
    if fallback_idx < len(group_names):
        return fallback_idx
    return max(0, len(group_names) - 1)

selected_group_a = None
selected_group_b = None
run_calc = False

with topcol2:
    selection_col, button_col = st.columns([7, 3], vertical_alignment="bottom")
    with selection_col:
        grp_cols = st.columns(2)
        with grp_cols[0]:
            selected_group_a = st.selectbox(
                "Group 1",
                options=group_names,
                index=_default_index(last_group_a, 0),
                key="feature_comp_group_a",
            )
        with grp_cols[1]:
            selected_group_b = st.selectbox(
                "Group 2",
                options=group_names,
                index=_default_index(last_group_b, 1),
                key="feature_comp_group_b",
            )
    with button_col:
        run_calc = st.button("Calculate", type="primary", width='stretch')

if selected_group_a:
    set_current_group_a(selected_group_a)
if selected_group_b:
    set_current_group_b(selected_group_b)

midcol1, midcol2 = st.columns(2)



def _example_lookup(df_egs: pd.DataFrame):
    """Map feature -> list[example_dict]."""
    if not isinstance(df_egs, pd.DataFrame) or df_egs.empty:
        return {}
    # expected columns: feature, eg_text, eg_html, slice_id, ...
    out: dict = {}
    for _, r in df_egs.iterrows():
        feat = r.get("feature")
        if not feat:
            continue
        out.setdefault(feat, []).append(r.to_dict())
    return out


if run_calc and selected_group_a and selected_group_b:
    group_a_data = saved_groups.get(selected_group_a, {})
    group_b_data = saved_groups.get(selected_group_b, {})
    name_a = group_a_data.get("name") or selected_group_a
    name_b = group_b_data.get("name") or selected_group_b
    q_a = group_a_data.get("query_str") or "1==1"
    q_b = group_b_data.get("query_str") or "1==1"

    with midcol1:
        st.write(f"### {name_a} vs {name_b}")

    groups_train = [(name_a, q_a), (name_b, q_b)]

    with midcol2:
        status_window = get_status_window()

        try:
            with log_progress("Calculating feature statistics"):
                df_smpl_feats = load_comparison_stats(tuple(groups_train))
                df_smpl_feats = df_smpl_feats.sample(frac=1)

            top_feats_g1 = (
                df_smpl_feats[df_smpl_feats["target"] == name_a]
                .sort_values("feat_rank1")
                .head(NUM_DISTINCTIVE_FEATS)["feat"]
                .astype(str)
                .tolist()
            )
            top_feats_g2 = (
                df_smpl_feats[df_smpl_feats["target"] == name_b]
                .sort_values("feat_rank2")
                .head(NUM_DISTINCTIVE_FEATS)["feat"]
                .astype(str)
                .tolist()
            )

            with log_progress("Fetching slice IDs and examples"):
                print("Getting slice IDs...")
                slice_ids_g1 = load_slice_ids(q_a)
                slice_ids_g2 = load_slice_ids(q_b)

                # random.shuffle(slice_ids_g1)
                # random.shuffle(slice_ids_g2)

                with log_progress("Fetching cached examples"):
                    df_egs_g1 = load_slice_feat_examples(
                        slice_ids_g1, top_feats_g1, num_egs=NUM_EG_PER_FEAT
                    )
                    df_egs_g2 = load_slice_feat_examples(
                        slice_ids_g2, top_feats_g2, num_egs=NUM_EG_PER_FEAT
                    )

                with log_progress("Looking up examples for group 1"):
                    egs_g1 = _example_lookup(df_egs_g1)
                with log_progress("Looking up examples for group B"):
                    egs_g2 = _example_lookup(df_egs_g2)

                print(
                    f"Found examples for {len(egs_g1)} features in group A and {len(egs_g2)} features in group B."
                )

        except Exception as e:
            st.error(f"Error calculating feature statistics: {e}")
            df_smpl_feats = pd.DataFrame()
            egs_g1, egs_g2 = {}, {}

    render_feature_summary(df_smpl_feats, name_a, name_b, egs_g1, egs_g2)
