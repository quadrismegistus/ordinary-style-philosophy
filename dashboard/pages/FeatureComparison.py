import sys, os

# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path:
    sys.path.append(PATH_DASHBOARD)
from osp import *

NUM_DISTINCTIVE_FEATS = 25
EG_TXT_WINDOW = 60
EG_TXT_WINDOW_LEFT = 25
EG_TXT_WINDOW_RIGHT = 35
EG_NUM_EG = 25

import streamlit as st
import pandas as pd
from collections import Counter
from streamlit_local_storage import LocalStorage
from utils import *

DEFAULT_COMPARISONS = []
for (name_a, query_a), (name_b, query_b) in COMPARISONS:
    DEFAULT_COMPARISONS.append(
        {
            "label": f"{name_a} vs {name_b}",
            "group_a": {"name": name_a, "query_str": query_a},
            "group_b": {"name": name_b, "query_str": query_b},
        }
    )
DEFAULT_COMPARISON = DEFAULT_COMPARISONS[0]


def _load_saved_comparisons():
    try:
        return {k: STASH_DASHBOARD_COMPARISONS[k] for k in STASH_DASHBOARD_COMPARISONS.keys()}
    except Exception:
        return {}

st.set_page_config(page_title="Most Distinctive Features", layout="wide")

topcol1, topcol2 = st.columns(2)
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

ls = LocalStorage()

# 1. Define hardcoded default comparison
# Import COMPARISONS from osp.constants

# Build default comparisons from COMPARISONS constant

# 2. Retrieve saved comparison from localStorage
saved_data = ls.getItem("osp_comparison_groups")

# 3. Build options list
comparison_options = list(DEFAULT_COMPARISONS)

# 3a. Add stash comparisons
stash_comps = _load_saved_comparisons()
for name, comp in sorted(stash_comps.items()):
    g1 = comp.get("group_a", {}) if isinstance(comp, dict) else {}
    g2 = comp.get("group_b", {}) if isinstance(comp, dict) else {}
    if not g1 or not g2:
        continue
    label = comp.get("name") or f"{g1.get('name','G1')} vs {g2.get('name','G2')}"
    comparison_options.append(
        {
            "label": f"{label} (stash)",
            "group_a": g1,
            "group_b": g2,
        }
    )

if saved_data and isinstance(saved_data, dict):
    g1 = saved_data.get("group_a", {})
    g2 = saved_data.get("group_b", {})
    if g1 and g2:
        label = f"{g1.get('name', 'Group 1')} vs {g2.get('name', 'Group 2')} (Saved)"
        comparison_options.append({"label": label, "group_a": g1, "group_b": g2})

# 4. Check URL parameters to set initial selection or add URL-based option
q_a_url = st.query_params.get("q_a")
q_b_url = st.query_params.get("q_b")
n_a_url = st.query_params.get("n_a")
n_b_url = st.query_params.get("n_b")

initial_index = 0
if q_a_url and q_b_url:
    # Check if these queries match an existing option
    match_idx = next((i for i, opt in enumerate(comparison_options) 
                      if opt["group_a"]["query_str"] == q_a_url and opt["group_b"]["query_str"] == q_b_url), None)
    
    if match_idx is not None:
        initial_index = match_idx
    else:
        # Add a new transient option from the URL parameters
        url_opt = {
            "label": f"{n_a_url or 'G1'} vs {n_b_url or 'G2'} (URL)",
            "group_a": {"name": n_a_url or "G1", "query_str": q_a_url},
            "group_b": {"name": n_b_url or "G2", "query_str": q_b_url},
        }
        comparison_options.append(url_opt)
        initial_index = len(comparison_options) - 1

with topcol2:
    # 5. Display dropdown
    selected_comparison = st.selectbox(
        "Select Comparison",
        options=comparison_options,
        index=initial_index,
        format_func=lambda x: x["label"],
    )
    
    # Update URL parameters when selection changes
    if selected_comparison:
        st.query_params["q_a"] = selected_comparison["group_a"]["query_str"]
        st.query_params["q_b"] = selected_comparison["group_b"]["query_str"]
        st.query_params["n_a"] = selected_comparison["group_a"]["name"]
        st.query_params["n_b"] = selected_comparison["group_b"]["name"]

midcol1, midcol2 = st.columns(2)


@st.cache_data
def load_slice_feat_examples(slice_ids, feats, num_egs=25, max_slices=10_000):
    """
    Return a dataframe of cached examples for the given feature list, sampled from the
    provided slice_ids. Uses STASH_FEAT_EXAMPLES2 via osp.features.get_slice_feat_egs().
    """
    # Make sure inputs are stable / hashable-ish for Streamlit cache
    slice_ids_l = list(slice_ids) if slice_ids is not None else None
    feats_l = list(feats) if feats is not None else None
    odf = get_slice_feat_egs(
        slice_ids=slice_ids_l,
        feats=feats_l,
        num_egs=num_egs,
        max_slices=max_slices,
    )
    print("loaded slice feat examples")
    print(f"df has shape: {odf.shape}")
    print(f"df has columns: {', '.join(odf.columns)}")
    return odf

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


# Show the selected comparison details
if selected_comparison:
    with midcol1:
        st.write(f"### {selected_comparison['label']}")

    # Load sample and features once for both groups
    q_a = selected_comparison["group_a"]["query_str"]
    q_b = selected_comparison["group_b"]["query_str"]
    name_a = selected_comparison["group_a"]["name"]
    name_b = selected_comparison["group_b"]["name"]

    groups_train = [(name_a, q_a), (name_b, q_b)]

    with midcol2:
        # Get the global status window
        status_window = get_status_window()

        with st.status("Calculating feature statistics...", expanded=False) as status:
            try:
                # Use the status window to capture stdout
                with status_window.capture("Calculating feature statistics"):
                    with logmap(f"logmapping"):
                        print('done!!')
                    df_smpl_feats = (
                        get_balanced_slice_sample_feats(
                            groups_train, with_diff_rows=True
                        )
                        .reset_index()
                        .sample(frac=1)
                    )

                # --- Cached examples (per group) ---
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

                with status_window.capture("Fetching slice IDs and examples"):
                    print("Getting slice IDs...")
                    slice_ids_g1 = get_slice_ids(q_a)
                    slice_ids_g2 = get_slice_ids(q_b)

                    random.shuffle(slice_ids_g1)
                    random.shuffle(slice_ids_g2)

                    print("Fetching cached examples...")
                    df_egs_g1 = load_slice_feat_examples(
                        slice_ids_g1, top_feats_g1, num_egs=100
                    )  # .drop_duplicates(['slice_id', 'feat']).sample(frac=1).iloc[:EG_NUM_EG]
                    df_egs_g2 = load_slice_feat_examples(
                        slice_ids_g2, top_feats_g2, num_egs=100
                    )  # .drop_duplicates(['slice_id', 'feat']).sample(frac=1).iloc[:EG_NUM_EG]

                    egs_g1 = _example_lookup(df_egs_g1)
                    egs_g2 = _example_lookup(df_egs_g2)

                    print(
                        f"Found examples for {len(egs_g1)} features in G1 and {len(egs_g2)} features in G2."
                    )

                status.update(
                    label="Feature statistics and examples complete!",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                st.error(f"Error calculating feature statistics: {e}")
                df_smpl_feats = pd.DataFrame()
                egs_g1, egs_g2 = {}, {}
                status.update(
                    label="Error calculating feature statistics", state="error"
                )

    col1, col2 = st.columns(2)

    # helper to render metrics
    def render_feat_metrics(df, target, title, examples_dict, sort_by="feat_rank"):
        def _format(ex):
            slice_id = ex.get("slice_id")
            href = f"/Passages?slice_id={slice_id}"
            meta = get_text_metadata(slice_id)
            title = meta.get("title", "Unknown")
            author = meta.get("author", "Unknown")
            year = meta.get("year", "Unknown")
            journal = meta.get("journal", "Unknown")
            signature = f'—{author}, "{title}", <i>{journal}</i> ({year})'
            html = ex.get("eg_html")
            word_id = ex.get("word_id")
            href_sent = f'/Sentence?slice_id={slice_id}&sent_id={ex.get("sent_id")+1}'
            if word_id is not None:
                href_sent += f'&word_id={word_id}'
            out = f'<a href="{href_sent}" target="_blank">{html}</a> <a href="{href}" target="_blank" style="vertical-align: bottom; text-decoration: none; color: inherit; "><br/><small>{signature}</small></a>'
            return out

        # st.markdown(f"#### {title}")
        if df.empty:
            st.write("No data available.")
            return

        df_target = (
            df[df["target"] == target].sort_values(sort_by).head(NUM_DISTINCTIVE_FEATS)
        )
        if df_target.empty and " - " in target:
            # Try reverse difference if it's a difference target and not found
            t1, t2 = target.split(" - ")
            rev_target = f"{t2} - {t1}"
            df_target = (
                df[df["target"] == rev_target]
                .sort_values(sort_by)
                .head(NUM_DISTINCTIVE_FEATS)
            )

        if df_target.empty:
            st.write("No distinctive features found.")
            return

        # Prepare Group 1 and Group 2 lookups for badges
        lookup_g1 = df[df["target"] == name_a].set_index("feat")
        lookup_g2 = df[df["target"] == name_b].set_index("feat")

        # Display metrics in a grid or list
        metric_num = 0
        for _, row in df_target.iterrows():
            feat = row["feat"]
            if feat in BAD_SLICE_FEATS:
                continue
            z = row["z"]
            raw = row["raw"]

            # Get values for both groups for the badge subtraction string
            try:
                row_g1 = lookup_g1.loc[feat]
                row_g2 = lookup_g2.loc[feat]
                if isinstance(row_g1, pd.DataFrame):
                    row_g1 = row_g1.iloc[0]
                if isinstance(row_g2, pd.DataFrame):
                    row_g2 = row_g2.iloc[0]

                raw_a, raw_b = row_g1["raw"], row_g2["raw"]
                z_a, z_b = row_g1["z"], row_g2["z"]
            except (KeyError, IndexError):
                raw_a = raw_b = z_a = z_b = 0

            # Show 3 columns inside each of the main cols
            feat_desc = FEAT2DESC.get(feat, feat)
            metric_num += 1
            feat_type, feat_name = feat.split("_", 1)
            # feat_hdr = f"{feat_name} ({feat_type})\n*{feat_desc}*"
            feat_hdr = f"{feat_desc}"
            st.divider()
            st.markdown(f"#### {metric_num}. {feat_hdr}")

            ex_list = (examples_dict or {}).get(feat) or []
            if isinstance(ex_list, dict):
                ex_list = [ex_list]
            ex_list_text = [ex.get("eg_text", "") for ex in ex_list]
            ex_list_text = center_starred_keywords(
                ex_list_text,
                window_left=EG_TXT_WINDOW_LEFT,
                window_right=EG_TXT_WINDOW_RIGHT,
                keep_asterisks=False,
            )
            # xmargin = ' '*10
            # ex_list_text = [f'{xstr}{xmargin}# {ex.get("slice_id")}' for xstr, ex in zip(ex_list_text, ex_list)]
            # all_eg_text = "\n".join(ex_list_text)
            # st.code(all_eg_text, height=128,line_numbers=False,language="markdown")
            eg_height = 185
            container_div = f'<div style="height: {eg_height}px; overflow: auto; scrollbar-width: none; font-family: monospace; font-size: 12px; white-space: nowrap; background: transparent; margin: 0; padding: 0; line-height: 1.0; display: flex; flex-direction: column;">'
            all_eg_text_html = [container_div]
            for eg_text, ex in zip(ex_list_text, ex_list):
                all_eg_text_html.append(
                    f'<p style="margin: 0; padding: 0;"><a href="/Passages?slice_id={ex.get("slice_id")}" target="_blank" style="text-decoration: none; color: inherit;">{eg_text}</a></p>'
                )
            all_eg_text_html.append("</div>")
            all_eg_text_html = "\n".join(all_eg_text_html)

            all_html = "\n\n<br/><br/>".join(
                [
                    _format(ex)
                    for ex in ex_list
                    if ex.get("eg_html") and ex.get("slice_id")
                ]
            )
            all_html_wrapped = (
                f'<div style="white-space: nowrap; overflow-x: auto;">{all_html}</div>'
            )
            all_html_wrapped = f"{container_div}{all_html_wrapped}</div>"

            st.components.v1.html(all_html_wrapped, height=eg_height, scrolling=False)

            # if ex_list:
            #     with st.expander("Examples", expanded=False):

            #         st.components.v1.html(all_html_wrapped, height=220, scrolling=True)

            c1, c1b, c2, c2b, c3 = st.columns(
                [3, 0.5, 3, 0.5, 3], vertical_alignment="top"
            )
            with c1b:
                st.text("–")
            with c2b:
                st.text("=")

            if target == name_b:
                # Group 2 column: show G2, then G1, then G2-G1
                raw_a2 = f"{raw_a:.0f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.0f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2_rev = (
                    f"{raw_b-raw_a:.0f}" if raw_b - raw_a > 1 else f"{raw_b-raw_a:.2f}"
                )
                with c1:
                    st.metric(
                        label=f"{name_b} (G2)", value=f"{raw_b2}", delta=f"{z_b:+.2f}z"
                    )
                with c2:
                    st.metric(
                        label=f"{name_a} (G1)", value=f"{raw_a2}", delta=f"{z_a:+.2f}z"
                    )
                with c3:
                    st.metric(
                        label="Δ (G2-G1)",
                        value=f"{raw_diff2_rev}",
                        delta=f"{z_b-z_a:+.2f}z",
                    )
            else:
                # Group 1 column: Always show G1, G2, then G1-G2
                raw_a2 = f"{raw_a:.0f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.0f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2 = (
                    f"{raw_a-raw_b:.0f}" if raw_a - raw_b > 1 else f"{raw_a-raw_b:.2f}"
                )
                with c1:
                    st.metric(
                        label=f"{name_a} (G1)", value=f"{raw_a2}", delta=f"{z_a:+.2f}z"
                    )
                with c2:
                    st.metric(
                        label=f"{name_b} (G2)", value=f"{raw_b2}", delta=f"{z_b:+.2f}z"
                    )
                with c3:
                    st.metric(
                        label="Δ (G1-G2)",
                        value=f"{raw_diff2}",
                        delta=f"{z_a-z_b:+.2f}z",
                    )

            # Examples dropdown (collapsed by default): tabs for Text vs HTML

    with col1:
        st.info(f"##### Group 1: {name_a}")
        # st.code(q_a, language="python")
        render_feat_metrics(
            df_smpl_feats, name_a, f"{name_a} (G1)", egs_g1, sort_by="feat_rank1"
        )

    with col2:
        st.error(f"##### Group 2: {name_b}")
        # st.code(q_b, language="python")
        render_feat_metrics(
            df_smpl_feats, name_b, f"{name_b} (G2)", egs_g2, sort_by="feat_rank2"
        )
