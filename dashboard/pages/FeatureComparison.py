import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *

import streamlit as st
from utils import *


def _load_saved_comparisons():
    try:
        return {k: STASH_DASHBOARD_COMPARISONS[k] for k in STASH_DASHBOARD_COMPARISONS.keys()}
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

# 1. Retrieve choices (stash-backed)
last_comp_name = get_current_comparison_name()
comparison_param = st.query_params.get("comparison")

def _label_to_param(label: str) -> str:
    return str(label).replace(" ", "_") if label is not None else ""

def _find_index_for_param(param_val, options):
    if not param_val:
        return None
    for i, opt in enumerate(options):
        if _label_to_param(opt.get("label", "")) == str(param_val):
            return i
    return None

# 2. Build options list strictly from stash
stash_comps = _load_saved_comparisons()
comparison_options = []
for name, comp in sorted(stash_comps.items()):
    g1 = comp.get("group_a", {}) if isinstance(comp, dict) else {}
    g2 = comp.get("group_b", {}) if isinstance(comp, dict) else {}
    if not g1 or not g2:
        continue
    label = comp.get("name") or f"{g1.get('name','G1')} vs {g2.get('name','G2')}"
    comparison_options.append(
        {
            "label": label,
            "group_a": g1,
            "group_b": g2,
        }
    )

# 3. Initial selection based on last remembered comparison (if present)
initial_index = 0
if comparison_param:
    match_idx = _find_index_for_param(comparison_param, comparison_options)
    if match_idx is not None:
        initial_index = match_idx
elif last_comp_name:
    match_idx = next(
        (
            i
            for i, opt in enumerate(comparison_options)
            if opt.get("label") == last_comp_name
        ),
        None,
    )
    if match_idx is not None:
        initial_index = match_idx

# Guard: if no comparisons available, prompt user to seed or save one
if not comparison_options:
    st.warning("No saved comparisons found in HashStash. Create one in Sampling or seed defaults in Settings.")
    st.stop()

selected_comparison = None
run_calc = False

with topcol2:
    topcol2a, topcol2b = st.columns([7,3], vertical_alignment="bottom")
    # 5. Display dropdown
    with topcol2a:
        selected_comparison = st.selectbox(
            "Group Comparison",
            options=comparison_options,
            index=initial_index,
            format_func=lambda x: x["label"],
        )
    with topcol2b:
        run_calc = st.button("Calculate", type="primary", use_container_width=True)
        auto_submit = bool(comparison_param) and selected_comparison is not None
        if auto_submit:
            run_calc = True
    
    # Update URL parameters and global state when selection changes
    if selected_comparison:
        # only track comparison in URL
        for k in ("q_a", "q_b", "n_a", "n_b"):
            st.query_params.pop(k, None)
        st.query_params["comparison"] = _label_to_param(selected_comparison["label"])

        set_current_comparison(selected_comparison["label"], selected_comparison)

midcol1, midcol2 = st.columns(2)


@st.cache_data
def load_slice_ids(query):
    """Cache the slice ID lookup."""
    return get_slice_ids(query)


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


# Show the selected comparison details only after button click
if selected_comparison and run_calc:
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

        # with st.status("Calculating feature statistics...", expanded=False) as status:
        try:
            with log_progress("Calculating feature statistics"):
                # Use cached function from utils
                df_smpl_feats = load_comparison_stats(tuple(groups_train))
                df_smpl_feats = df_smpl_feats.sample(frac=1)

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

            with log_progress("Fetching slice IDs and examples"):
                print("Getting slice IDs...")
                # Use cached function
                slice_ids_g1 = load_slice_ids(q_a)
                slice_ids_g2 = load_slice_ids(q_b)

                random.shuffle(slice_ids_g1)
                random.shuffle(slice_ids_g2)

                with log_progress("Fetching cached examples"):    
                    df_egs_g1 = load_slice_feat_examples(
                        slice_ids_g1, top_feats_g1, num_egs=3
                    )  # .drop_duplicates(['slice_id', 'feat']).sample(frac=1).iloc[:EG_NUM_EG]
                    df_egs_g2 = load_slice_feat_examples(
                        slice_ids_g2, top_feats_g2, num_egs=3
                    )  # .drop_duplicates(['slice_id', 'feat']).sample(frac=1).iloc[:EG_NUM_EG]

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
            # status.update(
                # label="Error calculating feature statistics", state="error"
            # )

    col1, col2 = st.columns(2)

    # helper to render metrics
    def render_feat_metrics(
        df, target, title, examples_dict, sort_by="feat_rank", active_feat_type=None
    ):
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

        # Track feature type and global ordering; filtering happens upstream
        df_target = df_target.copy()
        df_target["feat_type"] = df_target["feat"].apply(
            lambda x: x.split("_", 1)[0] if isinstance(x, str) and "_" in x else str(x)
        )
        df_target["global_rank"] = range(1, len(df_target) + 1)

        if active_feat_type:
            df_target = df_target[df_target["feat_type"] == active_feat_type]
            if df_target.empty:
                st.write("No distinctive features found.")
                return

        for _, row in df_target.iterrows():
            feat = row["feat"]
            if feat in BAD_SLICE_FEATS:
                continue
            global_rank = int(row.get("global_rank", 0)) or 0

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

            feat_typex,feat_namex = feat.split("_", 1)
            feat_desc_default = f'{feat_namex} ({feat_typex})'
            feat_desc = FEAT2DESC.get(feat, feat_desc_default)
            st.divider()
            st.markdown(f"#### {global_rank}. {feat_desc}")

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
            eg_height = 185
            container_div = f'<div style="height: {eg_height}px; overflow: hidden; scrollbar-width: none; font-family: monospace; font-size: 12px; white-space: nowrap; background: transparent; margin: 0; padding: 0; line-height: 1.0; display: flex; flex-direction: column;">'
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
                f'<div style="white-space: nowrap;">{all_html}</div>'
            )
            all_html_wrapped = f"{container_div}{all_html_wrapped}</div>"

            st.components.v1.html(all_html_wrapped, height=eg_height, scrolling=False)

            c1, c1b, c2, c2b, c3 = st.columns(
                [3, 0.5, 3, 0.5, 3], vertical_alignment="top"
            )
            with c1b:
                st.text("–")
            with c2b:
                st.text("=")

            if target == name_b:
                raw_a2 = f"{raw_a:.1f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.1f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2_rev = (
                    f"{raw_b-raw_a:.1f}" if raw_b - raw_a > 1 else f"{raw_b-raw_a:.2f}"
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
                raw_a2 = f"{raw_a:.1f}" if raw_a > 1 else f"{raw_a:.2f}"
                raw_b2 = f"{raw_b:.1f}" if raw_b > 1 else f"{raw_b:.2f}"
                raw_diff2 = (
                    f"{raw_a-raw_b:.1f}" if raw_a - raw_b > 1 else f"{raw_a-raw_b:.2f}"
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


    # Shared tabs so both columns respond to the same selection
    # Build ordered feat type list (avg rank ascending) from the combined targets
    df_smpl_feats['feat_type'] = df_smpl_feats['feat'].apply(lambda x: x.split("_", 1)[0])
    type_ranks = df_smpl_feats.groupby("feat_type").min(numeric_only=True).feat_rank.to_dict()

    def _avg_rank(ftype: str):
        return int(type_ranks.get(ftype)) or 0
        # vals = type_ranks.get(ftype) or []
        # return sum(vals) / len(vals) if vals else float("inf")

    nice_names = {
        "sent": "Clause forms",
        "pos": "Parts of speech",
        "deprel": "Dependency relations",
        "phrase": "Phrasal forms",
        "ttr": "Diversity metrics",
    }

    feat_types_all = sorted(type_ranks.keys(), key=_avg_rank)

    tab_labels = ["All"] + [
        f"{nice_names.get(ftype, ftype.title())} (min rank = {(_avg_rank(ftype))})"
        for ftype in feat_types_all
    ]
    tabs = st.tabs(tab_labels)

    for tab, ftype in zip(tabs, [None] + feat_types_all):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"##### Group 1: {name_a}")
                render_feat_metrics(
                    df_smpl_feats,
                    name_a,
                    f"{name_a} (G1)",
                    egs_g1,
                    sort_by="feat_rank1",
                    active_feat_type=ftype,
                )

            with col2:
                st.error(f"##### Group 2: {name_b}")
                render_feat_metrics(
                    df_smpl_feats,
                    name_b,
                    f"{name_b} (G2)",
                    egs_g2,
                    sort_by="feat_rank2",
                    active_feat_type=ftype,
                )
