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
from collections import Counter
import plotly.express as px
from streamlit_local_storage import LocalStorage
from utils import *
from osp import (
    STASH_DASHBOARD_GROUPS,
    STASH_DASHBOARD_COMPARISONS,
    COMPARISONS,
)

st.set_page_config(page_title="Classification", layout="wide")
ls = LocalStorage()



@st.cache_data
def load_metadata():
    df = get_corpus_metadata().copy()
    for c in ["discipline", "period", "journal"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


@st.cache_data
def load_parsed_slice_ids():
    return get_parsed_slice_ids()

@st.cache_data
def load_all_slice_ids():
    # Uses osp/slices.py:get_all_text_slice_ids (text_id__slice_num:02d)
    return get_all_text_slice_ids()


def load_saved_groups():
    try:
        return {k: STASH_DASHBOARD_GROUPS[k] for k in STASH_DASHBOARD_GROUPS.keys()}
    except Exception:
        return {}


def _load_saved_comparisons():
    try:
        return {k: STASH_DASHBOARD_COMPARISONS[k] for k in STASH_DASHBOARD_COMPARISONS.keys()}
    except Exception:
        return {}


def _save_comparison(name: str, payload: dict):
    STASH_DASHBOARD_COMPARISONS[name] = payload


def _seed_comparisons_from_constants(df_meta: pd.DataFrame):
    """Seed default groups/comparisons from osp.constants.COMPARISONS."""
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
        _save_comparison(comp_name, comp_payload)
        seeded.append(comp_name)
    return seeded


def select_saved_group(title: str, key_prefix: str, saved_groups: dict):
    # st.subheader(title)
    names = sorted(saved_groups.keys())
    if not names:
        st.info("No saved groups found. Create one on the Groups page.")
        return "", {}, ""

    default_index = 0 if key_prefix == "grp_a" else (1 if len(names) > 1 else 0)
    selected_name = st.selectbox(
        title,
        options=names,
        index=default_index,
        key=f"{key_prefix}_select",
    )
    data = saved_groups.get(selected_name, {}) if selected_name else {}
    query_str = data.get("query_str", "") if isinstance(data, dict) else ""
    query_struct = data.get("query_struct", {}) if isinstance(data, dict) else {}
    # st.code(query_str or "True")
    return selected_name or title, query_struct, query_str


def run_query(df: pd.DataFrame, query_str: str):
    q = query_str if query_str else "1==1"
    return df.query(q)


df_meta = load_metadata()
saved_groups = load_saved_groups()

if not saved_groups:
    st.warning("No saved groups found. Create a group on the Groups page first.")
    st.stop()


left, right = st.columns(2)

with left:
    st.title("Comparisons")
    st.caption("Select two groups to compare.")


def get_period_text_dist(df_sel: pd.DataFrame):
    if df_sel.empty or "period" not in df_sel.columns:
        return pd.DataFrame(columns=["period", "n_texts"])
    vc = df_sel["period"].fillna("").astype(str).value_counts()
    out = vc.rename_axis("period").reset_index(name="n_texts")
    out = out[out["period"].astype(str).str.strip() != ""]
    return out.sort_values("period")


def slice_id_to_text_id(slice_id: str):
    if not isinstance(slice_id, str):
        slice_id = str(slice_id)
    if "__" in slice_id:
        # ID format is: f"{text_id}__{slice_num:02d}"
        return slice_id.split("__", 1)[0]
    return slice_id


def get_slice_counts_by_text(slice_ids):
    return Counter(slice_id_to_text_id(sid) for sid in slice_ids)


def get_all_slice_counts_by_text():
    return get_slice_counts_by_text(load_all_slice_ids())


def get_parsed_slice_counts_by_text():
    return get_slice_counts_by_text(load_parsed_slice_ids())


def get_period_slice_dist(df_sel: pd.DataFrame, text_id2nslices: Counter, colname="n_slices"):
    if df_sel.empty or "period" not in df_sel.columns:
        return pd.DataFrame(columns=["period", colname])
    df_tmp = df_sel[["period"]].copy()
    df_tmp["text_id"] = df_sel.index.astype(str)
    df_tmp[colname] = df_tmp["text_id"].map(lambda tid: int(text_id2nslices.get(tid, 0)))
    out = df_tmp.groupby("period")[colname].sum().rename_axis("period").reset_index()
    out = out[out["period"].astype(str).str.strip() != ""].sort_values("period")
    return out


def get_slice_stats(
    df_sel: pd.DataFrame, text_id2nslices_total: Counter, text_id2nslices_parsed: Counter
):
    if df_sel.empty:
        return {
            "n_texts": 0,
            "n_texts_with_total_slices": 0,
            "n_texts_with_parsed_slices": 0,
            "n_total_slices": 0,
            "n_parsed_slices": 0,
            "df_period_total_slices": pd.DataFrame(columns=["period", "n_total_slices"]),
            "df_period_parsed_slices": pd.DataFrame(columns=["period", "n_parsed_slices"]),
        }

    text_ids = df_sel.index.astype(str).tolist()
    nslices_total = [int(text_id2nslices_total.get(tid, 0)) for tid in text_ids]
    nslices_parsed = [int(text_id2nslices_parsed.get(tid, 0)) for tid in text_ids]
    n_total_slices = int(sum(nslices_total))
    n_parsed_slices = int(sum(nslices_parsed))
    n_texts_with_total_slices = int(sum(1 for n in nslices_total if n > 0))
    n_texts_with_parsed_slices = int(sum(1 for n in nslices_parsed if n > 0))

    df_period_total_slices = get_period_slice_dist(
        df_sel, text_id2nslices_total, colname="n_total_slices"
    )
    df_period_parsed_slices = get_period_slice_dist(
        df_sel, text_id2nslices_parsed, colname="n_parsed_slices"
    )

    return {
        "n_texts": int(len(df_sel)),
        "n_texts_with_total_slices": n_texts_with_total_slices,
        "n_texts_with_parsed_slices": n_texts_with_parsed_slices,
        "n_total_slices": n_total_slices,
        "n_parsed_slices": n_parsed_slices,
        "df_period_total_slices": df_period_total_slices,
        "df_period_parsed_slices": df_period_parsed_slices,
    }

@st.cache_data
def load_slice_sample(
    query_a: str,
    query_b: str,
    sample_size: int | None,
    balance: bool,
    replace: bool,
    label_a: str,
    label_b: str,
):
    """
    Uses osp.slices.get_balanced_slice_sample() to define a balanced sample.
    Returns a slice-level DataFrame with metadata + [text_id, slice_id, _target].
    """
    q1 = query_a if query_a else "1==1"
    q2 = query_b if query_b else "1==1"
    groups_train = [(label_a, q1), (label_b, q2)]
    return get_balanced_slice_sample(
        groups_train,
        sample_size=sample_size,
        balance=balance,
        replace=replace,
        verbose=False,
    )





with right:
    name_a, query_a_struct, query_a_str = select_saved_group("Group 1", "grp_a", saved_groups)
    name_b, query_b_struct, query_b_str = select_saved_group("Group 2", "grp_b", saved_groups)
    label_a = name_a or "Group 1"
    label_b = name_b or "Group 2"
    comparison_name = st.text_input(
        "Comparison name",
        value=f"{label_a} vs {label_b}",
        help="Name to save this comparison in HashStash.",
    )

    # if st.button("Seed default comparisons (from constants)", use_container_width=True):
    #     try:
    #         seeded = _seed_comparisons_from_constants(df_meta)
    #         st.success(f"Seeded {len(seeded)} comparisons.")
    #     except Exception as e:
    #         st.error(f"Could not seed comparisons: {e}")

    # sample_size_input = st.number_input(
    #     "Sample size per group (0 = auto)",
    #     min_value=0,
    #     value=0,
    #     step=50,
    #     help="If balanced, defaults to the smaller group. If unbalanced, 0 keeps all.",
    # )
    # balance = st.checkbox(
    #     "Balance groups (same # of slices each)", value=True, help="Use equal counts from both groups."
    # )
    # replace = st.checkbox(
    #     "Sample with replacement",
    #     value=False,
    #     help="Allow repeated slices if sample size exceeds available.",
    # )
    col1, col2 = st.columns(2)
    with col1:
        submit = st.button("Compare", type="secondary", use_container_width=True)
    with col2:
        save = st.button("Save comparison", use_container_width=True)

if save:
    # Save both group names and their queries to localStorage
    saved_data = {
        "name": comparison_name.strip() or f"{label_a} vs {label_b}",
        "group_a": {
            "name": label_a,
            "query_str": query_a_str,
            "query_struct": query_a_struct,
        },
        "group_b": {
            "name": label_b,
            "query_str": query_b_str,
            "query_struct": query_b_struct,
        },
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        _save_comparison(saved_data["name"], saved_data)
        st.success(f"Comparison '{saved_data['name']}' saved to HashStash.")
    except Exception as e:
        st.error(f"Could not save comparison: {e}")
    ls.setItem("osp_comparison_groups", saved_data)

if submit:
    try:
        df_a_group = run_query(df_meta, query_a_str)
        df_b_group = run_query(df_meta, query_b_str)
    except Exception as e:
        st.error(f"Query error: {e}")
        df_a_group, df_b_group = pd.DataFrame(), pd.DataFrame()

    text_id2nslices_total = get_all_slice_counts_by_text()
    text_id2nslices_parsed = get_parsed_slice_counts_by_text()

    # Use get_balanced_slice_sample to define the *sample* (slice-level metadata)
    # sample_size = int(sample_size_input) if sample_size_input and sample_size_input > 0 else None
    sample_size = None
    balance = True
    replace = False

    try:
        df_slice_sample = load_slice_sample(
            query_a=query_a_str,
            query_b=query_b_str,
            sample_size=sample_size,
            balance=balance,
            replace=replace,
            label_a=label_a,
            label_b=label_b,
        )
    except Exception as e:
        st.error(f"Sampler error (get_balanced_slice_sample): {e}")
        df_slice_sample = pd.DataFrame()

    df_slice_g1 = (
        df_slice_sample[df_slice_sample["_target"] == label_a].copy()
        if not df_slice_sample.empty and "_target" in df_slice_sample.columns
        else pd.DataFrame()
    )
    df_slice_g2 = (
        df_slice_sample[df_slice_sample["_target"] == label_b].copy()
        if not df_slice_sample.empty and "_target" in df_slice_sample.columns
        else pd.DataFrame()
    )

    parsed_set = set(load_parsed_slice_ids())

    # # --- Model performance summary on the balanced sample (parsed slices only) ---
    # sample_parsed_slice_ids = []
    # slice_probs_md = ""
    # try:
    #     if not df_slice_sample.empty and "slice_id" in df_slice_sample.columns:
    #         sample_slice_ids = df_slice_sample["slice_id"].astype(str).tolist()
    #         sample_parsed_slice_ids = [sid for sid in sample_slice_ids if sid in parsed_set]
    #         if sample_parsed_slice_ids:
    #             slice_probs_md = describe_slice_probs(sample_parsed_slice_ids)
    # except Exception as e:
    #     slice_probs_md = f"Could not compute slice prediction summary: {e}"

    # if slice_probs_md:
    #     st.markdown(slice_probs_md)


    def slice_df_period_texts(df_slice: pd.DataFrame):
        if df_slice.empty or "period" not in df_slice.columns:
            return pd.DataFrame(columns=["period", "n_texts"])
        if "text_id" not in df_slice.columns:
            return pd.DataFrame(columns=["period", "n_texts"])
        dfu = df_slice[["period", "text_id"]].drop_duplicates()
        out = dfu.groupby("period").size().rename("n_texts").reset_index()
        out = out[out["period"].astype(str).str.strip() != ""].sort_values("period")
        return out

    def slice_df_period_slices(df_slice: pd.DataFrame, parsed_only: bool):
        if df_slice.empty or "period" not in df_slice.columns or "slice_id" not in df_slice.columns:
            return pd.DataFrame(columns=["period", "n_slices"])
        dfx = df_slice
        if parsed_only:
            dfx = dfx[dfx["slice_id"].astype(str).isin(parsed_set)]
        out = dfx.groupby("period").size().rename("n_slices").reset_index()
        out = out[out["period"].astype(str).str.strip() != ""].sort_values("period")
        return out

    # Sample text-level metadata (unique texts appearing in the slice sample)
    text_ids_s1 = (
        df_slice_g1["text_id"].astype(str).unique().tolist() if "text_id" in df_slice_g1.columns else []
    )
    text_ids_s2 = (
        df_slice_g2["text_id"].astype(str).unique().tolist() if "text_id" in df_slice_g2.columns else []
    )
    df_a_sample = df_meta.loc[df_meta.index.intersection(text_ids_s1)].copy()
    df_b_sample = df_meta.loc[df_meta.index.intersection(text_ids_s2)].copy()

    # Slice-level counts for sample
    n_slices_s1 = int(len(df_slice_g1))
    n_slices_s2 = int(len(df_slice_g2))
    n_parsed_s1 = (
        int(df_slice_g1["slice_id"].astype(str).isin(parsed_set).sum())
        if not df_slice_g1.empty and "slice_id" in df_slice_g1.columns
        else 0
    )
    n_parsed_s2 = (
        int(df_slice_g2["slice_id"].astype(str).isin(parsed_set).sum())
        if not df_slice_g2.empty and "slice_id" in df_slice_g2.columns
        else 0
    )

    st.session_state["classification_last_run"] = {
        "group_a_name": label_a,
        "group_b_name": label_b,
        "groups": {
            "group_a": {
                "query": query_a_str,
                "n_texts": int(len(df_a_group)),
                "df_period_texts": get_period_text_dist(df_a_group),
                "slice_stats": get_slice_stats(
                    df_a_group, text_id2nslices_total, text_id2nslices_parsed
                ),
            },
            "group_b": {
                "query": query_b_str,
                "n_texts": int(len(df_b_group)),
                "df_period_texts": get_period_text_dist(df_b_group),
                "slice_stats": get_slice_stats(
                    df_b_group, text_id2nslices_total, text_id2nslices_parsed
                ),
            },
        },
        "samples": {
            "group_a": {
                "query": query_a_str,
                "n_texts": int(len(df_a_sample)),
                "raw_n_texts": int(len(df_a_group)),
                "df_period_texts": slice_df_period_texts(df_slice_g1),
                "slice_stats": {
                    "n_total_slices": n_slices_s1,
                    "n_parsed_slices": n_parsed_s1,
                    "df_period_total_slices": slice_df_period_slices(df_slice_g1, parsed_only=False).rename(columns={"n_slices": "n_total_slices"}),
                    "df_period_parsed_slices": slice_df_period_slices(df_slice_g1, parsed_only=True).rename(columns={"n_slices": "n_parsed_slices"}),
                },
            },
            "group_b": {
                "query": query_b_str,
                "n_texts": int(len(df_b_sample)),
                "raw_n_texts": int(len(df_b_group)),
                "df_period_texts": slice_df_period_texts(df_slice_g2),
                "slice_stats": {
                    "n_total_slices": n_slices_s2,
                    "n_parsed_slices": n_parsed_s2,
                    "df_period_total_slices": slice_df_period_slices(df_slice_g2, parsed_only=False).rename(columns={"n_slices": "n_total_slices"}),
                    "df_period_parsed_slices": slice_df_period_slices(df_slice_g2, parsed_only=True).rename(columns={"n_slices": "n_parsed_slices"}),
                },
            },
        },
        # "slice_probs_md": slice_probs_md,
        # "slice_probs_n": int(len(sample_parsed_slice_ids)),
    }


run_data = st.session_state.get("classification_last_run")
if run_data:
    label_a = run_data.get("group_a_name", "Group 1")
    label_b = run_data.get("group_b_name", "Group 2")
    saved_comparisons = _load_saved_comparisons()
    if saved_comparisons:
        st.markdown("#### Saved comparisons")
        comp_names = sorted(saved_comparisons.keys())
        selected_comp = st.selectbox(
            "Saved comparisons (HashStash)",
            options=[""] + comp_names,
            format_func=lambda x: x or "Select comparison",
        )
        if selected_comp:
            comp = saved_comparisons.get(selected_comp, {})
            st.info(
                f"Name: {comp.get('name')}\n\n"
                f"Group A: {comp.get('group_a', {}).get('name')}\n"
                f"Query A: {comp.get('group_a', {}).get('query_str')}\n\n"
                f"Group B: {comp.get('group_b', {}).get('name')}\n"
                f"Query B: {comp.get('group_b', {}).get('query_str')}\n\n"
                f"Saved at: {comp.get('saved_at')}"
            )

    def fmt_int(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return str(x)

    def fmt_signed_int(x):
        try:
            xi = int(x)
            return f"{xi:+,}"
        except Exception:
            return str(x)

    def get_counts(d: dict):
        d = d or {}
        ss = (d.get("slice_stats", {}) or {}) if isinstance(d, dict) else {}
        n_texts = int(d.get("n_texts", 0) or 0) if isinstance(d, dict) else 0
        n_slices = int(ss.get("n_total_slices", 0) or 0)
        n_parsed = int(ss.get("n_parsed_slices", 0) or 0)
        return n_texts, n_slices, n_parsed

    def to_period_df(d, group_name: str, col_in: str, col_out: str):
        dfp = d.get(col_in, pd.DataFrame())
        if not isinstance(dfp, pd.DataFrame) or dfp.empty:
            return pd.DataFrame(columns=["period", "group", col_out])
        dfp2 = dfp.copy()
        dfp2["group"] = group_name
        if col_out != dfp2.columns[-1]:
            val_cols = [c for c in dfp2.columns if c not in {"period", "group"}]
            if val_cols:
                dfp2 = dfp2.rename(columns={val_cols[0]: col_out})
        return dfp2[["period", "group", col_out]]

    def plot_group_hist(df_long: pd.DataFrame, title: str, key: str):
        if df_long.empty:
            st.caption(f"No data for: {title}")
            return
        df_long = df_long.copy()
        df_long["count"] = (
            pd.to_numeric(df_long["count"], errors="coerce").fillna(0).astype(int)
        )
        period_order = sorted(
            [p for p in df_long["period"].astype(str).tolist() if p.strip()]
        )
        try:
            base_theme = st.get_option("theme.base")
        except Exception:
            base_theme = "light"
        template = "plotly_dark" if str(base_theme).lower() == "dark" else "plotly_white"

        fig = px.bar(
            df_long,
            x="period",
            y="count",
            color="group",
            barmode="group",
            category_orders={"period": period_order, "group": [label_a, label_b]},
            color_discrete_map={label_a: "#2166ac", label_b: "#b2182b"},
            text="count",
            # title=title,
            template=template,
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            legend_title_text="",
        )
        fig.update_yaxes(tickformat=",", title_text="Count")
        fig.update_xaxes(title_text="Period")
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key=key,
        )

    # --- Metrics + charts: top-level Corpus vs Sample, then per-row metrics + matching chart ---
    # st.markdown("### Metrics")

    groups = run_data.get("groups", {}) or {}
    samples = run_data.get("samples", {}) or {}

    # slice_probs_md = run_data.get("slice_probs_md") or ""
    slice_probs_n = int(run_data.get("slice_probs_n", 0) or 0)

    g1_group = groups.get("group_a", {}) or {}
    g2_group = groups.get("group_b", {}) or {}
    g1_sample = samples.get("group_a", {}) or {}
    g2_sample = samples.get("group_b", {}) or {}

    g1_g_texts, g1_g_slices, g1_g_parsed = get_counts(g1_group)
    g2_g_texts, g2_g_slices, g2_g_parsed = get_counts(g2_group)
    g1_s_texts, g1_s_slices, g1_s_parsed = get_counts(g1_sample)
    g2_s_texts, g2_s_slices, g2_s_parsed = get_counts(g2_sample)

    top_corpus, top_sample = st.columns(2, gap="large")

    def render_metric_row(label: str, g1_val: int, g2_val: int, *, signed_delta: bool = True):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.metric(label_a, fmt_int(g1_val))
        with c2:
            st.metric(label_b, fmt_int(g2_val))
        with c3:
            delta_val = int(g1_val) - int(g2_val)
            st.metric('Δ (G1 - G2)', fmt_signed_int(delta_val) if signed_delta else fmt_int(delta_val))

    def build_section_period_longs(section_key: str):
        sec = run_data.get(section_key, {}) or {}
        g1 = sec.get("group_a", {}) or {}
        g2 = sec.get("group_b", {}) or {}
        s1 = g1.get("slice_stats", {}) or {}
        s2 = g2.get("slice_stats", {}) or {}

        df_texts_long = pd.concat(
            [
                to_period_df(g1, label_a, "df_period_texts", "count"),
                to_period_df(g2, label_b, "df_period_texts", "count"),
            ],
            ignore_index=True,
        )
        df_parsed_slices_long = pd.concat(
            [
                to_period_df(s1, label_a, "df_period_parsed_slices", "count"),
                to_period_df(s2, label_b, "df_period_parsed_slices", "count"),
            ],
            ignore_index=True,
        )
        df_total_slices_long = pd.concat(
            [
                to_period_df(s1, label_a, "df_period_total_slices", "count"),
                to_period_df(s2, label_b, "df_period_total_slices", "count"),
            ],
            ignore_index=True,
        )
        return df_texts_long, df_parsed_slices_long, df_total_slices_long

    corpus_texts_long, corpus_parsed_long, corpus_all_long = build_section_period_longs("groups")
    sample_texts_long, sample_parsed_long, sample_all_long = build_section_period_longs("samples")



    with top_corpus:
        st.markdown("### Corpus")
        # h1, h2, h3 = st.columns(3, gap="large")
        # with h1:
        #     st.markdown("#### Group 1")
        # with h2:
        #     st.markdown("#### Group 2")
        # with h3:
        #     st.markdown("#### G1 - G2")

        st.markdown("#### Number of parsed passages (1Kw) in corpus")
        render_metric_row("Num Parsed Slices", g1_g_parsed, g2_g_parsed)
        plot_group_hist(corpus_parsed_long, "Parsed slices by period", key="corpus_parsed_slices_by_period")
        
        # st.markdown("#### Number of passages (1Kw) in corpus")
        # render_metric_row("Num Slices", g1_g_slices, g2_g_slices)
        # plot_group_hist(corpus_all_long, "All slices by period", key="corpus_all_slices_by_period")

        st.markdown("#### Number of texts in corpus")
        render_metric_row("Num Texts", g1_g_texts, g2_g_texts)
        plot_group_hist(corpus_texts_long, "Texts by period", key="corpus_texts_by_period")
        

    with top_sample:
        st.markdown("### Sample")
        # h1, h2, h3 = st.columns(3, gap="large")
        # with h1:
        #     st.markdown("#### Group 1")
        # with h2:
        #     st.markdown("#### Group 2")
        # with h3:
        #     st.markdown("#### G1 - G2")

        st.markdown("#### Number of parsed passages (1Kw) in sample")
        render_metric_row("Num Parsed Slices", g1_s_parsed, g2_s_parsed)
        plot_group_hist(sample_parsed_long, "Parsed slices by period", key="sample_parsed_slices_by_period")

        # st.markdown("#### Number of passages (1Kw) in sample")
        # render_metric_row("Num Slices", g1_s_slices, g2_s_slices)
        # plot_group_hist(sample_all_long, "All slices by period", key="sample_all_slices_by_period")
    
        st.markdown("#### Number of texts in sample")
        render_metric_row("Num Texts", g1_s_texts, g2_s_texts)
        plot_group_hist(sample_texts_long, "Texts by period", key="sample_texts_by_period")


