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
from collections import Counter
import plotly.express as px
from utils import *

st.set_page_config(page_title="Classification", layout="wide")

st.title("Classification")
st.caption("Select two groups to train a classifier (UI only for now).")


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


def _get_options(df, col):
    if col not in df.columns:
        return []
    opts = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip()])
    return opts


def _vals_to_query_in(col: str, vals):
    """
    Build a self-contained pandas DataFrame.query() clause like:
      discipline in ['Philosophy', 'Literature']

    Important: get_balanced_cv_data does df_meta.query(query_str) directly, so the
    query must NOT rely on @variables (no local_dict injection).
    """
    if not vals:
        return None
    # Use repr() to safely quote strings (handles apostrophes, etc.)
    vals_lit = "[" + ", ".join(repr(v) for v in vals) + "]"
    return f"{col} in {vals_lit}"


def build_metadata_query(discipline=None, period=None, journal=None):
    clauses = []
    for col, vals in [
        ("discipline", discipline or []),
        ("period", period or []),
        ("journal", journal or []),
    ]:
        clause = _vals_to_query_in(col, vals)
        if clause:
            clauses.append(clause)
    return " and ".join(sorted(clauses)) if clauses else ""


def group_selector(df_meta: pd.DataFrame, title: str, key_prefix: str):
    st.subheader(title)

    discipline_opts = _get_options(df_meta, "discipline")
    period_opts = _get_options(df_meta, "period")
    journal_opts = _get_options(df_meta, "journal")

    col1,col2,col3 = st.columns([3,3,4])

    with col1:
        sel_discipline = st.multiselect(
            "Discipline",
            options=discipline_opts,
            default=["Philosophy" if key_prefix == "grp_a" else "Literature"],
            key=f"{key_prefix}_discipline",
        )
    with col2:
        sel_period = st.multiselect(
            "Period",
            options=period_opts,
            default=[],
            key=f"{key_prefix}_period",
        )

    with col3:
        sel_journal = st.multiselect(
            "Journal",
            options=journal_opts,
            default=[],
            key=f"{key_prefix}_journal",
        )
    query_struct = {
        "discipline": sel_discipline,
        "period": sel_period,
        "journal": sel_journal,
    }
    query_str = build_metadata_query(
        discipline=sel_discipline, period=sel_period, journal=sel_journal
    )
    return query_struct, query_str


df_meta = load_metadata()

left, mid, right = st.columns([5,1,5], vertical_alignment="center", gap="small")

with left:
    query_a_struct, query_a_str = group_selector(df_meta, "Group 1", "grp_a")
    if query_a_str:
        st.code(query_a_str)

with right:
    query_b_struct, query_b_str = group_selector(df_meta, "Group 2", "grp_b")
    if query_b_str:
        st.code(query_b_str)



def run_query(df: pd.DataFrame, query_str: str):
    q = query_str if query_str else "True"
    return df.query(q)


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
def load_slice_sample(query_a: str, query_b: str):
    """
    Uses osp.slices.get_balanced_slice_sample() to define a balanced sample.
    Returns a slice-level DataFrame with metadata + [text_id, slice_id, _target].
    """
    q1 = query_a if query_a else "True"
    q2 = query_b if query_b else "True"
    groups_train = [("Group 1", q1), ("Group 2", q2)]
    return get_balanced_slice_sample(groups_train, verbose=False)





# col1, col2, col3 = st.columns([5, 1, 5])
with mid:
    submit = st.button("Compare", type="primary")



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
    try:
        df_slice_sample = load_slice_sample(query_a=query_a_str, query_b=query_b_str)
    except Exception as e:
        st.error(f"Sampler error (get_balanced_slice_sample): {e}")
        df_slice_sample = pd.DataFrame()

    df_slice_g1 = (
        df_slice_sample.query('_target=="Group 1"').copy()
        if not df_slice_sample.empty and "_target" in df_slice_sample.columns
        else pd.DataFrame()
    )
    df_slice_g2 = (
        df_slice_sample.query('_target=="Group 2"').copy()
        if not df_slice_sample.empty and "_target" in df_slice_sample.columns
        else pd.DataFrame()
    )

    parsed_set = set(load_parsed_slice_ids())

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
    }


run_data = st.session_state.get("classification_last_run")
if run_data:
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
            category_orders={"period": period_order, "group": ["Group 1", "Group 2"]},
            color_discrete_map={"Group 1": "#2166ac", "Group 2": "#b2182b"},
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
            st.metric(f'Group 1', fmt_int(g1_val))
        with c2:
            st.metric(f'Group 2', fmt_int(g2_val))
        with c3:
            delta_val = int(g1_val) - int(g2_val)
            st.metric(f'Δ (G1 - G2)', fmt_signed_int(delta_val) if signed_delta else fmt_int(delta_val))

    def build_section_period_longs(section_key: str):
        sec = run_data.get(section_key, {}) or {}
        g1 = sec.get("group_a", {}) or {}
        g2 = sec.get("group_b", {}) or {}
        s1 = g1.get("slice_stats", {}) or {}
        s2 = g2.get("slice_stats", {}) or {}

        df_texts_long = pd.concat(
            [
                to_period_df(g1, "Group 1", "df_period_texts", "count"),
                to_period_df(g2, "Group 2", "df_period_texts", "count"),
            ],
            ignore_index=True,
        )
        df_parsed_slices_long = pd.concat(
            [
                to_period_df(s1, "Group 1", "df_period_parsed_slices", "count"),
                to_period_df(s2, "Group 2", "df_period_parsed_slices", "count"),
            ],
            ignore_index=True,
        )
        df_total_slices_long = pd.concat(
            [
                to_period_df(s1, "Group 1", "df_period_total_slices", "count"),
                to_period_df(s2, "Group 2", "df_period_total_slices", "count"),
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

        st.markdown("#### Number of texts in corpus")
        render_metric_row("Num Texts", g1_g_texts, g2_g_texts)
        plot_group_hist(corpus_texts_long, "Texts by period", key="corpus_texts_by_period")

        st.markdown("#### Number of passages (1K words) in corpus")
        render_metric_row("Num Slices", g1_g_slices, g2_g_slices)
        plot_group_hist(corpus_all_long, "All slices by period", key="corpus_all_slices_by_period")

        st.markdown("#### Number of parsed passages in corpus")
        render_metric_row("Num Parsed Slices", g1_g_parsed, g2_g_parsed)
        plot_group_hist(corpus_parsed_long, "Parsed slices by period", key="corpus_parsed_slices_by_period")

    with top_sample:
        st.markdown("### Sample")
        # h1, h2, h3 = st.columns(3, gap="large")
        # with h1:
        #     st.markdown("#### Group 1")
        # with h2:
        #     st.markdown("#### Group 2")
        # with h3:
        #     st.markdown("#### G1 - G2")

        st.markdown("#### Number of texts in sample")
        render_metric_row("Num Texts", g1_s_texts, g2_s_texts)
        plot_group_hist(sample_texts_long, "Texts by period", key="sample_texts_by_period")

        st.markdown("#### Number of passages (1K words) in sample")
        render_metric_row("Num Slices", g1_s_slices, g2_s_slices)
        plot_group_hist(sample_all_long, "All slices by period", key="sample_all_slices_by_period")

        st.markdown("#### Number of parsed passages (1K words) in sample")
        render_metric_row("Num Parsed Slices", g1_s_parsed, g2_s_parsed)
        plot_group_hist(sample_parsed_long, "Parsed slices by period", key="sample_parsed_slices_by_period")


