import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *

import streamlit as st

st.set_page_config(page_title="Groups", layout="wide")


@st.cache_data
def load_metadata():
    df = get_corpus_metadata().copy()
    for c in [
        "discipline",
        "period",
        "journal",
        "decade",
        "century",
        "halfcentury",
        "century_discipline",
        "halfcentury_discipline",
        "period_discipline",
        "decade_discipline",
        "century_journal",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def _get_options(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    opts = sorted(
        [x for x in df[col].dropna().astype(str).unique().tolist() if str(x).strip()]
    )
    return opts


def _vals_to_query_in(col: str, vals):
    if not vals:
        return None
    vals_lit = "[" + ", ".join(repr(v) for v in vals) + "]"
    return f"{col} in {vals_lit}"


def _contains_clause(col: str, text: str):
    if not text:
        return None
    return f"{col}.str.contains({text!r}, case=False, na=False)"


def _range_clause(col: str, vmin, vmax):
    if vmin is None or vmax is None:
        return None
    return f"({col} >= {int(vmin)} and {col} <= {int(vmax)})"


def build_metadata_query(
    *,
    discipline=None,
    period=None,
    decade=None,
    journal=None,
    year_range=None,
    title_contains=None,
    author_contains=None,
    custom_query=None,
):
    clauses = []
    for col, vals in [
        ("discipline", discipline or []),
        ("period", period or []),
        ("decade", decade or []),
        ("journal", journal or []),
    ]:
        clause = _vals_to_query_in(col, vals)
        if clause:
            clauses.append(clause)

    if year_range and len(year_range) == 2:
        yr_clause = _range_clause("year", year_range[0], year_range[1])
        if yr_clause:
            clauses.append(yr_clause)

    for col, text_val in [
        ("title", title_contains),
        ("author", author_contains),
    ]:
        clause = _contains_clause(col, text_val)
        if clause:
            clauses.append(clause)

    if custom_query and str(custom_query).strip():
        clauses.append(f"({str(custom_query).strip()})")

    return " and ".join(clauses) if clauses else ""


def run_query(df: pd.DataFrame, query_str: str):
    q = query_str if query_str else "1==1"
    return df.query(q)


def _load_saved_groups():
    try:
        return {k: STASH_DASHBOARD_GROUPS[k] for k in STASH_DASHBOARD_GROUPS.keys()}
    except Exception:
        return {}


def _save_group(name: str, payload: dict):
    STASH_DASHBOARD_GROUPS[name] = payload


df_meta = load_metadata()

st.title("Groups")
st.caption("Create and manage reusable groups from corpus metadata.")

create_tab, saved_tab = st.tabs(["Create group", "Saved groups"])

with create_tab:
    group_name = st.text_input("Group name", value="New Group")

    discipline_opts = _get_options(df_meta, "discipline")
    period_opts = _get_options(df_meta, "period")
    decade_opts = _get_options(df_meta, "decade")
    journal_opts = _get_options(df_meta, "journal")
    min_year = int(df_meta["year"].min()) if "year" in df_meta.columns else None
    max_year = int(df_meta["year"].max()) if "year" in df_meta.columns else None

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        sel_discipline = st.multiselect(
            "Discipline",
            options=discipline_opts,
            default=[],
            help="Filter by discipline values in metadata.",
        )
        sel_period = st.multiselect(
            "Period",
            options=period_opts,
            default=[],
            help="Period buckets built from year.",
        )
    with fcol2:
        sel_decade = st.multiselect(
            "Decade",
            options=decade_opts,
            default=[],
            help="Decade buckets (e.g., 1950).",
        )
        sel_journal = st.multiselect(
            "Journal",
            options=journal_opts,
            default=[],
        )
    with fcol3:
        filter_by_year = st.checkbox(
            "Filter by year range",
            value=False,
            help="Toggle to constrain by publication year.",
        )
        if filter_by_year and min_year is not None and max_year is not None:
            sel_year_range = st.slider(
                "Year range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
            )
        else:
            sel_year_range = None
        title_contains = st.text_input(
            "Title contains",
            value="",
            help="Case-insensitive substring match on title.",
        )
        author_contains = st.text_input(
            "Author contains",
            value="",
            help="Case-insensitive substring match on author.",
        )

    custom_query = st.text_area(
        "Additional query (pandas DataFrame.query syntax)",
        value="",
        placeholder="publisher == 'Johns Hopkins University Press'",
        help="Optional advanced expression appended to the generated clauses.",
    )

    query_struct = {
        "discipline": sel_discipline,
        "period": sel_period,
        "decade": sel_decade,
        "journal": sel_journal,
        "year_range": sel_year_range,
        "title_contains": title_contains,
        "author_contains": author_contains,
        "custom_query": custom_query,
    }

    query_str = build_metadata_query(
        discipline=sel_discipline,
        period=sel_period,
        decade=sel_decade,
        journal=sel_journal,
        year_range=sel_year_range,
        title_contains=title_contains,
        author_contains=author_contains,
        custom_query=custom_query,
    )

    if query_str:
        st.markdown("#### Query")
        st.code(query_str, language="python")
        df_filtered = run_query(df_meta, query_str)
    else:
        df_filtered = df_meta
   
    # print(f'Filtering metadata with query: "{query_str}"')
    # df_filtered = run_query(df_meta, query_str)

    top_metrics, _ = st.columns([1, 1])
    with top_metrics:
        st.metric("Matching texts", f"{len(df_filtered):,}")

    st.markdown("#### Matching metadata")
    st.dataframe(
        df_filtered[DISPLAY_META_FIELDS],
        use_container_width=True,
        height=480,
    )
    if st.button("Save group", type="primary"):
        if not group_name.strip():
            st.error("Please provide a group name.")
        else:
            payload = {
                "name": group_name.strip(),
                "query_str": query_str or "1==1",
                "query_struct": query_struct,
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "n_texts": int(len(df_filtered)),
            }
            try:
                _save_group(group_name.strip(), payload)
                st.success(f"Saved group '{group_name.strip()}' to HashStash.")
            except Exception as e:
                st.error(f"Could not save group: {e}")


with saved_tab:
    saved_groups = _load_saved_groups()
    saved_names = sorted(saved_groups.keys())
    if not saved_names:
        st.info("No saved groups yet. Create one in the tab above.")
    else:
        df_saved = pd.DataFrame(saved_groups.values())
        if not df_saved.empty:
            df_saved = df_saved[["name", "n_texts", "saved_at", "query_str"]].rename(
                columns={
                    "name": "Name",
                    "n_texts": "Matching texts",
                    "saved_at": "Saved at",
                    "query_str": "Query",
                }
            )
        st.markdown("#### Saved groups")
        st.dataframe(
            df_saved,
            use_container_width=True,
            height=320,
        )

        selected_saved = st.selectbox(
            "Inspect saved group",
            options=[""] + saved_names,
            format_func=lambda x: x or "Select saved group",
        )
        if selected_saved:
            saved = saved_groups.get(selected_saved, {})
            st.info(
                f"Name: {saved.get('name')}\n\nQuery: {saved.get('query_str')}\n\nSaved at: {saved.get('saved_at')}"
            )