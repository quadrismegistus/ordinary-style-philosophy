import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="Corpus Info", layout="wide")


@st.cache_data
def load_metadata():
    df = get_corpus_metadata(min_year=0, max_year=3000)[DISPLAY_META_FIELDS].copy()
    df = df.fillna("").astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    return df


def _year_bounds(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None, None
    return int(s.min()), int(s.max())


def _year_label(min_year, max_year):
    if min_year is None or max_year is None:
        return "N/A"
    return f"{min_year} – {max_year}"


def _count_nonempty_unique(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.nunique()


df_meta = load_metadata()

n_texts = len(df_meta)
n_texts_fmt = f"{n_texts:,}"
min_year, max_year = _year_bounds(df_meta["year"]) if "year" in df_meta.columns else (None, None)
authors_count = _count_nonempty_unique(df_meta["author"]) if "author" in df_meta.columns else 0
n_disciplines = (
    int(df_meta["discipline"].nunique()) if "discipline" in df_meta.columns else 0
)
n_journals = (
    int(df_meta["journal"].nunique()) if "journal" in df_meta.columns else 0
)

st.title("Corpus")
st.caption("Overview of the corpus metadata.")

# summary_tab, table_tab = st.tabs(["Summary", "Table"])

# with summary_tab:
st.markdown("### Summary")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Year range", _year_label(min_year, max_year))
with mcol2:
    st.metric("Texts", n_texts_fmt)
with mcol3:
    st.metric("Authors", f"{authors_count:,}")
with mcol4:
    st.metric("Journals", f"{n_journals:,}")

# By-discipline metrics
if "discipline" in df_meta.columns:
    rows = []
    for disc, sub in df_meta.groupby("discipline"):
        disc = str(disc).strip()
        if not disc:
            continue
        d_min_year, d_max_year = _year_bounds(sub["year"]) if "year" in sub.columns else (None, None)
        d_authors = _count_nonempty_unique(sub["author"]) if "author" in sub.columns else 0
        d_journals = _count_nonempty_unique(sub["journal"]) if "journal" in sub.columns else 0
        rows.append(
            {
                "Discipline": disc,
                "Year range": _year_label(d_min_year, d_max_year),
                "# Texts": len(sub),
                "# Authors": d_authors,
                "# Journals": d_journals,
            }
        )
    if rows:
        rows = sorted(rows, key=lambda r: r["# Texts"], reverse=True)
        for r in rows:
            disc = r['Discipline']
            color = '#1f77b4' if disc == 'Philosophy' else '#d62728'
            st.markdown(f"### <font color='{color}'>{disc}</font>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Year range", r["Year range"])
            with c2:
                st.metric("Texts", f"{r['# Texts']:,}")
            with c3:
                st.metric("Authors", f"{r['# Authors']:,}")
            with c4:
                st.metric("Journals", f"{r['# Journals']:,}")
