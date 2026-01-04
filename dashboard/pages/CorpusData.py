import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="Corpus Data", layout="wide")


@st.cache_data
def load_metadata():
    df = get_corpus_metadata(min_year=0, max_year=3000)[DISPLAY_META_FIELDS].copy()
    df = df.fillna("").astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    return df


df_meta = load_metadata()

gb = GridOptionsBuilder.from_dataframe(df_meta)
gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

# # Precompute option lists for categorical filters
# _cat_cols = ["journal", "discipline", "period", "publisher"]
# for col in _cat_cols:
#     if col in df_meta.columns:
#         vals = sorted(v for v in df_meta[col].dropna().unique().tolist() if str(v).strip())
#         gb.configure_column(
#             col,
#             filter="agSetColumnFilter",
#             filterParams={"values": vals},
#         )
# link_renderer = JsCode(
#     """
#     function(params) {
#         if (!params.value) { return ''; }
#         const safe = String(params.value);
#         return `<a href="${safe}" target="_blank">${safe}</a>`;
#     }
#     """
# )
# # gb.configure_column("url", headerName="url", cellRenderer=link_renderer)
# # gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
# gb.configure_side_bar()
# if "year" in df_meta.columns:
#     gb.configure_column("year", sort="asc")
grid_options = gb.build()

# # Enable text filtering for string columns unless a specific filter is already set
# for col in df_meta.columns:
#     if df_meta[col].dtype == 'object':
#         grid_options["columnDefs"] = [
#             {**colDef, "filter": "agTextColumnFilter"}
#             if (colDef["field"] == col and colDef.get("filter") not in {"agSetColumnFilter"})
#             else colDef
#             for colDef in grid_options.get("columnDefs", [])
#         ]

# with table_tab:
# st.markdown("## Data")

st.title("Metadata")
st.caption("Metadata from the corpus.")

AgGrid(
    df_meta,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    fit_columns_on_grid_load=True,
    height=600,
    theme='streamlit',
    allow_unsafe_jscode=True,
    enable_enterprise_modules=True,
)