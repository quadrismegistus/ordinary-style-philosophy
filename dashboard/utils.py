import sys, os
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import contextlib
import io
from datetime import datetime

# Setup paths to import 'osp'
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)

from osp import *

# --- State Management Helpers ---

def get_current_comparison_name():
    """Get the currently selected comparison name from session state or HashStash."""
    if "current_comparison_name" in st.session_state:
        return st.session_state["current_comparison_name"]
    val = STASH_DASHBOARD_STATE.get("osp_last_comparison_name")
    if val:
        st.session_state["current_comparison_name"] = val
    return val

def set_current_comparison(name, comp_dict=None):
    """Set the current comparison in session state and HashStash."""
    changed = False
    if st.session_state.get("current_comparison_name") != name:
        st.session_state["current_comparison_name"] = name
        STASH_DASHBOARD_STATE["osp_last_comparison_name"] = name
        changed = True
        
    if comp_dict and st.session_state.get("current_comparison_data") != comp_dict:
        st.session_state["current_comparison_data"] = comp_dict
        STASH_DASHBOARD_STATE["osp_last_comparison"] = comp_dict
        changed = True
    return changed

def get_current_group_a():
    if "current_group_a" in st.session_state:
        return st.session_state["current_group_a"]
    val = STASH_DASHBOARD_STATE.get("osp_last_group_a")
    if val:
        st.session_state["current_group_a"] = val
    return val

def set_current_group_a(name):
    if st.session_state.get("current_group_a") == name:
        return False
    st.session_state["current_group_a"] = name
    STASH_DASHBOARD_STATE["osp_last_group_a"] = name
    return True

def get_current_group_b():
    if "current_group_b" in st.session_state:
        return st.session_state["current_group_b"]
    val = STASH_DASHBOARD_STATE.get("osp_last_group_b")
    if val:
        st.session_state["current_group_b"] = val
    return val

def set_current_group_b(name):
    if st.session_state.get("current_group_b") == name:
        return False
    st.session_state["current_group_b"] = name
    STASH_DASHBOARD_STATE["osp_last_group_b"] = name
    return True

# --- Caching Helpers ---

@st.cache_data
def get_cached_all_feats(normalize=True):
    """Cached version of get_all_feats to avoid slow disk I/O on every run."""
    return get_all_feats(normalize=normalize)

@cache
def load_comparison_stats(groups_train):
    """
    Cache the heavy feature aggregation for a given comparison.
    Uses an in-process LRU cache to avoid Streamlit element replay issues.
    groups_train is a tuple of ((name_a, query_a), (name_b, query_b)).
    """
    return get_balanced_slice_sample_feats(list(groups_train), with_diff_rows=True).reset_index()

# --- Existing Constants ---
DEFAULT_SLICE_LEN = 1000

featcols = [
    "feat_desc",
    "mean_Philosophy",
    "mean_Literature",
    "score_mean3",
    "score_mean_diff_3-1",
    "score_mean_diff_3-2",
]


# ============================================================================
# BARE-BONES LOG WINDOW
# ============================================================================


def _make_log_html(log_lines, height=50,  width=256):
    lines_html = "".join(
        [f'<div style="margin:0;padding:0;">{">" if not line.startswith("(") else " "} {line.strip()}</div>' for i, line in enumerate(log_lines)]
    )
    return f"""
    <div style="position: fixed; bottom: 0; left: 0; width: {width}px; height: {height}px; 
                        z-index: 999999; overflow: scroll; scrollbar-width: none; background: transparent; max-height: {height}px; max-width: {width}px;
                        margin: 0; padding: 5px; display: flex; flex-direction: column-reverse;
                        box-sizing: border-box;">
                <div style="font-family: monospace; font-size: 10px; white-space: pre-wrap; 
                            color: #666; margin: 0; padding: 0; border: none; background: transparent;
                            display: flex; flex-direction: column;">
                    {lines_html}
                </div>
            </div>
    """


class StatusWindow:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatusWindow, cls).__new__(cls)
            cls._instance.placeholder = None
        return cls._instance

    def __init__(self):
        if "logs" not in st.session_state:
            st.session_state.logs = []

    def write(self, msg):
        """Append a message to the logs and update session state."""
        st.session_state.logs.append(str(msg))
        if len(st.session_state.logs) > 500:
            st.session_state.logs = st.session_state.logs[-500:]

        # If we have a placeholder, update it immediately
        if self.placeholder:
            log_lines = (
                st.session_state.logs if st.session_state.logs else ["No logs yet..."]
            )
            html_log = _make_log_html(log_lines)
            self.placeholder.markdown(html_log, unsafe_allow_html=True)
    @contextlib.contextmanager
    def capture(self, label=None):
        """Redirect stdout to this log window."""
        import time
        
        start_time = time.time()
        if label:
            self.write(label)

        class LogStream:
            def __init__(self, window, original_stdout):
                self.window = window
                self.original_stdout = original_stdout

            def write(self, text):
                self.original_stdout.write(text)
                if text.strip():
                    self.window.write(text.strip())

            def flush(self):
                self.original_stdout.flush()

        old_stdout = sys.stdout
        sys.stdout = LogStream(self, old_stdout)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            elapsed = time.time() - start_time
            if label and elapsed > 1:
                self.write(f"({elapsed:.2f}s)")

    def render(self):
        """Display the log window at the top left of the screen."""
        self.placeholder = st.empty()
        log_lines = (
            st.session_state.logs if st.session_state.logs else ["No logs yet..."]
        )
        html_log = _make_log_html(log_lines)
        # self.placeholder.markdown(html_log, unsafe_allow_html=True)


def get_status_window():
    return StatusWindow()


def render_status_window():
    get_status_window().render()


# Utility Functions
def split_into_slices(text, slice_len=DEFAULT_SLICE_LEN):
    """Splits text into chunks of roughly slice_len recognized words."""
    tokens = tokenize_agnostic(text)
    slices = []
    current_slice = []
    word_count = 0

    for token in tokens:
        current_slice.append(token)
        if token.strip().isalpha():
            word_count += 1

        if word_count >= slice_len:
            slices.append("".join(current_slice))
            current_slice = []
            word_count = 0

    if current_slice:
        slices.append("".join(current_slice))

    return slices


# Cache the NLP processing to avoid redundant slow computations
@st.cache_data
def process_text(text):
    return get_nlp_doc(text)


@st.cache_data
def load_preds_feats():
    return get_preds_feats()


def load_feats():
    return load_preds_feats()[1]


# Cache feature weights loading
@st.cache_data
def load_weights():
    return get_current_feat_weights()


@st.cache_data
def load_slice_ids(query):
    return get_slice_ids(query)


@st.cache_data
def load_slice_feat_examples(slice_ids, feats, num_egs=25, max_slices=10_000):
    slice_ids_l = list(slice_ids) if slice_ids is not None else None
    feats_l = list(feats) if feats is not None else None
    slice_ids_l.sort()
    feats_l.sort()
    with log_progress('loading slice feat examples'):
        odf = get_slice_feat_egs(
            slice_ids=slice_ids_l,
            feats=feats_l,
            num_egs=num_egs,
            max_slices=max_slices,
        )
    return odf


# @st.cache_data
# def load_new_preds_feats(_text_input, cache_key=None):
#     """
#     _text_input can be a string (custom text) or a stanza Document.
#     cache_key is used for Streamlit hashing since Document is unhashable.
#     """
#     doc = process_text(_text_input) if isinstance(_text_input, str) else _text_input

#     # Check if cache_key looks like a slice_id (text_id__slice_num)
#     slice_id = cache_key if isinstance(cache_key, str) and "__" in cache_key else None
#     df_new_preds, df_new_feats = get_new_preds_feats(doc, slice_id=slice_id)

#     # Keep track of non-numeric columns we need
#     needed_cols = ["feature", "feat_type", "feat_name"]
#     # Ensure columns exist before groupby
#     for col in needed_cols:
#         if col not in df_new_feats.columns:
#             if col == 'feat_name':
#                 df_new_feats['feat_name'] = [str(x).split('_', 1)[-1] if '_' in str(x) else str(x) for x in df_new_feats['feature']]
#             elif col == 'feat_type':
#                 df_new_feats['feat_type'] = [str(x).split('_')[0] for x in df_new_feats['feature']]

#     df_new_feats_grouped = (
#         df_new_feats.groupby(needed_cols).mean(numeric_only=True).reset_index()
#     )

#     # Sort and filter for display
#     df_new_feats_display = df_new_feats_grouped.sort_values(
#         "score_mean_diff_3-1", ascending=False
#     )
#     df_new_feats_display = df_new_feats_display[
#         needed_cols + [c for c in featcols if c in df_new_feats_display.columns]
#     ]
#     df_new_feats_display["feat_desc"] = df_new_feats_display["feature"].map(
#         lambda x: FEAT2DESC.get(x, "")
#     )

#     if "comparison" not in df_new_preds.columns:
#         # Fallback if comparison is missing (e.g. from old stashed data)
#         # We can try to get it from the models or just use a placeholder
#         df_new_preds["comparison"] = "Unknown Comparison"

#     df_new_preds = (
#         df_new_preds.groupby("comparison")
#         .mean(numeric_only=True)
#         .sort_values("prob_Philosophy", ascending=False)
#     )
#     if "run" in df_new_preds.columns:
#         df_new_preds = df_new_preds.drop(columns=["run"])

#     return df_new_preds, df_new_feats_display, df_new_feats_grouped


def plot_predictive_features(df_new_feats):
    figld = []
    for i, row in df_new_feats.iterrows():
        # Scale specific features as in notebook
        s1, s2, s3 = row["mean_Philosophy"], row["mean_Literature"], row["score_mean3"]
        if row["feat_name"].startswith("num_words_in") or row["feat_name"] == "height":
            s1, s2, s3 = s1 / 10, s2 / 10, s3 / 10

        for grp in [1, 2]:
            d = {
                "feature": row["feature"],
                "feat_type": row["feat_type"],
                "feat_name": row["feat_name"],
                "target": ("Philosophy" if grp == 1 else "Literature"),
                "score_training": s1 if grp == 1 else s2,
                "score_new": s3,
            }
            if d["score_training"] > 1 and d["score_new"] > 1:
                figld.append(d)

    if not figld:
        st.warning("No features found with scores > 1 for plotting.")
        return

    figdf = pd.DataFrame(figld)
    figdf["odds_ratio"] = figdf["score_new"] / figdf["score_training"]
    figdf["odds_ratio_log"] = np.log(figdf["odds_ratio"])
    figdf["odds_ratio_log_abs"] = figdf["odds_ratio_log"].abs()

    def format_ratio(ratio):
        if ratio < 1:
            return f"-{1/ratio:.1f}x"
        return f"{ratio:.1f}x"

    figdf["ratio_label"] = figdf["odds_ratio"].apply(format_ratio)
    figdf["clean_feat_name"] = (
        figdf["feat_name"].str.replace("num_", "# ").str.replace("_", " ")
    )

    # Create charts for each target
    for target in ["Philosophy", "Literature"]:
        df_target = figdf[figdf["target"] == target].copy()
        if df_target.empty:
            continue

        # Create Altair Selection for highlighting
        selection = alt.selection_point(fields=["feature"], on="click")

        base = alt.Chart(df_target).encode(
            x=alt.X(
                "score_new:Q",
                scale=alt.Scale(type="log", domain=[1, 250]),
                title="This Text",
            ),
            y=alt.Y(
                "score_training:Q",
                scale=alt.Scale(type="log", domain=[1, 250]),
                title=f"Other {target} Texts",
            ),
            tooltip=[
                alt.Tooltip("clean_feat_name:N", title="Feature"),
                alt.Tooltip("score_training:Q", format=".2f", title="Other Text Score"),
                alt.Tooltip("score_new:Q", format=".2f", title="This Text Score"),
                alt.Tooltip("ratio_label:N", title="Ratio"),
            ],
        )

        points = (
            base.mark_point(filled=True)
            .encode(
                color=alt.condition(
                    selection,
                    alt.Color(
                        "odds_ratio_log:Q",
                        scale=alt.Scale(scheme="redblue", domainMid=0),
                        title="Log Odds Ratio",
                    ),
                    alt.value("lightgray"),
                ),
                size=alt.Size(
                    "odds_ratio_log_abs:Q",
                    scale=alt.Scale(range=[50, 400]),
                    title="Abs Log Odds",
                ),
                opacity=alt.condition(selection, alt.value(0.6), alt.value(0.1)),
            )
            .add_params(selection)
            .interactive()
        )

        text = (
            base.mark_text(align="left", baseline="middle", dx=7, fontSize=13)
            .transform_calculate(
                # Jitter the y-position for labels to reduce overlap
                jittered_y="datum.score_training * pow(1.2, random() - 0.5)"
            )
            .encode(
                y=alt.Y("jittered_y:Q", scale=alt.Scale(type="log")),
                text="clean_feat_name:N",
                opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1)),
            )
        )

        # Diagonal line
        line_val = [1, 250]
        line_df = pd.DataFrame({"x": line_val, "y": line_val})
        line = (
            alt.Chart(line_df)
            .mark_line(color="gray", strokeDash=[5, 5], opacity=0.5)
            .encode(x="x:Q", y="y:Q")
        )

        layered = alt.layer(line, points, text).properties(
            height=600, title=f"Predictive Features: {target}"
        )

        st.altair_chart(layered, width='stretch')


def display_slice_predictions(
    doc, color_column, word_feat_type, view_mode="Annotated", cache_key=None
):
    df_slice_preds, df_slice_feats_display, df_slice_feats_grouped = (
        load_new_preds_feats(doc, cache_key=cache_key)
    )
    st.markdown("##### Predictions for Slice")
    df_plot_preds = df_slice_preds.reset_index().melt(
        id_vars="comparison", var_name="Class", value_name="Probability"
    )
    df_plot_preds = df_plot_preds[df_plot_preds["Class"].str.contains("Philosophy")]
    df_plot_preds["Probability"] = df_plot_preds["Probability"].astype(float)

    pred_chart = (
        alt.Chart(df_plot_preds)
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("comparison:N", title="Comparison"),
            color="Class:N",
            tooltip=["comparison", "Class", alt.Tooltip("Probability", format=".2%")],
        )
        .properties(height=200)
    )
    st.altair_chart(pred_chart, width='stretch')


def display_slice_analysis(
    doc,
    color_column,
    word_feat_type,
    view_mode="Annotated",
    cache_key=None,
    sort_col='Sent Num',
    ascending=True,
):
    """Reusable component for displaying prediction chart, annotated passage, and feature plot for a slice."""
    # st.markdown("##### Annotated Passage")
    # display_doc_annotated(
    #     doc,
    #     color=color_column,
    #     word_feat_type=word_feat_type,
    #     key_prefix=f"slice_{cache_key or 'main'}",
    # )

    # Sentence-level feature table (includes HTML-colored sentences)
    # try:
    # st.markdown("##### Sentence Metrics")
    # df_sent_feats = get_sents_feats_df(
    #     doc, per_n_words=None, html=True, with_weights=True
    # ).round(3)
    # Allow simple sorting since st.dataframe won't render HTML.
    # sort_col = st.selectbox(
    #     "Sort sentence table by:",
    #     df_sent_feats.columns.tolist(),
    #     index=(
    #         df_sent_feats.columns.get_loc("Sent Num")
    #         if "Sent Num" in df_sent_feats.columns
    #         else 0
    #     ),
    # )
    # df_sorted = df_sent_feats.sort_values(sort_col, ascending=ascending)

    # html_table = get_doc_html2(doc, color=color_column, word_feat_type=word_feat_type)
    html_df = get_doc_html_table(doc, color_by=color_column, word_feat_type=word_feat_type)
    html_df = html_df.round(3).drop(columns=['sent']).reset_index()
    # html_df['sent_id'] = html_df['sent_id'].apply(
    #     lambda x: f'<a name="sent_{x}">{x}</a>'
    # )
    html_df = html_df.rename(columns={'sent_id':'Sent Num'})
    
    if cache_key:
        html_df['View'] = html_df['Sent Num'].apply(
            lambda x: f'<a href="./Sentence?slice_id={cache_key}&sent_id={x}" target="_blank">🔍</a>'
        )

    # Move Sent Num and View to front
    cols = html_df.columns.tolist()
    if 'View' in cols:
        cols.insert(0, cols.pop(cols.index('View')))
    cols.insert(0, cols.pop(cols.index('Sent Num')))
    html_df = html_df[cols]

    feats = [x for x in html_df.columns if x not in ['html', 'View', 'Sent Num']]
    
    sort_options = []
    for f in feats:
        sort_options.append(f"{f} ↑")
        sort_options.append(f"{f} ↓")
    
    current_label = f"{sort_col} {'↑' if ascending else '↓'}"
    default_index = sort_options.index(current_label) if current_label in sort_options else 0
    
    formcol1,formcol2 = st.columns([10,10],vertical_alignment="top")
    with formcol1:
        st.markdown("#### Passage breakdown by sentence, clause, and feature")
    with formcol2:
        selected_sort = st.selectbox(
            # "Sort sentence table by:",
            "Sort sentence table by:",
            sort_options,
            index=default_index,
        )
        sort_col = selected_sort[:-2]
        ascending = selected_sort.endswith("↑")

    html_df = html_df.sort_values(sort_col, ascending=ascending)
    html_str = html_df.rename(columns={'html':'Sentence'}).to_html(escape=False, border=0, index=False)
    # Add CSS to show only top border on table cells (not headers)
    html_str = f"""
    <style>
        table td {{
            padding: 8px;
            margin: 0;
        }}
        table th {{
            padding: 8px;
            margin: 0;
        }}
        table td, table th {{
            border-spacing: 0;
            border-collapse: collapse;
        }}
        table {{
            margin: 0;
            padding: 0;
            border-spacing: 0;
            border-collapse: collapse;
            border-radius: 10px;
            background: white;
            border: 0px;
            font-family: "Source Sans", sans-serif;
        }}
        table, tr, th, td, .dataframe, .dataframe th, .dataframe td {{
            font-family: "Source Sans", sans-serif !important;
            text-align: left;
            border: 1px solid #ddd;
        }}
    </style>
    {html_str}
    """
    st.components.v1.html(html_str, height=500, scrolling=True)

    # # Render HTML column; other columns show as text
    # html_table = df_sorted.to_html(escape=False)
    # # Force column widths (Sent column at least 300px)
    # colgroup = "<colgroup><col style='width:80px'><col style='width:300px'><col style='width:120px'></colgroup>"
    # html_table = html_table.replace(
    #     "<table", f"<table style='table-layout:fixed;width:100%;'"
    # )
    # html_table = html_table.replace(
    #     "<table style='table-layout:fixed;width:100%;'>",
    #     f"<table style='table-layout:fixed;width:100%;'>\n{colgroup}",
    #     1,
    # )
    # st.markdown(html_table, unsafe_allow_html=True)

    # # Clickable Sent Num to open SVG popup
    # st.markdown("###### Sentence Viewer")

    # # Define dialog for displaying sentence diagram
    # if hasattr(st, "dialog"):

    #     @st.dialog("Sentence Structure", width="large")
    #     def show_sent_dialog(sent, sent_num):
    #         st.caption(f"Sentence {sent_num}: {sent.text}")
    #         render_kwargs = dict(color_by=color_column)
    #         html_content = render_sent_displacy(
    #             sent, jupyter=False, **render_kwargs
    #         )
    #         st.components.v1.html(html_content, height=600, scrolling=True)

    # else:

    #     def show_sent_dialog(sent, sent_num):
    #         with st.expander(f"Sentence {sent_num} Diagram", expanded=True):
    #             render_kwargs = dict(color_by=color_column)
    #             html_content = render_sent_displacy(
    #                 sent, jupyter=False, **render_kwargs
    #             )
    #             st.components.v1.html(html_content, height=600, scrolling=True)

    # # Render buttons in sorted order
    # for sent_num in df_sorted.index:
    #     try:
    #         sent_idx = int(sent_num) - 1
    #         sent = doc.sentences[sent_idx]
    #     except Exception:
    #         continue
    #     if st.button(f"View {sent_num}", key=f"sent_svg_btn_{sent_num}"):
    #         show_sent_dialog(sent, sent_num)

    # except Exception as e:
        # st.warning(f"Unable to render sentence metrics: {e}")


def display_slice_feature_distribution(
    doc, color_column, word_feat_type, view_mode="Annotated", cache_key=None
):
    """Reusable component for displaying an annotated passage."""
    df_slice_preds, df_slice_feats_display, df_slice_feats_grouped = (
        load_new_preds_feats(doc, cache_key=cache_key)
    )
    st.dataframe(df_slice_feats_display)


def plot_weight_distribution(doc, color_column="weight_z", title=None):
    """
    Plots the distribution of feature weights for words/sentences in the doc.
    """
    if title is None:
        title = f"Distribution of {color_column}"

    df_slice_feats = get_slice_feats_by_word(doc, weight_cols=[color_column])

    if df_slice_feats.empty:
        st.warning("No feature data found for distribution plot.")
        return

    # Create density plot (histogram using lines)
    # Using density transform to get smooth lines, separated by feat_type

    # Filter for sentence features separately to drop duplicates
    df_sent = df_slice_feats[df_slice_feats["feat_type"] == "sent"].drop_duplicates(
        "sent_i"
    )
    df_others = df_slice_feats[df_slice_feats["feat_type"] != "sent"]
    df_plot = pd.concat([df_sent, df_others])

    chart = (
        alt.Chart(df_plot)
        .transform_density(
            density=color_column, as_=[color_column, "density"], groupby=["feat_type"]
        )
        .transform_joinaggregate(max_density="max(density)", groupby=["feat_type"])
        .transform_calculate(scaled_density="datum.density / datum.max_density")
        .mark_line()
        .encode(
            x=alt.X(
                f"{color_column}:Q",
                title="Feature Weight",
                scale=alt.Scale(domain=[-2, 2]),
            ),
            y=alt.Y("scaled_density:Q", title="Relative Density"),
            color=alt.Color("feat_type:N", title="Feature Type"),
            tooltip=[
                alt.Tooltip(f"{color_column}:Q", format=".4f"),
                alt.Tooltip("scaled_density:Q", format=".2f", title="Rel. Density"),
                "feat_type",
            ],
        )
        .properties(title=title, height=200)
        .interactive()
    )

    st.altair_chart(chart, width='stretch')


def setup_sidebar():
    # Get feature weights to populate options
    try:
        df_weights = load_weights()
        color_options = [
            c for c in df_weights.columns if df_weights[c].dtype in ["float64", "int64"]
        ]
    except Exception as e:
        st.error(f"Error loading feature weights: {e}")
        color_options = ["score_z_diff"]

    with st.sidebar:
        st.header("Settings")
        view_mode = st.radio(
            "View mode:", options=["Classic", "Annotated"], index=1, horizontal=True
        )
        word_feat_type = st.selectbox(
            "Color words by:", options=["deprel", "pos"], index=0
        )

        # Determine default weight column
        default_color = (
            "weight_z"
            if "weight_z" in color_options
            else ("weight" if "weight" in color_options else 0)
        )
        color_idx = (
            color_options.index(default_color) if isinstance(default_color, str) else 0
        )

        color_column = st.selectbox(
            "Weight column:", options=color_options, index=color_idx
        )

        st.divider()
        st.info("Blue: Positive weight | Orange: Negative weight")

    return word_feat_type, color_column, view_mode


newtext = """By a `denoting phrase' I mean a phrase such as any one of the following: a man, some man, any man, every man, all men, 
    the present King of England, the presenting King of France, the center of mass of the solar system at the first instant of the 
    twentieth century, the revolution of the earth round the sun, the revolution of the sun round the earth. Thus a phrase is 
    denoting solely in virtue of its form. We may distinguish three cases: (1) A phrase may be denoting, and yet not denote 
    anything; e.g., `the present King of France'. (2) A phrase may denote one definite object; e.g., `the present King of England' 
    denotes a certain man. (3) A phrase may denote ambiguously; e.g. `a man' denotes not many men, but an ambiguous man. The 
    interpretation of such phrases is a matter of considerably difficulty; indeed, it is very hard to frame any theory not 
    susceptible of formal refutation. All the difficulties with which I am acquainted are met, so far as I can discover, by the 
    theory which I am about to explain."""


@contextlib.contextmanager
def log_progress(msg):
    """Context manager that combines status_window.capture and st.spinner.
    Prioritizes status_window - ensures it captures all output even if spinner fails."""
    status_window = get_status_window()
    with status_window.capture(msg):
        try:
            with st.spinner(msg):
                yield
        finally:
            pass


