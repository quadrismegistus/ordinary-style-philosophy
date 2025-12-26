import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_PAGES = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_PAGES)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_DASHBOARD not in sys.path: sys.path.append(PATH_DASHBOARD)

import streamlit as st
import altair as alt
import pandas as pd
from utils import *

st.set_page_config(page_title="Predict Custom Input", layout="wide")

st.title("Predict Custom Input")

word_feat_type, color_column, view_mode = setup_sidebar()

@st.cache_data
def process_all_slices(text, color_column, word_feat_type):
    """Processes all slices and returns a list of (preds, feats, html) per slice."""
    slices = split_into_slices(text, slice_len=DEFAULT_SLICE_LEN)
    results = []
    progress_text = "Analyzing slices..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, slice_txt in enumerate(slices):
        doc = get_nlp_doc(slice_txt)
        df_preds, df_feats_display, df_feats_grouped = load_new_preds_feats(slice_txt, cache_key=slice_txt)
        html_output = get_doc_html(doc, color=color_column, word_feat_type=word_feat_type)
        results.append({
            'index': i,
            'text': slice_txt,
            'preds': df_preds,
            'feats_display': df_feats_display,
            'feats_grouped': df_feats_grouped,
            'html': html_output,
            'doc': doc # Store doc for streamlit annotation if needed
        })
        my_bar.progress((i + 1) / len(slices), text=f"Analyzed {i+1}/{len(slices)} slices")
    my_bar.empty()
    return results

# Main layout: Two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    text_input = st.text_area("Paste text here to analyze:", height=300, placeholder="Type or paste text here...", value=newtext)

if text_input:
    with st.spinner("Analyzing all slices..."):
        slice_results = process_all_slices(text_input, color_column, word_feat_type)
    
    # Global aggregates for the top level
    all_preds = pd.concat([r['preds'].assign(slice=r['index']) for r in slice_results])
    avg_preds = all_preds.groupby('comparison').mean(numeric_only=True).drop(columns=['slice'])
    
    with right_col:
        st.markdown("### Overall Predictions (Average)")
        df_plot_preds = avg_preds.reset_index().melt(id_vars='comparison', var_name='Class', value_name='Probability')
        df_plot_preds['Probability'] = df_plot_preds['Probability'].astype(float)
        
        num_comparisons = df_plot_preds['comparison'].nunique()
        chart_height = max(100, num_comparisons * 50)
        pred_chart = alt.Chart(df_plot_preds).mark_bar().encode(
            x=alt.X('Probability:Q'),
            y=alt.Y('comparison:N', title='Comparison'),
            color='Class:N',
            tooltip=['comparison', 'Class', alt.Tooltip('Probability', format='.2%')]
        ).properties(height=chart_height)
        st.altair_chart(pred_chart, use_container_width=True)

    # Navigation tabs below the input/summary row
    st.divider()
    tab1, tab2 = st.tabs(["Full Text Analysis", "Slice Explorer"])
    
    # ---------------- TAB 1: FULL TEXT ANALYSIS ----------------
    with tab1:
        t1_col1, t1_col2 = st.columns([1, 1])
        
        with t1_col1:
            st.markdown("### Full Text Analysis")
            total_words = sum([len(get_recog_words(r['text'])) for r in slice_results])
            st.metric("Total Recognized Words", total_words)
            st.info(f"The text has been analyzed in {len(slice_results)} slices of ~{DEFAULT_SLICE_LEN} words each.")
            
            st.markdown("### Annotated Passage")
            if view_mode == "Annotated":
                for r in slice_results:
                    display_doc_annotated(r['doc'], color=color_column, word_feat_type=word_feat_type)
            else:
                full_html = "".join([r['html'] for r in slice_results])
                st.markdown(full_html, unsafe_allow_html=True)
            
            # Add download button for results
            all_feats_display = pd.concat([r['feats_display'].assign(slice=r['index']) for r in slice_results])
            csv = all_feats_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed Features as CSV",
                data=csv,
                file_name='osp_analysis_results.csv',
                mime='text/csv',
            )

        with t1_col2:
            st.markdown("### Global Features")
            all_feats_grouped = pd.concat([r['feats_grouped'].assign(slice=r['index']) for r in slice_results])
            avg_feats_grouped = all_feats_grouped.groupby(["feature", "feat_type", "feat_name"]).mean(numeric_only=True).reset_index()
            plot_predictive_features(avg_feats_grouped)

    # ---------------- TAB 2: SLICE EXPLORER ----------------
    with tab2:
        st.markdown("### Probability Timeline")
        
        timeline_data = []
        for r in slice_results:
            comp_name = 'Philosophy vs Literature'
            if comp_name in r['preds'].index:
                p_phi = r['preds'].loc[comp_name, 'prob_Philosophy']
                timeline_data.append({'Slice': r['index'], 'Prob. Philosophy': p_phi})
            else:
                first_comp = r['preds'].index[0]
                prob_cols = [c for c in r['preds'].columns if c.startswith('prob_')]
                if prob_cols:
                    p_phi = r['preds'].loc[first_comp, prob_cols[0]]
                    timeline_data.append({'Slice': r['index'], 'Prob. Philosophy': p_phi})
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            timeline_chart = alt.Chart(df_timeline).mark_line(point=True).encode(
                x=alt.X('Slice:O'),
                y=alt.Y('Prob. Philosophy:Q', scale=alt.Scale(domain=[0, 1])),
                tooltip=['Slice', alt.Tooltip('Prob. Philosophy:Q', format='.2%')]
            ).properties(height=300).interactive()
            st.altair_chart(timeline_chart, use_container_width=True)
        
        st.divider()
        
        selected_slice_idx = st.selectbox("Select a slice to inspect:", 
                                         options=range(len(slice_results)),
                                         format_func=lambda i: f"Slice {i}: {slice_results[i]['text'][:100]}...")
        
        res = slice_results[selected_slice_idx]
        display_slice_analysis(res['doc'], color_column, word_feat_type, view_mode=view_mode, cache_key=res['text'])

else:
    with left_col:
        st.info("Please paste some text to see the visualization.")

