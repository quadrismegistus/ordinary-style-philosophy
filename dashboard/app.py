import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *
import streamlit as st
import altair as alt
import numpy as np

newtext= """By a `denoting phrase' I mean a phrase such as any one of the following: a man, some man, any man, every man, all men, the present King of England, the presenting King of France, the center of mass of the solar system at the first instant of the twentieth century, the revolution of the earth round the sun, the revolution of the sun round the earth. Thus a phrase is denoting solely in virtue of its form. We may distinguish three cases: (1) A phrase may be denoting, and yet not denote anything; e.g., `the present King of France'. (2) A phrase may denote one definite object; e.g., `the present King of England' denotes a certain man. (3) A phrase may denote ambiguously; e.g. `a man' denotes not many men, but an ambiguous man. The interpretation of such phrases is a matter of considerably difficulty; indeed, it is very hard to frame any theory not susceptible of formal refutation. All the difficulties with which I am acquainted are met, so far as I can discover, by the theory which I am about to explain."""

featcols = [
    "feat_desc",
    "score_mean1",
    "score_mean2",
    "score_mean3",
    # "score_mean_div_3-1",
    # "score_mean_div_3-2",
    # "score_z1",
    # "score_z2",
    # "score_z3",
    "score_mean_diff_3-1",
    "score_mean_diff_3-2",
    # "score_z_diff_3-1",
    # "score_mean_diff_abs_3-1",
    # "score_z_diff_abs_3-1",
]

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
def load_new_preds_feats(text_input):
    doc = process_text(text_input)
    df_new_preds, df_new_feats = get_new_preds_feats(doc)
    
    # Keep track of non-numeric columns we need
    needed_cols = ["feature", "feat_type", "feat_name"]
    df_new_feats_grouped = df_new_feats.groupby(needed_cols).mean(numeric_only=True).reset_index()
    
    # Sort and filter for display
    df_new_feats_display = df_new_feats_grouped.sort_values("score_mean_diff_3-1", ascending=False)
    df_new_feats_display = df_new_feats_display[needed_cols + [c for c in featcols if c in df_new_feats_display.columns]]
    df_new_feats_display['feat_desc'] = df_new_feats_display['feature'].map(lambda x: FEAT2DESC.get(x, ''))
    
    df_new_preds = df_new_preds.groupby("comparison").mean(numeric_only=True).sort_values("prob_Philosophy",ascending=False).drop(columns=['run'])
    
    return df_new_preds, df_new_feats_display, df_new_feats_grouped

def plot_predictive_features(df_new_feats):
    figld = []
    for i, row in df_new_feats.iterrows():
        # Scale specific features as in notebook
        s1, s2, s3 = row['score_mean1'], row['score_mean2'], row['score_mean3']
        if row['feat_name'].startswith('num_words_in') or row['feat_name'] == 'height':
            s1, s2, s3 = s1 / 10, s2 / 10, s3 / 10
            
        for grp in [1, 2]:
            d = {
                'feature': row['feature'],
                'feat_type': row['feat_type'],
                'feat_name': row['feat_name'],
                'target': ('Philosophy' if grp == 1 else 'Literature'),
                'score_training': s1 if grp == 1 else s2,
                'score_new': s3,
            }
            if d['score_training'] > 1 and d['score_new'] > 1:
                figld.append(d)
    
    if not figld:
        st.warning("No features found with scores > 1 for plotting.")
        return
        
    figdf = pd.DataFrame(figld)
    figdf['odds_ratio'] = figdf['score_new'] / figdf['score_training']
    figdf['odds_ratio_log'] = np.log(figdf['odds_ratio'])
    figdf['odds_ratio_log_abs'] = figdf['odds_ratio_log'].abs()
    
    def format_ratio(ratio):
        if ratio < 1:
            return f'-{1/ratio:.1f}x'
        return f'{ratio:.1f}x'
    
    figdf['ratio_S'] = figdf['odds_ratio'].apply(format_ratio)
    figdf['clean_feat_name'] = figdf['feat_name'].str.replace("num_","# ").str.replace("_"," ")

    # Create Altair Selection for highlighting
    selection = alt.selection_point(fields=['feature'], on='click')
    
    base = alt.Chart(figdf).encode(
        x=alt.X('score_training:Q', scale=alt.Scale(type='log', domain=[1, 250]), title='Training Set Feature Score'),
        y=alt.Y('score_new:Q', scale=alt.Scale(type='log', domain=[1, 250]), title='New Text Feature Score'),
        tooltip=[
            alt.Tooltip('clean_feat_name:N', title='Feature'),
            alt.Tooltip('target:N', title='Comparison'),
            alt.Tooltip('score_training:Q', format='.2f', title='Training Score'),
            alt.Tooltip('score_new:Q', format='.2f', title='New Score'),
            alt.Tooltip('ratio_label:N', title='Ratio')
        ]
    )

    points = base.mark_point(filled=True).encode(
        color=alt.condition(
            selection,
            alt.Color('odds_ratio_log:Q', scale=alt.Scale(scheme='redblue', domainMid=0), title='Log Odds Ratio'),
            alt.value('lightgray')
        ),
        size=alt.Size('odds_ratio_log_abs:Q', scale=alt.Scale(range=[50, 400]), title='Abs Log Odds'),
        shape=alt.Shape('target:N', title='Target'),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2)),
    ).add_params(
        selection
    ).interactive()
    
    text = base.mark_text(
        align='left',
        baseline='middle',
        dx=7,
        fontSize=10
    ).transform_calculate(
        # Jitter the y-position for labels to reduce overlap
        jittered_y='datum.score_new * pow(1.2, random() - 0.5)'
    ).encode(
        y=alt.Y('jittered_y:Q', scale=alt.Scale(type='log')),
        text='clean_feat_name:N',
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.1))
    )

    # Diagonal line
    line_val = [min(figdf['score_new'].min(), figdf['score_training'].min()), 
                max(figdf['score_new'].max(), figdf['score_training'].max())]
    line_df = pd.DataFrame({'x': line_val, 'y': line_val})
    line = alt.Chart(line_df).mark_line(color='gray', strokeDash=[5, 5], opacity=0.5).encode(
        x='x:Q',
        y='y:Q'
    )

    layered = alt.layer(line, points, text, data=figdf).properties(
        height=800
    )

    # Note: on_select="rerun" is not yet supported for layered charts in Streamlit
    st.altair_chart(layered, use_container_width=True)

st.set_page_config(page_title="Ordinary Style Philosophy Visualizer", layout="wide")

st.title("Ordinary Style Philosophy Visualizer")

# Get feature weights to populate options
try:
    df_weights = load_weights()
    color_options = [c for c in df_weights.columns if df_weights[c].dtype in ['float64', 'int64']]
except Exception as e:
    st.error(f"Error loading feature weights: {e}")
    color_options = ['score_z_diff']

with st.sidebar:
    st.header("Settings")
    word_feat_type = st.selectbox("Color words by:", options=['deprel', 'pos'], index=0)
    color_column = st.selectbox("Weight column:", options=color_options, index=color_options.index('score_z_diff') if 'score_z_diff' in color_options else 0)
    
    st.info("Blue: Positive weight | Orange: Negative weight")

# Main layout: Two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    text_input = st.text_area("Paste text here to visualize:", height=300, placeholder="Type or paste text here...", value=newtext)

if text_input:
    with st.spinner("Processing text with Stanza..."):
        try:
            # Generate doc from text (using cached function)
            doc = process_text(text_input)
            
            # Generate HTML visualization
            html_output = get_doc_html(
                doc, 
                color=color_column, 
                word_feat_type=word_feat_type
            )
            df_new_preds, df_new_feats_display, df_new_feats_grouped = load_new_preds_feats(text_input)
            
            with right_col:
                st.markdown("### Predictions")
                # Melt the predictions for bar plotting
                df_plot_preds = df_new_preds.reset_index().melt(id_vars='comparison', var_name='Class', value_name='Probability')
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

                st.markdown("### Features")
                plot_predictive_features(df_new_feats_grouped)

            with left_col:
                st.markdown("### Passage")
                st.markdown(html_output, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
else:
    with left_col:
        st.info("Please paste some text to see the visualization.")

