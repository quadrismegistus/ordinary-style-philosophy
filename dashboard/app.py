import sys, os
import streamlit as st

# Setup paths to import 'osp'
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)

from osp import get_nlp_doc, get_doc_html, get_current_feat_weights

# Cache the NLP processing to avoid redundant slow computations
@st.cache_data
def process_text(text):
    return get_nlp_doc(text)

# Cache feature weights loading
@st.cache_data
def load_weights():
    return get_current_feat_weights()

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

text_input = st.text_area("Paste text here to visualize:", height=300, placeholder="Type or paste text here...")

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
            
            st.markdown("### Visualization")
            st.markdown(html_output, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
else:
    st.info("Please paste some text to see the visualization.")
