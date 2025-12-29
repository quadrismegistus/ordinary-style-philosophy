import streamlit as st
import sys, os
import pandas as pd
import stanza

# Setup paths to import 'osp'
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DASHBOARD = os.path.dirname(PATH_HERE)
PATH_REPO = os.path.dirname(PATH_DASHBOARD)
if PATH_REPO not in sys.path:
    sys.path.append(PATH_REPO)

from osp import *
from osp.constants import STASH_SLICES_NLP
from osp.sentences import render_sent_displacy, get_sent_html
from osp.nlp_utils import get_sent_tree_full, get_nlp_doc

st.set_page_config(layout="wide", page_title="Sentence Analysis")

# Helper to get svgling if available
try:
    import svgling
    HAS_SVGLING = True
except ImportError:
    HAS_SVGLING = False

def get_sentence_from_params():
    params = st.query_params
    
    doc = None
    sent_idx = 0
    
    if "slice_id" in params:
        slice_id = params["slice_id"]
        if slice_id in STASH_SLICES_NLP:
            docstr = STASH_SLICES_NLP[slice_id]
            doc = stanza.Document.from_serialized(docstr)
        else:
            st.error(f"Slice ID {slice_id} not found in stash.")
            return None, None
            
        if "sent_id" in params:
            try:
                # Assuming 1-indexed from user/URL
                sent_idx = int(params["sent_id"]) - 1
            except ValueError:
                sent_idx = 0
    
    elif "txt" in params:
        txt = params["txt"]
        doc = get_nlp_doc(txt)
        sent_idx = 0
        
    return doc, sent_idx

doc, sent_idx = get_sentence_from_params()

if doc and 0 <= sent_idx < len(doc.sentences):
    sent = doc.sentences[sent_idx]
    
    st.title("Sentence Analysis")
    
    # 1. Display plain text
    st.markdown(f'<blockquote><span style="font-size: 1.5em; font-family: Baskerville;">{sent.text}</span></blockquote>', unsafe_allow_html=True)

    color_options = ["weight_z", "weight", "score_z_diff", "pos", "deprel"]
    color_column = st.sidebar.selectbox("Color by:", color_options, index=0)

    # 4. Sentence HTML (Annotated)
    sent_html = get_sent_html(sent, color=color_column)
    st.markdown(sent_html, unsafe_allow_html=True)
    
    # col1, col2 = st.columns(2)
    
    
    # with col2:
        # 3. Constituency Tree (nltk/svgling)
    st.subheader("Constituency tree")
    tree = get_sent_tree(sent)
    if HAS_SVGLING:
        svg_obj = svgling.draw_tree(tree)
        # svgling's _repr_svg_ returns the SVG string
        st.components.v1.html(svg_obj._repr_svg_(), height=500, scrolling=True)
    else:
        st.code(str(tree))
    


    # 2. Dependency Tree (displacy SVG)
    st.subheader("Dependency relations")
    # Get color setting from sidebar/params if available, else default


    html_content = render_sent_displacy(sent, color_by=color_column, jupyter=False)

    # Inject script to scroll to bottom-left
    # displaCy puts words at the bottom. We want to see them immediately.
    html_with_scroll = f"""
    <div style="zoom: 0.8;">
    {html_content}
    </div>
    <script>
    // Scroll to bottom left on load
    setTimeout(function() {{
    window.scrollTo(0, document.body.scrollHeight);
    }}, 100);
    </script>
    """
    st.components.v1.html(html_with_scroll, height=500, scrolling=True)
    


else:
    st.title("Sentence Analysis")
    st.info("Please provide `slice_id` and `sent_id`, or `txt` via URL parameters.")
    st.write("Examples:")
    st.code("?slice_id=phil/1900-1925/00000001__01&sent_id=1")
    st.code("?txt=The%20present%20King%20of%20France%20is%20bald.")

    # Show some available slice IDs if possible
    if st.checkbox("Show some available slice IDs"):
        ids = list(STASH_SLICES_NLP.keys())[:20]
        st.write(ids)

