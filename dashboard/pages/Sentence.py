import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osp import *
from utils import *

import streamlit as st

st.set_page_config(layout="wide", page_title="Sentence Analysis")

# 1. Retrieve last choice
last_slice_id = STASH_DASHBOARD_STATE.get("osp_last_slice_id")
last_sent_id = STASH_DASHBOARD_STATE.get("osp_last_sent_id")
st.title("Sentence")

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
    
    slice_id = params.get("slice_id") or last_slice_id
    if slice_id:
        if slice_id in STASH_SLICES_NLP:
            docstr = STASH_SLICES_NLP[slice_id]
            doc = stanza.Document.from_serialized(docstr)
        else:
            st.error(f"Slice ID {slice_id} not found in stash.")
            return None, None
            
        sent_id = params.get("sent_id") or last_sent_id
        if sent_id:
            try:
                # Assuming 1-indexed from user/URL
                sent_idx = int(sent_id) - 1
            except ValueError:
                sent_idx = 0
    
    elif "txt" in params:
        txt = params["txt"]
        doc = get_nlp_doc(txt)
        sent_idx = 0
        
    return doc, sent_idx

status_window = get_status_window()
with log_progress("Getting sentence from params"):
    doc, sent_idx = get_sentence_from_params()

if not doc or not sent_idx:
    with log_progress("Getting random sentence"):
        doc = get_random_doc()
    sent_idx = random.randint(0, len(doc.sentences) - 1)


if doc and 0 <= sent_idx < len(doc.sentences):
    sent = doc.sentences[sent_idx]
    
    # st.title("Sentence Analysis")
    
    # 1. Display plain text
    st.markdown(f'<blockquote><span style="font-size: 1.5em; font-family: Baskerville;">{sent.text}</span></blockquote>', unsafe_allow_html=True)

    color_options = ["weight_z", "weight", "score_z_diff", "pos", "deprel"]
    # color_column = st.sidebar.selectbox("Color by:", color_options, index=0)
    color_column = color_options[0]  # Default to first option

    # 4. Sentence HTML (Annotated)
    word_id = st.query_params.get("word_id")
    if word_id:
        try:
            word_id = int(word_id)
        except ValueError:
            word_id = None
    
    with log_progress("Getting sentence HTML"), st.spinner("Getting sentence HTML"):
        sent_html = get_sent_html(sent, color=color_column, highlight_word_id=word_id)
    
    st.markdown(sent_html, unsafe_allow_html=True)
    
    # Auto-track last viewed sentence
    slice_id = st.query_params.get("slice_id")
    sent_id = st.query_params.get("sent_id")
    if slice_id and st.session_state.get("last_viewed_slice_id") != slice_id:
        st.session_state["last_viewed_slice_id"] = slice_id
        STASH_DASHBOARD_STATE["osp_last_slice_id"] = slice_id
    if sent_id and st.session_state.get("last_viewed_sent_id") != sent_id:
        st.session_state["last_viewed_sent_id"] = sent_id
        STASH_DASHBOARD_STATE["osp_last_sent_id"] = sent_id
    
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
    



else:
    st.info("Please provide `slice_id` and `sent_id`, or `txt` via URL parameters.")
    st.write("Examples:")
    st.code("?slice_id=phil/1900-1925/00000001__01&sent_id=1")
    st.code("?txt=The%20present%20King%20of%20France%20is%20bald.")

    # Show some available slice IDs if possible
    if st.checkbox("Show some available slice IDs"):
        ids = list(STASH_SLICES_NLP.keys())[:20]
        st.write(ids)

