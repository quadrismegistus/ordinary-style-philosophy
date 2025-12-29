import sys, os
# Setup paths to import 'osp' and 'utils'
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
if PATH_REPO not in sys.path: sys.path.append(PATH_REPO)
if PATH_HERE not in sys.path: sys.path.append(PATH_HERE)

import streamlit as st
from utils import *

about_page = st.Page("pages/About.py", title="About", icon=":material/info:")
custom_page = st.Page("pages/Custom.py", title="Custom Input", icon=":material/edit_note:")
features_explore_page = st.Page("pages/Features.py", title="Explorer", icon=":material/local_pizza:")
feature_comp_page = st.Page("pages/FeatureComparison.py", title="Most Distinctive Features", icon=":material/compare:")
predict_page = st.Page("pages/Predictions.py", title="Predictions", icon=":material/psychology_alt:")
passages_page = st.Page("pages/Passages.py", title="Passages", icon=":material/visibility:")
sample_page = st.Page("pages/Sampling.py", title="Samples", icon=":material/model_training:")
sentence_page = st.Page("pages/Sentence.py", title="Sentences", icon=":material/text_snippet:")
group_page = st.Page("pages/Groups.py", title="Groups", icon=":material/group:")
corpus_info_page = st.Page("pages/CorpusInfo.py", title="Info", icon=":material/menu_book:")
corpus_data_page = st.Page("pages/CorpusData.py", title="Metadata", icon=":material/menu_book:")


pages = {
    "Corpus": [
        corpus_info_page,
        corpus_data_page,
    ],
    "Sample": [
        group_page,
        sample_page,
    ],
    "Features": [
        sentence_page,
        feature_comp_page,
        features_explore_page,
    ],
    "Predictions": [
        predict_page,
    ],
    "Passages": [
        passages_page,
        custom_page,
    ],
    "Project": [
        about_page,
    ],
}


pg = st.navigation(pages)

st.set_page_config(page_title="Ordinary Style Philosophy", layout="wide")

# Initialize and render the status window in the sidebar
get_status_window().render()

pg.run()
