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
features_page = st.Page("pages/Features.py", title="Feature Explorer", icon=":material/local_pizza:")
feature_comp_page = st.Page("pages/FeatureComparison.py", title="Feature Comparison", icon=":material/compare:")
predict_page = st.Page("pages/Predictions.py", title="Predictions", icon=":material/psychology_alt:")
passages_page = st.Page("pages/Passages.py", title="Passages", icon=":material/visibility:")
sample_page = st.Page("pages/Sampling.py", title="Samples", icon=":material/model_training:")

pages = {
    "Info": [
        about_page,
    ],
    "Corpus": [
        sample_page,
    ],
    "Features": [
        features_page,
        feature_comp_page,
    ],
    "Predictions": [
        predict_page,
    ],
    "Passages": [
        passages_page,
        custom_page,
    ],
}


pg = st.navigation(pages)

st.set_page_config(page_title="Ordinary Style Philosophy", layout="wide")

pg.run()
