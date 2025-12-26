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
predict_page = st.Page("pages/Predictions.py", title="Predictions", icon=":material/psychology_alt:")
passages_page = st.Page("pages/Passages.py", title="Passages", icon=":material/visibility:")
pg = st.navigation([about_page, predict_page, features_page, custom_page, passages_page])

st.set_page_config(page_title="Ordinary Style Philosophy", layout="wide")

pg.run()
