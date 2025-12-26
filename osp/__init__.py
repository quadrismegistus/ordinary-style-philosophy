# Core imports
import json
import random
import numpy as np
from collections import Counter
import string
from pprint import pprint
import os
from tqdm import tqdm
import orjsonl
import pandas as pd
from hashstash import HashStash, stashed_result
import nltk
from collections import defaultdict
import stanza
import re
import plotnine as p9
from functools import lru_cache
import json
from IPython.display import HTML,display,Markdown
import html
import multiprocessing as mp


# Configure cache and plotnine
cache = lru_cache(maxsize=None)
p9.options.figure_size = (10, 8)
p9.options.dpi = 300
pd.options.display.max_colwidth = 200

# Import all modules
from .constants import *
from .text_processing import *
from .data_loaders import *
from .nlp_utils import *
from .word_freqs import *
from .slices import *
from .features import *
from .statistics import *
from .classify import *
from .passages import *
from .sentences import *

