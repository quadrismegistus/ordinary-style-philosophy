import string
from pprint import pprint
import os
from tqdm import tqdm
import orjsonl
import pandas as pd
from hashstash import HashStash, stashed_result
import nltk
import stanza
import re

TOTAL_PMLA = 71902
TOTAL_JSTOR = 12412004
TOTAL_JSTOR_DATA = 32783
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_HERE, 'data')
FN_PMLA = os.path.join(PATH_DATA, 'raw/LitStudiesJSTOR.jsonl')
FN_JSTOR = os.path.join(PATH_DATA, 'raw/jstor_metadata_2025-11-28.jsonl.gz')
FN_JSTOR_DATA = os.path.join(PATH_DATA, 'raw/jstor_data.jsonl.gz')
NLP = None
NLP_STASH = HashStash('osp_nlp')
PMLA_STASH = HashStash('osp_pmla')
JSTOR_STASH = HashStash('osp_jstor')
DF_STASH = HashStash('osp_df', serializer='pickle')

def iter_jsonl(fn, total=None):
    yield from tqdm(orjsonl.stream(fn), total=total)

def iter_pmla():
    yield from iter_jsonl(FN_PMLA, total=TOTAL_PMLA)

def iter_jstor():
    yield from iter_jsonl(FN_JSTOR, total=TOTAL_JSTOR)

def get_pmla_df():
    df = pd.DataFrame(iter_pmla())
    df['year'] = df['datePublished'].apply(lambda x: x.split('-')[0]).apply(int)
    df['decade'] = df['year'] // 10 * 10
    return df[df.docSubType=="research-article"]


import re

def dehyphenate(text):
    """
    Removes hyphenation at line breaks, e.g. 'artist-\nically' or 'artist- ically' -> 'artistically'
    Handles both space and newline after hyphens.
    """
    # Handle hyphen followed by newline/whitespace, optionally some more whitespace, then continuing letters
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'-\s+', '', text)

    # INSERT_YOUR_CODE
    # if ? is between two letters, replace it with em-dash
    text = re.sub(r'(?<=[a-zA-Z])\?(?=[a-zA-Z])', '—', text)
    return text


def filter_pmla_pages(article):
    newpages = []
    for page in article:
        # remove leading all-caps words (likely running heads) from the page
        words = page.split()
        i = 0
        for w in words:
            # Match word if it's all caps (with possible initial punctuation and optional trailing punctuation/numbers)
            if w.isdigit() or re.match(r"^[A-Z]+[.,:;?\-–—'\d]*$", w):
                i += 1
            else:
                break
        newpages.append(' '.join(words[i:]))
    return dehyphenate(' '.join(newpages))


def get_nlp():
    global NLP
    if NLP is None:
        NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner,depparse,constituency', verbose=0)
    return NLP

def get_nlp_doc(txt):
    nlp = get_nlp()
    doc = nlp(txt)
    return doc

def get_nlp_df(doc):
    ld=[]
    for i,sent in enumerate(doc.sentences):
        for ii,word in enumerate(sent.words):
            d = {
                'sent_id': i+1,
                'word_id': word.id,
                'word': word.text,
                'lemma': word.lemma,
                'pos': word.pos,
                'deprel': word.deprel,
                'head': word.head,
            }
            ld.append(d)
    return pd.DataFrame(ld)





def is_in_sbar(tree):
    parent = tree.parent()
    while parent is not None:
        if parent.label() == 'SBAR':
            return True
        parent = parent.parent()
    return False

def get_num_independent_clauses(tree):
    return len([t for t in tree.subtrees(lambda t: t.label() == 'S' and not is_in_sbar(t))])

def get_num_dependent_clauses(tree):
    return len([t for t in tree.subtrees(lambda t: t.label() == 'SBAR')])

def get_num_words_in_dependent_clauses(tree):
    return sum([len(t.leaves()) for t in tree.subtrees(lambda t: t.label() == 'SBAR')])

def get_num_words(tree):
    return len(tree.leaves())

def get_num_words_in_independent_clauses(tree):
    return get_num_words(tree) - get_num_words_in_dependent_clauses(tree)

def get_num_punct(tree, punct_type=None):
    if punct_type is None:
        out = len([t for t in tree.subtrees() if t.label() in string.punctuation])
        return out + get_num_parens(tree)
    else:
        return len([t for t in tree.subtrees() if t.label() == punct_type])

def get_num_parens(tree, paren_set={'-LRB-', '-RRB-'}):
    return len([t for t in tree.subtrees() if t.label() in paren_set])

def get_sent_tree(sent):
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    treestr = str(sent.constituency)
    return nltk.ParentedTree.fromstring(treestr)

def get_tree_stats(tree):
    if not isinstance(tree, nltk.ParentedTree):
        tree = get_sent_tree(tree)
    d = {
        'num_words': get_num_words(tree),
        'height': tree.height(),
        'num_independent_clauses': get_num_independent_clauses(tree),
        'num_dependent_clauses': get_num_dependent_clauses(tree),
        'num_words_in_dependent_clauses': get_num_words_in_dependent_clauses(tree),
        'num_words_in_independent_clauses': get_num_words_in_independent_clauses(tree),
        'num_punct': get_num_punct(tree),
        'num_punct_colon': get_num_punct(tree, ':'),
        'num_punct_comma': get_num_punct(tree, ','),
        'num_parens': get_num_parens(tree),
    }
    return d

@DF_STASH.stashed_result
def get_jstor_data():
    df = pd.DataFrame(iter_jsonl(FN_JSTOR_DATA, total=TOTAL_JSTOR_DATA))
    df = df.rename(columns={'iid': 'id'})
    ids = set(df.id)

    ld = [d for d in iter_jstor() if d['item_id'] in ids]
    df2 = pd.DataFrame(ld)
    df2 = df2.rename(columns={'item_id': 'id'})
    df2['year'] = df2['published_date'].fillna('').apply(str).apply(lambda x: x.split('-')[0]).apply(int)
    df2['decade'] = df2['year'] // 10 * 10
    return df2.merge(df, on='id', how='left')