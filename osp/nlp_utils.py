from . import *

def get_nlp():
    import osp.constants as constants
    
    if constants.NLP is None:
        constants.NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner,depparse,constituency', verbose=0)
    return constants.NLP


def get_nlp_doc(txt, force=False, id=None):
    from .constants import NLP_STASH
    
    key = (id, txt)
    if not force and key in NLP_STASH:
        return stanza.Document.from_serialized(NLP_STASH[key])
    nlp = get_nlp()
    doc = nlp(txt)
    NLP_STASH[key] = doc.to_serialized()
    return doc


def get_nlp_df(doc):
    ld = []
    for i, sent in enumerate(doc.sentences):
        for ii, word in enumerate(sent.words):
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

def tokenize_clauses_simple(tree): # -> return both whether "IC" or "DC"
    return [
        (t.leaves(), "IC" if t.label() == 'S' and not is_in_sbar(t) else "DC") 
        for t in tree.subtrees(lambda t: t.label() == 'S')
    ]

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


def get_pos_stats(tree):
    from .constants import pos_names
    
    pos_counter = Counter()
    for x in tree.productions():
        if x.is_lexical():
            pos = x.lhs().symbol()[0]
            if pos in pos_names:
                pos_counter[pos_names[pos]] += 1
    return pos_counter


def get_sent_str(sent):
    tokens = []
    for token in sent.tokens:
        token_d = token.to_dict()[0]
        tokens.append(token_d['text'])
        if not "SpaceAfter=No" in str(token_d.get('misc','')):
            tokens.append(' ')
    return ''.join(tokens)


def get_tree_stats(tree):
    if not isinstance(tree, nltk.ParentedTree):
        sentstr = get_sent_str(tree)
        tree = get_sent_tree(tree)
    else:
        sentstr = ' '.join(tree.leaves())
    
    dpos = get_pos_stats(tree)
    d = {
        'sent': sentstr,
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
        **{f'num_words_{pos}': dpos[pos] for pos in dpos}
    }
    return d


def get_sent_stats(sent, norm=True):
    tree = get_sent_tree(sent)
    data = {}
    data['num_words'] = get_num_words(tree)
    data['height'] = tree.height()
    data['num_words_in_dependent_clauses'] = get_num_words_in_dependent_clauses(tree)
    data['num_words_in_independent_clauses'] = get_num_words_in_independent_clauses(tree)
    data['num_independent_clauses'] = get_num_independent_clauses(tree)
    data['num_dependent_clauses'] = get_num_dependent_clauses(tree)

    if norm:
        for k in data:
            if k != 'num_words':
                data[k] = data[k] / data['num_words'] * 1000
    return data


def get_word_context(doc, sent_i, word_i, context_len=2):
    from .text_processing import remove_left_right_punct
    
    sent = doc.sentences[sent_i]

    prev_context = ''
    next_context = ''

    words_forward = sent.words[word_i+1:]
    words_backward = reversed(sent.words[:word_i])
    for w in words_forward:
        if len(next_context) < context_len:
            next_context += w.text + ' '
        else:
            break
    for w in words_backward:
        if len(prev_context) < context_len:
            prev_context = w.text + ' ' + prev_context + ' '
        else:
            break

    word = sent.words[word_i]
    out = f'{prev_context.strip()} {word.text.upper()} {next_context.strip()}'
    out = out.replace('\n', ' ').replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(':', ':').replace(' ;', ';')
    out = out.replace('( ', '(').replace('[ ', '[').strip().replace(' )', ')').replace(' ]', ']').replace('"', ' ').replace("'", " ")
    return remove_left_right_punct(out.strip()).strip()

