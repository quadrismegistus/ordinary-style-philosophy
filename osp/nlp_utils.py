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


def _stanza_constituency_to_nltk(node):
    """
    Recursively convert a stanza constituency node to an nltk.ParentedTree.
    """
    # stanza nodes expose: label, children, is_leaf, and for leaves value/text
    if hasattr(node, "is_leaf") and node.is_leaf:
        # Leaf node: use its label as POS/tag and its value/text as the child
        leaf_text = getattr(node, "value", None) or getattr(node, "text", "")
        return nltk.ParentedTree(node.label, [leaf_text])
    children = getattr(node, "children", []) or []
    return nltk.ParentedTree(node.label, [_stanza_constituency_to_nltk(c) for c in children])


def get_sent_tree_full(sent):
    """
    Build an nltk.ParentedTree directly from stanza's constituency object.
    This preserves full phrase labels (e.g., S, SBAR, NP) as provided by stanza.
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    # If stanza produced a constituency parse, convert it; otherwise fall back.
    if hasattr(sent, "constituency") and sent.constituency:
        try:
            return _stanza_constituency_to_nltk(sent.constituency)
        except Exception:
            # Fall back to the simpler string-based construction
            treestr = str(sent.constituency)
            return nltk.ParentedTree.fromstring(treestr)
    # No constituency available; fall back to previous behavior.
    return get_sent_tree(sent)


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


# =============================================================================
# Clause Extraction using Stanza Dependency and Constituency Parsing
# =============================================================================

# Dependency relations that typically introduce subordinate/dependent clauses
# Based on Universal Dependencies: https://universaldependencies.org/u/dep/
CLAUSE_DEPRELS = {
    'ccomp',      # clausal complement (that-clauses, reported speech)
    'xcomp',      # open clausal complement (infinitive/gerund complements)
    'advcl',      # adverbial clause (when, if, because, etc.)
    'acl',        # adnominal clause / relative clause
    'acl:relcl',  # relative clause (explicit)
    'csubj',      # clausal subject
    'csubj:pass', # clausal subject (passive)
    'parataxis',  # loosely attached clause
}


def get_clauses_v2(sent):
    """
    Extract clauses from a Stanza sentence using dependency parsing.
    
    This approach identifies clause boundaries based on dependency relations:
    - The root verb heads the main clause
    - Words with deprels like ccomp, xcomp, advcl, acl head subordinate clauses
    - Each word is assigned to a clause by following its dependency path
    
    Returns a DataFrame with columns:
    - word_id: 1-indexed word position
    - word: the word text
    - clause_id: which clause the word belongs to
    - clause_type: 'main' for the main clause, 'sub' for subordinate clauses
    - clause_head_id: the head word id of this clause
    - clause_deprel: the dependency relation introducing the clause
    - pos: the POS tag
    - deprel: the word's dependency relation
    - head: the word's head
    
    Reference: https://stanfordnlp.github.io/stanza/depparse.html
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    
    words = sent.words
    n = len(words)
    
    if n == 0:
        return pd.DataFrame()
    
    # Identify clause heads
    # A clause head is: (1) the root, or (2) a word with deprel in CLAUSE_DEPRELS
    clause_heads = {}  # clause_id -> (head_word_id, clause_type, clause_deprel)
    clause_id = 0
    
    for i, word in enumerate(words):
        word_id = i + 1
        if word.head == 0:  # Root of the sentence = main clause head
            clause_id += 1
            clause_heads[clause_id] = (word_id, 'main', 'root')
        elif word.deprel in CLAUSE_DEPRELS:
            clause_id += 1
            clause_heads[clause_id] = (word_id, 'sub', word.deprel)
    
    if not clause_heads:
        # No clauses found, treat entire sentence as one main clause
        clause_heads[1] = (1, 'main', 'root')
    
    # Build a set of clause head word_ids for quick lookup
    clause_head_ids = {head_id: cid for cid, (head_id, _, _) in clause_heads.items()}
    
    # Map each word to its clause by walking up the dependency tree
    word_to_clause = {}
    
    def find_clause_for_word(word_id, visited=None):
        """Find which clause a word belongs to by walking up the dependency tree."""
        if visited is None:
            visited = set()
        
        if word_id in word_to_clause:
            return word_to_clause[word_id]
        
        if word_id in visited or word_id <= 0 or word_id > n:
            return 1  # Default to first clause on cycle or invalid
        visited.add(word_id)
        
        # Check if this word is a clause head
        if word_id in clause_head_ids:
            return clause_head_ids[word_id]
        
        # Otherwise, follow the head
        head_id = words[word_id - 1].head
        if head_id == 0:
            # We're at root; find first main clause
            for cid, (hid, ctype, _) in clause_heads.items():
                if ctype == 'main':
                    return cid
            return 1
        
        return find_clause_for_word(head_id, visited)
    
    for i in range(1, n + 1):
        word_to_clause[i] = find_clause_for_word(i)
    
    # Build output DataFrame
    rows = []
    last_clause_id = None
    clause_num = -1
    for i, word in enumerate(words):
        word_id = i + 1
        cid = word_to_clause.get(word_id, 1)
        chead_id, ctype, cdeprel = clause_heads.get(cid, (1, 'main', 'root'))
        if last_clause_id != cid:
            last_clause_id = cid
            clause_num += 1
        
        rows.append({
            # 'word_num':word_id,
            'clause_i': clause_num,
            'clause_id': cid,
            'clause_type': ctype,
            'clause_head_id': chead_id,
            'clause_deprel': cdeprel,
            'word_i': i,
            'word': word.text,
            'word_pos': word.xpos or word.upos,
            'word_deprel': word.deprel,
            'word_head': word.head,
        })
    
    return pd.DataFrame(rows)


def get_clauses_with_spans(sent):
    """
    Extract clauses from a Stanza sentence and return both word-level and clause-level data.
    
    Returns a tuple:
    - df_words: DataFrame with word-level clause assignments (from get_clauses_v2)
    - df_clauses: DataFrame with clause-level summary:
        - clause_id
        - clause_type: 'main' or 'sub'
        - clause_deprel: the deprel introducing the clause
        - clause_head_word: the head word of the clause
        - num_words: number of words in the clause
        - text: the clause text (words joined)
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    
    df_words = get_clauses_v2(sent)
    
    if df_words.empty:
        return df_words, pd.DataFrame()
    
    # Summarize clauses
    clause_rows = []
    for cid, grp in df_words.groupby('clause_id'):
        ctype = grp['clause_type'].iloc[0]
        cdeprel = grp['clause_deprel'].iloc[0]
        chead_id = grp['clause_head_id'].iloc[0]
        chead_word = sent.words[chead_id - 1].text if chead_id <= len(sent.words) else ''
        
        # Get clause text (words in order of appearance)
        clause_text = ' '.join(grp.sort_values('word_id')['word'].tolist())
        
        clause_rows.append({
            'clause_id': cid,
            'clause_type': ctype,
            'clause_deprel': cdeprel,
            'clause_head_word': chead_word,
            'num_words': len(grp),
            'text': clause_text,
        })
    
    df_clauses = pd.DataFrame(clause_rows)
    return df_words, df_clauses


def get_clause_nodes_from_constituency(sent):
    """
    Use Stanza's constituency parse to identify clause-level nodes (S, SBAR, SINV, SQ, SBARQ).
    
    This walks the constituency tree directly from Stanza's parse (not string conversion).
    
    Returns a list of dicts with:
    - label: the constituency label (S, SBAR, etc.)
    - parent_label: the parent node's label
    - depth: depth in the tree
    - is_subordinate: True if this node is under an SBAR
    - text: the words under this clause node joined
    
    Reference: https://stanfordnlp.github.io/stanza/constituency.html
    """
    if isinstance(sent, str):
        sent = get_nlp_doc(sent).sentences[0]
    
    if not hasattr(sent, 'constituency') or sent.constituency is None:
        return []
    
    # Clause-level labels in PTB-style constituency
    clause_labels = {'S', 'SBAR', 'SINV', 'SQ', 'SBARQ'}
    
    clauses = []
    
    def collect_leaves(node):
        """Recursively collect leaf words from a constituency node."""
        if node.is_leaf():
            return [node.label]
        words = []
        for child in node.children:
            words.extend(collect_leaves(child))
        return words
    
    def walk_tree(node, parent_label=None, depth=0, in_sbar=False):
        """Walk the constituency tree to find clause nodes."""
        if node.is_leaf():
            return
        
        label = node.label
        is_now_in_sbar = in_sbar or (label == 'SBAR')
        
        if label in clause_labels:
            words = collect_leaves(node)
            
            clauses.append({
                'label': label,
                'parent_label': parent_label,
                'depth': depth,
                'is_subordinate': in_sbar,  # Was already in SBAR before this node
                'text': ' '.join(words),
                'num_words': len(words),
            })
        
        for child in node.children:
            walk_tree(child, parent_label=label, depth=depth + 1, in_sbar=is_now_in_sbar)
    
    walk_tree(sent.constituency)
    return clauses

