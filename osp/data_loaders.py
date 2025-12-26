from . import *

def rename_journal(journal):
    if 'Erkenntnis' in journal:
        return 'Erkenntnis'
    if 'Ethics' in journal:
        return 'Ethics'
    return journal


def is_non_content_pos(pos):
    return pos and pos[0] not in {'n','v','j'}


def iter_jsonl(fn, total=None):
    yield from tqdm(orjsonl.stream(fn), total=total)


def iter_pmla():
    from .constants import FN_PMLA, TOTAL_PMLA
    yield from iter_jsonl(FN_PMLA, total=TOTAL_PMLA)


def iter_jstor():
    from .constants import FN_JSTOR, TOTAL_JSTOR
    yield from iter_jsonl(FN_JSTOR, total=TOTAL_JSTOR)


def get_pmla_df():
    df = pd.DataFrame(iter_pmla())
    df['year'] = df['datePublished'].apply(lambda x: x.split('-')[0]).apply(int)
    df['decade'] = df['year'] // 10 * 10
    return df[df.docSubType=="research-article"]


@DF_STASH.stashed_result
def get_jstor_data():
    from .constants import FN_JSTOR_DATA, TOTAL_JSTOR_DATA
    df = pd.DataFrame(iter_jsonl(FN_JSTOR_DATA, total=TOTAL_JSTOR_DATA))
    df = df.rename(columns={'iid': 'id'})
    ids = set(df.id)

    ld = [d for d in iter_jstor() if d['item_id'] in ids]
    df2 = pd.DataFrame(ld)
    df2 = df2.rename(columns={'item_id': 'id'})
    df2['year'] = df2['published_date'].fillna('').apply(str).apply(lambda x: x.split('-')[0]).apply(int)
    df2['decade'] = df2['year'] // 10 * 10
    return df2.merge(df, on='id', how='left')


def periodize_year(year, periodize_by=10):
    minval = year // periodize_by * periodize_by
    maxval = minval + periodize_by
    return f'{minval}-{maxval}'


def get_half_century(year):
    """
    Returns a string representing whether the year is in the early or late half
    of its century. 'eC18' for early 18th century, 'lC19' for late 19th, etc.
    Early: yyyy < {century}50, Late: yyyy >= {century}50.
    """
    century = year // 100 + 1
    cutoff = (century - 1) * 100 + 50
    half = 'e' if year < cutoff else 'l'
    return f"{half}C{century}"


@cache
def get_corpus_metadata(path=None, periodize_by=25, min_year=None, max_year=None):
    from .constants import METADATA, PATH_METADATA, MIN_YEAR, MAX_YEAR
    import osp.constants as constants
    
    if path is None:
        path = PATH_METADATA
    if min_year is None:
        min_year = MIN_YEAR
    if max_year is None:
        max_year = MAX_YEAR
    
    if constants.METADATA is not None:
        return constants.METADATA
    df = pd.read_csv(path).set_index('id')
    df = df.query('year>=@min_year and year<@max_year')
    df['decade'] = df['year'] // 10 * 10
    df['period'] = df['year'].apply(lambda y: periodize_year(y, periodize_by))
    df['century'] = df['year'].apply(lambda y: f'C{y//100+1}')
    df['halfcentury'] = df['year'].apply(get_half_century)
    df['century_discipline'] = df['century'] + ' ' + df['discipline']
    df['halfcentury_discipline'] = df['halfcentury'] + ' ' + df['discipline']
    df['period_discipline'] = df['period'] + ' ' + df['discipline']
    df['decade_discipline'] = df['decade'].apply(str) + ' ' + df['discipline']
    df['century_journal'] = df['century'] + ' ' + df['journal']
    df['journal_orig'] = df['journal']
    df['journal'] = df['journal'].fillna('').apply(rename_journal)
    constants.METADATA = df
    return df


def get_text_metadata(id):
    id = id.split('__')[0]
    df_meta = get_corpus_metadata()
    return df_meta.loc[id] if id in df_meta.index else {}


def get_corpus_txt(id):
    from .constants import PATH_TXT
    from .text_processing import dehyphenate
    
    fn = os.path.join(PATH_TXT, id + '.txt')
    with open(fn, 'r') as f:
        txt = f.read()
    sents = nltk.sent_tokenize(txt)
    sents = [dehyphenate(s).replace('\n', ' ').strip() for s in sents if not any(x.isdigit() for x in s)]
    sents = [s for s in sents if s and s[0].isalpha() and s[0].isupper()]
    return '\n'.join(sents)


def iter_corpus_txt():
    df = get_corpus_metadata()
    for id in df.index:
        yield id, get_corpus_txt(id)


def count_tokens(id, force=False):
    from .constants import STASH_COUNTS
    from .text_processing import tokenize
    
    if not force and id in STASH_COUNTS:
        return STASH_COUNTS[id]
    txt = get_corpus_txt(id)
    d = dict(Counter(tokenize(txt)))
    STASH_COUNTS[id] = d
    return d


def get_total_text_counts(path=None, force=False):
    from .constants import STASH_COUNTS, PATH_TOTAL_TEXT_COUNTS
    
    if path is None:
        path = PATH_TOTAL_TEXT_COUNTS
    
    if not force and os.path.exists(path):
        try:
            return json.load(open(path))
        except:
            pass
    text_counts = Counter()
    for id, counts in tqdm(STASH_COUNTS.items(), total=len(STASH_COUNTS)):
        if counts:
            for word, count in counts.items():
                text_counts[word] += 1
    out = dict(text_counts.most_common())
    with open(path, 'w') as f:
        json.dump(out, f)
    return out


def get_top_words(n=10, remove_stopwords=True, except_first_n=100):
    from .text_processing import is_stopword
    
    d = get_total_text_counts()
    d = Counter({k:v for k,v in d.items() if not is_stopword(k)})
    out = [w for w,c in d.most_common(n+except_first_n)]
    if except_first_n and len(out)>except_first_n:
        out = out[except_first_n:]
    return out

@cache
def get_worddb(path=None):
    from .constants import PATH_WORDDB
    
    if path is None:
        path = PATH_WORDDB
    
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep='\t').set_index('word').fillna('')
    
    def is_non_content_pos_worddb(pos, word):
        pos = str(pos)
        if pos and pos[0] in {'n','v','j'}:
            return False
        if word.endswith('ly'):
            return False
        if pos and pos[0] == "m":
            return False
        return True
    
    df['is_non_content_pos'] = [is_non_content_pos_worddb(x, y) for x, y in zip(df.pos, df.index)]
    return df

def get_recog_words(words):
    ok_words = set(get_ok_words())
    return [w for w in words if w.lower().strip() in ok_words]

@stashed_result
def get_non_content_words_orig():
    worddb = get_worddb()
    worddb.is_non_content_pos.value_counts()
    return list(worddb[worddb.is_non_content_pos].index)


def load_non_content_words(path=None):
    from .constants import PATH_NON_CONTENT_WORDS
    
    if path is None:
        path = PATH_NON_CONTENT_WORDS
    
    return pd.read_excel(path).set_index('word')


def get_non_content_words(path=None):
    from .constants import PATH_NON_CONTENT_WORDS
    
    if path is None:
        path = PATH_NON_CONTENT_WORDS
    
    df = load_non_content_words(path)
    return df[df.incl.str.lower().str.strip() != "n"].index.tolist()


def load_mdw_data(path=None):
    from .constants import PATH_MDW_DATA
    
    if path is None:
        path = PATH_MDW_DATA
    
    return pd.read_csv(path).set_index('word')


def get_mdw_words(path=None, wordset="top", absval=True):
    from .constants import PATH_MDW_DATA
    
    if path is None:
        path = PATH_MDW_DATA
    
    s = load_mdw_data(path).query(f'wordset=="{wordset}"').mean(numeric_only=True, axis=1)
    if absval:
        s = s.apply(abs)
    return s.sort_values(ascending=False)


def get_top_mdw_words(n=100, path=None, wordset="top", absval=True):
    from .constants import PATH_MDW_DATA
    
    if path is None:
        path = PATH_MDW_DATA
    
    s = get_mdw_words(path, wordset=wordset, absval=absval)
    return s.head(n).index.tolist()


@cache
def get_ok_words():
    import osp.constants as constants
    
    if constants.OK_WORDS is None:
        worddb = get_worddb().fillna('')
        constants.OK_WORDS = {word for word in worddb.index}
    return constants.OK_WORDS


def get_wordsets(wordsets=None, n_words=100):
    from .constants import WORDSETS
    
    if wordsets is None:
        wordsets = WORDSETS
    
    wordsets_words = {}
    for wordset in wordsets:
        if wordset == 'non_content':
            wordsets_words[wordset] = get_non_content_words()[:n_words]
        else:
            wordsets_words[wordset] = get_top_mdw_words(n=n_words, wordset=wordset)
    return wordsets_words

