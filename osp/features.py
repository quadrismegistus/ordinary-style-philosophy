from . import *

def get_pos_counts(ids, incl_deprel=True, incl_pos=True):
    from .constants import STASH_POS_COUNTS
    
    ids = set(ids)
    index = []
    rows = []
    for id in STASH_POS_COUNTS.keys():
        if id in ids or id.split('__')[0] in ids:
            index.append(id)
            dat = STASH_POS_COUNTS[id]
            odx = {}
            for dk, dv in dat.items():
                is_deprel = dk == dk.lower()
                if (incl_deprel and is_deprel) or (incl_pos and not is_deprel):
                    odx[dk] = dv
            rows.append(odx)
    return pd.DataFrame(rows, index=index).rename_axis('id').fillna(0).applymap(int)

# @stashed_result
def get_pos_word_counts(ids):
    from .constants import STASH_FEAT2WORD2COUNT
    
    ids = set(ids)
    all_feat2word2count = defaultdict(Counter)
    for id in STASH_FEAT2WORD2COUNT.keys():
        if id in ids or id.split('__')[0] in ids:
            for feat, word2count in STASH_FEAT2WORD2COUNT[id].items():
                all_feat2word2count[feat].update(word2count)
    return all_feat2word2count

# @stashed_result
def get_pos_word_egs(ids):
    from .constants import STASH_FEAT2WORD2EG
    
    ids = set(ids)
    all_feat2word2eg = defaultdict(dict)
    for id in STASH_FEAT2WORD2EG.keys():
        if id in ids or id.split('__')[0] in ids:
            for feat, word2eg in STASH_FEAT2WORD2EG[id].items():
                all_feat2word2eg[feat].update(word2eg)
    return all_feat2word2eg


def get_egs(word2count, n=None, min_count=None, word2eg={}, incl_count=False):
    from .constants import FEAT_N, FEAT_MIN_COUNT
    word2count = Counter(word2count) if not isinstance(word2count, Counter) else word2count
    if n is None:
        n = FEAT_N
    
    total = word2count.total()
    o = []
    for w, c in word2count.most_common(n):
        c = int(round(c/total*1000))
        if not min_count or c >= min_count or len(o) >= n:
            o.append(f'{w} ({c})' if incl_count else f'{w}')
            break
    return ' '.join(o)


def get_slice_feats(id):
    from .constants import STASH_POS_COUNTS, STASH_SENT_FEAT_COUNTS, BAD_SLICE_FEATS
    
    posfeat_counts = STASH_POS_COUNTS.get(id, {})
    deprel_counts = {k: v for k, v in posfeat_counts.items() if k == k.lower()}
    pos_counts = {k: v for k, v in posfeat_counts.items() if k == k.upper()}

    sent_feat_counts = STASH_SENT_FEAT_COUNTS.get(id, {})
    sent_feat_counts_df = pd.DataFrame(sent_feat_counts)

    sent_sums = sent_feat_counts_df.sum(numeric_only=True)
    num_words = sent_sums['num_words']
    num_sents = len(sent_feat_counts)
    sent_sums['num_sents'] = num_sents
    total_clauses = sent_sums['num_independent_clauses'] + sent_sums['num_dependent_clauses']
    out_sent_d = {
        'avg_num_sents': num_sents / num_words * 1000,
        'avg_num_words_per_sent': num_words / num_sents,
        'avg_height': sent_sums['height'] / num_sents,
        'perc_dependent_clauses': sent_sums['num_dependent_clauses'] / total_clauses,
        'perc_independent_clauses': sent_sums['num_independent_clauses'] / total_clauses,
        'avg_num_words_per_clause': num_words / total_clauses,
    }
    out = {
        **{f'sent_{k}': v for k, v in out_sent_d.items()},
        **{f'pos_{k}': v for k, v in pos_counts.items() if k and k[0].isalpha()},
        **{f'deprel_{k}': v for k, v in deprel_counts.items() if k and k[0].isalpha()},
    }
    return {k: v for k, v in sorted(out.items()) if not any(k.startswith(b) for b in BAD_SLICE_FEATS)}


def get_valid_feat_keys():
    from .constants import STASH_POS_COUNTS, STASH_SENT_FEAT_COUNTS
    
    return set(STASH_POS_COUNTS.keys()) & set(STASH_SENT_FEAT_COUNTS.keys())


def get_slice_feat_counts(id, bad_feats=None):
    from .constants import STASH_SLICE_FEATS, BAD_SLICE_FEATS
    
    if bad_feats is None:
        bad_feats = BAD_SLICE_FEATS
    
    out_d = {}
    d = STASH_SLICE_FEATS.get(id)
    if not d:
        return {}
    for k, v in d.items():
        if k not in bad_feats and '_' in k:
            k1, k2 = k.split('_', 1)
            if k2.isalpha():
                out_d[k] = v
    return out_d


@cache
def get_all_feats(normalize=True, **kwargs):
    odf = get_all_feats_stashed()
    for c in odf.columns:
        odf[c] = pd.to_numeric(odf[c], errors='coerce').fillna(0)
    odf = odf.fillna(0)

    # replace less-than-0 values with 0
    odf = odf.applymap(lambda x: 0 if x < 0 else x)

    if normalize:
        for c in odf.columns:
            cmean = odf[c].mean()
            cstd = odf[c].std()
            odf[c] = (odf[c] - cmean) / cstd
    
    return odf[[c for c in odf.columns if c not in BAD_SLICE_FEATS]]


@stashed_result
def get_all_feats_stashed():
    from .constants import STASH_SLICE_FEATS
    
    stash = STASH_SLICE_FEATS
    all_keys = stash.keys_l()
    out_keys = []
    out_values = []
    for key in tqdm(all_keys, desc='getting all feat counts'):
        value = stash.get(key)
        if value is not None:
            out_keys.append(key)
            out_values.append(value)
    
    df_all_feats = pd.DataFrame(out_values, index=out_keys).rename_axis('id')
    return df_all_feats
    

def get_feat_counts(ids, normalize=True, renormalize=False, **kwargs):
    df_all_feats = get_all_feats(normalize=normalize)
    idset = set(ids)
    slice_ids = [id for id in df_all_feats.index if id.split('__', 1)[0] in idset]
    df_slice_feats = df_all_feats.loc[slice_ids]
    if renormalize:
        df_slice_feats = df_slice_feats.copy()
        for c in df_slice_feats.columns:
            cmean = df_slice_feats[c].mean()
            cstd = df_slice_feats[c].std()
            df_slice_feats[c] = (df_slice_feats[c] - cmean) / cstd
    return df_slice_feats


def extract_slice_feats(docstr, context_len=None, force=False, return_dict=True):
    from .constants import CONTEXT_LEN, BAD_POS, BAD_DEPREL
    from .nlp_utils import get_sent_stats, get_word_context
    
    if context_len is None:
        context_len = CONTEXT_LEN
    
    doc = stanza.Document.from_serialized(docstr) if isinstance(docstr, str) else docstr
    if doc is None:
        return {}

    o = []
    allwords=[]
    for sent_i, sent in enumerate(doc.sentences):
        for word_i, word in enumerate(sent.words):
            if word.pos in BAD_POS or word.deprel in BAD_DEPREL:
                continue
            allwords.append(word.text.lower())
            sent_d = get_sent_stats(sent)

            pos = word.xpos
            deprel = word.deprel
            eg_word = word.text.lower()
            eg_context = get_word_context(doc, sent_i, word_i, context_len=context_len).strip()
            
            out_d = {
                'sent_i': sent_i,
                'word_i': word_i,
                'word': eg_word,
                'pos': pos,
                'deprel': deprel,
                'context': eg_context,
                **{f'sent_{k}': v for k, v in sent_d.items() if k not in {'sent', 'num_words'}}
            }
            o.append(out_d)
    df1 = pd.DataFrame(o)
    if not return_dict:
        return df1
    
    odx = df1.drop(columns=['sent_i', 'word_i']).mean(numeric_only=True)
    orig_d = df1.drop(columns=['sent_i', 'word_i']).mean(numeric_only=True).to_dict()
    pos_d = df1.pos.value_counts().to_dict()
    deprel_d = df1.deprel.value_counts().to_dict()
    ttr = len(set(allwords)) / len(allwords) * 1000

    allwords_recog = get_recog_words(allwords)
    ttr_recog = len(set(allwords_recog)) / len(allwords_recog) * 1000

    num_recog_words = len(allwords_recog)
    num_words = len(allwords)
    perc_recog_words = num_recog_words / num_words * 1000

    odx = {
        **orig_d,
        **{f'pos_{k}': (v/sum(pos_d.values()))*1000 for k, v in pos_d.items()},
        **{f'deprel_{k}': (v/sum(deprel_d.values()))*1000 for k, v in deprel_d.items()},
        'ttr': ttr,
        'ttr_recog': ttr_recog,
        'num_words': num_words,
        'num_recog_words': num_recog_words,
    }
    return odx

# @stashed_result
def get_mdw_feats(groups_train, feat_n=10, feat_n_egs=5, **kwargs):
    name1,q1 = groups_train[0]
    name2,q2 = groups_train[1]
    ids1=get_corpus_metadata().query(q1).index.tolist()
    ids2=get_corpus_metadata().query(q2).index.tolist()
    
    df_scores_z = get_balanced_cv_data(groups_train)
    df_scores_raw = get_balanced_cv_data(groups_train, normalize=False)
    
    cols = set(df_scores_z.columns) & set(df_scores_raw.columns)
    hdr = '_group'
    words_grp1 = get_pos_word_counts(ids1)
    words_grp2 = get_pos_word_counts(ids2)
    egs_grp1 = get_pos_word_egs(ids1)
    egs_grp2 = get_pos_word_egs(ids2)
    feats = [f for f in cols if f[0]!='_']
    o = []
    for feat in feats:
        feat_name = feat.split('_')[-1]
        dfx_z = df_scores_z.groupby(hdr)[feat].mean()
        dfx_raw = df_scores_raw.groupby(hdr)[feat].mean()
        words1 = get_top_word_egs(words_grp1.get(feat_name, {}), n=feat_n)
        words2 = get_top_word_egs(words_grp2.get(feat_name, {}), n=feat_n)
        egs1 = get_top_egs(egs_grp1.get(feat_name, {}), words1, n=feat_n_egs)
        egs2 = get_top_egs(egs_grp2.get(feat_name, {}), words2, n=feat_n_egs)
        words1_str = get_top_word_egs_str(words1)
        words2_str = get_top_word_egs_str(words2)
        egs1_str = get_top_egs_str(egs1)
        egs2_str = get_top_egs_str(egs2)
        out_d = {
            'feature': feat,
            'feat_desc': FEAT2DESC.get(feat_name, ''),
            'comparison': f'{name1} vs {name2}',
            # 'group0': 'Unseen',
            'group1': name1,
            'group2': name2,
            # 'score_mean0': dfx_raw.loc['Unseen'],
            'score_mean1': dfx_raw.loc[name1],
            'score_mean2': dfx_raw.loc[name2],
            # 'score_mean_diff': dfx_raw.loc[name1] - dfx_raw.loc[name2],
            # 'score_mean_diff_abs': abs(dfx_raw.loc[name1] - dfx_raw.loc[name2]),
            # 'score_mean_diff_pct': abs(dfx_raw.loc[name1] - dfx_raw.loc[name2]) / dfx_raw.loc[name2] if dfx_raw.loc[name2]!=0 else None,
            # 'score_mean_div': dfx_raw.loc[name1] / dfx_raw.loc[name2] if dfx_raw.loc[name2]!=0 else None,
            # 'score_mean_div_abs': abs(dfx_raw.loc[name1] / dfx_raw.loc[name2] if dfx_raw.loc[name2]!=0 else None),
            # 'score_z0': dfx_z.loc['Unseen'],
            'score_z1': dfx_z.loc[name1],
            'score_z2': dfx_z.loc[name2],
            # 'score_z_diff': dfx_z.loc[name1] - dfx_z.loc[name2],
            # 'score_z_diff_abs': abs(dfx_z.loc[name1] - dfx_z.loc[name2]),
            # 'score_z_diff_pct': abs(dfx_z.loc[name1] - dfx_z.loc[name2]) / dfx_z.loc[name2] if dfx_z.loc[name2]!=0 else None,
            # 'score_z_div': dfx_z.loc[name1] / dfx_z.loc[name2] if dfx_z.loc[name2]!=0 else None,
            # 'score_z_div_abs': abs(dfx_z.loc[name1] / dfx_z.loc[name2] if dfx_z.loc[name2]!=0 else None),
            'words1': words1_str,
            'words2': words2_str,
            'egs1': egs1_str,
            'egs2': egs2_str,
        }
        o.append(out_d)
    odf=pd.DataFrame(o)
    return odf

def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False


def get_top_word_egs(word2count, n=None, min_count=None, incl_count=False):
    # from .constants import FEAT_N, FEAT_MIN_COUNT
    word2count = Counter({x:int(i) for x,i in word2count.items() if is_numeric(i)})
    if n is None:
        n = FEAT_N
    
    total = word2count.total()
    words = []
    for w, c in word2count.most_common(n):
        c = int(round(c/total*1000))
        if not min_count or c >= min_count or len(words) >= n:
            words.append((w, c))
        else:
            break
    return [w if incl_count else (w, c) for w, c in words]

def get_top_egs(egs_grp, words, n=None, min_count=None, incl_count=False):
    egs = []
    words_l = [w[0] if isinstance(w, tuple) else w for w in set(words)]
    random.shuffle(words_l)
    for w in words_l:
        eg=egs_grp.get(w, '')
        if eg:
            egs.append(eg)
        if len(egs) >= n:
            break
    return egs

def get_top_egs_str(top_egs):
    return '; '.join(f'“{e}”' for e in top_egs)

def get_top_word_egs_str(top_words):
    return ', '.join(f'{w[0]} ({w[1]})' if isinstance(w, tuple) else str(w) for w in top_words)



def get_balanced_cv_data(groups_train, target_col='discipline', balance=True, normalize=NORMALIZE_DATA, **kwargs):
    df_meta = get_corpus_metadata()
    name1, query1 = groups_train[0]
    name2, query2 = groups_train[1]

    df_meta1 = df_meta.query(query1)
    df_meta2 = df_meta.query(query2)

    
    df_scores_all = get_all_feats(normalize=normalize, **kwargs).fillna(0)
    df_scores1 = get_feat_counts(df_meta1.index.tolist(), normalize=normalize, renormalize=False, **kwargs)
    df_scores2 = get_feat_counts(df_meta2.index.tolist(), normalize=normalize, renormalize=False, **kwargs)

    for dfx in [df_scores_all, df_scores1, df_scores2]:
        dfx['_target'] = [get_text_metadata(i).get(target_col) for i in dfx.index]
        dfx.dropna(subset=['_target'], inplace=True)
    
    if balance:
        minsize = min(len(df_scores1), len(df_scores2))
        df_scores1 = df_scores1.sample(n=minsize).assign(_group=name1)
        df_scores2 = df_scores2.sample(n=minsize).assign(_group=name2)
    
    df_scores_cv = pd.concat([df_scores1, df_scores2]).assign(_type='CV')
    df_scores_rest = df_scores_all.drop(df_scores_cv.index).assign(_type='Unseen', _group='Unseen')
    df_scores = pd.concat([df_scores_cv, df_scores_rest])
    return df_scores


def get_current_feat_weights(comparisons=None):
    df = pd.read_excel(PATH_FEAT_WEIGHTS)
    df = df.drop(columns=['Unnamed: 0','run'])
    if comparisons is not None:
        df = df.query('comparison in @comparisons')
    
    return df.groupby('feature').mean(numeric_only=True)
    



def gen_all_slice_feats(force=False, batch_n=100, num_proc=1):
    ids = get_parsed_slice_ids()
    if not force:
        ids_not_done = [id for id in ids if id not in STASH_SLICE_FEATS]
        if len(ids_not_done) == 0:
            return
        ids = ids_not_done
    
    for batch_i in tqdm(list(range(0,len(ids),batch_n)),position=0):
        batch_ids = ids[batch_i:batch_i+batch_n]
        batch_docstrs = [STASH_SLICES_NLP[id] for id in batch_ids]

        with mp.Pool(num_proc) as pool:
            iterr = pool.imap(_do_gen_all_slice_feats, batch_docstrs)
            for id,res in zip(batch_ids,iterr):
                STASH_SLICE_FEATS[id] = res

def _do_gen_all_slice_feats(docstr):
    try:
        return extract_slice_feats(docstr)
    except Exception as e:
        print(f'!! {e}')
        return None
