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


# @cache
def get_all_feats(normalize=True, feat_types=None, **kwargs):
    odf = get_all_feats_stashed()
    
    if feat_types:
        bad_cols = [c for c in odf.columns if not c or c[0]=='_' or c.split('_')[0] not in feat_types]
        odf = odf.drop(columns=bad_cols)
    
    for c in odf.columns:
        odf[c] = pd.to_numeric(odf[c], errors='coerce').fillna(0)
    odf = odf.fillna(0)

    # # replace less-than-0 values with 0
    # odf = odf.applymap(lambda x: 0 if x < 0 else x)

    if normalize:
        for c in odf.columns:
            cmean = odf[c].mean()
            cstd = odf[c].std()
            odf[c] = (odf[c] - cmean) / cstd
    
    return odf[[c for c in odf.columns if c not in BAD_SLICE_FEATS]]


# @stashed_result
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
    

def get_feat_counts(ids, normalize=True, renormalize=False, feat_types=CV_FEAT_TYPES, **kwargs):
    df_all_feats = get_all_feats(normalize=normalize, feat_types=feat_types)
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
        'ttr_mean': ttr,
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


def get_balanced_cv_data(groups_train, target_col='discipline', balance=True, normalize=NORMALIZE_DATA, feat_types=CV_FEAT_TYPES, **kwargs):
    df_meta = get_corpus_metadata()
    name1, query1 = groups_train[0]
    name2, query2 = groups_train[1]

    df_meta1 = df_meta.query(query1)
    df_meta2 = df_meta.query(query2)

    
    df_scores_all = get_all_feats(normalize=normalize, feat_types=feat_types, **kwargs).fillna(0)
    df_scores1 = get_feat_counts(df_meta1.index.tolist(), normalize=normalize, renormalize=False, feat_types=feat_types, **kwargs)
    df_scores2 = get_feat_counts(df_meta2.index.tolist(), normalize=normalize, renormalize=False, feat_types=feat_types, **kwargs)

    for dfx in [df_scores_all, df_scores1, df_scores2]:
        dfx['_target'] = [get_text_metadata(i).get(target_col,'') for i in dfx.index]
        dfx.dropna(subset=['_target'], inplace=True)
    
    if balance:
        minsize = min(len(df_scores1), len(df_scores2))
        df_scores1 = df_scores1.sample(n=minsize).assign(_group=name1)
        df_scores2 = df_scores2.sample(n=minsize).assign(_group=name2)
    
    df_scores_cv = pd.concat([df_scores1, df_scores2]).assign(_type='CV')
    df_scores_rest = df_scores_all.drop(df_scores_cv.index).assign(_type='Unseen', _group='Unseen')
    df_scores = pd.concat([df_scores_cv, df_scores_rest])
    return df_scores


@cache
def get_current_feat_weights(*args,group_by=('feature',), **kwargs):
    from .classify import get_preds_feats
    df_preds, df_feats, d_models = get_preds_feats(*args, **kwargs)
    odf = df_feats.groupby(list(group_by)).mean(numeric_only=True)
    odf['weight_z'] = (odf['weight'] - odf['weight'].mean()) / odf['weight'].std()
    return odf



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

def get_nice_df_feats(df_feats=None):
    if df_feats is None:
        df_preds, df_feats, d_models = get_preds_feats()

    out_ld = []
    for feat,featdf in df_feats.groupby('feature'):
        out_d = {'feature':feat}
        vals_P = []
        vals_L = []
        out_d2 = {}
        for cmp,cmp_df in featdf.groupby('comparison'):
            cmp_prd = cmp.split(' ')[0].split('-')[0]
            cmp_key_P = f'P{cmp_prd}'
            val_P = float(cmp_df.score_mean1.mean())
            vals_P.append(val_P)
            out_d2[cmp_key_P] = val_P
            
            cmp_key_L = f'L{cmp_prd}'
            val_L = float(cmp_df.score_mean2.mean())
            vals_L.append(val_L)
            # out_d2[cmp_key_L] = val_L
        out_d2['P2000/P1925'] = np.log(out_d2['P2000']/out_d2['P1925']) if out_d2['P1925'] else np.nan
        out_d['P'] = np.mean(vals_P)
        out_d['L'] = np.mean(vals_L)
        out_d['P/L'] = np.log(out_d['P'] / out_d['L'])
        out_ld.append({**out_d, **out_d2})
    
    odf = pd.DataFrame(out_ld)
    odf = odf.round(2).sort_values('P/L',ascending=False).dropna()
    return odf

def get_dashboard_df_feats(df_feats=None):
    from .classify import get_preds_feats
    if df_feats is None:
        df_preds, df_feats, d_models = get_preds_feats()

    period2cmp = {x.split('-')[0]:x for x in df_feats.comparison.unique()}
    

    out_ld = []
    for feat,featdf in df_feats.round(3).groupby('feature'):
        out_d = {'feature':feat}
        vals = defaultdict(list)
        vals2 = defaultdict(list)

        first_period = None
        first_P = None
        first_L = None
        first_W = None
        for period,cmpname in sorted(period2cmp.items()):
            cmp_df = featdf.query('comparison==@cmpname')
            vals['W'].append(w:=float(cmp_df.weight.mean()))
            vals['P'].append(p:=float(cmp_df.score_mean1.mean()))
            vals['L'].append(l:=float(cmp_df.score_mean2.mean()))
            vals['P/L'].append(np.log(p / l) if l else np.nan)

            if first_period is None:
                first_period = period
                first_P = p
                first_L = l
                first_W = w
                vals2[f'P/P{first_period}'].append(0)
                vals2[f'L/L{first_period}'].append(0)
                vals2[f'W/W{first_period}'].append(0)
            else:
                vals2[f'P/P{first_period}'].append(np.log(p / first_P) if first_P else np.nan)
                vals2[f'L/L{first_period}'].append(np.log(l / first_L) if first_L else np.nan)
                vals2[f'W/W{first_period}'].append(w / first_W if first_W else np.nan)

        out_d2 = {}
        for feat,feat_vals in vals.items():
            for prd,feat_val in zip(sorted(period2cmp.keys()),feat_vals):
                out_d2[f'{feat}{prd}' if not '/' in feat else feat.replace('/',f'{prd}/')+prd] = feat_val
        for feat,feat_vals in vals2.items():
            for prd,feat_val in zip(sorted(period2cmp.keys()),feat_vals):
                key=feat.replace('/',f'{prd}/')
                key_l = key.split('/',1)
                if len(key_l)==2 and key_l[0]==key_l[1]:
                    continue
                out_d2[key] = feat_val
        
        out_vals = {f'vals_{k}': v for k,v in vals.items()}
        out_vals2 = {f'vals_{k}': v for k,v in vals2.items()}
        out_d2 = {k:float(v) for k,v in out_d2.items()}
        out_out = {**out_d, **out_d2, **out_vals, **out_vals2}
        out_ld.append(out_out)
    
    odf = pd.DataFrame(out_ld)
    odf['feat_desc'] = [FEAT2DESC.get(feat, '') for feat in odf.feature]
    odf = odf.round(3)#.sort_values('P/L',ascending=False).dropna()
    return odf[[c for c in COLS_FEATS if c in odf.columns]].set_index('feature')

    
def get_slice_feats_by_word(doc, weight_cols = ['weight','score_mean1','score_mean2']):
    df_feat_weights = get_current_feat_weights()
    df_slice_feats = extract_slice_feats(doc, return_dict=False)

    o = []
    df_slice_feats_sent = df_slice_feats.drop_duplicates(subset=['sent_i'])
    for i,row in df_slice_feats.iterrows():
        meta_d = {
            'sent_i':row['sent_i'],
            'word_i':row['word_i'],
        }
        pos = row['pos']
        deprel = row['deprel']
        out_d1 = {**meta_d, 'feature':f'pos_{pos}', 'value':1}
        out_d2 = {**meta_d, 'feature':f'deprel_{deprel}', 'value':1}
        o.extend([out_d1,out_d2])

    for i,row in df_slice_feats_sent.iterrows():
        meta_d = {
            'sent_i':row['sent_i'],
            'word_i':row['word_i'],
        }
        for c in row.index:
            if c.startswith('sent_') and c.split('_')[-1] not in ['i','id']:
                out_d = {**meta_d, 'feature':c, 'value':row[c]}
                o.append(out_d)

    odf_slice_feats = pd.DataFrame(o)
    odf_slice_feats = odf_slice_feats.merge(df_feat_weights[weight_cols], on='feature', how='left').dropna()
    odf_slice_feats['feat_type'] = odf_slice_feats.feature.str.split('_').str[0]
    return odf_slice_feats