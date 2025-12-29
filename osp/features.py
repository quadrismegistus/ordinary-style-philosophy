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


# def get_slice_feats(id):
#     from .constants import STASH_POS_COUNTS, STASH_SENT_FEAT_COUNTS, BAD_SLICE_FEATS
    
#     posfeat_counts = STASH_POS_COUNTS.get(id, {})
#     deprel_counts = {k: v for k, v in posfeat_counts.items() if k == k.lower()}
#     pos_counts = {k: v for k, v in posfeat_counts.items() if k == k.upper()}

#     sent_feat_counts = STASH_SENT_FEAT_COUNTS.get(id, {})
#     sent_feat_counts_df = pd.DataFrame(sent_feat_counts)

#     sent_sums = sent_feat_counts_df.sum(numeric_only=True)
#     num_words = sent_sums['num_words']
#     num_sents = len(sent_feat_counts)
#     sent_sums['num_sents'] = num_sents
#     total_clauses = sent_sums['num_independent_clauses'] + sent_sums['num_dependent_clauses']
#     out_sent_d = {
#         'avg_num_sents': num_sents / num_words * 1000,
#         'avg_num_words_per_sent': num_words / num_sents,
#         'avg_height': sent_sums['height'] / num_sents,
#         'perc_dependent_clauses': sent_sums['num_dependent_clauses'] / total_clauses,
#         'perc_independent_clauses': sent_sums['num_independent_clauses'] / total_clauses,
#         'avg_num_words_per_clause': num_words / total_clauses,
#     }
#     out = {
#         **{f'sent_{k}': v for k, v in out_sent_d.items()},
#         **{f'pos_{k}': v for k, v in pos_counts.items() if k and k[0].isalpha()},
#         **{f'deprel_{k}': v for k, v in deprel_counts.items() if k and k[0].isalpha()},
#     }
#     return {k: v for k, v in sorted(out.items()) if not any(k.startswith(b) for b in BAD_SLICE_FEATS)}


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
def get_all_feats(normalize=NORMALIZE_FEAT_DATA, feat_types=None, **kwargs):
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
@STASH_ALL_FEATS.stashed_result
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


def get_balanced_cv_data(groups_train, target_col='discipline', balance=True, normalize=NORMALIZE_CLASSIFY_DATA, feat_types=CV_FEAT_TYPES, **kwargs):
    """
    Returns a slice-level feature matrix with two partitions:
    - _type=="CV": balanced sample of slice_ids for the two groups
    - _type=="Unseen": all remaining slices

    Sampling is now driven by osp.slices.get_balanced_slice_sample() (metadata-level),
    then features are pulled for the sampled slice_ids.
    """
    from .slices import get_balanced_slice_sample, get_text_id2slice_ids

    sample_size = kwargs.pop("sample_size", None)

    df_meta = get_corpus_metadata()
    name1, query1 = groups_train[0]
    name2, query2 = groups_train[1]

    df_scores_all = get_all_feats(normalize=normalize, feat_types=feat_types, **kwargs).fillna(0)

    # --- Choose CV slice ids ---
    slice_ids_g1 = []
    slice_ids_g2 = []

    if balance:
        df_slice_sample = get_balanced_slice_sample(
            groups_train, sample_size=sample_size, verbose=False
        )
        if not df_slice_sample.empty and "slice_id" in df_slice_sample.columns:
            slice_ids_g1 = (
                df_slice_sample.query("_target==@name1")["slice_id"].astype(str).tolist()
            )
            slice_ids_g2 = (
                df_slice_sample.query("_target==@name2")["slice_id"].astype(str).tolist()
            )
    else:
        df_meta1 = df_meta.query(query1)
        df_meta2 = df_meta.query(query2)
        text2slice_ids = get_text_id2slice_ids()
        slice_ids_g1 = [
            slice_id
            for text_id in df_meta1.index
            for slice_id in text2slice_ids.get(text_id, [])
        ]
        slice_ids_g2 = [
            slice_id
            for text_id in df_meta2.index
            for slice_id in text2slice_ids.get(text_id, [])
        ]
        if sample_size is not None:
            sample_size = int(sample_size)
            if sample_size > 0:
                slice_ids_g1 = (
                    random.sample(slice_ids_g1, min(sample_size, len(slice_ids_g1)))
                    if slice_ids_g1
                    else []
                )
                slice_ids_g2 = (
                    random.sample(slice_ids_g2, min(sample_size, len(slice_ids_g2)))
                    if slice_ids_g2
                    else []
                )

    # Keep only slice ids that we actually have features for
    idx_all = set(df_scores_all.index.astype(str))
    slice_ids_g1 = [sid for sid in slice_ids_g1 if sid in idx_all]
    slice_ids_g2 = [sid for sid in slice_ids_g2 if sid in idx_all]

    df_scores1 = df_scores_all.loc[slice_ids_g1].copy() if slice_ids_g1 else df_scores_all.iloc[0:0].copy()
    df_scores2 = df_scores_all.loc[slice_ids_g2].copy() if slice_ids_g2 else df_scores_all.iloc[0:0].copy()

    # Attach target labels (discipline, etc.) per slice id
    for dfx in [df_scores_all, df_scores1, df_scores2]:
        dfx["_target"] = [get_text_metadata(i).get(target_col, "") for i in dfx.index]
        dfx.dropna(subset=["_target"], inplace=True)

    # Attach CV group labels
    df_scores1 = df_scores1.assign(_group=name1)
    df_scores2 = df_scores2.assign(_group=name2)

    df_scores_cv = pd.concat([df_scores1, df_scores2]).assign(_type="CV")
    df_scores_rest = (
        df_scores_all.drop(df_scores_cv.index, errors="ignore")
        .assign(_type="Unseen", _group="Unseen")
    )
    return pd.concat([df_scores_cv, df_scores_rest])


@cache
def get_current_feat_weights(*args,group_by=('feature',), **kwargs):
    from .classify import get_preds_feats
    df_preds, df_feats, d_models = get_preds_feats(*args, **kwargs)
    odf = df_feats.groupby(list(group_by)).mean(numeric_only=True)
    odf['weight_z'] = (odf['weight'] - odf['weight'].mean()) / odf['weight'].std()
    return odf

@cache
@STASH_PARSED_SLICE_IDS.stashed_result
def get_parsed_slice_ids():
    return list(STASH_SLICES_NLP.keys())

def gen_all_slice_feats(force=False, num_proc=1, lim=None):
    ids = get_parsed_slice_ids()
    if not force:
        ids_done = set(STASH_SLICE_FEATS.keys())
        ids = [id for id in ids if id not in ids_done]
        if len(ids) == 0:
            return
    
    ids = ids[:lim] if lim else ids
    
    if num_proc < 2:
        for id in tqdm(ids):
            docstr = STASH_SLICES_NLP.get(id, None)
            res = extract_slice_feats(docstr)
            STASH_SLICE_FEATS[id] = res
        return
    
    def iter_objs():
        for id in ids:
            docstr = STASH_SLICES_NLP.get(id, None)
            yield docstr
    
    with mp.Pool(num_proc) as p:
        iterr = p.imap(_do_gen_all_slice_feats, iter_objs(), chunksize=1)
        iterr = zip(ids, iterr)
        iterr = tqdm(iterr, total=len(ids))
        for id, res in iterr:
            STASH_SLICE_FEATS[id] = res

def _do_gen_all_slice_feats(docstr):
    return extract_slice_feats(docstr)

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
            val_P = float(cmp_df.mean_Philosophy.mean())
            vals_P.append(val_P)
            out_d2[cmp_key_P] = val_P
            
            cmp_key_L = f'L{cmp_prd}'
            val_L = float(cmp_df.mean_Literature.mean())
            vals_L.append(val_L)
            # out_d2[cmp_key_L] = val_L
        out_d2['P2000/P1925'] = np.log(out_d2['P2000']/out_d2['P1925']) if out_d2['P1925'] else np.nan
        out_d['P'] = np.mean(vals_P)
        out_d['L'] = np.mean(vals_L)
        out_d['P/L'] = np.log(out_d['P'] / out_d['L'])
        out_ld.append({**out_d, **out_d2})
    
    odf = pd.DataFrame(out_ld)
    odf = odf.round(2).sort_values('P/L',ascending=False).fillna(0)
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
            vals['P'].append(p:=float(cmp_df.mean_Philosophy.mean()))
            vals['L'].append(l:=float(cmp_df.mean_Literature.mean()))
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

    
def get_slice_feats_by_word(doc, weight_cols = ['weight','mean_Philosophy','mean_Literature']):
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



def extract_pos_feats_sent(sent):
    counter = Counter()
    for word in sent.words:
        pos = word.xpos
        if pos not in BAD_POS and pos and pos[0].isalpha():
            counter[pos] += 1
    return counter

def extract_pos_feats(doc):
    counter = Counter()
    for sent in doc.sentences:
        sent_pos_feats = extract_pos_feats_sent(sent)
        for pos,count in sent_pos_feats.items():
            counter[pos] += count
    return counter


def extract_deprel_feats_sent(sent):
    counter = Counter()
    for word in sent.words:
        deprel = word.deprel
        if deprel not in BAD_DEPREL:
            counter[deprel] += 1
    return counter

def extract_deprel_feats(doc):
    counter = Counter()
    for sent in doc.sentences:
        sent_deprel_feats = extract_deprel_feats_sent(sent)
        for deprel,count in sent_deprel_feats.items():
            counter[deprel] += count
    return counter


def extract_phrase_feats_sent(sent, top_n=MAX_FEATSET_FEATS):
    from .sentences import get_phrase_counts_sent
    return get_phrase_counts_sent(sent)

def extract_phrase_feats(doc, top_n=MAX_FEATSET_FEATS):
    from .sentences import get_phrase_counts_doc
    return get_phrase_counts_doc(doc)

def extract_ttr_feats(doc, within_pos = ["NOUN","ADJ","VERB","ADV"], max_tokens=1000,normalize=True):
    counter = Counter()
    pos2counter = defaultdict(Counter)
    ntok=-1
    for sent in doc.sentences:
        for word in sent.words:
            ntok += 1
            pos = word.pos
            #if pos and pos[0].isalpha() and pos != 'PUNCT':
            if pos not in within_pos:
                pos = 'OTHER'
            tok = word.text.lower()
            pos2counter[pos][tok] += 1
            counter[tok] += 1
            if ntok > max_tokens:
                break
        if ntok > max_tokens:   # this is a hack to stop the loop early
            break
    
    def d2ttr(d):
        num_types = len(d)
        num_tokens = sum(d.values())
        if not normalize:
            return num_types
        return num_types / num_tokens if num_tokens > 0 else np.nan

    out_d = {
        'mean': d2ttr(counter),
        **{
            f'{pos}': d2ttr(pos2counter[pos])
            for pos in pos2counter
        }
    }
    return out_d




def extract_syntax_feats_sent(sent, incl_formula=False, max_n_clauses=5):
    from .sentences import get_syntax_df
    from .nlp_utils import get_clause_form

    df = get_syntax_df(sent)
    df = df[df.word_deprel!='punct']
    df_clause = df.drop_duplicates('clause_i')
    num_ic = len(df_clause.query('clause_type!="sub"'))
    num_dc = len(df_clause.query('clause_type=="sub"'))

    # clause_form = get_clause_form(sent)
    # num_ic = clause_form.count('IC')
    # num_dc = clause_form.count('DC')
    num_c = df.clause_id.nunique()
    num_c_star = df.clause_i.nunique()
    # num_c = len(clause_form.split('(')) - 1
    # num_c_star = len(clause_form.split('('))

    
    if num_ic < 1: num_ic = 1
    out_d = {}
    out_d['IC']=num_ic
    out_d['DC']=num_dc
    out_d['C']=num_c
    out_d['C*']=num_c_star

    df_words = df#[df.word_deprel!='punct']
    out_d['DCw']=len(df_words[df_words.clause_type=='sub'])
    out_d['ICw']=len(df_words[df_words.clause_type!='sub'])

    avg_s = df.max(numeric_only=True).round(0)
    out_d['Wd'] = int(avg_s['word_depth'])
    out_d['Cd'] = int(avg_s['clause_depth']) + 1

    if incl_formula and num_c <= max_n_clauses:
        out_d[clause_form]=1
        # out_d[f'{clause_form}w']=len(df)
    
    return out_d

def extract_syntax_feats(doc):
    df = pd.DataFrame([extract_syntax_feats_sent(sent) for sent in doc.sentences])
    return {k:int(v) for k,v in df.sum(numeric_only=True).items()}


def extract_slice_feats(docstr, normalize=NORMALIZE_FEAT_DATA):
    doc = stanza.Document.from_serialized(docstr) if isinstance(docstr, (str,bytes)) else docstr
    if doc is None:
        return {}

    feats_d = {}
    feats_d['pos'] = extract_pos_feats(doc)
    feats_d['deprel'] = extract_deprel_feats(doc)
    feats_d['phrase'] = extract_phrase_feats(doc)
    feats_d['ttr'] = extract_ttr_feats(doc,normalize=normalize)
    feats_d['sent'] = extract_syntax_feats(doc)
    
    out_d = {
        **{'pos_'+k: v for k,v in feats_d['pos'].items()},
        **{'deprel_'+k: v for k,v in feats_d['deprel'].items()},
        **{'phrase_'+k: v for k,v in feats_d['phrase'].items()},
        **{'ttr_'+k: v for k,v in feats_d['ttr'].items()},
        **{'sent_'+k: v for k,v in extract_syntax_feats(doc).items()},
    }

    if normalize:
        num_words = sum(len(sent.words) for sent in doc.sentences)
        for k,v in out_d.items():
            out_d[k] = v / num_words * 1000
    return out_d





def get_diff_rows(df_smpl_feats,group1='Philosophy',group2='Literature'):
    new_rows = []
    for feat,feat_df in df_smpl_feats.groupby('feat'):
        feat2means = feat_df.groupby('target').mean(numeric_only=True)
        z1 = feat2means.loc[group1,'z']
        z2 = feat2means.loc[group2,'z']
        z_diff = z1 - z2
        raw1 = feat2means.loc[group1,'raw']
        raw2 = feat2means.loc[group2,'raw']
        raw_diff = raw1 - raw2
        new_row = {
            'feat': feat,
            'target': f'{group1} - {group2}',
            'z': z_diff,
            'raw': raw_diff,
        }
        new_rows.append(new_row)
    odf = pd.DataFrame(new_rows)
    return odf
    


def get_balanced_slice_sample_feats(groups_train, df_smpl=None, sample_size=None, balance=False, with_diff_rows=True):
    name1, query1 = groups_train[0]
    name2, query2 = groups_train[1]

    df_smpl = get_balanced_slice_sample(
        groups_train, 
        sample_size=sample_size, 
        balance=balance
    ) if df_smpl is None else df_smpl

    print(f"Loading features for {len(df_smpl)} slices...")
    df_all_feats_z = get_all_feats(normalize=True)
    df_all_feats_raw = get_all_feats(normalize=False)

    print(f"Filtering valid slices...")
    valid_ids = set(df_smpl.slice_id) & set(df_all_feats_z.index)
    df_smpl_valid = df_smpl[df_smpl.slice_id.isin(valid_ids)].set_index('slice_id')
    df_smpl_feats_z = df_all_feats_z.loc[df_smpl_valid.index].copy()
    df_smpl_feats_z['_target'] = df_smpl_valid['_target']
    
    df_smpl_feats_raw = df_all_feats_raw.loc[df_smpl_valid.index].copy()

    print(f"Aggregating feature statistics...")
    # Optimized approach using melt
    feat_cols = [c for c in df_smpl_feats_z.columns if c and c[0] != '_']
    
    df_z_melt = df_smpl_feats_z.reset_index().melt(
        id_vars=['slice_id', '_target'], 
        value_vars=feat_cols,
        var_name='feat', 
        value_name='z'
    )
    df_raw_melt = df_smpl_feats_raw.reset_index().melt(
        id_vars=['slice_id'], 
        value_vars=feat_cols,
        var_name='feat', 
        value_name='raw'
    )
    
    odf = df_z_melt.merge(df_raw_melt, on=['slice_id', 'feat'])
    odf = odf.rename(columns={'_target': 'target'})
    odf = odf.groupby(['feat','target']).mean(numeric_only=True).reset_index()

    print(f"Computing difference rows...")
    df_diff_rows = get_diff_rows(odf, group1=name1, group2=name2)
    feat2zdiff = df_diff_rows.groupby('feat').z.mean()
    feat2rawdiff = df_diff_rows.groupby('feat').raw.mean()
    # name1,q1 = groups_train[0]
    # name2,q2 = groups_train[1]
    # feat2z1 = odf[odf.target == name1].groupby('feat').z.mean()
    # feat2z2 = odf[odf.target == name2].groupby('feat').z.mean()

    # feat2rank1 = feat2z1.rank(method='dense', ascending=False).apply(int)
    # feat2rank2 = feat2z2.rank(method='dense', ascending=True).apply(int)
    # feat2diff_rank = df_diff_rows.groupby('feat').feat_diff_rank.first().to_dict()
    # odf['feat_diff_rank'] = odf['feat'].map(feat2diff_rank)
    odf = pd.concat([
        odf.set_index(['feat','target']), 
        df_diff_rows.set_index(['feat','target'])
    ]).reset_index()
    odf['z_abs'] = odf['z'].abs()
    
    odf['z_diff'] = odf['feat'].map(feat2zdiff)
    odf['z_diff_abs'] = odf['z_diff'].abs()
    
    odf['raw_diff'] = odf['feat'].map(feat2rawdiff)
    odf['raw_diff_abs'] = odf['raw_diff'].abs()

    odf['feat_rank'] = odf['z_diff_abs'].rank(method='dense', ascending=False).apply(int)


    odf['feat_rank1'] = odf['z_diff'].rank(method='dense', ascending=False).apply(int)
    odf['feat_rank2'] = odf['z_diff'].rank(method='dense', ascending=True).apply(int)
    
    # feat2max_abs_z = odf.groupby('feat').z_abs.max()
    # ranked_feat2max_abs_z = feat2max_abs_z.rank(method='dense', ascending=False).apply(int)    
    # odf['feat_rank'] = odf['feat'].map(ranked_feat2max_abs_z)
    return odf.sort_values(['feat_rank', 'z'],ascending=[True, False]).set_index(['feat','target'])

# def get_slices_feats(slice_ids, normalize=True):
#     from .slices import get_slice_ids
#     slice_ids = get_slice_ids(slice_ids)
#     df_all_feats = get_all_feats(normalize=normalize)
#     valid_ids = list(set(slice_ids) & set(df_all_feats.index))
#     df_slices_feats = df_all_feats.loc[valid_ids]
#     return df_slices_feats

def get_slices_feats(slice_ids):
    out = []
    for slice_id in slice_ids:
        res_d = STASH_SLICE_FEATS.get(slice_id, None)
        if res_d:
            out.append(res_d)
    return pd.DataFrame(out, index=slice_ids).rename_axis('slice_id')

# def get_feat_egs(df_feats, feats, n=1):
#     out_feat2egs = {}
#     if df_feats is None or df_feats.empty:
#         print("get_feat_egs: df_feats is empty")
#         return out_feat2egs
        
#     print(f"get_feat_egs: searching for examples for {len(feats)} features in {len(df_feats)} slices")
    
#     # Pre-filter features present in df_feats
#     valid_feats = [f for f in feats if f in df_feats.columns]
    
#     for feat in valid_feats:
#         feat_egs = []
#         top_slices = df_feats.sort_values(by=feat, ascending=False)
        
#         # Look at top slices for this feature
#         slices_checked = 0
#         for slice_id in top_slices.index:
#             # Check if feature has a value in this slice
#             if top_slices.loc[slice_id, feat] <= 0:
#                 break
            
#             slices_checked += 1
#             if slices_checked > 100: # Don't search forever
#                 break
                
#             egs_feat2word2eg = STASH_FEAT2WORD2EG.get(slice_id, {})
#             if not egs_feat2word2eg:
#                 # Try getting it directly if .get() is weird
#                 try:
#                     egs_feat2word2eg = STASH_FEAT2WORD2EG[slice_id]
#                 except:
#                     continue
            
#             if not egs_feat2word2eg:
#                 continue
            
#             # The key in the stash might be 'NOUN' while the feature is 'pos_NOUN'
#             word2eg = egs_feat2word2eg.get(feat)
#             if not word2eg:
#                 feat_parts = feat.split('_', 1)
#                 if len(feat_parts) > 1:
#                     word2eg = egs_feat2word2eg.get(feat_parts[1])
            
#             if not word2eg:
#                 continue
            
#             # Found examples for this feature in this slice
#             words = [w for w in word2eg.keys() if word2eg[w]]
#             if not words:
#                 continue
                
#             word = random.choice(words)
#             eg = word2eg[word]
            
#             # Clean up example text
#             eg_text = str(eg).strip()
#             # If example is too long, truncate it
#             if len(eg_text) > 250:
#                 eg_text = eg_text[:250] + "..."
                
#             # Add metadata to the link
#             meta = get_text_metadata(slice_id)
#             title = meta.get('title', slice_id)
#             author = meta.get('author', 'Unknown')
            
#             feat_egs.append(f'* **{word}**: “{eg_text}” — *{author}, [{title}](/Passages?slice_id={slice_id})*')
            
#             if n and len(feat_egs) >= n:
#                 out_feat2egs[feat] = feat_egs
#                 break
    
#     print(f"get_feat_egs: found examples for {len(out_feat2egs)} features")
#     return out_feat2egs        

# @STASH_FEAT_GROUP_EGS.stashed_result
# def get_feat_group_egs(feats, groups=None, n=None, normalize=False):
#     from .slices import get_slice_ids
#     if isinstance(feats, str):
#         feats = [feats]
#     if groups is None:
#         groups = COMPARISONS[0]
    
#     name1,query1 = groups[0]
#     name2,query2 = groups[1]

#     slice_ids1 = get_slice_ids(query1)
#     slice_ids2 = get_slice_ids(query2)

#     # Use un-normalized feature counts by default so we can reliably find "top slices"
#     # with positive values (needed for example extraction).
#     df_feats1 = get_slices_feats(slice_ids1, normalize=normalize)
#     df_feats2 = get_slices_feats(slice_ids2, normalize=normalize)

#     return get_feat_egs(df_feats1, feats, n), get_feat_egs(df_feats2, feats, n)


def get_slices_feats(slice_ids):
    out = []
    for slice_id in slice_ids:
        res_d = STASH_SLICE_FEATS.get(slice_id, None)
        if res_d:
            out.append(res_d)
    return pd.DataFrame(out, index=slice_ids).rename_axis('slice_id')

def get_feat_egs(df_feats, feats, num_egs=10):
    out_feat2egs = []
    for feat in feats:
        feat_egs = []
        done_words = set()
        if feat not in df_feats.columns:
            continue
        top_slices = df_feats.sort_values(by=feat, ascending=False)
        for slice_id in top_slices.index:
            egs_feat2word2eg = STASH_FEAT2WORD2EG.get(slice_id, {})
            if not egs_feat2word2eg:
                continue
            word2eg = egs_feat2word2eg.get(feat, egs_feat2word2eg.get(feat.split('_',1)[-1], {}))
            word2eg = {k:v for k,v in word2eg.items() if k not in done_words}
            if not word2eg:
                continue
            
            word = random.choice(list(word2eg.keys()))
            eg = word2eg[word]
            feat_egs.append({'feat': feat, 'word': word, 'eg': eg, 'slice_id': slice_id})
            done_words.add(word)
            if len(done_words)>=num_egs:
                out_feat2egs.extend(feat_egs)
                break
    return pd.DataFrame(out_feat2egs)

def get_feat_group_egs(feats, groups=None, num_egs=10):
    # from .slices import get_slice_ids
    if isinstance(feats, str):
        feats = [feats]
    if groups is None:
        groups = COMPARISONS[0]
    
    name1,query1 = groups[0]
    name2,query2 = groups[1]
    
    slice_ids1 = get_slice_ids(query1)
    slice_ids2 = get_slice_ids(query2)
    
    df_feats1 = get_slices_feats(slice_ids1)
    df_feats2 = get_slices_feats(slice_ids2)

    odf1 = get_feat_egs(df_feats1, feats, num_egs=num_egs)
    odf2 = get_feat_egs(df_feats2, feats, num_egs=num_egs)

    return pd.concat([odf1.assign(group=name1), odf2.assign(group=name2)])


# @STASH_FEAT_EG_CACHE.stashed_result
def get_slice_feat_egs(slice_ids=None, feats=None, num_egs=10, max_slices=1000):
    if slice_ids is None:
        slice_ids = get_parsed_slice_ids()
    if isinstance(feats, str):
        feats = [feats]
    elif not feats:
        feats = [x for x in FEAT2DESC.keys() if x.split('_')[0] in {'pos','deprel'} and x not in BAD_SLICE_FEATS]
    
    random.shuffle(slice_ids)
    egs = {feat:[] for feat in feats}
    iterr = slice_ids[:max_slices]
    for slice_id in iterr:
        min_feat_len = min(len(egs[f]) for f in egs)
        if min_feat_len >= num_egs:
            break
        
        res_ld = STASH_FEAT_EXAMPLES2.get(slice_id, None)
        if isinstance(res_ld, list) and res_ld:
            for d in res_ld:
                feat = d['feature']
                if (not feats or feat in set(feats)) and len(egs[feat]) < num_egs:
                    d['slice_id'] = slice_id
                    egs[feat].append(d)
    return pd.DataFrame([vx for vl in egs.values() for vx in vl]).sample(frac=1)