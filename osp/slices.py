from . import *

def get_text_slices(id, force=False, slice_len=1000):
    from .constants import SLICE_LEN
    from .data_loaders import get_corpus_txt, get_ok_words
    
    if slice_len is None:
        slice_len = SLICE_LEN
    
    stash = STASH_SLICES
    if not force and id in stash:
        return {int(k): v for k, v in stash[id].items()}
    txt = get_corpus_txt(id)
    slices = dict(iter_txt_slices(txt, slice_len, get_ok_words()))
    stash[id] = {int(k): v for k, v in slices.items()}
    return slices


def get_text_freqs(id, slice_len=None, force=False):
    from .constants import SLICE_LEN
    from .text_processing import count_recog_words
    
    if slice_len is None:
        slice_len = SLICE_LEN
    
    stash = STASH_FREQS_SLICES
    if not force and id in stash:
        return {int(k): v for k, v in stash[id].items()}
    slices = get_text_slices(id)
    freqs = {
        int(slice_num): dict(count_recog_words(txt, slice_len))
        for slice_num, txt in slices.items()
    }
    stash[id] = freqs
    return freqs


def iter_slice_word_freqs(df_meta=None):
    from .data_loaders import get_corpus_metadata
    
    df_meta = get_corpus_metadata() if df_meta is None else df_meta
    for id in tqdm(df_meta.index):
        for slice_num, freqs in get_text_freqs(id).items():
            yield id, slice_num, freqs


def get_words_freqs_slices(words, slice_len=None):
    from .constants import SLICE_LEN
    
    if slice_len is None:
        slice_len = SLICE_LEN
    
    stash = STASH_WORD_FREQS_SLICES
    if not any(w not in stash for w in words):
        word2text2count = {
            w: stash[w]
            for w in words
        }
    else:        
        word2text2count = defaultdict(dict)
        for id, slice_num, freqs in iter_slice_word_freqs():
            for w, c in freqs.items():
                if w in words:
                    word2text2count[w][f'{id}__{slice_num:02d}'] = c

        for k, v in tqdm(list(word2text2count.items()), desc='saving to stash'):
            stash[k] = v

        for w in words:
            if w not in stash:
                stash[w] = {}

    return pd.DataFrame(word2text2count).rename_axis('id__slice').fillna(0).applymap(int).astype(int)


def iter_txt_slices(txt, slice_len, ok_words):
    """Helper function to iterate over text slices."""
    from .text_processing import tokenize_agnostic
    
    words = []
    for token in tokenize_agnostic(txt):
        if token.strip().isalpha() and token.lower() in ok_words:
            words.append(token)
    
    slice_num = 0
    while len(words) >= slice_len:
        slice_words = words[:slice_len]
        yield slice_num, ' '.join(slice_words)
        words = words[slice_len:]
        slice_num += 1


def get_text_slice_ids(id, n_slices=10):
    from .constants import STASH_SLICES_NLP
    
    return [
        f'{id}__{slice_id:02d}'
        for slice_id in range(1,n_slices+1)
        if f'{id}__{slice_id:02d}' in STASH_SLICES_NLP
    ]

import itertools

@STASH_ALL_TEXT_SLICE_IDS.stashed_result
def get_all_text_slice_ids(lim=None):
    iterr = STASH_SLICES.items()
    iterr = itertools.islice(iterr,lim)
    iterr = tqdm(iterr,total=lim if lim is not None else len(STASH_SLICES))
    return [
        f'{text_id}__{int(slice_id):02d}'
        for text_id,d in iterr
        for slice_id in d.keys()
    ]

@cache
def get_text_id2slice_ids():
    from .features import get_parsed_slice_ids
    out = defaultdict(list)
    for k in get_parsed_slice_ids():
        out[k.split('__')[0]].append(k)
    return out


def get_balanced_slice_sample(groups_train, sample_size=None, verbose=True):
    name1, query1 = groups_train[0]
    name2, query2 = groups_train[1]

    df_meta = get_corpus_metadata()
    df_meta_g1 = df_meta.query(query1)
    df_meta_g2 = df_meta.query(query2)
    if not len(df_meta_g1) or not len(df_meta_g2):
        return pd.DataFrame()


    text2slice_ids = get_text_id2slice_ids()
    slice_ids_g1 = [slice_id for text_id in df_meta_g1.index for slice_id in text2slice_ids[text_id]]
    slice_ids_g2 = [slice_id for text_id in df_meta_g2.index for slice_id in text2slice_ids[text_id]]

    min_size = min(len(slice_ids_g1), len(slice_ids_g2))
    if sample_size is None or sample_size > min_size:
        sample_size = min_size

    slice_ids_g1 = random.sample(slice_ids_g1, sample_size)
    slice_ids_g2 = random.sample(slice_ids_g2, sample_size)

    df_slice_ids_g1 = pd.DataFrame(slice_ids_g1, columns=['slice_id']).assign(_target=name1)
    df_slice_ids_g2 = pd.DataFrame(slice_ids_g2, columns=['slice_id']).assign(_target=name2)
    
    df_slice_ids = pd.concat([df_slice_ids_g1, df_slice_ids_g2])
    df_slice_ids['text_id'] = df_slice_ids.slice_id.str.split('__').str[0]
    df_slice_ids = df_slice_ids[['text_id', 'slice_id','_target']]
    # df_meta is indexed by text_id; merge on index to preserve metadata columns.
    odf = df_meta[DF_PREDS_METADATA_COLS].merge(
        df_slice_ids, left_index=True, right_on='text_id', how='right'
    )
    
    return odf
    

TARGET_NICKNAMES = {'Philosophy':'Phil', 'Literature':'Lit'}

def get_slice_info_df_preds(slice_ids):
    from .classify import get_current_pred_probs
    if isinstance(slice_ids, str): slice_ids = [slice_ids]
    df_all_preds = get_current_pred_probs()
    df_preds = df_all_preds.loc[slice_ids]
    targets = df_preds['target'].unique()
    comparisons = df_preds['comparison'].unique()
    predict_types = df_preds['predict_type'].unique()
    
    def describe_probs_target(target, dfx):
        probf = f'prob_{target}'
        out_d = {}
        out_d2 = {}
        out_d3 = {}
        o = []

        avg_prob = dfx[probf].mean()
        num_correct = len(dfx.query(f'{probf}>=0.5'))
        out_d['prob'] = avg_prob
        # out_d['perc_correct'] = num_correct / len(dfx)

        tname = TARGET_NICKNAMES.get(target, target)
        for cmp in sorted(comparisons):
            cmpname = cmp.split('-')[0]
            dfx_cmp = dfx[dfx['comparison']==cmp]
            avg_cmp_prob = dfx_cmp[probf].mean()
            num_cmp_correct = len(dfx_cmp.query(f'{probf}>=0.5'))
            out_d[f'prob_{cmpname}'] = avg_cmp_prob
            # out_d2[f'perc_correct_{cmpname}'] = num_cmp_correct / len(dfx_cmp)
            for pt in predict_types:
                dfx_cmp_pt = dfx_cmp[dfx_cmp['predict_type']==pt]
                support = dfx_cmp_pt.iloc[0]['support']
                num_runs = dfx_cmp_pt.iloc[0]['num_runs']
                avg_cmp_pt_prob = dfx_cmp_pt[probf].mean()
                num_cmp_pt_correct = len(dfx_cmp_pt.query(f'{probf}>=0.5'))
                perc_cmp_pt_correct = num_cmp_pt_correct / len(dfx_cmp_pt)
                out_d3[f'prob_{cmpname}_{pt}'] = avg_cmp_pt_prob
                outx = {
                    'target':target,
                    'comparison':cmp,
                    'predict_type':pt,
                    'prob_correct':avg_cmp_pt_prob,
                    'perc_correct':perc_cmp_pt_correct,
                    'num_correct':num_cmp_pt_correct,
                    'num_runs':num_runs,
                    'support':support,
                    'num_samples':len(dfx_cmp_pt),
                }
                o.append(outx)
        return o

    ld = []
    for target,target_df in df_preds.groupby('target'):
        out_l = describe_probs_target(target, target_df)
        ld.extend(out_l)
    return pd.DataFrame(ld).set_index('target')




def describe_slice_probs(slice_ids, width=90, para='\n\n'):
    import textwrap
    out=[]
    dfx = get_slice_info_df_preds(slice_ids)
    dfx_q = dfx.select_dtypes(include=['number'])
    median_cols = ['support','num_correct']
    sum_cols = ['num_samples']
    avg_cols = [c for c in dfx_q.columns if c not in set(median_cols+sum_cols)]
    dfx_target = dfx.groupby('target').agg(
        {
            **{c:'mean' for c in avg_cols},
            **{c:'sum' for c in sum_cols},
            **{c:'median' for c in median_cols},
        }
    )
    
    # target_cols = prob_correct	perc_correct	num_correct	support	num_samples

    num_cmps = dfx.comparison.nunique()
    for target,row in dfx_target.iterrows():
        out.append(f'''- **{target}** (n={int(row.num_samples):,}) was predicted successfully **{row.perc_correct*100:.1f}%** of the time (across {int(row.num_runs):,} model runs each of {num_cmps} comparisons), with an average confidence of **{row.prob_correct*100:.1f}%**. ''')
    
    for target,row in dfx_target.iterrows():
        target_comparison_df = dfx.query('target==@target')
        target_comparison_df.sort_values(['prob_correct','perc_correct'],ascending=False,inplace=True)
        best_cmp = target_comparison_df.iloc[0]
        worst_cmp = target_comparison_df.iloc[-1]
        support = int(best_cmp.support)
        out.append(f'''- Each run of the model had **{support:,}** samples divided evenly.''')
        out.append(f'''- The best performing comparison was **{best_cmp.comparison.split(" ")[0]}** ({best_cmp.prob_correct*100:.1f}% confidence),  with a success rate of **{best_cmp.perc_correct*100:.1f}%**.''')
        out.append(f'''- The worst performing comparison was **{worst_cmp.comparison.split(" ")[0]}** ({worst_cmp.prob_correct*100:.1f}% confidence), with a success rate of **{worst_cmp.perc_correct*100:.1f}%**.''' )
        out
        break

    return para.join(textwrap.fill(x, width=width) for x in out)
