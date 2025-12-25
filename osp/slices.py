from . import *

def get_text_slices(id, force=False, slice_len=1000):
    from .constants import SLICE_LEN
    from .data_loaders import get_corpus_txt, get_ok_words
    
    if slice_len is None:
        slice_len = SLICE_LEN
    
    stash = HashStash(f'osp_slices_{slice_len}')
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
    
    stash = HashStash(f'osp_freqs_slices_{slice_len}')
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
    
    stash = HashStash(f'osp_word_freqs_slices_{slice_len}')
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

