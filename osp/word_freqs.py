from . import *

def gen_word_freqs_raw(lim=None):
    from .constants import PATH_WORDFREQS_TSV, STASH_COUNTS
    
    ndone = 0
    with open(PATH_WORDFREQS_TSV, 'a+') as f:
        for text_id, text_counts in tqdm(STASH_COUNTS.items(), total=len(STASH_COUNTS), desc='gen_word_freqs', position=0):
            total_count = sum(text_counts.values())
            for word, count in text_counts.items():
                f.write(f'{word}\t{text_id}\t{count}\n')


def iter_word_freqs_raw():
    from .constants import PATH_WORDFREQS_TSV, PATH_WORDFREQS_TSV_NL
    
    with open(PATH_WORDFREQS_TSV, 'r') as f:
        for line in tqdm(f, total=PATH_WORDFREQS_TSV_NL, desc='get_word_freqs', position=0):
            ln = line.strip()
            if not ln or not ln.count('\t') == 2:
                continue
            word, text_id, count = ln.split('\t')
            yield word, text_id, int(count)


def load_word_freqs(words):
    from .constants import STASH_WORD_FREQS
    
    dd = {}
    for w in words:
        if w in STASH_WORD_FREQS:
            dd[w] = STASH_WORD_FREQS[w]
    return pd.DataFrame(dd).rename_axis('text_id')


def get_word_freqs(words, force=False):
    from .constants import STASH_WORD_FREQS
    
    words_done_df = load_word_freqs(words)
    words_done = words_done_df.columns
    words_not_done = [w for w in tqdm(words, desc='finding word freqs', position=0) if w not in words_done]
    if words_not_done:
        word2textid2count = defaultdict(lambda: defaultdict(int))
        for k, v in words_done_df.to_dict().items():
            word2textid2count[k] = v

        for word, text_id, count in iter_word_freqs_raw():
            if word in words_not_done:
                word2textid2count[word][text_id] += count
        
        for word in words_not_done:
            text_id_count = word2textid2count[word]
            STASH_WORD_FREQS[word] = text_id_count
            word2textid2count[word] = text_id_count
        words_done_df = pd.DataFrame(word2textid2count)
    return words_done_df.fillna(0).applymap(int).astype(int)


@stashed_result
def get_text_lens():
    textid2len = Counter()
    for word, text_id, count in iter_word_freqs_raw():
        textid2len[text_id] += count
    return dict(textid2len)


def get_word_group_counts(words_list):
    df = get_word_freqs(words_list)
    return df.rename_axis('id')


def get_counts_wordset(wordset):
    data = get_word_group_counts(wordset)
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0)
    return data

