from . import *
import multiprocessing as mp


def get_eg_from_word(word, window=10, window_chars=100, html=True):
    word_i = word.id - 1
    sent = word.sent
    words = sent.words
    tokens = sent.tokens
    try:
        prev_tokens = tokens[:word_i]
        this_token = tokens[word_i]
        next_tokens = tokens[word_i+1:]
    except:
        return ""
    
    if window:
        window_rad = window//2
        words = words[word_i-window_rad:word_i+window_rad+1]
        tokens = tokens[word_i-window_rad:word_i+window_rad+1]
    if html:
        return get_sent_html(words,show_labels=True,highlight_word_id=word.id)
    else:
        
        midstr = f'*{word.text.upper()}*{this_token.spaces_after}'

        prev_text = detokenize_stanza(prev_tokens)[-40:]
        next_text = detokenize_stanza(next_tokens)[:60 - len(midstr)]
        return f'{prev_text}{midstr}{next_text}'


def extract_feat_examples(doc, max_per_feat=1, window=10, window_chars=100):
    sents = doc.sentences
    random.shuffle(sents)
    egs = defaultdict(list)
    feat2word2count = defaultdict(Counter)
    for sent in sents:
        for word in sent.words:
            for feat_type in ['deprel','pos']:
                feat_val = word.deprel if feat_type == 'deprel' else word.xpos
                if feat_val:
                    feat = f'{feat_type}_{feat_val}'
                    egs[feat].append(word)
                    feat2word2count[feat][word.text.lower()] += 1
    
    o = []
    for feat,words in egs.items():
        if len(words) > max_per_feat:
            words = random.sample(words, max_per_feat)
        else:
            random.shuffle(words)
        
        for word in words:
            word_str = word.text.lower()
            count = feat2word2count[feat][word_str]
            total = feat2word2count[feat].total()
            perc = round(count/total*100, 1) if total else 0
            odx = {
                'feature': feat,
                'word': word_str,
                'count': count,
                'perc': perc,
                'eg_text': get_eg_from_word(word, html=False, window=window, window_chars=window_chars),
                'eg_html': get_eg_from_word(word, html=True, window=window, window_chars=window_chars),
                'sent_id':word.sent.id,
                'word_id':word.id,
            }
            o.append(odx)
    return o
    



def _do_gen_feat_examples(args):
    docstr,max_per_feat,window = args
    doc = stanza.Document.from_serialized(docstr)
    return extract_feat_examples(doc, max_per_feat=max_per_feat, window=window)

def gen_feat_examples(max_per_feat=3, window=10, force=False, num_proc=1, lim=None, batch_size=100):
    ids_done = set(STASH_FEAT_EXAMPLES2.keys())
    ids_todo = set(get_parsed_slice_ids()) - ids_done if not force else get_parsed_slice_ids()
    ids_todo = list(ids_todo)
    random.shuffle(ids_todo)
    ids_todo = ids_todo[:lim]

    if num_proc < 2:
        for id in tqdm(ids_todo):
            _do_gen_feat_examples((STASH_SLICES_NLP[id], max_per_feat, window))
        return
    
    def iter_objs():
        for id in ids_todo:
            docstr = STASH_SLICES_NLP.get(id,None)
            yield (docstr, max_per_feat, window)
    
    with mp.Pool(num_proc) as p:
        iterr = p.imap(_do_gen_feat_examples, iter_objs(), chunksize=1)
        iterr = zip(ids_todo, iterr)
        iterr = tqdm(iterr, total=len(ids_todo))
        for id,data in iterr:
            STASH_FEAT_EXAMPLES2[id] = data



def pad_starred_keyword(line, left_pad=40,keep_asterisks=True):
    m = re.search(r"\*[A-Z][A-Z0-9_-]*\*", line)
    if not m:
        return line
    start, end = m.span()

    left_text = line[:start]
    right_text = line[end:]
    word_text = line[start:end]
    word_text_unstarred = word_text[1:-1]
    
    # get to left of keyword, padding if nec
    left_pad = left_pad - start
    if left_pad > 0:
        left_text = ' '*left_pad + left_text
    
    # trim if nec
    word = word_text_unstarred if not keep_asterisks else word_text
    return left_text + word + right_text
    
def center_starred_keyword(line, window=50, window_left=20, window_right=40, keep_asterisks=True):
    line = line.replace("\n", " ").replace("\t", " ")
    window_left = window // 2 if not window_left else window_left
    window_right = window // 2 if not window_right else window_right
    m = re.search(r"\*[A-Z][A-Z0-9_-]*\*", line)
    if not m:
        return line
    start, end = m.span()
    left_text = line[start-window_left if start-window_left > 0 else 0:start]
    right_text = line[end:end+window_right]
    word_text = line[start:end]
    word_text_unstarred = word_text[1:-1]
    
    if len(left_text) < window_left:
        needs_padding = window_left - len(left_text)
        left_text = ' '*needs_padding + left_text
    if len(right_text) < window_right:
        needs_padding = window_right - len(right_text)
        right_text = right_text + ' '*needs_padding

    word = word_text_unstarred if not keep_asterisks else word_text
    return left_text + word + right_text

def center_starred_keywords(lines, **kwargs):
    lines = [center_starred_keyword(line, **kwargs) for line in lines]
    # pad right side to maximum
    if not lines: return lines
    max_len = max(len(line) for line in lines)
    lines = [line + ' '*(max_len - len(line)) for line in lines]
    return lines