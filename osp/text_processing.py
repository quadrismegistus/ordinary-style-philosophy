from . import *

def dehyphenate(text):
    """
    Removes hyphenation at line breaks, e.g. 'artist-\nically' or 'artist- ically' -> 'artistically'
    Handles both space and newline after hyphens.
    """
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'-\s+', '', text)
    # if ? is between two letters, replace it with em-dash
    text = re.sub(r'(?<=[a-zA-Z])\?(?=[a-zA-Z])', '—', text)
    return text


def filter_pmla_pages(article):
    newpages = []
    for page in article:
        # remove leading all-caps words (likely running heads) from the page
        words = page.split()
        i = 0
        for w in words:
            # Match word if it's all caps (with possible initial punctuation and optional trailing punctuation/numbers)
            if w.isdigit() or re.match(r"^[A-Z]+[.,:;?\-–—'\d]*$", w):
                i += 1
            else:
                break
        newpages.append(' '.join(words[i:]))
    return dehyphenate(' '.join(newpages))


def tokenize(txt):
    tokens = txt.lower().split()
    cleaned = []
    for t in tokens:
        cleaned_token = t.lstrip("".join([c for c in t if not c.isalpha()])).rstrip("".join([c for c in t if not c.isalpha()]))
        if cleaned_token:
            cleaned.append(cleaned_token)
    return cleaned


def tokenize_agnostic(txt: str):
    """Tokenize text in a language-agnostic way.

    Args:
        txt: The input text.

    Returns:
        A list of tokens.
    """
    return re.findall(r"[\w']+|[.,!?; -—–'\n]", txt)


def remove_left_right_punct(text):
    """
    Removes leading and trailing punctuation from a string.
    """
    return text.strip(string.punctuation)


def is_stopword(word):
    return len(word) < 4


def get_recog_words(txt):
    from .data_loaders import get_ok_words
    return [
        w.lower() for w in tokenize_agnostic(txt)
        if w.strip().isalpha() and w.lower() in get_ok_words()
    ]


def count_recog_words(txt, n=None):
    from .constants import SLICE_LEN
    if n is None:
        n = SLICE_LEN
    return Counter(get_recog_words(txt.lower())[:n])

