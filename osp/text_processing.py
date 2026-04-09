import re
import string
from collections import Counter
from html.parser import HTMLParser


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



def html_to_latex(html: str) -> str:
    def escape_latex(text: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(ch, ch) for ch in text)

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out = []
            self.list_stack = []
            self.style_stack = []

        def _append(self, s: str):
            if s:
                self.out.append(s)

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()

            if tag == "ul":
                self._append("\n\\begin{itemize}\n")
                self.list_stack.append("itemize")
            elif tag == "ol":
                self._append("\n\\begin{enumerate}\n")
                self.list_stack.append("enumerate")
            elif tag == "li":
                self._append("\\item ")
            elif tag in ("b", "strong"):
                self._append("\\textbf{")
                self.style_stack.append("}")
            elif tag in ("i", "em"):
                self._append("\\textit{")
                self.style_stack.append("}")
            elif tag == "u":
                self._append("\\underline{")
                self.style_stack.append("}")
            elif tag == "br":
                self._append(" \\\\\n")

        def handle_endtag(self, tag):
            tag = tag.lower()

            if tag in ("ul", "ol"):
                if self.list_stack:
                    env = self.list_stack.pop()
                    self._append(f"\n\\end{{{env}}}\n")
            elif tag == "li":
                self._append("\n")
            elif tag in ("b", "strong", "i", "em", "u"):
                if self.style_stack:
                    self._append(self.style_stack.pop())

        def handle_data(self, data):
            if not data:
                return
            txt = re.sub(r"\s+", " ", data)
            self._append(escape_latex(txt))

    p = _Parser()
    p.feed(html)
    p.close()

    latex = "".join(p.out)
    latex = re.sub(r"[ \t]+\n", "\n", latex)
    latex = re.sub(r"\n{3,}", "\n\n", latex)

    out = latex.strip()
    out = out.replace('\n', ' ')
    while '  ' in out:
        out = out.replace('  ', ' ')
    return out



def shorten_eg(eg, max_len=40):
    eg_pre,w,eg_post = eg.split('*',2)
    radius = (max_len - len(w)) // 2
    eg_pre = eg_pre[-radius:].lstrip()
    eg_post = eg_post[:max_len-len(eg_pre)-len(w)].rstrip()
    return f'...{eg_pre}XXXEMPHXXX{{{w.lower()}}}{eg_post}...'
