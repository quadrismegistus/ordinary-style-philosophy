"""Tests for osp.text_processing — pure string functions, no data deps."""
from osp.text_processing import (
    dehyphenate,
    tokenize,
    tokenize_agnostic,
    remove_left_right_punct,
    is_stopword,
    html_to_latex,
    filter_pmla_pages,
)


class TestDehyphenate:
    def test_hyphen_newline(self):
        assert dehyphenate("arti-\nstically") == "artistically"

    def test_hyphen_space(self):
        assert dehyphenate("arti- stically") == "artistically"

    def test_hyphen_space_newline(self):
        assert dehyphenate("arti- \n stically") == "artistically"

    def test_no_hyphen(self):
        assert dehyphenate("hello world") == "hello world"

    def test_question_mark_between_letters(self):
        assert dehyphenate("some?thing") == "some—thing"

    def test_question_mark_at_end(self):
        assert dehyphenate("what?") == "what?"


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        result = tokenize("'hello,' she said.")
        assert "hello" in result
        assert "said" in result

    def test_empty(self):
        assert tokenize("") == []


class TestTokenizeAgnostic:
    def test_basic(self):
        tokens = tokenize_agnostic("Hello, world!")
        assert "Hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_preserves_case(self):
        tokens = tokenize_agnostic("The Cat")
        assert "The" in tokens
        assert "Cat" in tokens

    def test_em_dash(self):
        tokens = tokenize_agnostic("this—that")
        assert "this" in tokens
        assert "—" in tokens
        assert "that" in tokens


class TestRemoveLeftRightPunct:
    def test_strips_quotes(self):
        assert remove_left_right_punct('"hello"') == "hello"

    def test_strips_period(self):
        assert remove_left_right_punct("hello.") == "hello"

    def test_no_punct(self):
        assert remove_left_right_punct("hello") == "hello"

    def test_internal_punct_preserved(self):
        assert remove_left_right_punct("it's") == "it's"


class TestIsStopword:
    def test_short_word(self):
        assert is_stopword("the") is True
        assert is_stopword("a") is True

    def test_long_word(self):
        assert is_stopword("philosophy") is False
        assert is_stopword("word") is False


class TestHtmlToLatex:
    def test_bold(self):
        result = html_to_latex("<b>hello</b>")
        assert r"\textbf{hello}" in result

    def test_italic(self):
        result = html_to_latex("<i>hello</i>")
        assert r"\textit{hello}" in result

    def test_special_chars(self):
        result = html_to_latex("10% of $100")
        assert r"\%" in result
        assert r"\$" in result

    def test_plain_text(self):
        assert html_to_latex("hello world") == "hello world"

    def test_nested(self):
        result = html_to_latex("<b><i>bold italic</i></b>")
        assert r"\textbf" in result
        assert r"\textit" in result


class TestFilterPmlaPages:
    def test_strips_running_heads(self):
        pages = ["RUNNING HEAD 123 The actual text begins here"]
        result = filter_pmla_pages(pages)
        assert "The actual text begins here" in result

    def test_preserves_normal_text(self):
        pages = ["This is normal text."]
        result = filter_pmla_pages(pages)
        assert "This is normal text." in result
