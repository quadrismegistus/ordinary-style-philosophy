"""Tests for NLP utility functions (require stanza)."""
import nltk

from osp.nlp_utils import (
    get_sent_tree,
    get_num_words,
    get_num_independent_clauses,
    get_num_dependent_clauses,
    get_clauses_v2,
    render_clause_form,
    get_tree_stats,
)


class TestSentTree:
    def test_returns_parented_tree(self, stanza_doc_simple):
        sent = stanza_doc_simple.sentences[0]
        tree = get_sent_tree(sent)
        assert isinstance(tree, nltk.ParentedTree)

    def test_num_words(self, stanza_doc_simple):
        sent = stanza_doc_simple.sentences[0]
        tree = get_sent_tree(sent)
        # "The cat sat on the mat ." = 7 tokens
        assert get_num_words(tree) == 7


class TestTreeStats:
    def test_returns_dict(self, stanza_doc_simple):
        sent = stanza_doc_simple.sentences[0]
        stats = get_tree_stats(sent)
        assert isinstance(stats, dict)
        assert "num_words" in stats
        assert "height" in stats
        assert "num_independent_clauses" in stats
        assert stats["num_words"] == 7

    def test_complex_sentence(self, stanza_doc_complex):
        sent = stanza_doc_complex.sentences[1]
        stats = get_tree_stats(sent)
        assert stats["num_dependent_clauses"] >= 1


class TestClausesV2:
    def test_simple_sentence(self, stanza_doc_simple):
        sent = stanza_doc_simple.sentences[0]
        df = get_clauses_v2(sent)
        assert not df.empty
        assert "clause_id" in df.columns
        assert "clause_type" in df.columns
        assert "word" in df.columns
        # Simple sentence should have one main clause
        assert (df["clause_type"] == "main").any()

    def test_complex_sentence_has_subordinate(self, stanza_doc_complex):
        sent = stanza_doc_complex.sentences[0]
        df = get_clauses_v2(sent)
        # "Although the argument is compelling, ..." should have a sub clause
        assert (df["clause_type"] == "sub").any()


class TestRenderClauseForm:
    def test_simple(self, stanza_doc_simple):
        sent = stanza_doc_simple.sentences[0]
        form = render_clause_form(sent)
        assert "IC" in form

    def test_complex(self, stanza_doc_complex):
        sent = stanza_doc_complex.sentences[0]
        form = render_clause_form(sent)
        assert "IC" in form
        assert "DC" in form
