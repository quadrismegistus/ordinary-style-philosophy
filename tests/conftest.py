"""Shared fixtures for OSP tests."""
import pytest


@pytest.fixture(scope="session")
def stanza_nlp():
    """A shared Stanza pipeline (expensive to load, so session-scoped)."""
    import stanza
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse,constituency",
        verbose=False,
    )


@pytest.fixture(scope="session")
def stanza_doc_simple(stanza_nlp):
    """A parsed single-sentence document: 'The cat sat on the mat.'"""
    return stanza_nlp("The cat sat on the mat.")


@pytest.fixture(scope="session")
def stanza_doc_complex(stanza_nlp):
    """A parsed multi-sentence document with subordinate clauses."""
    text = (
        "Although the argument is compelling, the philosopher rejected it. "
        "She believed that truth requires evidence, which is hard to obtain."
    )
    return stanza_nlp(text)


@pytest.fixture
def ok_words():
    """A small set of recognized words for slicing tests."""
    return {"the", "cat", "sat", "on", "mat", "dog", "ran", "big", "small",
            "is", "was", "and", "but", "of", "in", "to", "a", "it", "that",
            "this", "with", "for", "not", "are", "be", "has", "from", "or",
            "an", "can", "had", "all", "each", "which", "do", "how", "if",
            "will", "up", "about", "out", "many", "then", "them", "would",
            "like", "so", "these", "her", "long", "make", "thing", "see",
            "him", "two", "way", "could", "she", "he", "we", "what",
            "although", "argument", "compelling", "philosopher", "rejected",
            "believed", "truth", "requires", "evidence", "hard", "obtain"}
