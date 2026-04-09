"""Tests for feature extraction (require stanza)."""
from osp.features import (
    extract_pos_feats,
    extract_deprel_feats,
    extract_ttr_feats,
    extract_slice_feats,
    extract_syntax_feats,
)


class TestExtractPosFeats:
    def test_returns_counter(self, stanza_doc_simple):
        counts = extract_pos_feats(stanza_doc_simple)
        assert isinstance(counts, dict)
        # "The cat sat on the mat." has DT, NN, VBD, IN, NN, .
        assert "DT" in counts or "NN" in counts
        assert all(isinstance(v, int) for v in counts.values())

    def test_complex_doc(self, stanza_doc_complex):
        counts = extract_pos_feats(stanza_doc_complex)
        assert len(counts) > 0
        total = sum(counts.values())
        assert total > 0


class TestExtractDeprelFeats:
    def test_returns_counter(self, stanza_doc_simple):
        counts = extract_deprel_feats(stanza_doc_simple)
        assert isinstance(counts, dict)
        assert "nsubj" in counts
        assert "root" in counts

    def test_complex_has_more_relations(self, stanza_doc_complex):
        counts = extract_deprel_feats(stanza_doc_complex)
        # Complex sentences should have more diverse deprels
        assert len(counts) > 3


class TestExtractTtrFeats:
    def test_returns_dict(self, stanza_doc_simple):
        ttr = extract_ttr_feats(stanza_doc_simple)
        assert isinstance(ttr, dict)
        assert "mean" in ttr

    def test_values_between_zero_and_one(self, stanza_doc_simple):
        ttr = extract_ttr_feats(stanza_doc_simple, normalize=True)
        for k, v in ttr.items():
            assert 0.0 <= v <= 1.0, f"TTR {k}={v} out of range"


class TestExtractSyntaxFeats:
    def test_returns_dict(self, stanza_doc_simple):
        feats = extract_syntax_feats(stanza_doc_simple)
        assert isinstance(feats, dict)
        assert "IC" in feats
        assert "DC" in feats

    def test_complex_has_dependent_clauses(self, stanza_doc_complex):
        feats = extract_syntax_feats(stanza_doc_complex)
        assert feats["DC"] >= 1


class TestExtractSliceFeats:
    def test_returns_prefixed_dict(self, stanza_doc_simple):
        feats = extract_slice_feats(stanza_doc_simple)
        assert isinstance(feats, dict)
        assert len(feats) > 0
        # Features should be prefixed with their type
        prefixes = {k.split("_")[0] for k in feats}
        assert "pos" in prefixes
        assert "deprel" in prefixes

    def test_values_are_numeric(self, stanza_doc_simple):
        feats = extract_slice_feats(stanza_doc_simple)
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"{k}={v} is not numeric"

    def test_normalize_false(self, stanza_doc_simple):
        feats_norm = extract_slice_feats(stanza_doc_simple, normalize=True)
        feats_raw = extract_slice_feats(stanza_doc_simple, normalize=False)
        # Raw counts should generally be integers, normalized should differ
        assert feats_norm != feats_raw
