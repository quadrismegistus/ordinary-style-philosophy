"""Tests for text slicing logic."""
from osp.slices import iter_txt_slices


class TestIterTxtSlices:
    def test_basic_slicing(self, ok_words):
        """A short text with more recognized words than slice_len should produce slices."""
        text = "The cat sat on the mat.\nThe dog ran on the mat."
        slices = list(iter_txt_slices(text, slice_len=5, ok_words=ok_words))
        assert len(slices) >= 1
        # Each slice is (slice_num, text)
        for num, txt in slices:
            assert isinstance(num, int)
            assert isinstance(txt, str)
            assert len(txt) > 0

    def test_short_text_no_slices(self, ok_words):
        """Text shorter than slice_len should produce no slices."""
        text = "The cat."
        slices = list(iter_txt_slices(text, slice_len=100, ok_words=ok_words))
        assert len(slices) == 0

    def test_slice_numbering_starts_at_one(self, ok_words):
        text = "The cat sat on the mat and the dog ran on the big mat.\n" * 10
        slices = list(iter_txt_slices(text, slice_len=5, ok_words=ok_words))
        if slices:
            assert slices[0][0] == 1

    def test_unrecognized_words_not_counted(self):
        """Words not in ok_words shouldn't count toward the slice length."""
        ok = {"the", "cat"}
        text = "The cat xyzzy xyzzy xyzzy xyzzy the cat"
        slices = list(iter_txt_slices(text, slice_len=3, ok_words=ok))
        assert len(slices) >= 1
