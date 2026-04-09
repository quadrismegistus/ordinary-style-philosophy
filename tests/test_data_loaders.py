"""Tests for pure utility functions in osp.data_loaders."""
from osp.data_loaders import periodize_year, get_half_century, rename_journal


class TestPeriodizeYear:
    def test_decade(self):
        assert periodize_year(1955, 10) == "1950-1960"

    def test_quarter_century(self):
        assert periodize_year(1955, 25) == "1950-1975"

    def test_boundary(self):
        assert periodize_year(1950, 25) == "1950-1975"

    def test_early_year(self):
        assert periodize_year(1901, 25) == "1900-1925"


class TestGetHalfCentury:
    def test_early(self):
        assert get_half_century(1920) == "eC20"

    def test_late(self):
        assert get_half_century(1975) == "lC20"

    def test_boundary(self):
        assert get_half_century(1950) == "lC20"

    def test_21st_century(self):
        assert get_half_century(2010) == "eC21"


class TestRenameJournal:
    def test_erkenntnis(self):
        assert rename_journal("Erkenntnis (1975-)") == "Erkenntnis"

    def test_ethics(self):
        assert rename_journal("Ethics and Policy") == "Ethics"

    def test_passthrough(self):
        assert rename_journal("PMLA") == "PMLA"
