"""
Tests for Wikipedia-based race prediction functionality.
"""

import pandas as pd
import pytest

from ethnicolr.pred_wiki_ln import pred_wiki_ln
from ethnicolr.pred_wiki_name import pred_wiki_name

from .helpers import assert_prediction_quality, validate_race_prediction_consistency


class TestWikiLastNamePrediction:
    """Test Wikipedia-based last name prediction."""

    def test_basic_lastname_prediction(self, sample_wiki_names):
        """Test basic last name prediction using Wikipedia model."""
        result = pred_wiki_ln(sample_wiki_names, "last")

        # Validate prediction quality
        assert_prediction_quality(result, "wiki")

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

        # Should preserve original data
        assert len(result) == len(sample_wiki_names)

    def test_lastname_prediction_with_confidence(self, sample_wiki_names):
        """Test last name prediction with confidence intervals."""
        result = pred_wiki_ln(sample_wiki_names, "last", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "wiki", with_confidence=True)

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

    def test_lastname_prediction_extensive(self, extensive_names):
        """Test last name prediction on extensive dataset."""
        # Create simplified mapping for wiki predictions
        wiki_df = extensive_names.copy()
        wiki_df["expected_wiki"] = wiki_df["expected_major"].map({
            "white": "GreaterEuropean,British",
            "asian": "Asian,GreaterEastAsian,EastAsian",
            "hispanic": "GreaterEuropean,WestEuropean,Hispanic",
            "black": "GreaterAfrican,Africans"
        })

        result = pred_wiki_ln(wiki_df, "last")
        assert_prediction_quality(result, "wiki")

        # Check reasonable accuracy for major groups
        european_mask = wiki_df["expected_major"] == "white"
        if european_mask.any():
            european_results = result[european_mask]
            european_predictions = european_results["race"].str.contains("GreaterEuropean")
            european_accuracy = european_predictions.mean()
            assert european_accuracy >= 0.5, f"European accuracy too low: {european_accuracy}"


class TestWikiFullNamePrediction:
    """Test Wikipedia-based full name (first + last) prediction."""

    def test_basic_fullname_prediction(self, sample_wiki_names):
        """Test basic full name prediction using Wikipedia model."""
        result = pred_wiki_name(sample_wiki_names, "last", "first")

        # Validate prediction quality
        assert_prediction_quality(result, "wiki")

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

        # Should preserve original data
        assert len(result) == len(sample_wiki_names)

    def test_fullname_prediction_with_confidence(self, sample_wiki_names):
        """Test full name prediction with confidence intervals."""
        result = pred_wiki_name(sample_wiki_names, "last", "first", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "wiki", with_confidence=True)

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

    def test_fullname_handles_duplicates(self):
        """Test that full name prediction preserves duplicates correctly."""
        dupe_df = pd.DataFrame([
            {"last": "O'Neil", "first": "John"},
            {"last": "ONeil", "first": "John"},  # Similar but different
            {"last": "Smith", "first": "John"},
            {"last": "Smith", "first": "John"},  # Exact duplicate
        ])

        result = pred_wiki_name(dupe_df, "last", "first")

        # Should preserve all rows
        assert len(result) == len(dupe_df)

        # Should preserve original name values
        assert result["last"].tolist() == dupe_df["last"].tolist()
        assert result["first"].tolist() == dupe_df["first"].tolist()

        # Should have processing status column
        assert "processing_status" in result.columns
        assert result["processing_status"].eq("processed").all()

    def test_fullname_handles_special_characters(self, edge_case_names):
        """Test full name prediction with special characters and edge cases."""
        result = pred_wiki_name(edge_case_names, "last", "first")

        # Should handle all names without crashing
        assert len(result) == len(edge_case_names)

        # Should have processing status for all names
        assert "processing_status" in result.columns

        # Most names should be processed successfully
        processed_count = (result["processing_status"] == "processed").sum()
        total_count = len(result)
        success_rate = processed_count / total_count
        assert success_rate >= 0.7, f"Processing success rate too low: {success_rate}"

    def test_fullname_handles_accents_and_initials(self):
        """Test specific handling of accented characters and initials."""
        tricky_df = pd.DataFrame([
            {"last": "Szathmáry", "first": "Emöke"},      # Hungarian with accents
            {"last": "McMillan", "first": "A."},          # Initial with period
            {"last": "FitzGerald", "first": "John"},      # Irish prefix
            {"last": "Edwards", "first": "N"},            # Single letter initial
            {"last": "Sauder", "first": "E"},             # Single letter
            {"last": "Aguilar", "first": "J."},           # Initial with period
        ])

        result = pred_wiki_name(tricky_df, "last", "first")

        # Should process all names
        assert len(result) == len(tricky_df)

        # All should be marked as processed (the 2024 improvements should handle these)
        assert result["processing_status"].eq("processed").all()

        # Should have normalization tracking columns
        assert "name_normalized" in result.columns
        assert "__name" in result.columns


class TestWikiModelComparison:
    """Test comparison between last name and full name wiki models."""

    def test_lastname_vs_fullname_consistency(self, sample_wiki_names):
        """Test that last name and full name models produce consistent results."""
        ln_result = pred_wiki_ln(sample_wiki_names, "last")
        name_result = pred_wiki_name(sample_wiki_names, "last", "first")

        # Both should have same number of rows
        assert len(ln_result) == len(name_result)

        # Both should pass quality checks
        assert_prediction_quality(ln_result, "wiki")
        assert_prediction_quality(name_result, "wiki")

        # Full name model might be more accurate, but both should make predictions
        assert all(ln_result["race"].notna())
        assert all(name_result["race"].notna())

    def test_confidence_interval_consistency(self, sample_wiki_names):
        """Test confidence intervals are consistent between models."""
        ln_conf = pred_wiki_ln(sample_wiki_names, "last", conf_int=0.9)
        name_conf = pred_wiki_name(sample_wiki_names, "last", "first", conf_int=0.9)

        # Both should have confidence interval columns
        assert_prediction_quality(ln_conf, "wiki", with_confidence=True)
        assert_prediction_quality(name_conf, "wiki", with_confidence=True)

        # Should have same set of mean columns
        ln_mean_cols = [col for col in ln_conf.columns if col.endswith('_mean')]
        name_mean_cols = [col for col in name_conf.columns if col.endswith('_mean')]
        assert set(ln_mean_cols) == set(name_mean_cols)


class TestWikiErrorHandling:
    """Test error handling and edge cases for Wiki models."""

    def test_empty_names(self):
        """Test handling of empty or invalid names."""
        empty_df = pd.DataFrame([
            {"last": "", "first": "John"},
            {"last": "Smith", "first": ""},
            {"last": "   ", "first": "Mary"},  # Whitespace
        ])

        # Should not crash
        ln_result = pred_wiki_ln(empty_df, "last")
        name_result = pred_wiki_name(empty_df, "last", "first")

        assert len(ln_result) == len(empty_df)
        assert len(name_result) == len(empty_df)

        # Full name model should track processing status
        assert "processing_status" in name_result.columns

    def test_missing_column_error(self, sample_wiki_names):
        """Test that appropriate errors are raised for missing columns."""
        with pytest.raises((KeyError, ValueError)):
            pred_wiki_ln(sample_wiki_names, "nonexistent_column")

        with pytest.raises((KeyError, ValueError)):
            pred_wiki_name(sample_wiki_names, "last", "nonexistent_column")

    def test_very_long_names(self):
        """Test handling of unusually long names."""
        long_names_df = pd.DataFrame([
            {"last": "a" * 100, "first": "John"},  # Very long last name
            {"last": "Smith", "first": "b" * 100},  # Very long first name
            {"last": "Normal", "first": "Jane"},     # Normal name for comparison
        ])

        # Should handle without crashing
        result = pred_wiki_name(long_names_df, "last", "first")
        assert len(result) == 3
        assert_prediction_quality(result, "wiki")

    def test_unicode_handling(self):
        """Test proper handling of various Unicode characters."""
        unicode_df = pd.DataFrame([
            {"last": "Müller", "first": "Björn"},      # Germanic umlauts
            {"last": "José", "first": "María"},        # Spanish accents
            {"last": "Ξενοφῶν", "first": "Greek"},     # Greek characters
            {"last": "Владимир", "first": "Russian"},   # Cyrillic
            {"last": "张", "first": "伟"},              # Chinese characters
        ])

        # Should process without crashing
        result = pred_wiki_name(unicode_df, "last", "first")
        assert len(result) == 5

        # Most should be processed (depending on model training data)
        assert "processing_status" in result.columns
