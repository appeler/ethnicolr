"""
Tests for census-based race prediction functionality.
"""

import pandas as pd
import pytest

from ethnicolr import estimate_census_surname, lookup_census_surname

from .helpers import assert_prediction_quality


class TestCensusLookup:
    """Test the lookup_census_surname statistical lookup function."""

    def test_census_2000_lookup(self, sample_census_names):
        """Test census 2000 last name lookup."""
        result = lookup_census_surname(sample_census_names, "last", year=2000)

        # Check that census percentage columns are added
        assert "pctwhite" in result.columns
        assert "pctblack" in result.columns
        assert "pctapi" in result.columns
        assert "pcthispanic" in result.columns

        # Should preserve original data
        assert len(result) == len(sample_census_names)
        assert all(result["last"] == sample_census_names["last"])

    def test_census_2010_lookup(self, sample_census_names):
        """Test census 2010 last name lookup."""
        result = lookup_census_surname(sample_census_names, "last", year=2010)

        # Check that census percentage columns are added
        assert "pctwhite" in result.columns
        assert "pctblack" in result.columns
        assert "pctapi" in result.columns
        assert "pcthispanic" in result.columns
        assert "pctaian" in result.columns
        assert "pct2prace" in result.columns

        # Should preserve original data
        assert len(result) == len(sample_census_names)
        assert all(result["last"] == sample_census_names["last"])

    def test_census_2020_lookup(self, sample_census_names):
        """Test census 2020 last name lookup."""
        result = lookup_census_surname(sample_census_names, "last", year=2020)

        # Check that census percentage columns are added
        assert "pctwhite" in result.columns
        assert "pctblack" in result.columns
        assert "pctapi" in result.columns
        assert "pcthispanic" in result.columns
        assert "pctaian" in result.columns
        assert "pct2prace" in result.columns

        # Should preserve original data
        assert len(result) == len(sample_census_names)
        assert all(result["last"] == sample_census_names["last"])

    def test_census_2020_vs_2010_consistency(self, sample_census_names):
        """Test that 2020 and 2010 data have consistent structure and similar values."""
        result_2010 = lookup_census_surname(sample_census_names, "last", year=2010)
        result_2020 = lookup_census_surname(sample_census_names, "last", year=2020)

        # Same columns
        assert set(result_2010.columns) == set(result_2020.columns)

        # Same number of rows
        assert len(result_2010) == len(result_2020)

        # Values should be in reasonable ranges (0-100)
        for col in ["pctwhite", "pctblack", "pctapi", "pcthispanic"]:
            valid_2020 = result_2020[col].dropna()
            assert (valid_2020 >= 0).all()
            assert (valid_2020 <= 100).all()

    def test_census_lookup_with_missing_names(self):
        """Test census lookup with names not in census data."""
        df = pd.DataFrame(
            [{"last": "veryrareuncommonname", "id": 1}, {"last": "smith", "id": 2}]
        )

        result = lookup_census_surname(df, "last", year=2010)
        assert len(result) == 2
        assert "pctwhite" in result.columns


class TestCensusPrediction:
    """Test the census-based ML prediction models."""

    @pytest.mark.parametrize("year", [2000, 2010])
    def test_basic_prediction(self, sample_census_names, year):
        """Test basic census prediction for both census years."""
        result = estimate_census_surname(sample_census_names, "last", year=year)

        # Validate prediction quality
        assert_prediction_quality(result, "census")

        # Check that most predictions are reasonable (ML models aren't 100% accurate)
        # Should get at least 60% accuracy on this small sample
        if "true_race" in result.columns:
            accuracy = (result["race"] == result["true_race"]).mean()
            assert accuracy >= 0.6, f"Prediction accuracy too low: {accuracy}"

        # Should preserve original data
        assert len(result) == len(sample_census_names)

    @pytest.mark.parametrize("year", [2000, 2010])
    def test_prediction_with_uncertainty_intervals(self, sample_census_names, year):
        """Test census prediction with MC-dropout uncertainty summaries."""
        result = estimate_census_surname(
            sample_census_names, "last", year=year, uncertainty_level=0.9
        )

        # Validate prediction quality including MC-dropout uncertainty summaries
        assert_prediction_quality(result, "census", with_uncertainty=True)

        # Check that most predictions are reasonable (ML models aren't 100% accurate)
        if "true_race" in result.columns:
            accuracy = (result["race"] == result["true_race"]).mean()
            assert accuracy >= 0.6, f"Prediction accuracy too low: {accuracy}"

    def test_prediction_extensive_dataset(self, extensive_names):
        """Test census prediction on larger, more realistic dataset."""
        result = estimate_census_surname(extensive_names, "last", year=2010)

        # Validate prediction quality
        assert_prediction_quality(result, "census")

        # Check that we get reasonable predictions for major groups
        # race_counts = result["race"].value_counts()

        # Should predict white for most European surnames
        european_subset = extensive_names[extensive_names["expected_major"] == "white"]
        european_results = result[result.index.isin(european_subset.index)]
        white_accuracy = (european_results["race"] == "white").mean()
        assert white_accuracy >= 0.7, (
            f"White prediction accuracy too low: {white_accuracy}"
        )

        # Should predict asian for most Asian surnames
        asian_subset = extensive_names[extensive_names["expected_major"] == "asian"]
        asian_results = result[result.index.isin(asian_subset.index)]
        asian_accuracy = (asian_results["race"] == "api").mean()
        assert asian_accuracy >= 0.7, (
            f"Asian prediction accuracy too low: {asian_accuracy}"
        )

    def test_prediction_preserves_input_columns(self, sample_census_names):
        """Test that prediction preserves all input columns."""
        result = estimate_census_surname(sample_census_names, "last", year=2010)

        # Should have all original columns plus prediction columns
        for col in sample_census_names.columns:
            assert col in result.columns

        # Original values should be unchanged (except last names are
        # normalized to title case)
        for col in sample_census_names.columns:
            if col == "last":
                # Names are normalized to title case during prediction
                assert all(result[col] == sample_census_names[col].str.title())
            else:
                assert all(result[col] == sample_census_names[col])

    def test_different_column_names(self):
        """Test prediction with non-standard column names."""
        df = pd.DataFrame(
            [
                {"surname": "smith", "expected": "white"},
                {"surname": "zhang", "expected": "api"},
            ]
        )

        result = estimate_census_surname(df, "surname", year=2010)
        assert_prediction_quality(result, "census")
        assert len(result) == 2

    def test_prediction_deterministic(self, sample_census_names):
        """Test that predictions are deterministic (same input -> same output)."""
        result1 = estimate_census_surname(sample_census_names, "last", year=2010)
        result2 = estimate_census_surname(sample_census_names, "last", year=2010)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.parametrize("conf_level", [0.8, 0.9, 0.95])
    def test_confidence_interval_levels(self, sample_census_names, conf_level):
        """Test different MC-dropout uncertainty summary levels."""
        result = estimate_census_surname(
            sample_census_names, "last", year=2010, uncertainty_level=conf_level
        )
        assert_prediction_quality(result, "census", with_uncertainty=True)

        # Should have MC-dropout uncertainty summary columns
        mean_cols = [col for col in result.columns if col.endswith("_mc_mean")]
        assert len(mean_cols) == 4  # One for each race

    def test_census_year_consistency(self, sample_census_names):
        """Test that different census years produce consistent result structure."""
        result_2000 = estimate_census_surname(sample_census_names, "last", year=2000)
        result_2010 = estimate_census_surname(sample_census_names, "last", year=2010)

        # Both should have same columns structure (except for census-specific columns)
        race_cols_2000 = [
            col
            for col in result_2000.columns
            if col in ["api", "black", "hispanic", "white"]
        ]
        race_cols_2010 = [
            col
            for col in result_2010.columns
            if col in ["api", "black", "hispanic", "white"]
        ]

        assert set(race_cols_2000) == set(race_cols_2010)
        assert len(result_2000) == len(result_2010)
