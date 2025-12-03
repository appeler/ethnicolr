"""
Tests for North Carolina voter registration-based race prediction functionality.
"""

import pandas as pd
import pytest

from ethnicolr.pred_nc_reg_name import pred_nc_reg_name

from .helpers import assert_prediction_quality, validate_race_prediction_consistency


class TestNorthCarolinaPrediction:
    """Test North Carolina voter registration-based prediction model."""

    def test_basic_nc_prediction(self, sample_nc_names):
        """Test basic North Carolina name prediction."""
        result = pred_nc_reg_name(sample_nc_names, "last", "first")

        # Validate prediction quality
        assert_prediction_quality(result, "nc")

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

        # Should preserve original data
        assert len(result) == len(sample_nc_names)

    def test_nc_prediction_with_confidence(self, sample_nc_names):
        """Test North Carolina prediction with confidence intervals."""
        result = pred_nc_reg_name(sample_nc_names, "last", "first", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "nc", with_confidence=True)

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

    def test_nc_extensive_validation(self, extensive_names):
        """Test NC prediction on extensive dataset with adapted expectations."""
        # Create NC-compatible test data with realistic expectations
        nc_test_data = []
        for _, row in extensive_names.iterrows():
            expected_major = row["expected_major"]

            # Map to NC categories (Hispanic/Non-Hispanic + Race)
            if expected_major == "hispanic":
                expected_nc = "HL+W"  # Hispanic + White is most common
            elif expected_major == "white":
                expected_nc = "NL+W"  # Non-Hispanic White
            elif expected_major == "black":
                expected_nc = "NL+B"  # Non-Hispanic Black
            elif expected_major == "asian":
                expected_nc = "NL+A"  # Non-Hispanic Asian
            else:
                expected_nc = "NL+O"  # Non-Hispanic Other

            nc_test_data.append(
                {
                    "last": row["last"],
                    "first": row["first"],
                    "expected_nc": expected_nc,
                    "expected_major": expected_major,
                }
            )

        nc_df = pd.DataFrame(nc_test_data)
        result = pred_nc_reg_name(nc_df, "last", "first")

        # Validate prediction quality
        assert_prediction_quality(result, "nc")

        # Test accuracy for major ethnic groups
        # Hispanic names should often predict HL+ categories
        hispanic_mask = nc_df["expected_major"] == "hispanic"
        if hispanic_mask.any():
            hispanic_results = result[hispanic_mask]
            hispanic_predictions = hispanic_results["race"].str.startswith("HL+")
            hispanic_accuracy = hispanic_predictions.mean()
            assert hispanic_accuracy >= 0.3, (
                f"Hispanic prediction accuracy too low: {hispanic_accuracy}"
            )

        # Non-Hispanic whites should often predict NL+W
        white_mask = nc_df["expected_major"] == "white"
        if white_mask.any():
            white_results = result[white_mask]
            white_accuracy = (white_results["race"] == "NL+W").mean()
            assert white_accuracy >= 0.3, (
                f"White prediction accuracy too low: {white_accuracy}"
            )

    def test_nc_race_categories_comprehensive(self, sample_nc_names):
        """Test that NC model produces all expected race categories."""
        result = pred_nc_reg_name(sample_nc_names, "last", "first")

        # Should have all 12 NC race categories as columns
        expected_categories = [
            "HL+A",
            "HL+B",
            "HL+I",
            "HL+M",
            "HL+O",
            "HL+W",  # Hispanic
            "NL+A",
            "NL+B",
            "NL+I",
            "NL+M",
            "NL+O",
            "NL+W",  # Non-Hispanic
        ]

        for category in expected_categories:
            assert category in result.columns, f"Missing category: {category}"

        # All probability columns should exist
        assert_prediction_quality(result, "nc")

    @pytest.mark.parametrize("conf_level", [0.8, 0.9, 0.95])
    def test_nc_confidence_levels(self, sample_nc_names, conf_level):
        """Test different confidence interval levels for NC model."""
        result = pred_nc_reg_name(sample_nc_names, "last", "first", conf_int=conf_level)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "nc", with_confidence=True)

        # Should have 12 mean columns (one for each race category)
        mean_cols = [col for col in result.columns if col.endswith("_mean")]
        assert len(mean_cols) == 12

        # All mean columns should sum to approximately 1.0
        mean_sums = result[mean_cols].sum(axis=1)
        assert all(abs(mean_sums - 1.0) <= 0.1)

    def test_nc_hispanic_vs_nonhispanic_distinction(self):
        """Test that NC model properly distinguishes Hispanic vs Non-Hispanic."""
        # Test with clearly Hispanic and Non-Hispanic names
        test_names = pd.DataFrame(
            [
                {"last": "Garcia", "first": "Jose", "expected_type": "HL"},
                {"last": "Rodriguez", "first": "Maria", "expected_type": "HL"},
                {"last": "Smith", "first": "John", "expected_type": "NL"},
                {"last": "Johnson", "first": "Emily", "expected_type": "NL"},
            ]
        )

        result = pred_nc_reg_name(test_names, "last", "first")
        assert_prediction_quality(result, "nc")

        # Check that Hispanic/Non-Hispanic distinction makes sense
        for i, row in test_names.iterrows():
            predicted_race = result.iloc[i]["race"]
            expected_prefix = row["expected_type"]

            assert predicted_race.startswith(expected_prefix), (
                f"Expected {expected_prefix}+ category but got {predicted_race} for {row['first']} {row['last']}"
            )


class TestNorthCarolinaErrorHandling:
    """Test error handling and edge cases for NC model."""

    def test_nc_missing_columns(self, sample_nc_names):
        """Test error handling for missing columns."""
        with pytest.raises((KeyError, ValueError)):
            pred_nc_reg_name(sample_nc_names, "nonexistent_last", "first")

        with pytest.raises((KeyError, ValueError)):
            pred_nc_reg_name(sample_nc_names, "last", "nonexistent_first")

    def test_nc_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame(columns=["last", "first"])

        # Should handle gracefully
        result = pred_nc_reg_name(empty_df, "last", "first")
        assert len(result) == 0

        # Should still have expected columns
        expected_cols = [
            "HL+A",
            "HL+B",
            "HL+I",
            "HL+M",
            "HL+O",
            "HL+W",
            "NL+A",
            "NL+B",
            "NL+I",
            "NL+M",
            "NL+O",
            "NL+W",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_nc_single_row(self):
        """Test handling of single-row DataFrames."""
        single_df = pd.DataFrame([{"last": "smith", "first": "john"}])

        result = pred_nc_reg_name(single_df, "last", "first")
        assert len(result) == 1
        assert_prediction_quality(result, "nc")

    def test_nc_special_characters(self, edge_case_names):
        """Test NC model with special characters in names."""
        result = pred_nc_reg_name(edge_case_names, "last", "first")

        # Should handle without crashing
        assert len(result) == len(edge_case_names)
        assert_prediction_quality(result, "nc")

    def test_nc_null_and_empty_names(self):
        """Test handling of null and empty name values."""
        problematic_df = pd.DataFrame(
            [
                {"last": None, "first": "John"},
                {"last": "Smith", "first": None},
                {"last": "", "first": "Jane"},
                {"last": "Doe", "first": ""},
                {"last": "   ", "first": "Bob"},  # Whitespace only
            ]
        )

        # Should not crash
        result = pred_nc_reg_name(problematic_df, "last", "first")
        assert len(result) == len(problematic_df)


class TestNorthCarolinaPerformance:
    """Test performance characteristics of NC model."""

    def test_nc_deterministic_results(self, sample_nc_names):
        """Test that NC model produces deterministic results."""
        result1 = pred_nc_reg_name(sample_nc_names, "last", "first")
        result2 = pred_nc_reg_name(sample_nc_names, "last", "first")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_nc_column_preservation(self, sample_nc_names):
        """Test that original columns are preserved."""
        # Add extra columns to test data
        test_df = sample_nc_names.copy()
        test_df["id"] = range(len(test_df))
        test_df["notes"] = "test_data"

        result = pred_nc_reg_name(test_df, "last", "first")

        # Original columns should be preserved
        assert "id" in result.columns
        assert "notes" in result.columns
        assert all(result["id"] == test_df["id"])
        assert all(result["notes"] == test_df["notes"])

    def test_nc_confidence_vs_regular_consistency(self, sample_nc_names):
        """Test consistency between regular and confidence interval predictions."""
        regular = pred_nc_reg_name(sample_nc_names, "last", "first")
        with_conf = pred_nc_reg_name(sample_nc_names, "last", "first", conf_int=0.9)

        # Both should have same basic structure
        assert len(regular) == len(with_conf)

        # Regular probability columns should be identical
        race_cols = [
            col
            for col in regular.columns
            if col
            in [
                "HL+A",
                "HL+B",
                "HL+I",
                "HL+M",
                "HL+O",
                "HL+W",
                "NL+A",
                "NL+B",
                "NL+I",
                "NL+M",
                "NL+O",
                "NL+W",
            ]
        ]

        for col in race_cols:
            pd.testing.assert_series_equal(
                regular[col], with_conf[col], check_names=False
            )

        # Predicted race should be the same
        pd.testing.assert_series_equal(
            regular["race"], with_conf["race"], check_names=False
        )


class TestNorthCarolinaUniqueFeatures:
    """Test features unique to the NC model."""

    def test_nc_twelve_category_coverage(self):
        """Test that NC model can predict all 12 categories."""
        # Create names likely to trigger different categories
        diverse_names = pd.DataFrame(
            [
                # Hispanic categories
                {"last": "Garcia", "first": "Jose"},  # HL+W likely
                {"last": "Martinez", "first": "Carlos"},  # HL+W likely
                {"last": "Chen", "first": "Maria"},  # HL+A possible
                {"last": "Johnson", "first": "Carlos"},  # HL+B possible
                # Non-Hispanic categories
                {"last": "Smith", "first": "John"},  # NL+W likely
                {"last": "Williams", "first": "James"},  # NL+B likely
                {"last": "Kim", "first": "Sarah"},  # NL+A likely
                {"last": "Patel", "first": "David"},  # NL+A likely
                {"last": "Running Bear", "first": "Tom"},  # NL+I possible
                {"last": "Nakamura", "first": "Lisa"},  # NL+A likely
            ]
        )

        result = pred_nc_reg_name(diverse_names, "last", "first")
        assert_prediction_quality(result, "nc")

        # Should predict at least several different categories
        unique_predictions = result["race"].nunique()
        assert unique_predictions >= 3, (
            f"Only predicted {unique_predictions} unique categories"
        )

        # Should have both Hispanic and Non-Hispanic predictions
        has_hispanic = result["race"].str.startswith("HL+").any()
        has_nonhispanic = result["race"].str.startswith("NL+").any()
        assert has_hispanic, "No Hispanic (HL+) predictions found"
        assert has_nonhispanic, "No Non-Hispanic (NL+) predictions found"

    def test_nc_probability_distribution_sanity(self, sample_nc_names):
        """Test that probability distributions make intuitive sense."""
        result = pred_nc_reg_name(sample_nc_names, "last", "first")

        # For each prediction, the highest probability should correspond to predicted race
        race_cols = [
            "HL+A",
            "HL+B",
            "HL+I",
            "HL+M",
            "HL+O",
            "HL+W",
            "NL+A",
            "NL+B",
            "NL+I",
            "NL+M",
            "NL+O",
            "NL+W",
        ]

        for _i, row in result.iterrows():
            predicted_race = row["race"]
            predicted_prob = row[predicted_race]

            # Predicted race should have highest (or tied for highest) probability
            max_prob = row[race_cols].max()
            assert predicted_prob == max_prob, (
                f"Predicted race {predicted_race} doesn't have max probability"
            )
