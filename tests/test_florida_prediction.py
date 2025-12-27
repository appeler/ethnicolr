"""
Tests for Florida voter registration-based race prediction functionality.
"""

import pandas as pd
import pytest

from ethnicolr.pred_fl_reg_ln import pred_fl_reg_ln
from ethnicolr.pred_fl_reg_ln_five_cat import pred_fl_reg_ln_five_cat
from ethnicolr.pred_fl_reg_name import pred_fl_reg_name
from ethnicolr.pred_fl_reg_name_five_cat import pred_fl_reg_name_five_cat

from .helpers import assert_prediction_quality, validate_race_prediction_consistency


class TestFloridaLastNamePrediction:
    """Test Florida voter registration-based last name prediction models."""

    def test_four_category_lastname_prediction(self, sample_florida_names):
        """Test 4-category last name prediction model."""
        result = pred_fl_reg_ln(sample_florida_names, "last")

        # Validate prediction quality
        assert_prediction_quality(result, "florida_4cat")

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

        # Should preserve original data
        assert len(result) == len(sample_florida_names)

    def test_five_category_lastname_prediction(self, sample_florida_names):
        """Test 5-category last name prediction model."""
        result = pred_fl_reg_ln_five_cat(sample_florida_names, "last")

        # Validate prediction quality
        assert_prediction_quality(result, "florida_5cat")

        # Should preserve original data
        assert len(result) == len(sample_florida_names)

        # Should include 'other' category
        assert "other" in result.columns

    def test_four_category_with_confidence(self, sample_florida_names):
        """Test 4-category last name prediction with confidence intervals."""
        result = pred_fl_reg_ln(sample_florida_names, "last", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "florida_4cat", with_confidence=True)

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

    def test_five_category_with_confidence(self, sample_florida_names):
        """Test 5-category last name prediction with confidence intervals."""
        result = pred_fl_reg_ln_five_cat(sample_florida_names, "last", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "florida_5cat", with_confidence=True)


class TestFloridaFullNamePrediction:
    """Test Florida voter registration-based full name prediction models."""

    def test_four_category_fullname_prediction(self, sample_florida_names):
        """Test 4-category full name prediction model."""
        result = pred_fl_reg_name(sample_florida_names, "last", "first")

        # Validate prediction quality
        assert_prediction_quality(result, "florida_4cat")

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

        # Should preserve original data
        assert len(result) == len(sample_florida_names)

    def test_five_category_fullname_prediction(self, sample_florida_names):
        """Test 5-category full name prediction model."""
        result = pred_fl_reg_name_five_cat(sample_florida_names, "last", "first")

        # Validate prediction quality
        assert_prediction_quality(result, "florida_5cat")

        # Should preserve original data
        assert len(result) == len(sample_florida_names)

        # Should include 'other' category
        assert "other" in result.columns

    def test_four_category_fullname_with_confidence(self, sample_florida_names):
        """Test 4-category full name prediction with confidence intervals."""
        result = pred_fl_reg_name(sample_florida_names, "last", "first", conf_int=0.9)

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "florida_4cat", with_confidence=True)

        # Check that predictions match expected races for test data
        assert validate_race_prediction_consistency(result)

    def test_five_category_fullname_with_confidence(self, sample_florida_names):
        """Test 5-category full name prediction with confidence intervals."""
        result = pred_fl_reg_name_five_cat(
            sample_florida_names, "last", "first", conf_int=0.9
        )

        # Validate prediction quality including confidence intervals
        assert_prediction_quality(result, "florida_5cat", with_confidence=True)


class TestFloridaModelComparison:
    """Test comparisons between different Florida prediction models."""

    def test_lastname_vs_fullname_four_category(self, sample_florida_names):
        """Compare 4-category last name vs full name models."""
        ln_result = pred_fl_reg_ln(sample_florida_names, "last")
        name_result = pred_fl_reg_name(sample_florida_names, "last", "first")

        # Both should have same number of rows and structure
        assert len(ln_result) == len(name_result)

        # Both should pass quality checks
        assert_prediction_quality(ln_result, "florida_4cat")
        assert_prediction_quality(name_result, "florida_4cat")

        # Should have same race categories
        race_cols_ln = [
            col
            for col in ln_result.columns
            if col in ["asian", "hispanic", "nh_black", "nh_white"]
        ]
        race_cols_name = [
            col
            for col in name_result.columns
            if col in ["asian", "hispanic", "nh_black", "nh_white"]
        ]
        assert set(race_cols_ln) == set(race_cols_name)

    def test_lastname_vs_fullname_five_category(self, sample_florida_names):
        """Compare 5-category last name vs full name models."""
        ln_result = pred_fl_reg_ln_five_cat(sample_florida_names, "last")
        name_result = pred_fl_reg_name_five_cat(sample_florida_names, "last", "first")

        # Both should have same number of rows and structure
        assert len(ln_result) == len(name_result)

        # Both should pass quality checks
        assert_prediction_quality(ln_result, "florida_5cat")
        assert_prediction_quality(name_result, "florida_5cat")

        # Should have same race categories including 'other'
        expected_cols = ["asian", "hispanic", "nh_black", "nh_white", "other"]
        race_cols_ln = [col for col in ln_result.columns if col in expected_cols]
        race_cols_name = [col for col in name_result.columns if col in expected_cols]
        assert set(race_cols_ln) == set(race_cols_name)
        assert len(race_cols_ln) == 5

    def test_four_vs_five_category_consistency(self, sample_florida_names):
        """Compare 4-category vs 5-category models for consistency."""
        four_cat = pred_fl_reg_ln(sample_florida_names, "last")
        five_cat = pred_fl_reg_ln_five_cat(sample_florida_names, "last")

        # Both should pass their respective quality checks
        assert_prediction_quality(four_cat, "florida_4cat")
        assert_prediction_quality(five_cat, "florida_5cat")

        # 4-category columns should be subset of 5-category
        four_cat_cols = ["asian", "hispanic", "nh_black", "nh_white"]
        for col in four_cat_cols:
            assert col in four_cat.columns
            assert col in five_cat.columns

        # 5-category should have additional 'other' column
        assert "other" in five_cat.columns
        assert "other" not in four_cat.columns


class TestFloridaExtensiveValidation:
    """Test Florida models on extensive datasets."""

    def test_extensive_four_category_accuracy(self, extensive_names):
        """Test 4-category model accuracy on extensive dataset."""
        # Map expected races to Florida categories
        fl_mapping = {
            "white": "nh_white",
            "black": "nh_black",
            "hispanic": "hispanic",
            "asian": "asian",
        }

        test_df = extensive_names.copy()
        test_df["expected_fl"] = test_df["expected_major"].map(fl_mapping)

        result = pred_fl_reg_ln(test_df, "last")
        assert_prediction_quality(result, "florida_4cat")

        # Test accuracy for major groups
        for expected_race, fl_race in fl_mapping.items():
            subset_mask = test_df["expected_major"] == expected_race
            if subset_mask.any():
                subset_results = result[subset_mask]
                accuracy = (subset_results["race"] == fl_race).mean()
                assert accuracy >= 0.25, (
                    f"{expected_race} -> {fl_race} accuracy too low: {accuracy}"
                )

    def test_extensive_five_category_other_usage(self, extensive_names):
        """Test that 5-category model appropriately uses 'other' category."""
        result = pred_fl_reg_ln_five_cat(extensive_names, "last")
        assert_prediction_quality(result, "florida_5cat")

        # Should make some 'other' predictions (though maybe not many for common names)
        race_counts = result["race"].value_counts()
        assert len(race_counts) >= 4  # Should use at least 4 categories

        # 'Other' category should exist even if not frequently predicted
        assert "other" in result.columns
        other_prob_sum = result["other"].sum()
        assert other_prob_sum >= 0  # Should be non-negative


class TestFloridaConfidenceIntervals:
    """Test confidence interval functionality across Florida models."""

    @pytest.mark.parametrize(
        "model_func,model_type",
        [
            (pred_fl_reg_ln, "florida_4cat"),
            (pred_fl_reg_ln_five_cat, "florida_5cat"),
        ],
    )
    @pytest.mark.parametrize("conf_level", [0.8, 0.9, 0.95])
    def test_lastname_confidence_levels(
        self, sample_florida_names, model_func, model_type, conf_level
    ):
        """Test different confidence levels for last name models."""
        result = model_func(sample_florida_names, "last", conf_int=conf_level)
        assert_prediction_quality(result, model_type, with_confidence=True)

        # Should have appropriate number of mean columns
        mean_cols = [col for col in result.columns if col.endswith("_mean")]
        expected_count = 4 if model_type == "florida_4cat" else 5
        assert len(mean_cols) == expected_count

    @pytest.mark.parametrize(
        "model_func,model_type",
        [
            (pred_fl_reg_name, "florida_4cat"),
            (pred_fl_reg_name_five_cat, "florida_5cat"),
        ],
    )
    @pytest.mark.parametrize("conf_level", [0.8, 0.9, 0.95])
    def test_fullname_confidence_levels(
        self, sample_florida_names, model_func, model_type, conf_level
    ):
        """Test different confidence levels for full name models."""
        result = model_func(sample_florida_names, "last", "first", conf_int=conf_level)
        assert_prediction_quality(result, model_type, with_confidence=True)

        # Should have appropriate number of mean columns
        mean_cols = [col for col in result.columns if col.endswith("_mean")]
        expected_count = 4 if model_type == "florida_4cat" else 5
        assert len(mean_cols) == expected_count


class TestFloridaErrorHandling:
    """Test error handling for Florida models."""

    def test_missing_columns(self, sample_florida_names):
        """Test appropriate error handling for missing columns."""
        with pytest.raises((KeyError, ValueError)):
            pred_fl_reg_ln(sample_florida_names, "nonexistent_column")

        with pytest.raises((KeyError, ValueError)):
            pred_fl_reg_name(sample_florida_names, "last", "nonexistent_column")

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame(columns=["last", "first"])

        # Should handle gracefully
        result = pred_fl_reg_ln(empty_df, "last")
        assert len(result) == 0
        assert "asian" in result.columns  # Should still have expected columns

    def test_single_row(self):
        """Test handling of single-row DataFrames."""
        single_df = pd.DataFrame([{"last": "smith", "first": "john"}])

        result = pred_fl_reg_ln(single_df, "last")
        assert len(result) == 1
        assert_prediction_quality(result, "florida_4cat")

        result_name = pred_fl_reg_name(single_df, "last", "first")
        assert len(result_name) == 1
        assert_prediction_quality(result_name, "florida_4cat")


class TestFloridaPerformance:
    """Test performance characteristics of Florida models."""

    def test_deterministic_results(self, sample_florida_names):
        """Test that Florida models produce deterministic results."""
        # Run same prediction multiple times
        result1 = pred_fl_reg_ln(sample_florida_names, "last")
        result2 = pred_fl_reg_ln(sample_florida_names, "last")
        result3 = pred_fl_reg_name(sample_florida_names, "last", "first")
        result4 = pred_fl_reg_name(sample_florida_names, "last", "first")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result3, result4)

    def test_column_preservation(self, sample_florida_names):
        """Test that original columns are preserved in predictions."""
        # Add extra columns to test data
        test_df = sample_florida_names.copy()
        test_df["id"] = range(len(test_df))
        test_df["extra_info"] = "preserved"

        result = pred_fl_reg_ln(test_df, "last")

        # Original columns should be preserved
        assert "id" in result.columns
        assert "extra_info" in result.columns
        assert all(result["id"] == test_df["id"])
        assert all(result["extra_info"] == test_df["extra_info"])
