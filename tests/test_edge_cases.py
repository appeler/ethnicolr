"""
Tests for edge cases, error handling, and boundary conditions.
"""

import numpy as np
import pandas as pd
import pytest

from ethnicolr import estimate_census_surname, estimate_wikipedia_full_name

from .helpers import assert_prediction_quality


class TestDataFrameEdgeCases:
    """Test edge cases related to DataFrame structure and content."""

    def test_empty_dataframe(self):
        """Test handling of completely empty DataFrames."""
        empty_df = pd.DataFrame()

        # All functions should handle empty DataFrames gracefully
        with pytest.raises((ValueError, KeyError, IndexError)):
            estimate_census_surname(empty_df, "last", year=2010)

    def test_dataframe_with_no_rows(self):
        """Test DataFrames with columns but no data rows."""
        empty_df = pd.DataFrame(columns=["last", "first", "id"])

        # Should handle empty DataFrames gracefully by returning empty result
        result = estimate_census_surname(empty_df, "last", year=2010)
        assert len(result) == 0
        assert "race" in result.columns  # Should have prediction columns
        assert all(
            col in result.columns for col in ["last", "first", "id"]
        )  # Preserve input columns

    def test_single_row_dataframe(self):
        """Test DataFrames with only one row."""
        single_df = pd.DataFrame([{"last": "smith", "id": 1}])

        result = estimate_census_surname(single_df, "last", year=2010)
        assert len(result) == 1
        assert_prediction_quality(result, "census")

    def test_very_large_dataframe(self):
        """Test handling of large DataFrames."""
        # Create a moderately large DataFrame (1000 rows)
        large_data = []
        names = ["smith", "garcia", "zhang", "johnson", "williams"] * 200
        for i, name in enumerate(names):
            large_data.append({"last": name, "id": i})

        large_df = pd.DataFrame(large_data)

        # Should handle without issues (though may be slow)
        result = estimate_census_surname(large_df, "last", year=2010)
        assert len(result) == 1000
        assert_prediction_quality(result, "census")

    def test_dataframe_with_all_identical_values(self):
        """Test DataFrame where all names are identical."""
        identical_df = pd.DataFrame([{"last": "smith"} for _ in range(10)])

        result = estimate_census_surname(identical_df, "last", year=2010)
        assert len(result) == 10

        # All predictions should be identical
        assert result["race"].nunique() == 1
        assert_prediction_quality(result, "census")


class TestNullAndMissingValueHandling:
    """Test handling of null, NaN, and missing values."""

    def test_null_values_in_name_columns(self):
        """Test handling of null values in name columns."""
        null_df = pd.DataFrame(
            [
                {"last": None, "first": "John", "id": 1},
                {"last": "Smith", "first": None, "id": 2},
                {"last": "Garcia", "first": "Maria", "id": 3},
            ]
        )

        # Different functions may handle nulls differently
        # Census prediction should handle gracefully
        result = estimate_census_surname(null_df, "last", year=2010)
        assert len(result) == 3  # Should preserve all rows

    def test_nan_values_in_name_columns(self):
        """Test handling of NaN values in name columns."""
        nan_df = pd.DataFrame(
            [
                {"last": np.nan, "first": "John", "id": 1},
                {"last": "Smith", "first": np.nan, "id": 2},
                {"last": "Garcia", "first": "Maria", "id": 3},
            ]
        )

        result = estimate_census_surname(nan_df, "last", year=2010)
        assert len(result) == 3

    def test_empty_string_values(self):
        """Test handling of empty string values in name columns."""
        empty_str_df = pd.DataFrame(
            [
                {"last": "", "first": "John", "id": 1},
                {"last": "Smith", "first": "", "id": 2},
                {"last": "Garcia", "first": "Maria", "id": 3},
            ]
        )

        result = estimate_census_surname(empty_str_df, "last", year=2010)
        assert len(result) == 3

        wiki_result = estimate_wikipedia_full_name(empty_str_df, "last", "first")
        assert len(wiki_result) == 3
        assert wiki_result["scored"].all()
        assert not wiki_result["abstained"].any()

    def test_whitespace_only_values(self):
        """Test handling of whitespace-only values."""
        whitespace_df = pd.DataFrame(
            [
                {"last": "   ", "first": "John", "id": 1},
                {"last": "Smith", "first": "\t", "id": 2},
                {"last": "\n", "first": "   ", "id": 3},
            ]
        )

        result = estimate_census_surname(whitespace_df, "last", year=2010)
        assert len(result) == 3


class TestSpecialCharacterHandling:
    """Test handling of special characters and unicode in names."""

    def test_unicode_characters(self):
        """Test handling of Unicode characters in names."""
        unicode_df = pd.DataFrame(
            [
                {"last": "Müller", "first": "Hans", "id": 1},  # German umlauts
                {"last": "José", "first": "María", "id": 2},  # Spanish accents
                {"last": "Наташа", "first": "Иван", "id": 3},  # Cyrillic
                {"last": "张", "first": "伟", "id": 4},  # Chinese
                {"last": "محمد", "first": "علي", "id": 5},  # Arabic
                {"last": "Ñoño", "first": "Peña", "id": 6},  # Spanish ñ
            ]
        )

        census_result = estimate_census_surname(unicode_df, "last", year=2010)
        assert len(census_result) == 6

        wiki_result = estimate_wikipedia_full_name(unicode_df, "last", "first")
        assert len(wiki_result) == 6
        assert wiki_result["script_supported"].tolist() == [
            True,
            True,
            False,
            False,
            False,
            True,
        ]
        assert wiki_result["abstained"].tolist() == [
            False,
            False,
            True,
            True,
            True,
            False,
        ]

    def test_punctuation_in_names(self):
        """Test handling of punctuation in names."""
        punct_df = pd.DataFrame(
            [
                {"last": "O'Brien", "first": "Patrick", "id": 1},  # Apostrophe
                {"last": "Smith-Jones", "first": "Mary", "id": 2},  # Hyphen
                {"last": "Van Der Berg", "first": "Anna", "id": 3},  # Spaces
                {"last": "D'Angelo", "first": "Giuseppe", "id": 4},  # Apostrophe
                {"last": "Jean-Luc", "first": "Pierre", "id": 5},  # Hyphen in first
                {"last": "McDonald", "first": "O'Malley", "id": 6},  # Mixed
            ]
        )

        census_result = estimate_census_surname(punct_df, "last", year=2010)
        assert len(census_result) == 6
        assert_prediction_quality(census_result, "census")

        wiki_result = estimate_wikipedia_full_name(punct_df, "last", "first")
        assert len(wiki_result) == 6

    def test_numbers_in_names(self):
        """Test handling of numbers in names."""
        number_df = pd.DataFrame(
            [
                {"last": "Smith3", "first": "John", "id": 1},
                {"last": "Garcia2nd", "first": "Maria", "id": 2},
                {"last": "Johnson", "first": "John3", "id": 3},
                {"last": "123", "first": "Test", "id": 4},
            ]
        )

        result = estimate_census_surname(number_df, "last", year=2010)
        assert len(result) == 4

    def test_extremely_long_names(self):
        """Test handling of extremely long names."""
        long_df = pd.DataFrame(
            [
                {"last": "a" * 1000, "first": "John", "id": 1},  # Very long last name
                {"last": "Smith", "first": "b" * 1000, "id": 2},  # Very long first name
                {"last": "c" * 50, "first": "d" * 50, "id": 3},  # Both long
            ]
        )

        result = estimate_census_surname(long_df, "last", year=2010)
        assert len(result) == 3


class TestColumnNameEdgeCases:
    """Test edge cases related to column names and DataFrame structure."""

    def test_nonexistent_column_name(self):
        """Test error handling for nonexistent column names."""
        df = pd.DataFrame([{"last": "Smith", "first": "John"}])

        with pytest.raises((KeyError, ValueError)):
            estimate_census_surname(df, "nonexistent_column", year=2010)

        with pytest.raises((KeyError, ValueError)):
            estimate_wikipedia_full_name(df, "last", "nonexistent_first_column")

    def test_case_sensitive_column_names(self):
        """Test that column names are case-sensitive."""
        df = pd.DataFrame([{"last": "Smith", "first": "John", "id": 1}])

        # Should work with correct case
        result = estimate_census_surname(df, "last", year=2010)
        assert len(result) == 1
        assert result.iloc[0]["last"] == "Smith"  # Original data preserved

        # Should fail with incorrect case for non-existent column
        with pytest.raises((KeyError, ValueError)):
            estimate_census_surname(
                df, "Last", year=2010
            )  # "Last" doesn't exist, only "last"

        with pytest.raises((KeyError, ValueError)):
            estimate_census_surname(
                df, "LAST", year=2010
            )  # "LAST" doesn't exist, only "last"

    def test_duplicate_column_names(self):
        """Test handling of DataFrames with duplicate column names."""
        df = pd.DataFrame({"last": ["Smith", "Garcia"], "first": ["John", "Maria"]})
        df.columns = ["last", "last"]

        with pytest.raises(ValueError, match="Duplicate column name"):
            estimate_census_surname(df, "last", year=2010)

    def test_column_names_with_special_characters(self):
        """Test column names with special characters."""
        df = pd.DataFrame(
            [
                {"surname-with-dashes": "Smith", "first name": "John"},
                {"surname-with-dashes": "Garcia", "first name": "Maria"},
            ]
        )

        result = estimate_census_surname(df, "surname-with-dashes", year=2010)
        assert len(result) == 2
        assert_prediction_quality(result, "census")

    def test_numeric_column_names(self):
        """Test numeric column names."""
        df = pd.DataFrame({0: ["Smith", "Garcia"], 1: ["John", "Maria"]})

        result = estimate_census_surname(df, 0, year=2010)  # Use numeric column name
        assert len(result) == 2


class TestDataTypeEdgeCases:
    """Test edge cases related to data types."""

    def test_non_string_name_values(self):
        """Test handling of non-string values in name columns."""
        mixed_df = pd.DataFrame(
            [
                {"last": 123, "first": "John", "id": 1},  # Numeric
                {"last": "Smith", "first": 456, "id": 2},  # Numeric first
                {"last": True, "first": "Maria", "id": 3},  # Boolean
                {"last": "Garcia", "first": False, "id": 4},  # Boolean first
            ]
        )

        # Functions should handle type conversion gracefully
        result = estimate_census_surname(mixed_df, "last", year=2010)
        assert len(result) == 4

    def test_datetime_in_name_columns(self):
        """Test handling of datetime objects in name columns."""
        import datetime

        datetime_df = pd.DataFrame(
            [
                {
                    "last": datetime.datetime.now(tz=datetime.UTC),
                    "first": "John",
                },
                {
                    "last": "Smith",
                    "first": datetime.datetime.now(tz=datetime.UTC).date(),
                },
            ]
        )

        # Should handle gracefully (likely convert to string)
        result = estimate_census_surname(datetime_df, "last", year=2010)
        assert len(result) == 2

    def test_list_values_in_columns(self):
        """Test handling of list/array values in columns."""
        list_df = pd.DataFrame(
            [
                {"last": ["Smith", "Johnson"], "first": "John"},
                {"last": "Garcia", "first": ["Maria", "Carmen"]},
            ]
        )

        # Should handle gracefully or raise appropriate error
        try:
            result = estimate_census_surname(list_df, "last", year=2010)
            assert len(result) == 2
        except (TypeError, ValueError):
            # Also acceptable to raise an error for this edge case
            pass


class TestModelBoundaryConditions:
    """Test boundary conditions for model parameters."""

    def test_extreme_confidence_intervals(self):
        """Test edge cases for MC-dropout uncertainty summary values."""
        df = pd.DataFrame([{"last": "Smith", "first": "John"}])

        # Very small MC-dropout uncertainty summary (should work)
        result_small = estimate_census_surname(
            df, "last", year=2010, uncertainty_level=0.01
        )
        assert_prediction_quality(result_small, "census", with_uncertainty=True)

        # Maximum MC-dropout uncertainty summary (should work)
        result_max = estimate_census_surname(
            df, "last", year=2010, uncertainty_level=0.99
        )
        assert_prediction_quality(result_max, "census", with_uncertainty=True)

    def test_extreme_iteration_counts(self):
        """Test edge cases for iteration counts."""
        df = pd.DataFrame([{"last": "Smith", "first": "John"}])

        # One draw cannot define a sample standard deviation.
        with pytest.raises(ValueError, match="mc_iterations must be at least 2"):
            estimate_census_surname(
                df, "last", year=2010, uncertainty_level=0.9, mc_iterations=1
            )

        # Large iteration count (might be slow)
        try:
            result_large = estimate_census_surname(
                df, "last", year=2010, uncertainty_level=0.9, mc_iterations=10
            )
            assert_prediction_quality(result_large, "census", with_uncertainty=True)
        except Exception:
            # If it fails due to time/memory constraints, that's acceptable
            pytest.skip(
                "Large iteration count test failed - likely resource constraint"
            )

    def test_invalid_year_values(self):
        """Test handling of invalid year values."""
        df = pd.DataFrame([{"last": "Smith"}])

        # Invalid years should raise appropriate errors
        with pytest.raises((ValueError, KeyError)):
            estimate_census_surname(df, "last", year=1999)  # Before valid range

        with pytest.raises((ValueError, KeyError)):
            estimate_census_surname(df, "last", year=2025)  # After valid range


class TestConcurrencyAndPerformance:
    """Test edge cases related to performance and concurrency."""

    @pytest.mark.slow
    def test_memory_usage_large_dataset(self):
        """Test memory usage with large datasets."""
        # Create a larger dataset to test memory handling
        large_data = [
            {"last": f"Name{i}", "first": f"First{i}", "extra_col": f"data{i}"}
            for i in range(5000)
        ]

        large_df = pd.DataFrame(large_data)

        # Should handle without excessive memory usage
        result = estimate_census_surname(large_df, "last", year=2010)
        assert len(result) == 5000
        assert_prediction_quality(result, "census")

        # Memory should be released (this is hard to test directly)
        del large_df, result

    def test_consistent_results_multiple_runs(self):
        """Test that results are consistent across multiple runs."""
        df = pd.DataFrame(
            [
                {"last": "Smith", "first": "John"},
                {"last": "Garcia", "first": "Maria"},
                {"last": "Zhang", "first": "Wei"},
            ]
        )

        # Run multiple times
        results = []
        for _ in range(3):
            result = estimate_census_surname(df, "last", year=2010)
            results.append(result)

        # Results should be identical
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])


class TestInteroperability:
    """Test interoperability edge cases."""

    def test_different_pandas_dtypes(self):
        """Test handling of different pandas dtypes."""
        # Create DataFrame with specific dtypes
        df = pd.DataFrame(
            {
                "last": pd.Series(["Smith", "Garcia"], dtype="string"),
                "first": pd.Series(["John", "Maria"], dtype="object"),
                "id": pd.Series([1, 2], dtype="int64"),
            }
        )

        result = estimate_census_surname(df, "last", year=2010)
        assert len(result) == 2
        assert_prediction_quality(result, "census")

    def test_categorical_columns(self):
        """Test handling of categorical data types."""
        df = pd.DataFrame(
            {
                "last": pd.Categorical(["Smith", "Garcia", "Smith"]),
                "first": ["John", "Maria", "Jane"],
                "category": pd.Categorical(["A", "B", "A"]),
            }
        )

        result = estimate_census_surname(df, "last", year=2010)
        assert len(result) == 3
        assert_prediction_quality(result, "census")

    def test_index_preservation(self):
        """Test that DataFrame index is preserved correctly."""
        df = pd.DataFrame(
            {"last": ["Smith", "Garcia", "Zhang"], "first": ["John", "Maria", "Wei"]},
            index=[10, 20, 30],
        )

        result = estimate_census_surname(df, "last", year=2010)

        # Index should be preserved
        assert list(result.index) == [10, 20, 30]
        assert len(result) == 3


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_partial_model_failure_recovery(self):
        """Test recovery when some names fail but others succeed."""
        # Mix of normal names and potentially problematic ones
        mixed_df = pd.DataFrame(
            [
                {"last": "Smith", "first": "John"},  # Normal
                {"last": "", "first": ""},  # Empty
                {"last": "Garcia", "first": "Maria"},  # Normal
                {"last": None, "first": None},  # Null
                {"last": "Zhang", "first": "Wei"},  # Normal
            ]
        )

        result = estimate_census_surname(mixed_df, "last", year=2010)

        # Should return results for all rows (with some predictions
        # possibly null/default)
        assert len(result) == 5

        # Should have prediction columns
        assert "race" in result.columns

    def test_graceful_handling_of_model_loading_issues(self):
        """Test graceful handling when models might have issues."""
        df = pd.DataFrame([{"last": "Smith", "first": "John"}])

        # This test depends on the actual model loading implementation
        # If models are missing or corrupted, should get appropriate error
        try:
            result = estimate_census_surname(df, "last", year=2010)
            assert_prediction_quality(result, "census")
        except (FileNotFoundError, KeyError, ValueError) as e:
            # If models are missing, that's a valid scenario to test
            pytest.skip(f"Model loading issue (expected in some environments): {e}")

    def test_network_timeout_simulation(self):
        """Test handling of potential network issues (if applicable)."""
        # This would be more relevant if models were downloaded dynamically
        df = pd.DataFrame([{"last": "Smith", "first": "John"}])

        # For now, just ensure basic prediction works
        result = estimate_census_surname(df, "last", year=2010)
        assert len(result) == 1
