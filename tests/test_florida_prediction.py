"""Tests for the 2022 Florida voter name-pattern estimators."""

import inspect

import pandas as pd
import pytest

from ethnicolr import (
    estimate_florida_voter_full_name,
    estimate_florida_voter_surname,
)

from .helpers import assert_prediction_quality

FLORIDA_CATEGORIES = ["asian", "hispanic", "nh_black", "nh_white", "other"]


def test_florida_api_has_no_historical_model_selector():
    """The public API exposes the current model, not an artifact menu."""
    assert "model" not in inspect.signature(estimate_florida_voter_surname).parameters
    assert "model" not in inspect.signature(estimate_florida_voter_full_name).parameters


def test_surname_estimate(sample_florida_names):
    result = estimate_florida_voter_surname(sample_florida_names, "last")

    assert_prediction_quality(result, "florida_5cat")
    assert len(result) == len(sample_florida_names)
    assert set(FLORIDA_CATEGORIES).issubset(result.columns)
    assert result["model_id"].eq("florida-voter-surname").all()


def test_full_name_estimate(sample_florida_names):
    result = estimate_florida_voter_full_name(sample_florida_names, "last", "first")

    assert_prediction_quality(result, "florida_5cat")
    assert len(result) == len(sample_florida_names)
    assert set(FLORIDA_CATEGORIES).issubset(result.columns)
    assert "__ethnicolr_full_name" not in result.columns
    assert result["model_id"].eq("florida-voter-full-name").all()


@pytest.mark.parametrize("uncertainty_level", [0.8, 0.9, 0.95])
def test_surname_uncertainty_levels(sample_florida_names, uncertainty_level):
    result = estimate_florida_voter_surname(
        sample_florida_names,
        "last",
        uncertainty_level=uncertainty_level,
        mc_iterations=20,
    )

    assert_prediction_quality(result, "florida_5cat", with_uncertainty=True)
    assert len([column for column in result if column.endswith("_mc_mean")]) == 5


@pytest.mark.parametrize("uncertainty_level", [0.8, 0.9, 0.95])
def test_full_name_uncertainty_levels(sample_florida_names, uncertainty_level):
    result = estimate_florida_voter_full_name(
        sample_florida_names,
        "last",
        "first",
        uncertainty_level=uncertainty_level,
        mc_iterations=20,
    )

    assert_prediction_quality(result, "florida_5cat", with_uncertainty=True)
    assert len([column for column in result if column.endswith("_mc_mean")]) == 5


def test_surname_and_full_name_share_categories(sample_florida_names):
    surname_result = estimate_florida_voter_surname(sample_florida_names, "last")
    full_name_result = estimate_florida_voter_full_name(
        sample_florida_names, "last", "first"
    )

    assert (
        surname_result[FLORIDA_CATEGORIES].shape
        == full_name_result[FLORIDA_CATEGORIES].shape
    )


def test_extensive_surname_estimate_uses_multiple_categories(extensive_names):
    result = estimate_florida_voter_surname(extensive_names, "last")

    assert_prediction_quality(result, "florida_5cat")
    assert result["race"].nunique() >= 4
    assert result["other"].ge(0).all()


def test_missing_columns_raise(sample_florida_names):
    with pytest.raises(ValueError, match="Surname column"):
        estimate_florida_voter_surname(sample_florida_names, "missing")

    with pytest.raises(ValueError, match="First-name column"):
        estimate_florida_voter_full_name(sample_florida_names, "last", "missing")


def test_empty_data_returns_expected_columns():
    empty_data = pd.DataFrame(columns=["last", "first"])
    result = estimate_florida_voter_surname(empty_data, "last")

    assert result.empty
    assert set(FLORIDA_CATEGORIES).issubset(result.columns)


def test_single_row(sample_florida_names):
    one_name = sample_florida_names.head(1)

    surname_result = estimate_florida_voter_surname(one_name, "last")
    full_name_result = estimate_florida_voter_full_name(one_name, "last", "first")

    assert len(surname_result) == 1
    assert len(full_name_result) == 1


def test_results_are_deterministic(sample_florida_names):
    first_surname_result = estimate_florida_voter_surname(sample_florida_names, "last")
    second_surname_result = estimate_florida_voter_surname(sample_florida_names, "last")
    first_full_name_result = estimate_florida_voter_full_name(
        sample_florida_names, "last", "first"
    )
    second_full_name_result = estimate_florida_voter_full_name(
        sample_florida_names, "last", "first"
    )

    pd.testing.assert_frame_equal(first_surname_result, second_surname_result)
    pd.testing.assert_frame_equal(first_full_name_result, second_full_name_result)


def test_input_columns_are_preserved(sample_florida_names):
    data = sample_florida_names.copy()
    data["record_id"] = range(len(data))

    result = estimate_florida_voter_surname(data, "last")

    pd.testing.assert_series_equal(result["record_id"], data["record_id"])
