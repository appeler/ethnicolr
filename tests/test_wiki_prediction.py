"""Tests for Wikipedia race/ethnicity name-pattern estimators."""

import pandas as pd
import pytest

from ethnicolr import estimate_wikipedia_full_name, estimate_wikipedia_surname

from .helpers import assert_prediction_quality, validate_race_prediction_consistency


def test_wikipedia_surname_estimate(sample_wiki_names):
    result = estimate_wikipedia_surname(sample_wiki_names, "last")

    assert_prediction_quality(result, "wiki")
    assert validate_race_prediction_consistency(result)
    assert len(result) == len(sample_wiki_names)


def test_wikipedia_full_name_estimate(sample_wiki_names):
    result = estimate_wikipedia_full_name(sample_wiki_names, "last", "first")

    assert_prediction_quality(result, "wiki")
    assert validate_race_prediction_consistency(result)
    assert len(result) == len(sample_wiki_names)
    assert not any(column.startswith("__ethnicolr_full_name") for column in result)


@pytest.mark.parametrize("uncertainty_level", [0.8, 0.9, 0.95])
def test_wikipedia_surname_uncertainty(sample_wiki_names, uncertainty_level):
    result = estimate_wikipedia_surname(
        sample_wiki_names,
        "last",
        uncertainty_level=uncertainty_level,
        mc_iterations=20,
    )

    assert_prediction_quality(result, "wiki", with_uncertainty=True)


@pytest.mark.parametrize("uncertainty_level", [0.8, 0.9, 0.95])
def test_wikipedia_full_name_uncertainty(sample_wiki_names, uncertainty_level):
    result = estimate_wikipedia_full_name(
        sample_wiki_names,
        "last",
        "first",
        uncertainty_level=uncertainty_level,
        mc_iterations=20,
    )

    assert_prediction_quality(result, "wiki", with_uncertainty=True)


def test_full_name_preserves_duplicates_and_input_values():
    data = pd.DataFrame(
        [
            {"last": "O'Neil", "first": "John"},
            {"last": "ONeil", "first": "John"},
            {"last": "Smith", "first": "John"},
            {"last": "Smith", "first": "John"},
        ]
    )

    result = estimate_wikipedia_full_name(data, "last", "first")

    assert len(result) == len(data)
    assert result["last"].tolist() == data["last"].tolist()
    assert result["first"].tolist() == data["first"].tolist()


def test_full_name_reports_script_support():
    data = pd.DataFrame(
        [
            {"last": "Szathmáry", "first": "Emöke"},
            {"last": "Müller", "first": "Björn"},
            {"last": "Ξενοφῶν", "first": "Nikos"},
            {"last": "Владимир", "first": "Ivan"},
            {"last": "张", "first": "Wei"},
        ]
    )

    result = estimate_wikipedia_full_name(data, "last", "first")

    assert result.loc[:1, "script_supported"].all()
    assert not result.loc[2:, "script_supported"].any()
    assert result.loc[2:, "abstained"].all()
    assert result.loc[2:, "abstention_reason"].eq("unsupported-script").all()


def test_surname_and_full_name_share_output_categories(sample_wiki_names):
    surname_result = estimate_wikipedia_surname(sample_wiki_names, "last")
    full_name_result = estimate_wikipedia_full_name(sample_wiki_names, "last", "first")

    surname_probability_columns = {
        column
        for column in surname_result
        if column in full_name_result and column[0].isupper()
    }
    assert surname_probability_columns


def test_empty_names_abstain_without_dropping_rows():
    data = pd.DataFrame(
        [
            {"last": "", "first": ""},
            {"last": "Smith", "first": ""},
            {"last": "", "first": "Mary"},
        ]
    )

    surname_result = estimate_wikipedia_surname(data, "last")
    full_name_result = estimate_wikipedia_full_name(data, "last", "first")

    assert len(surname_result) == len(data)
    assert len(full_name_result) == len(data)
    assert surname_result.loc[0, "abstention_reason"] == "missing-name"
    assert full_name_result.loc[0, "abstention_reason"] == "missing-name"


def test_missing_columns_raise(sample_wiki_names):
    with pytest.raises(ValueError, match="Surname column"):
        estimate_wikipedia_surname(sample_wiki_names, "missing")

    with pytest.raises(ValueError, match="First-name column"):
        estimate_wikipedia_full_name(sample_wiki_names, "last", "missing")


def test_long_names_preserve_rows():
    data = pd.DataFrame(
        [
            {"last": "a" * 100, "first": "John"},
            {"last": "Smith", "first": "b" * 100},
            {"last": "Normal", "first": "Jane"},
        ]
    )

    result = estimate_wikipedia_full_name(data, "last", "first")

    assert len(result) == len(data)
    assert_prediction_quality(result, "wiki")


def test_common_european_surnames_are_recognized(extensive_names):
    european_rows = extensive_names[extensive_names["expected_major"] == "white"]
    result = estimate_wikipedia_surname(european_rows, "last")

    european_accuracy = result["race"].str.contains("GreaterEuropean").mean()
    assert european_accuracy >= 0.5
