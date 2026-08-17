"""Tests for the Ethnicolr 2.0 public naming and argument contract."""

from __future__ import annotations

import inspect

import pytest

import ethnicolr

COMMON_COLUMN_ARGUMENTS = {"data", "surname_column", "first_name_column"}
OLD_PUBLIC_NAMES = {
    "census_fn",
    "census_ln",
    "pred_census_ln",
    "pred_census_name",
    "pred_fl_reg_ln",
    "pred_fl_reg_ln_five_cat",
    "pred_fl_reg_name",
    "pred_fl_reg_name_five_cat",
    "pred_nc_reg_name",
    "pred_voter_name",
    "pred_wiki_ln",
    "pred_wiki_name",
    "pred_wiki_origin",
}


def test_public_function_family_uses_consistent_verbs() -> None:
    assert all(name.startswith(("estimate_", "lookup_")) for name in ethnicolr.__all__)
    assert OLD_PUBLIC_NAMES.isdisjoint(ethnicolr.__all__)
    assert all(not hasattr(ethnicolr, name) for name in OLD_PUBLIC_NAMES)


@pytest.mark.parametrize("function_name", ethnicolr.__all__)
def test_public_functions_use_explicit_column_arguments(function_name: str) -> None:
    parameters = inspect.signature(getattr(ethnicolr, function_name)).parameters

    assert "data" in parameters
    assert COMMON_COLUMN_ARGUMENTS.intersection(parameters)
    assert "df" not in parameters
    assert "lname_col" not in parameters
    assert "fname_col" not in parameters


@pytest.mark.parametrize(
    "function_name",
    [
        "estimate_census_surname",
        "estimate_florida_voter_full_name",
        "estimate_florida_voter_surname",
        "estimate_north_carolina_voter_full_name",
        "estimate_wikipedia_full_name",
        "estimate_wikipedia_origin",
        "estimate_wikipedia_surname",
    ],
)
def test_model_estimators_share_uncertainty_arguments(function_name: str) -> None:
    parameters = inspect.signature(getattr(ethnicolr, function_name)).parameters

    assert "uncertainty_level" in parameters
    assert "mc_iterations" in parameters
    assert "target_prior" in parameters
    assert "conformal_coverage" in parameters
