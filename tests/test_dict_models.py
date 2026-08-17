"""
Tests for the dictionary-based estimators (census first names, census
first+last naive Bayes, six-state voter-file name tables, Wilson intervals).
"""

import numpy as np
import pandas as pd
import pytest

from ethnicolr import (
    estimate_census_full_name,
    estimate_voter_file_full_name,
    lookup_census_first_name,
    lookup_census_surname,
)
from ethnicolr.name_dictionaries import (
    CENSUS_CATEGORIES,
    VOTER_FILE_CATEGORIES,
    wilson_interval,
)


@pytest.fixture
def names_data():
    return pd.DataFrame(
        {
            "last": ["smith", "garcia", "zhang", "xqzwv"],
            "first": ["tyrone", "guadalupe", "wei", "james"],
        }
    )


class TestCensusFirstNameLookup:
    def test_known_values_match_census_brief(self, names_data):
        result = lookup_census_first_name(names_data, "first")
        by_name = result.set_index("first")
        # Published in the C2020BR-13 brief
        assert by_name.loc["guadalupe", "pcthispanic"] >= 95
        assert by_name.loc["tyrone", "pctblack"] >= 75
        assert len(result) == len(names_data)

    def test_unmatched_name_is_nan(self):
        result = lookup_census_first_name(
            pd.DataFrame({"first": ["xqzwvblorp"]}), "first"
        )
        assert np.isnan(result["pctwhite"].iloc[0])

    def test_wilson_bounds_bracket_estimate(self, names_data):
        result = lookup_census_first_name(names_data, "first", uncertainty_level=0.95)
        matched = result.dropna(subset=["pctblack"])
        assert (matched["pctblack_lower"] <= matched["pctblack"]).all()
        assert (matched["pctblack_upper"] >= matched["pctblack"]).all()

    def test_invalid_uncertainty_level_raises(self, names_data):
        with pytest.raises(ValueError, match="between 0 and 1"):
            lookup_census_first_name(names_data, "first", uncertainty_level=95)


class TestWilsonInterval:
    def test_analytic_case(self):
        # p=0.5, n=100, 95%: Wilson interval approx [0.404, 0.596]
        lower_bound, upper_bound = wilson_interval(
            np.array([0.5]), np.array([100.0]), 0.95
        )
        assert abs(lower_bound[0] - 0.404) < 0.005
        assert abs(upper_bound[0] - 0.596) < 0.005

    def test_width_shrinks_with_n(self):
        small_sample_lower, small_sample_upper = wilson_interval(
            np.array([0.3]), np.array([100.0]), 0.95
        )
        large_sample_lower, large_sample_upper = wilson_interval(
            np.array([0.3]), np.array([10000.0]), 0.95
        )
        assert (large_sample_upper - large_sample_lower) < (
            small_sample_upper - small_sample_lower
        )


class TestCensusSurnameWilsonBounds:
    def test_bounds_added_and_tight_for_common_names(self):
        data = pd.DataFrame({"last": ["smith"]})
        result = lookup_census_surname(data, "last", year=2020, uncertainty_level=0.95)
        row = result.iloc[0]
        assert row["pctwhite_lower"] <= row["pctwhite"] <= row["pctwhite_upper"]
        assert row["pctwhite_upper"] - row["pctwhite_lower"] < 0.5


class TestCensusFullNameEstimate:
    def test_dictionary_only_rows_do_not_resolve_neural_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def fail_if_called(*arguments: object) -> None:
            raise AssertionError(f"unexpected neural model resolution: {arguments}")

        monkeypatch.setattr(
            "ethnicolr.name_dictionaries.resolve_model_bundle", fail_if_called
        )
        data = pd.DataFrame(
            {"last": ["smith", "garcia"], "first": ["tyrone", "guadalupe"]}
        )

        result = estimate_census_full_name(data, "last", "first")

        assert result["model_id"].str.startswith("census-name-dictionary-").all()

    def test_first_name_moves_posterior(self, names_data):
        result = estimate_census_full_name(names_data, "last", "first")
        by_last = result.set_index("last")
        # Smith alone is majority-white; Tyrone flips it
        assert by_last.loc["smith", "race"] == "black"
        assert by_last.loc["garcia", "race"] == "hispanic"
        assert by_last.loc["zhang", "race"] == "api"

    def test_evidence_basis_column(self, names_data):
        result = estimate_census_full_name(names_data, "last", "first")
        by_last = result.set_index("last")
        assert (
            by_last.loc["smith", "evidence_basis"]
            == "first-name-and-surname-dictionaries"
        )
        assert (
            by_last.loc["xqzwv", "evidence_basis"]
            == "surname-model-and-first-name-dictionary"
        )

    def test_probabilities_sum_to_one(self, names_data):
        result = estimate_census_full_name(names_data, "last", "first")
        full = result[
            result["evidence_basis"].isin(
                ["first-name-and-surname-dictionaries", "surname-dictionary"]
            )
        ]
        sums = full[CENSUS_CATEGORIES].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_bayes_math_hand_case(self):
        """Posterior ∝ p(r|l)·p(r|f)/π must beat either alone for aligned names."""
        data = pd.DataFrame({"last": ["washington"], "first": ["tyrone"]})
        combined = estimate_census_full_name(data, "last", "first")
        last_only = estimate_census_full_name(
            pd.DataFrame({"last": ["washington"], "first": [""]}), "last", "first"
        )
        assert combined["black"].iloc[0] > last_only["black"].iloc[0]

    def test_prior_shifts(self, names_data):
        base = estimate_census_full_name(names_data, "last", "first")
        prior = {
            category: 0.9 if category == "hispanic" else 0.02
            for category in CENSUS_CATEGORIES
        }
        adjusted = estimate_census_full_name(
            names_data, "last", "first", target_prior=prior
        )
        full = base["evidence_basis"].isin(
            ["first-name-and-surname-dictionaries", "surname-dictionary"]
        )
        assert (
            adjusted.loc[full, "hispanic"].to_numpy()
            >= base.loc[full, "hispanic"].to_numpy()
        ).all()

    def test_row_count_preserved(self, names_data):
        assert len(estimate_census_full_name(names_data, "last", "first")) == len(
            names_data
        )


class TestVoterFileFullNameEstimate:
    def test_known_predictions(self, names_data):
        result = estimate_voter_file_full_name(names_data, "last", "first")
        by_last = result.set_index("last")
        assert by_last.loc["smith", "race"] == "black"
        assert by_last.loc["garcia", "race"] == "hispanic"
        assert by_last.loc["zhang", "race"] == "asian"

    def test_basis_fallbacks(self):
        data = pd.DataFrame(
            {"last": ["xqzwvblorp", "smith"], "first": ["james", "zzzyblorp"]}
        )
        result = estimate_voter_file_full_name(data, "last", "first")
        assert result["evidence_basis"].tolist() == [
            "first-name-dictionary",
            "surname-dictionary",
        ]

    def test_none_basis_gives_nan(self):
        data = pd.DataFrame({"last": ["xqzwvblorp"], "first": ["zzzyblorp"]})
        result = estimate_voter_file_full_name(data, "last", "first")
        assert result["evidence_basis"].iloc[0] == "none"
        assert np.isnan(result["white"].iloc[0])

    def test_probabilities_sum_to_one(self, names_data):
        result = estimate_voter_file_full_name(names_data, "last", "first")
        scored_rows = result["evidence_basis"] != "none"
        assert np.allclose(
            result.loc[scored_rows, VOTER_FILE_CATEGORIES].sum(axis=1),
            1.0,
            atol=1e-6,
        )

    def test_prior_shifts(self, names_data):
        base = estimate_voter_file_full_name(names_data, "last", "first")
        prior = {
            category: 0.9 if category == "hispanic" else 0.025
            for category in VOTER_FILE_CATEGORIES
        }
        adjusted = estimate_voter_file_full_name(
            names_data, "last", "first", target_prior=prior
        )
        scored_rows = base["evidence_basis"] != "none"
        assert (
            adjusted.loc[scored_rows, "hispanic"].to_numpy()
            >= base.loc[scored_rows, "hispanic"].to_numpy()
        ).all()
