"""
Tests for the dictionary-based estimators (census first names, census
first+last naive Bayes, Rosenman voter dictionaries, Wilson intervals).
"""

import numpy as np
import pandas as pd
import pytest

from ethnicolr import census_fn, census_ln, pred_census_name, pred_voter_name
from ethnicolr.dict_models import CENSUS_CATS, VOTER_CATS, wilson_interval


@pytest.fixture
def names_df():
    return pd.DataFrame(
        {
            "last": ["smith", "garcia", "zhang", "xqzwv"],
            "first": ["tyrone", "guadalupe", "wei", "james"],
        }
    )


class TestCensusFn:
    def test_known_values_match_census_brief(self, names_df):
        result = census_fn(names_df, "first")
        by_name = result.set_index("first")
        # Published in the C2020BR-13 brief
        assert by_name.loc["guadalupe", "pcthispanic"] >= 95
        assert by_name.loc["tyrone", "pctblack"] >= 75
        assert len(result) == len(names_df)

    def test_unmatched_name_is_nan(self):
        result = census_fn(pd.DataFrame({"first": ["xqzwvblorp"]}), "first")
        assert np.isnan(result["pctwhite"].iloc[0])

    def test_wilson_bounds_bracket_estimate(self, names_df):
        result = census_fn(names_df, "first", conf_int=0.95)
        matched = result.dropna(subset=["pctblack"])
        assert (matched["pctblack_lb"] <= matched["pctblack"]).all()
        assert (matched["pctblack_ub"] >= matched["pctblack"]).all()

    def test_invalid_conf_int_raises(self, names_df):
        with pytest.raises(ValueError, match="between 0 and 1"):
            census_fn(names_df, "first", conf_int=95)


class TestWilsonInterval:
    def test_analytic_case(self):
        # p=0.5, n=100, 95%: Wilson interval approx [0.404, 0.596]
        lb, ub = wilson_interval(np.array([0.5]), np.array([100.0]), 0.95)
        assert abs(lb[0] - 0.404) < 0.005
        assert abs(ub[0] - 0.596) < 0.005

    def test_width_shrinks_with_n(self):
        lb1, ub1 = wilson_interval(np.array([0.3]), np.array([100.0]), 0.95)
        lb2, ub2 = wilson_interval(np.array([0.3]), np.array([10000.0]), 0.95)
        assert (ub2 - lb2) < (ub1 - lb1)


class TestCensusLnWilson:
    def test_bounds_added_and_tight_for_common_names(self):
        df = pd.DataFrame({"last": ["smith"]})
        result = census_ln(df, "last", 2020, conf_int=0.95)
        row = result.iloc[0]
        assert row["pctwhite_lb"] <= row["pctwhite"] <= row["pctwhite_ub"]
        assert row["pctwhite_ub"] - row["pctwhite_lb"] < 0.5


class TestPredCensusName:
    def test_first_name_moves_posterior(self, names_df):
        result = pred_census_name(names_df, "last", "first")
        by_last = result.set_index("last")
        # Smith alone is majority-white; Tyrone flips it
        assert by_last.loc["smith", "race"] == "black"
        assert by_last.loc["garcia", "race"] == "hispanic"
        assert by_last.loc["zhang", "race"] == "api"

    def test_basis_column(self, names_df):
        result = pred_census_name(names_df, "last", "first")
        by_last = result.set_index("last")
        assert by_last.loc["smith", "basis"] == "dict_both"
        assert by_last.loc["xqzwv", "basis"] == "lstm+dict_first"

    def test_probabilities_sum_to_one(self, names_df):
        result = pred_census_name(names_df, "last", "first")
        full = result[result["basis"].isin(["dict_both", "dict_last"])]
        sums = full[CENSUS_CATS].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6)

    def test_bayes_math_hand_case(self):
        """Posterior ∝ p(r|l)·p(r|f)/π must beat either alone for aligned names."""
        df = pd.DataFrame({"last": ["washington"], "first": ["tyrone"]})
        combined = pred_census_name(df, "last", "first")
        last_only = pred_census_name(
            pd.DataFrame({"last": ["washington"], "first": [""]}), "last", "first"
        )
        assert combined["black"].iloc[0] > last_only["black"].iloc[0]

    def test_prior_shifts(self, names_df):
        base = pred_census_name(names_df, "last", "first")
        prior = {c: (0.9 if c == "hispanic" else 0.02) for c in CENSUS_CATS}
        adjusted = pred_census_name(names_df, "last", "first", prior=prior)
        full = base["basis"].isin(["dict_both", "dict_last"])
        assert (
            adjusted.loc[full, "hispanic"].to_numpy()
            >= base.loc[full, "hispanic"].to_numpy()
        ).all()

    def test_row_count_preserved(self, names_df):
        assert len(pred_census_name(names_df, "last", "first")) == len(names_df)


class TestPredVoterName:
    def test_known_predictions(self, names_df):
        result = pred_voter_name(names_df, "last", "first")
        by_last = result.set_index("last")
        assert by_last.loc["smith", "race"] == "black"
        assert by_last.loc["garcia", "race"] == "hispanic"
        assert by_last.loc["zhang", "race"] == "asian"

    def test_basis_fallbacks(self):
        df = pd.DataFrame(
            {"last": ["xqzwvblorp", "smith"], "first": ["james", "zzzyblorp"]}
        )
        result = pred_voter_name(df, "last", "first")
        assert result["basis"].tolist() == ["dict_first", "dict_last"]

    def test_none_basis_gives_nan(self):
        df = pd.DataFrame({"last": ["xqzwvblorp"], "first": ["zzzyblorp"]})
        result = pred_voter_name(df, "last", "first")
        assert result["basis"].iloc[0] == "none"
        assert np.isnan(result["white"].iloc[0])

    def test_probabilities_sum_to_one(self, names_df):
        result = pred_voter_name(names_df, "last", "first")
        found = result["basis"] != "none"
        assert np.allclose(result.loc[found, VOTER_CATS].sum(axis=1), 1.0, atol=1e-6)

    def test_prior_shifts(self, names_df):
        base = pred_voter_name(names_df, "last", "first")
        prior = {c: (0.9 if c == "hispanic" else 0.025) for c in VOTER_CATS}
        adjusted = pred_voter_name(names_df, "last", "first", prior=prior)
        found = base["basis"] != "none"
        assert (
            adjusted.loc[found, "hispanic"].to_numpy()
            >= base.loc[found, "hispanic"].to_numpy()
        ).all()
