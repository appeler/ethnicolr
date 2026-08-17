"""
Tests for the calibration / prior-adjustment / conformal-set layer.
"""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from ethnicolr import estimate_census_surname, estimate_wikipedia_surname
from ethnicolr.torch_utils import (
    adjust_probabilities_for_prior,
    build_conformal_prediction_sets,
)

MODELS_DIR = Path(str(files("ethnicolr"))) / "models"


def all_stats_files() -> list[Path]:
    return sorted(MODELS_DIR.rglob("*_stats_pt.json")) + sorted(
        MODELS_DIR.rglob("*_stats_pytorch.json")
    )


class TestStatsArtifacts:
    def test_every_model_has_stats(self):
        assert len(all_stats_files()) == 9

    def test_stats_schema(self):
        for path in all_stats_files():
            model_statistics = json.loads(path.read_text())
            assert model_statistics["temperature"] > 0
            training_distribution = model_statistics["train_class_distribution"]
            assert abs(sum(training_distribution.values()) - 1.0) < 1e-6
            assert set(training_distribution) == set(model_statistics["classes"])
            for conformal_quantile in model_statistics["conformal_quantiles"].values():
                assert 0 < conformal_quantile <= 1.0
            metrics = model_statistics["metrics"]
            assert metrics["ece_post"] <= metrics["ece_pre"] + 0.01
            if not model_statistics.get("calibration_status", "").startswith(
                "invalid-"
            ):
                for level, empirical_coverage in metrics[
                    "conformal_empirical_coverage"
                ].items():
                    assert abs(empirical_coverage - float(level)) < 0.03, (
                        f"{path.name}: coverage@{level} = {empirical_coverage}"
                    )


class TestPriorAdjustment:
    def test_analytic_reweighting(self):
        category_probabilities = np.array([[0.6, 0.4]])
        categories = ["a", "b"]
        training_distribution = {"a": 0.5, "b": 0.5}
        target_prior = {"a": 0.9, "b": 0.1}
        adjusted_probabilities = adjust_probabilities_for_prior(
            category_probabilities,
            categories,
            target_prior,
            training_distribution,
        )
        expected_a = (0.6 * 0.9 / 0.5) / (0.6 * 0.9 / 0.5 + 0.4 * 0.1 / 0.5)
        assert np.allclose(adjusted_probabilities, [[expected_a, 1 - expected_a]])
        assert np.allclose(adjusted_probabilities.sum(axis=1), 1.0)

    def test_uniform_prior_matching_train_is_identity(self):
        category_probabilities = np.array([[0.2, 0.3, 0.5]])
        categories = ["a", "b", "c"]
        training_distribution = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        adjusted_probabilities = adjust_probabilities_for_prior(
            category_probabilities,
            categories,
            training_distribution,
            training_distribution,
        )
        assert np.allclose(adjusted_probabilities, category_probabilities)

    def test_missing_class_raises(self):
        with pytest.raises(ValueError, match="missing classes"):
            adjust_probabilities_for_prior(
                np.array([[0.5, 0.5]]), ["a", "b"], {"a": 1.0}, {"a": 0.5, "b": 0.5}
            )


class TestConformalPredictionSets:
    def test_sets_contain_most_likely_category_and_grow_with_coverage(self):
        category_probabilities = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
        categories = ["a", "b", "c"]
        lower_coverage_sets = build_conformal_prediction_sets(
            category_probabilities, categories, 0.5
        )
        higher_coverage_sets = build_conformal_prediction_sets(
            category_probabilities, categories, 0.95
        )
        for row_index, row_probabilities in enumerate(category_probabilities):
            most_likely_category = categories[int(row_probabilities.argmax())]
            assert most_likely_category in lower_coverage_sets[row_index]
            assert set(lower_coverage_sets[row_index]) <= set(
                higher_coverage_sets[row_index]
            )

    def test_quantile_one_returns_all_categories(self):
        category_probabilities = np.array([[0.5, 0.3, 0.2]])
        prediction_sets = build_conformal_prediction_sets(
            category_probabilities, ["a", "b", "c"], 1.0
        )
        assert len(prediction_sets[0]) == 3


class TestPredictionAPI:
    def test_census_prior_shifts_probabilities(self, sample_census_names):
        base = estimate_census_surname(sample_census_names.copy(), "last", year=2010)
        heavy_hispanic = {"api": 0.01, "black": 0.01, "hispanic": 0.97, "white": 0.01}
        adjusted = estimate_census_surname(
            sample_census_names.copy(), "last", year=2010, target_prior=heavy_hispanic
        )
        assert (adjusted["hispanic"] > base["hispanic"]).all()
        prob_cols = ["api", "black", "hispanic", "white"]
        assert np.allclose(adjusted[prob_cols].sum(axis=1), 1.0, atol=1e-5)

    def test_census_coverage_sets(self, sample_census_names):
        result = estimate_census_surname(
            sample_census_names.copy(), "last", year=2010, conformal_coverage=0.9
        )
        assert "race_set" in result.columns
        for _, row in result.iterrows():
            assert row["race"] in row["race_set"]

    def test_wiki_coverage_sets(self, sample_wiki_names):
        result = estimate_wikipedia_surname(
            sample_wiki_names.copy(), "last", conformal_coverage=0.9
        )
        assert "race_set" in result.columns
        for _, row in result.iterrows():
            assert isinstance(row["race_set"], list)
            assert row["race"] in row["race_set"]

    def test_prior_with_coverage_raises(self, sample_census_names):
        with pytest.raises(ValueError, match="cannot be used together"):
            estimate_census_surname(
                sample_census_names.copy(),
                "last",
                year=2010,
                target_prior={
                    "api": 0.25,
                    "black": 0.25,
                    "hispanic": 0.25,
                    "white": 0.25,
                },
                conformal_coverage=0.9,
            )

    def test_prior_with_uncertainty_level_raises(self, sample_census_names):
        with pytest.raises(
            ValueError,
            match="`target_prior` cannot be used when `uncertainty_level` is set",
        ):
            estimate_census_surname(
                sample_census_names.copy(),
                "last",
                year=2010,
                uncertainty_level=0.9,
                target_prior={
                    "api": 0.25,
                    "black": 0.25,
                    "hispanic": 0.25,
                    "white": 0.25,
                },
            )

    def test_invalid_coverage_level_raises(self, sample_census_names):
        with pytest.raises(ValueError, match="coverage must be one of"):
            estimate_census_surname(
                sample_census_names.copy(), "last", year=2010, conformal_coverage=0.5
            )

    def test_retrained_florida_model_returns_conformal_sets(self, sample_florida_names):
        from ethnicolr import estimate_florida_voter_surname

        result = estimate_florida_voter_surname(
            sample_florida_names.copy(), "last", conformal_coverage=0.9
        )
        assert result["calibration_status"].eq("validated-source-disjoint").all()
        assert result["race_set"].map(list).map(len).ge(1).all()

    def test_wiki_balanced_prior_runs(self, sample_wiki_names):
        model_statistics = json.loads(
            (MODELS_DIR / "wiki" / "lstm" / "wiki_ln_stats_pt.json").read_text()
        )
        categories = model_statistics["classes"]
        uniform_prior = {category: 1 / len(categories) for category in categories}
        result = estimate_wikipedia_surname(
            sample_wiki_names.copy(), "last", target_prior=uniform_prior
        )
        assert np.allclose(result[categories].sum(axis=1), 1.0, atol=1e-5)


class TestTemperatureApplied:
    def test_predictions_reflect_temperature(self, sample_census_names):
        """With T != 1 stored, probabilities differ from the raw-softmax run."""
        model_statistics = json.loads(
            (
                MODELS_DIR / "census" / "lstm" / "census2010_ln_stats_pytorch.json"
            ).read_text()
        )
        assert model_statistics["temperature"] != 1.0
        result = estimate_census_surname(sample_census_names.copy(), "last", year=2010)
        # Probabilities remain a valid distribution after scaling
        prob_cols = ["api", "black", "hispanic", "white"]
        assert np.allclose(result[prob_cols].sum(axis=1), 1.0, atol=1e-5)
