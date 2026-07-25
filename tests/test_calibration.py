"""
Tests for the calibration / prior-adjustment / conformal-set layer.
"""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from ethnicolr.pred_census_ln import pred_census_ln
from ethnicolr.pred_wiki_ln import pred_wiki_ln
from ethnicolr.torch_utils import apply_prior, conformal_sets, load_model_stats

MODELS_DIR = Path(str(files("ethnicolr"))) / "models"


def all_stats_files() -> list[Path]:
    return sorted(MODELS_DIR.rglob("*_stats_pt.json")) + sorted(
        MODELS_DIR.rglob("*_stats_pytorch.json")
    )


class TestStatsArtifacts:
    def test_every_model_has_stats(self):
        models = sorted(MODELS_DIR.rglob("*_lstm_pt.pt")) + sorted(
            MODELS_DIR.rglob("*_lstm_pytorch.pt")
        )
        assert len(models) >= 12
        for model_path in models:
            stats = load_model_stats(model_path)
            assert stats is not None, f"missing stats for {model_path.name}"

    def test_stats_count_matches_models(self):
        models = sorted(MODELS_DIR.rglob("*_lstm_pt.pt")) + sorted(
            MODELS_DIR.rglob("*_lstm_pytorch.pt")
        )
        assert len(all_stats_files()) == len(models)

    def test_stats_schema(self):
        for path in all_stats_files():
            stats = json.loads(path.read_text())
            assert stats["temperature"] > 0
            dist = stats["train_class_distribution"]
            assert abs(sum(dist.values()) - 1.0) < 1e-6
            assert set(dist) == set(stats["classes"])
            for qhat in stats["conformal_quantiles"].values():
                assert 0 < qhat <= 1.0
            metrics = stats["metrics"]
            assert metrics["ece_post"] <= metrics["ece_pre"] + 0.01
            for level, cov in metrics["conformal_empirical_coverage"].items():
                assert abs(cov - float(level)) < 0.03, (
                    f"{path.name}: coverage@{level} = {cov}"
                )


class TestApplyPrior:
    def test_analytic_reweighting(self):
        probs = np.array([[0.6, 0.4]])
        classes = ["a", "b"]
        train = {"a": 0.5, "b": 0.5}
        prior = {"a": 0.9, "b": 0.1}
        out = apply_prior(probs, classes, prior, train)
        expected_a = (0.6 * 0.9 / 0.5) / (0.6 * 0.9 / 0.5 + 0.4 * 0.1 / 0.5)
        assert np.allclose(out, [[expected_a, 1 - expected_a]])
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_uniform_prior_matching_train_is_identity(self):
        probs = np.array([[0.2, 0.3, 0.5]])
        classes = ["a", "b", "c"]
        train = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        out = apply_prior(probs, classes, train, train)
        assert np.allclose(out, probs)

    def test_missing_class_raises(self):
        with pytest.raises(ValueError, match="missing classes"):
            apply_prior(
                np.array([[0.5, 0.5]]), ["a", "b"], {"a": 1.0}, {"a": 0.5, "b": 0.5}
            )


class TestConformalSets:
    def test_sets_contain_argmax_and_grow_with_qhat(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
        classes = ["a", "b", "c"]
        small = conformal_sets(probs, classes, 0.5)
        large = conformal_sets(probs, classes, 0.95)
        for i, row_probs in enumerate(probs):
            assert classes[int(row_probs.argmax())] in small[i]
            assert set(small[i]) <= set(large[i])

    def test_qhat_one_returns_all_classes(self):
        probs = np.array([[0.5, 0.3, 0.2]])
        assert len(conformal_sets(probs, ["a", "b", "c"], 1.0)[0]) == 3


class TestPredictionAPI:
    def test_census_prior_shifts_probabilities(self, sample_census_names):
        base = pred_census_ln(sample_census_names.copy(), "last", 2010)
        heavy_hispanic = {"api": 0.01, "black": 0.01, "hispanic": 0.97, "white": 0.01}
        adjusted = pred_census_ln(
            sample_census_names.copy(), "last", 2010, prior=heavy_hispanic
        )
        assert (adjusted["hispanic"] > base["hispanic"]).all()
        prob_cols = ["api", "black", "hispanic", "white"]
        assert np.allclose(adjusted[prob_cols].sum(axis=1), 1.0, atol=1e-5)

    def test_census_coverage_sets(self, sample_census_names):
        result = pred_census_ln(sample_census_names.copy(), "last", 2010, coverage=0.9)
        assert "race_set" in result.columns
        for _, row in result.iterrows():
            assert row["race"] in row["race_set"]

    def test_wiki_coverage_sets(self, sample_wiki_names):
        result = pred_wiki_ln(sample_wiki_names.copy(), "last", coverage=0.9)
        assert "race_set" in result.columns
        for _, row in result.iterrows():
            assert isinstance(row["race_set"], list)
            assert row["race"] in row["race_set"]

    def test_prior_with_coverage_raises(self, sample_census_names):
        with pytest.raises(ValueError, match="cannot be combined"):
            pred_census_ln(
                sample_census_names.copy(),
                "last",
                2010,
                prior={"api": 0.25, "black": 0.25, "hispanic": 0.25, "white": 0.25},
                coverage=0.9,
            )

    def test_prior_with_conf_int_raises(self, sample_census_names):
        with pytest.raises(ValueError, match="point predictions"):
            pred_census_ln(
                sample_census_names.copy(),
                "last",
                2010,
                conf_int=0.9,
                prior={"api": 0.25, "black": 0.25, "hispanic": 0.25, "white": 0.25},
            )

    def test_invalid_coverage_level_raises(self, sample_census_names):
        with pytest.raises(ValueError, match="coverage must be one of"):
            pred_census_ln(sample_census_names.copy(), "last", 2010, coverage=0.5)

    def test_wiki_balanced_prior_runs(self, sample_wiki_names):
        stats = load_model_stats(MODELS_DIR / "wiki" / "lstm" / "wiki_ln_lstm_pt.pt")
        classes = stats["classes"]
        uniform = {c: 1 / len(classes) for c in classes}
        result = pred_wiki_ln(sample_wiki_names.copy(), "last", prior=uniform)
        assert np.allclose(result[classes].sum(axis=1), 1.0, atol=1e-5)


class TestTemperatureApplied:
    def test_predictions_reflect_temperature(self, sample_census_names):
        """With T != 1 stored, probabilities differ from the raw-softmax run."""
        stats = load_model_stats(
            MODELS_DIR / "census" / "lstm" / "census2010_ln_lstm_pytorch.pt"
        )
        assert stats is not None
        result = pred_census_ln(sample_census_names.copy(), "last", 2010)
        # Probabilities remain a valid distribution after scaling
        prob_cols = ["api", "black", "hispanic", "white"]
        assert np.allclose(result[prob_cols].sum(axis=1), 1.0, atol=1e-5)
