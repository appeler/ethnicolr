"""Tests for the shared name-inference result contract."""

import numpy as np
import pandas as pd
import pytest

from ethnicolr import (
    estimate_census_full_name,
    estimate_census_surname,
    estimate_voter_file_full_name,
    estimate_wikipedia_full_name,
    estimate_wikipedia_surname,
    lookup_census_first_name,
    lookup_census_surname,
)
from ethnicolr.inference import add_inference_metadata
from ethnicolr.torch_utils import artifact_revision

CONTRACT_COLUMNS = {
    "inference_contract_version",
    "estimate_type",
    "target",
    "input_scope",
    "predicted_label",
    "predicted_probability",
    "scored",
    "script_supported",
    "abstained",
    "abstention_reason",
    "model_id",
    "model_version",
    "model_revision",
    "reference_population",
    "calibration_reference",
    "calibration_status",
    "uncertainty_method",
    "uncertainty_level",
}


@pytest.mark.parametrize(
    "predict", [estimate_census_surname, estimate_wikipedia_surname]
)
def test_unusable_inputs_abstain_without_probabilities(predict) -> None:
    data = pd.DataFrame({"last": ["हेमा", "1234", "---", "Smith"]})
    result = predict(data, "last")
    probability_columns = (
        ["api", "black", "hispanic", "white"]
        if predict is estimate_census_surname
        else [
            column
            for column in result.columns
            if column.startswith(("Asian,", "GreaterAfrican,", "GreaterEuropean,"))
        ]
    )

    assert result["abstained"].tolist() == [True, True, True, False]
    assert result["script_supported"].tolist() == [False, False, False, True]
    assert result.loc[:2, "race"].isna().all()
    assert result.loc[:2, probability_columns].isna().all(axis=None)
    assert result.loc[3, probability_columns].sum() == pytest.approx(1.0)
    assert set(CONTRACT_COLUMNS) <= set(result.columns)
    assert result.loc[3, "estimate_type"] == "name-pattern estimate"
    assert result.loc[3, "inference_contract_version"] == "1.0"
    assert result["scored"].tolist() == [False, False, False, True]
    assert result.loc[3, "predicted_label"] == result.loc[3, "race"]
    assert result.loc[3, "predicted_probability"] == pytest.approx(
        result.loc[3, probability_columns].max()
    )
    assert result.loc[3, "model_revision"].startswith("sha256:")


@pytest.mark.parametrize("level", [0.0, 1.1, float("nan"), float("inf")])
def test_invalid_mc_dropout_level_raises(level: float) -> None:
    data = pd.DataFrame({"last": ["Smith"]})
    with pytest.raises(ValueError, match="uncertainty_level"):
        estimate_wikipedia_surname(data, "last", uncertainty_level=level)


@pytest.mark.parametrize("iterations", [0, 1])
def test_too_few_mc_dropout_iterations_raise(iterations: int) -> None:
    data = pd.DataFrame({"last": ["Smith"]})
    with pytest.raises(ValueError, match="mc_iterations"):
        estimate_census_surname(
            data, "last", uncertainty_level=0.9, mc_iterations=iterations
        )


def test_mc_dropout_columns_do_not_claim_confidence_bounds() -> None:
    result = estimate_wikipedia_surname(
        pd.DataFrame({"last": ["Smith"]}),
        "last",
        uncertainty_level=0.9,
        mc_iterations=3,
    )
    assert result.loc[0, "uncertainty_method"] == "mc-dropout"
    assert result.loc[0, "uncertainty_level"] == pytest.approx(0.9)
    assert any(column.endswith("_mc_lower") for column in result.columns)
    assert not any(column.endswith("_lb") for column in result.columns)
    assert np.isfinite(result.filter(like="_mc_std").to_numpy()).all()


def test_full_name_wrapper_preserves_unsupported_row_as_abstention() -> None:
    data = pd.DataFrame({"last": ["देवी", "Smith"], "first": ["हेमा", "John"]})
    result = estimate_wikipedia_full_name(data, "last", "first")

    assert result["abstained"].tolist() == [True, False]
    assert result.loc[0, "abstention_reason"] == "unsupported-script"
    assert "processing_status" not in result.columns
    assert pd.isna(result.loc[0, "race"])
    assert result.loc[1, "race"] is not None


@pytest.mark.parametrize(
    "predict", [estimate_census_surname, estimate_wikipedia_surname]
)
def test_empty_neural_input_has_stable_result_schema(predict) -> None:
    result = predict(pd.DataFrame({"last": pd.Series(dtype="string")}), "last")

    assert result.empty
    assert set(CONTRACT_COLUMNS) <= set(result.columns)
    assert "race" in result


def test_census_lookups_preserve_rows_and_explain_abstention() -> None:
    data = pd.DataFrame(
        {
            "last": ["Smith", None, "देवी", "---", "Qzxqzx"],
            "first": ["John", None, "हेमा", "1234", "Qzxqzx"],
        },
        index=[7, 8, 9, 10, 11],
    )

    last = lookup_census_surname(data.copy(), "last", year=2020, uncertainty_level=0.95)
    first = lookup_census_first_name(data.copy(), "first", uncertainty_level=0.95)

    for result in (last, first):
        assert result.index.tolist() == data.index.tolist()
        assert set(CONTRACT_COLUMNS) <= set(result.columns)
        assert result["abstained"].tolist() == [False, True, True, True, True]
        assert result["abstention_reason"].tolist()[1:] == [
            "missing-name",
            "unsupported-script",
            "no-letters",
            "out-of-dictionary",
        ]
        assert result.loc[7, "race"] is not None
        assert result.loc[7, "predicted_probability"] == pytest.approx(
            result.loc[7, f"pct{result.loc[7, 'race']}"] / 100
        )
        assert result.loc[7, "uncertainty_method"] == "wilson-score"


@pytest.mark.parametrize(
    "predict", [estimate_census_full_name, estimate_voter_file_full_name]
)
def test_full_name_dictionary_apis_abstain_on_unusable_input(predict) -> None:
    data = pd.DataFrame(
        {
            "last": ["Smith", None, "देवी", "---", "Qzxqzx"],
            "first": ["John", None, "हेमा", "1234", "Qzxqzx"],
        }
    )
    result = predict(data, "last", "first")

    assert len(result) == len(data)
    assert set(CONTRACT_COLUMNS) <= set(result.columns)
    assert not bool(result.loc[0, "abstained"])
    assert result.loc[1:3, "abstained"].all()
    assert result.loc[1:3, "race"].isna().all()
    assert result.loc[1, "abstention_reason"] == "missing-name"
    assert result.loc[2, "abstention_reason"] == "unsupported-script"
    assert result.loc[3, "abstention_reason"] == "no-letters"
    if predict is estimate_voter_file_full_name:
        assert result.loc[4, "abstained"]
        assert result.loc[4, "abstention_reason"] == "out-of-dictionary"


def test_inference_contract_rejects_contradictory_status() -> None:
    result = pd.DataFrame({"race": ["white"], "white": [0.8]})

    with pytest.raises(ValueError, match="unscored row must abstain"):
        add_inference_metadata(
            result,
            target="race-ethnicity",
            input_scope="last-name",
            scored=np.array([False]),
            script_supported=np.array([True]),
            abstained=np.array([False]),
            abstention_reasons=np.array([None], dtype=object),
            label_column="race",
            model_id="test",
            model_revision="sha256:test",
            reference_population="test",
            calibration_status="test",
        )


def test_artifact_revision_covers_the_full_bundle(tmp_path) -> None:
    weights = tmp_path / "weights.pt"
    vocab = tmp_path / "vocab.csv"
    weights.write_bytes(b"weights")
    vocab.write_bytes(b"vocab-v1")
    first = artifact_revision(weights, vocab)

    artifact_revision.cache_clear()
    vocab.write_bytes(b"vocab-v2")

    assert artifact_revision(weights, vocab) != first
