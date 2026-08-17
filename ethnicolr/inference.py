"""Shared result semantics for name-pattern estimates."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pandas as pd

from .torch_utils import name_support_reason

ESTIMATE_TYPE = "name-pattern estimate"
INFERENCE_CONTRACT_VERSION = "1.0"


def prepare_full_name_data(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
) -> tuple[pd.DataFrame, str]:
    """Copy input data and add a collision-safe full-name column."""
    if surname_column not in data.columns:
        raise ValueError(f"Surname column {surname_column!r} does not exist.")
    if first_name_column not in data.columns:
        raise ValueError(f"First-name column {first_name_column!r} does not exist.")
    if (data.columns == surname_column).sum() > 1:
        raise ValueError(f"Duplicate surname column {surname_column!r}.")
    if (data.columns == first_name_column).sum() > 1:
        raise ValueError(f"Duplicate first-name column {first_name_column!r}.")

    result = data.copy()
    full_name_column = "__ethnicolr_full_name"
    while full_name_column in result.columns:
        full_name_column += "_"
    result[full_name_column] = (
        result[surname_column].fillna("").astype(str).str.strip()
        + " "
        + result[first_name_column].fillna("").astype(str).str.strip()
    ).str.strip()
    return result, full_name_column


def validate_inference_options(
    *,
    uncertainty_level: float | None,
    target_prior: dict[str, float] | None,
    conformal_coverage: float | None,
) -> None:
    """Reject inference options whose statistical guarantees conflict."""
    if target_prior is not None and conformal_coverage is not None:
        raise ValueError(
            "`target_prior` and `conformal_coverage` cannot be used together. "
            "Choose target-prior adjustment or a conformal prediction set."
        )
    if target_prior is not None and uncertainty_level is not None:
        raise ValueError(
            "`target_prior` cannot be used when `uncertainty_level` is set. "
            "Set `uncertainty_level=None` or omit `target_prior`."
        )
    if conformal_coverage is not None and uncertainty_level is not None:
        raise ValueError(
            "`conformal_coverage` cannot be used when `uncertainty_level` is set. "
            "Set `uncertainty_level=None` or omit `conformal_coverage`."
        )


def combined_name_support(*columns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return script support and reasons for one or more name columns."""
    if not columns:
        raise ValueError("at least one name column is required")
    normalized = [column.astype("string").fillna("").astype(str) for column in columns]
    reasons = np.array(
        [
            name_support_reason(" ".join(values))
            for values in zip(*(column.tolist() for column in normalized), strict=True)
        ],
        dtype=object,
    )
    supported = np.fromiter((reason is None for reason in reasons), dtype=bool)
    return supported, reasons


def add_inference_metadata(
    result: pd.DataFrame,
    *,
    target: str,
    input_scope: str,
    scored: np.ndarray,
    script_supported: np.ndarray,
    abstained: np.ndarray,
    abstention_reasons: np.ndarray,
    label_column: str,
    label_to_probability_column: Mapping[str, str] | None = None,
    probability_scale: float = 1.0,
    model_id: Any,
    model_revision: Any,
    reference_population: Any,
    calibration_status: Any,
    calibration_reference: Any = pd.NA,
    uncertainty_method: str | None = None,
    uncertainty_level: float | None = None,
) -> pd.DataFrame:
    """Append the common inference contract to a result DataFrame."""
    row_count = len(result)
    scored = np.asarray(scored, dtype=bool)
    script_supported = np.asarray(script_supported, dtype=bool)
    abstained = np.asarray(abstained, dtype=bool)
    if any(
        len(values) != row_count for values in (scored, script_supported, abstained)
    ):
        raise ValueError("inference status arrays must match the result row count")
    if probability_scale <= 0:
        raise ValueError("probability_scale must be positive")
    if np.any(~scored & ~abstained):
        raise ValueError("an unscored row must abstain")
    if np.any(~script_supported & scored):
        raise ValueError("an unsupported-script row cannot be scored")

    if (result.columns == label_column).sum() != 1:
        raise ValueError(f"result must contain one {label_column!r} column")
    label_values = cast(pd.Series, result[label_column])
    predicted_labels = pd.array(label_values, dtype="string")
    probability_columns = label_to_probability_column or {}
    predicted_probabilities = np.full(row_count, np.nan, dtype=float)
    for row_position, label in enumerate(predicted_labels):
        if pd.isna(label):
            continue
        probability_column = probability_columns.get(str(label), str(label))
        predicted_probabilities[row_position] = (
            float(result.iloc[row_position][probability_column]) / probability_scale
        )

    result["inference_contract_version"] = INFERENCE_CONTRACT_VERSION
    result["estimate_type"] = ESTIMATE_TYPE
    result["target"] = target
    result["input_scope"] = input_scope
    result["predicted_label"] = predicted_labels
    result["predicted_probability"] = predicted_probabilities
    result["scored"] = scored
    result["script_supported"] = script_supported
    result["abstained"] = abstained
    result["abstention_reason"] = pd.array(abstention_reasons, dtype="string")
    result["model_id"] = model_id
    result["model_version"] = version("ethnicolr")
    result["model_revision"] = model_revision
    result["reference_population"] = reference_population
    result["calibration_reference"] = calibration_reference
    result["calibration_status"] = calibration_status
    result["uncertainty_method"] = uncertainty_method or pd.NA
    result["uncertainty_level"] = (
        uncertainty_level if uncertainty_level is not None else np.nan
    )
    return result
