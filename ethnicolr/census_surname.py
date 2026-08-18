"""Look up surname race/ethnicity proportions published by the U.S. Census."""

from __future__ import annotations

import importlib.resources as resources
import logging
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from .inference import (
    add_inference_metadata,
    combined_name_support,
    rename_conflicting_input_columns,
)
from .neural_name_model import NeuralNameModel
from .runtime_tables import CENSUS_SURNAME_SCHEMA, read_runtime_table
from .torch_utils import artifact_revision

logger = logging.getLogger(__name__)

CENSUS_SURNAME_FILES = {
    year: str(resources.files("ethnicolr") / f"data/census/census_{year}.parquet")
    for year in (2000, 2010, 2020)
}
CENSUS_PERCENTAGE_COLUMNS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]
CENSUS_CATEGORIES = ["white", "black", "api", "aian", "2prace", "hispanic"]


@lru_cache(maxsize=len(CENSUS_SURNAME_FILES))
def _load_census_surname_table(year: int) -> pd.DataFrame:
    """Load and cache one Census surname table."""
    if year not in CENSUS_SURNAME_FILES:
        raise ValueError("year must be 2000, 2010, or 2020")

    census_file = CENSUS_SURNAME_FILES[year]
    logger.info(f"Loading Census {year} surname data from {census_file}")
    table = read_runtime_table(
        census_file,
        CENSUS_SURNAME_SCHEMA,
        columns=["name", "count", *CENSUS_PERCENTAGE_COLUMNS],
    )
    return table.set_index("name")


def lookup_census_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    year: int = 2020,
    uncertainty_level: float | None = None,
) -> pd.DataFrame:
    """Append Census surname proportions and optional Wilson score bounds."""
    data = NeuralNameModel.validate_name_column(data, surname_column)
    normalized_surnames = (
        data[surname_column].astype("string").fillna("").str.strip().str.upper()
    )
    surname_table = _load_census_surname_table(year)
    matched_names = surname_table.reindex(normalized_surnames)
    script_supported, abstention_reasons = combined_name_support(
        cast(pd.Series, data[surname_column])
    )

    output_columns = set(CENSUS_PERCENTAGE_COLUMNS) | {"race"}
    if uncertainty_level is not None:
        for percentage_column in CENSUS_PERCENTAGE_COLUMNS:
            output_columns.update(
                {f"{percentage_column}_lower", f"{percentage_column}_upper"}
            )
    result = rename_conflicting_input_columns(data, output_columns)
    for percentage_column in CENSUS_PERCENTAGE_COLUMNS:
        result[percentage_column] = matched_names[percentage_column].to_numpy()

    if uncertainty_level is not None:
        if not 0 < uncertainty_level < 1:
            raise ValueError("uncertainty_level must be strictly between 0 and 1")
        from .name_dictionaries import wilson_interval

        sample_sizes = matched_names["count"].to_numpy(dtype=float)
        for percentage_column in CENSUS_PERCENTAGE_COLUMNS:
            proportions = result[percentage_column].to_numpy(dtype=float) / 100
            lower_bound, upper_bound = wilson_interval(
                proportions,
                sample_sizes,
                uncertainty_level,
            )
            result[f"{percentage_column}_lower"] = (lower_bound * 100).round(2)
            result[f"{percentage_column}_upper"] = (upper_bound * 100).round(2)
    matched_rows = np.isfinite(
        result[CENSUS_PERCENTAGE_COLUMNS].to_numpy(dtype=float)
    ).any(axis=1)
    predicted_categories = np.full(len(result), None, dtype=object)
    if matched_rows.any():
        matched_percentages = result.loc[
            matched_rows, CENSUS_PERCENTAGE_COLUMNS
        ].to_numpy(dtype=float)
        predicted_categories[matched_rows] = [
            CENSUS_CATEGORIES[category_index]
            for category_index in np.nanargmax(matched_percentages, axis=1)
        ]
    result["race"] = predicted_categories

    scored_rows = script_supported & matched_rows
    abstention_reasons[script_supported & ~matched_rows] = "out-of-dictionary"
    add_inference_metadata(
        result,
        target="race-ethnicity",
        input_scope="last-name",
        scored=scored_rows,
        script_supported=script_supported,
        abstained=~scored_rows,
        abstention_reasons=abstention_reasons,
        label_column="race",
        label_to_probability_column=dict(
            zip(CENSUS_CATEGORIES, CENSUS_PERCENTAGE_COLUMNS, strict=True)
        ),
        probability_scale=100,
        model_id=f"census-surname-{year}",
        model_revision=artifact_revision(Path(CENSUS_SURNAME_FILES[year])),
        reference_population=f"U.S. Census {year} surname table",
        calibration_status="not-applicable-dictionary",
        uncertainty_method=("wilson-score" if uncertainty_level is not None else None),
        uncertainty_level=uncertainty_level,
    )

    matched_count = int(scored_rows.sum())
    logger.info(f"Matched {matched_count} of {len(result)} surnames")
    return result
