"""Look up surname race/ethnicity proportions published by the U.S. Census."""

from __future__ import annotations

import importlib.resources as resources
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .inference import (
    add_inference_metadata,
    combined_name_support,
)
from .neural_name_model import NeuralNameModel
from .runtime_tables import CENSUS_SURNAME_SCHEMA, read_runtime_table
from .torch_utils import artifact_revision

if TYPE_CHECKING:
    import pandas as pd

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
    logger.info("Loading Census %s surname data from %s", year, census_file)
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
        cast("pd.Series", data[surname_column])
    )

    # name_dictionaries imports this module, so its helper loads at call time.
    from .name_dictionaries import _add_census_percentages

    result, matched_rows = _add_census_percentages(
        data, matched_names, uncertainty_level
    )

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
    logger.info("Matched %d of %d surnames", matched_count, len(result))
    return result
