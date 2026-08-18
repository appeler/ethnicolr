#!/usr/bin/env python
"""Dictionary-based race/ethnicity estimators.

These estimators use conditional frequencies from published name tables. A
neural model is used only as the documented surname fallback in
``estimate_census_full_name``.

- ``lookup_census_first_name``: first-name lookup against the Census 2020 first-name file
  (53,616 names covering ~94% of the enumerated population).
- ``estimate_census_full_name``: six-category posterior from first+last name via naive
  Bayes over the census first-name and surname tables, with the census LSTM
  as fallback for out-of-dictionary surnames.
- ``estimate_voter_file_full_name``: five-category estimate from exact
  first- and last-name frequencies in AL/FL/GA/LA/NC/SC voter files.

The first+last combination assumes conditional independence of first and last
name given race (naive Bayes). This is an explicit, documented approximation; see
docs/source/statistical_principles.md.
"""

from __future__ import annotations

import importlib.resources as resources
import json
import logging
from pathlib import Path
from statistics import NormalDist
from typing import cast

import numpy as np
import pandas as pd

from .census_surname import CENSUS_SURNAME_FILES
from .inference import (
    add_inference_metadata,
    combined_name_support,
    rename_conflicting_input_columns,
)
from .model_artifacts import resolve_model_bundle
from .neural_name_model import NeuralNameModel
from .runtime_tables import (
    CENSUS_FIRST_NAME_SCHEMA,
    CENSUS_SURNAME_SCHEMA,
    NAME_RACE_PROBABILITY_SCHEMA,
    read_runtime_table,
)
from .torch_utils import (
    adjust_probabilities_for_prior,
    artifact_revision,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DATA = resources.files("ethnicolr") / "data"
CENSUS_FIRST_NAME_2020_FILE = str(_DATA / "census/census_2020_first_names.parquet")
VOTER_FILE_FIRST_NAME_TABLE = str(_DATA / "rosenman/first_name_race.parquet")
VOTER_FILE_LAST_NAME_TABLE = str(_DATA / "rosenman/last_name_race.parquet")
VOTER_FILE_REFERENCE_STATS = str(_DATA / "rosenman/rosenman_stats.json")

CENSUS_PERCENTAGE_COLUMNS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]
CENSUS_CATEGORIES = ["white", "black", "api", "aian", "2prace", "hispanic"]
NEURAL_MODEL_CATEGORIES = ["api", "black", "hispanic", "white"]
VOTER_FILE_CATEGORIES = ["white", "black", "hispanic", "asian", "other"]

_MINIMUM_PROBABILITY = 1e-8


class _NameTables:
    """Lazy caches for the dictionary tables."""

    _census_first_name_table: pd.DataFrame | None = None
    _census_surname_tables: dict[int, pd.DataFrame] = {}
    _census_population_prior: np.ndarray | None = None
    _voter_file_name_probabilities: dict[str, pd.DataFrame] = {}
    _voter_file_population_marginal: np.ndarray | None = None

    @classmethod
    def census_first_name_table(cls) -> pd.DataFrame:
        if cls._census_first_name_table is None:
            data = read_runtime_table(
                CENSUS_FIRST_NAME_2020_FILE, CENSUS_FIRST_NAME_SCHEMA
            )
            cls._census_first_name_table = data.set_index("name")
        return cls._census_first_name_table

    @classmethod
    def census_surname_table(cls, year: int) -> pd.DataFrame:
        if year not in cls._census_surname_tables:
            path = CENSUS_SURNAME_FILES[year]
            data = read_runtime_table(
                path,
                CENSUS_SURNAME_SCHEMA,
                columns=["name", *CENSUS_PERCENTAGE_COLUMNS],
            )
            cls._census_surname_tables[year] = data.set_index("name")
        return cls._census_surname_tables[year]

    @classmethod
    def census_population_prior(cls) -> np.ndarray:
        """Return the person-level Census distribution implied by first names."""
        if cls._census_population_prior is None:
            data = cls.census_first_name_table()
            weights = (
                data[CENSUS_PERCENTAGE_COLUMNS].fillna(0).to_numpy()
                * data["count"].to_numpy()[:, None]
            )
            weighted_category_counts = weights.sum(axis=0)
            cls._census_population_prior = (
                weighted_category_counts / weighted_category_counts.sum()
            )
        assert cls._census_population_prior is not None
        return cls._census_population_prior

    @classmethod
    def voter_file_name_probabilities(cls, name_part: str) -> pd.DataFrame:
        if name_part not in cls._voter_file_name_probabilities:
            path = (
                VOTER_FILE_FIRST_NAME_TABLE
                if name_part == "first"
                else VOTER_FILE_LAST_NAME_TABLE
            )
            data = read_runtime_table(path, NAME_RACE_PROBABILITY_SCHEMA)
            # A list key always yields a DataFrame; the stubs widen it.
            cls._voter_file_name_probabilities[name_part] = cast(
                pd.DataFrame, data.set_index("name")[VOTER_FILE_CATEGORIES]
            )
        return cls._voter_file_name_probabilities[name_part]

    @classmethod
    def voter_file_population_marginal(cls) -> np.ndarray:
        if cls._voter_file_population_marginal is None:
            reference_statistics = json.loads(
                Path(VOTER_FILE_REFERENCE_STATS).read_text()
            )
            cls._voter_file_population_marginal = np.array(
                [
                    reference_statistics["voter_population_marginal"][category]
                    for category in VOTER_FILE_CATEGORIES
                ]
            )
        return cls._voter_file_population_marginal


def wilson_interval(
    proportions: np.ndarray,
    sample_sizes: np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (vectorized)."""
    critical_value = NormalDist().inv_cdf((1 + confidence_level) / 2)
    denominator = 1 + critical_value**2 / sample_sizes
    center = (proportions + critical_value**2 / (2 * sample_sizes)) / denominator
    half_width = (critical_value / denominator) * np.sqrt(
        proportions * (1 - proportions) / sample_sizes
        + critical_value**2 / (4 * sample_sizes**2)
    )
    return (
        np.clip(center - half_width, 0, 1),
        np.clip(center + half_width, 0, 1),
    )


def _normalize_names(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def lookup_census_first_name(
    data: pd.DataFrame,
    first_name_column: str,
    year: int = 2020,
    uncertainty_level: float | None = None,
) -> pd.DataFrame:
    """Append Census 2020 demographic percentages by first name.

    Mirrors :func:`ethnicolr.census_surname` for first names, using the Census
    Bureau's 2020 first-name file. With ``uncertainty_level``, adds exact Wilson
    score bounds (``<col>_lower``/``<col>_upper``) computed from the published
    name counts (sampling uncertainty only; the counts also carry the
    Bureau's small disclosure-avoidance noise).

    Args:
        data: Input DataFrame containing first names.
        first_name_column: Column name containing first names.
        year: Data year (only 2020 is available).
        uncertainty_level: Optional confidence level for Wilson bounds, e.g. 0.95.

    Returns:
        DataFrame with original data plus pctwhite/pctblack/pctapi/pctaian/
        pct2prace/pcthispanic columns (NaN for names not in the file).
    """
    if year != 2020:
        raise ValueError("First-name data is only available for 2020")
    data = NeuralNameModel.validate_name_column(data, first_name_column)

    first_name_table = _NameTables.census_first_name_table()
    normalized_first_names = _normalize_names(cast(pd.Series, data[first_name_column]))
    matched_names = first_name_table.reindex(normalized_first_names)
    script_supported, abstention_reasons = combined_name_support(
        cast(pd.Series, data[first_name_column])
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
            raise ValueError("uncertainty_level must be between 0 and 1")
        sample_sizes = matched_names["count"].to_numpy()
        for percentage_column in CENSUS_PERCENTAGE_COLUMNS:
            proportions = result[percentage_column].to_numpy() / 100
            lower_bound, upper_bound = wilson_interval(
                proportions, sample_sizes, uncertainty_level
            )
            result[f"{percentage_column}_lower"] = (lower_bound * 100).round(2)
            result[f"{percentage_column}_upper"] = (upper_bound * 100).round(2)

    matched_rows = np.asarray(
        np.isfinite(matched_names[CENSUS_PERCENTAGE_COLUMNS].to_numpy(dtype=float)).any(
            axis=1
        ),
        dtype=bool,
    )
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
    matched_count = int(matched_rows.sum())
    logger.info(f"Matched {matched_count} of {len(result)} first names")
    scored_rows = script_supported & matched_rows
    abstention_reasons[script_supported & ~matched_rows] = "out-of-dictionary"
    add_inference_metadata(
        result,
        target="race-ethnicity",
        input_scope="first-name",
        scored=scored_rows,
        script_supported=script_supported,
        abstained=~scored_rows,
        abstention_reasons=abstention_reasons,
        label_column="race",
        label_to_probability_column=dict(
            zip(CENSUS_CATEGORIES, CENSUS_PERCENTAGE_COLUMNS, strict=True)
        ),
        probability_scale=100,
        model_id="census-first-name-2020",
        model_revision=artifact_revision(Path(CENSUS_FIRST_NAME_2020_FILE)),
        reference_population="U.S. Census 2020 first-name table",
        calibration_status="not-applicable-dictionary",
        uncertainty_method="wilson-score" if uncertainty_level is not None else None,
        uncertainty_level=uncertainty_level,
    )
    return result


def _combine_name_probabilities(
    surname_probabilities: np.ndarray,
    first_name_probabilities: np.ndarray,
    reference_prior: np.ndarray,
) -> np.ndarray:
    """Naive-Bayes posterior: p(r|f,l) ∝ p(r|l)·p(r|f)/π(r)."""
    posterior = (
        np.maximum(surname_probabilities, _MINIMUM_PROBABILITY)
        * np.maximum(first_name_probabilities, _MINIMUM_PROBABILITY)
        / np.maximum(reference_prior, _MINIMUM_PROBABILITY)
    )
    return posterior / posterior.sum(axis=1, keepdims=True)


def estimate_census_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    year: int = 2020,
    target_prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict race/ethnicity from first and last name using census tables.

    Combines the census surname table p(race|last) and first-name table
    p(race|first) via naive Bayes (conditional independence of the two names
    given race). Six categories: white, black, api, aian, 2prace, hispanic.

    Out-of-dictionary surnames fall back to the census LSTM (four categories;
    aian/2prace are NaN on those rows). The ``evidence_basis`` column records
    the dictionary and model evidence used for each row.

    Args:
        data: Input DataFrame.
        surname_column: Last-name column.
        first_name_column: First-name column.
        year: Census year for the surname table (2000, 2010, or 2020;
            first-name data is 2020 regardless).
        target_prior: Optional target class distribution (see statistical principles
            docs); reweights posteriors from the Census population prior.

    Returns:
        DataFrame with probability columns per category, ``race`` (argmax),
        and ``evidence_basis``.
    """
    from .census_surname_model import (
        estimate_census_surname,
        get_census_surname_model_files,
    )

    if surname_column not in data.columns or first_name_column not in data.columns:
        raise ValueError(
            "surname_column and first_name_column must exist in the DataFrame"
        )

    result = data.copy()
    script_supported, abstention_reasons = combined_name_support(
        cast(pd.Series, result[surname_column]),
        cast(pd.Series, result[first_name_column]),
    )
    normalized_surnames = _normalize_names(cast(pd.Series, result[surname_column]))
    normalized_first_names = _normalize_names(
        cast(pd.Series, result[first_name_column])
    )

    surname_table = _NameTables.census_surname_table(year)
    first_name_table = _NameTables.census_first_name_table()
    census_population_prior = _NameTables.census_population_prior()

    surname_probabilities = (
        surname_table[CENSUS_PERCENTAGE_COLUMNS].reindex(normalized_surnames).to_numpy()
        / 100
    )
    first_name_probabilities = (
        first_name_table[CENSUS_PERCENTAGE_COLUMNS]
        .reindex(normalized_first_names)
        .to_numpy()
        / 100
    )
    # Renormalize over non-suppressed cells
    with np.errstate(invalid="ignore"):
        surname_probabilities = np.nan_to_num(surname_probabilities) / np.maximum(
            np.nan_to_num(surname_probabilities).sum(axis=1, keepdims=True),
            _MINIMUM_PROBABILITY,
        )
        first_name_probabilities = np.nan_to_num(first_name_probabilities) / np.maximum(
            np.nan_to_num(first_name_probabilities).sum(axis=1, keepdims=True),
            _MINIMUM_PROBABILITY,
        )

    surname_in_dictionary = ~np.isnan(
        surname_table[CENSUS_PERCENTAGE_COLUMNS[0]]
        .reindex(normalized_surnames)
        .to_numpy()
    )
    first_name_in_dictionary = ~np.isnan(
        first_name_table[CENSUS_PERCENTAGE_COLUMNS[0]]
        .reindex(normalized_first_names)
        .to_numpy()
    )

    row_count = len(result)
    category_probabilities = np.full((row_count, len(CENSUS_CATEGORIES)), np.nan)
    evidence_basis = np.empty(row_count, dtype=object)

    # Dictionary paths (six categories)
    both_names_in_dictionary = surname_in_dictionary & first_name_in_dictionary
    category_probabilities[both_names_in_dictionary] = _combine_name_probabilities(
        surname_probabilities[both_names_in_dictionary],
        first_name_probabilities[both_names_in_dictionary],
        census_population_prior,
    )
    evidence_basis[both_names_in_dictionary] = "first-name-and-surname-dictionaries"
    only_surname_in_dictionary = surname_in_dictionary & ~first_name_in_dictionary
    category_probabilities[only_surname_in_dictionary] = surname_probabilities[
        only_surname_in_dictionary
    ]
    evidence_basis[only_surname_in_dictionary] = "surname-dictionary"

    # Surname-model fallback for out-of-dictionary surnames (four categories)
    surname_not_in_dictionary = ~surname_in_dictionary
    neural_model_result = None
    if surname_not_in_dictionary.any():
        neural_model_input = result.loc[
            surname_not_in_dictionary, [surname_column]
        ].copy()
        neural_model_result = estimate_census_surname(
            neural_model_input, surname_column, year=year
        )
        neural_model_probabilities = neural_model_result[
            NEURAL_MODEL_CATEGORIES
        ].to_numpy()

        neural_category_positions = np.array(
            [CENSUS_CATEGORIES.index(category) for category in NEURAL_MODEL_CATEGORIES],
            dtype=np.intp,
        )
        four_category_reference_prior = (
            census_population_prior[neural_category_positions]
            / census_population_prior[neural_category_positions].sum()
        )

        first_name_probabilities_four_categories = first_name_probabilities[
            surname_not_in_dictionary
        ][:, neural_category_positions]
        first_name_supports_neural_categories = first_name_in_dictionary[
            surname_not_in_dictionary
        ] & (
            first_name_probabilities_four_categories.sum(axis=1) > _MINIMUM_PROBABILITY
        )
        first_name_probabilities_four_categories = (
            first_name_probabilities_four_categories
            / np.maximum(
                first_name_probabilities_four_categories.sum(axis=1, keepdims=True),
                _MINIMUM_PROBABILITY,
            )
        )

        combined_probabilities = np.where(
            first_name_supports_neural_categories[:, None],
            _combine_name_probabilities(
                neural_model_probabilities,
                first_name_probabilities_four_categories,
                four_category_reference_prior,
            ),
            neural_model_probabilities,
        )
        fallback_probabilities = np.full(
            (int(surname_not_in_dictionary.sum()), len(CENSUS_CATEGORIES)), np.nan
        )
        fallback_probabilities[:, neural_category_positions] = combined_probabilities
        category_probabilities[surname_not_in_dictionary] = fallback_probabilities
        evidence_basis[surname_not_in_dictionary] = np.where(
            first_name_supports_neural_categories,
            "surname-model-and-first-name-dictionary",
            "surname-model",
        )

    if target_prior is not None:
        finite_probabilities = np.isfinite(category_probabilities)
        uses_six_category_distribution = finite_probabilities.all(axis=1)
        if uses_six_category_distribution.any():
            category_probabilities[uses_six_category_distribution] = (
                adjust_probabilities_for_prior(
                    category_probabilities[uses_six_category_distribution],
                    CENSUS_CATEGORIES,
                    target_prior,
                    dict(zip(CENSUS_CATEGORIES, census_population_prior, strict=True)),
                )
            )
        uses_four_category_distribution = (
            finite_probabilities.any(axis=1) & ~uses_six_category_distribution
        )
        if uses_four_category_distribution.any():
            neural_category_positions = np.array(
                [
                    CENSUS_CATEGORIES.index(category)
                    for category in NEURAL_MODEL_CATEGORIES
                ],
                dtype=np.intp,
            )
            row_positions = np.flatnonzero(uses_four_category_distribution)
            four_category_reference_prior = (
                census_population_prior[neural_category_positions]
                / census_population_prior[neural_category_positions].sum()
            )
            four_category_target_prior = {
                category: target_prior[category]
                for category in NEURAL_MODEL_CATEGORIES
                if category in target_prior
            }
            if len(four_category_target_prior) != len(NEURAL_MODEL_CATEGORIES):
                raise ValueError(
                    f"target_prior must include all of {NEURAL_MODEL_CATEGORIES}"
                )
            category_probabilities[np.ix_(row_positions, neural_category_positions)] = (
                adjust_probabilities_for_prior(
                    category_probabilities[
                        np.ix_(row_positions, neural_category_positions)
                    ],
                    NEURAL_MODEL_CATEGORIES,
                    four_category_target_prior,
                    dict(
                        zip(
                            NEURAL_MODEL_CATEGORIES,
                            four_category_reference_prior,
                            strict=True,
                        )
                    ),
                )
            )

    category_probabilities[~script_supported] = np.nan
    evidence_basis[~script_supported] = "none"
    finite_probabilities = np.isfinite(category_probabilities)
    has_any_probability = finite_probabilities.any(axis=1)
    has_complete_distribution = finite_probabilities.all(axis=1)
    scored_rows = script_supported & has_complete_distribution
    abstention_reasons[script_supported & ~has_any_probability] = "out-of-vocabulary"
    abstention_reasons[
        script_supported & has_any_probability & ~has_complete_distribution
    ] = "insufficient-evidence"

    result = rename_conflicting_input_columns(
        result, {*CENSUS_CATEGORIES, "race", "evidence_basis"}
    )
    for category_index, category in enumerate(CENSUS_CATEGORIES):
        result[category] = category_probabilities[:, category_index]
    predicted_categories = np.full(row_count, None, dtype=object)
    if scored_rows.any():
        predicted_categories[scored_rows] = [
            CENSUS_CATEGORIES[category_index]
            for category_index in np.nanargmax(
                category_probabilities[scored_rows], axis=1
            )
        ]
    result["race"] = predicted_categories
    result["evidence_basis"] = evidence_basis

    first_name_table_path = Path(CENSUS_FIRST_NAME_2020_FILE)
    surname_table_path = Path(CENSUS_SURNAME_FILES[year])
    dictionary_revision = artifact_revision(first_name_table_path, surname_table_path)
    uses_neural_fallback = surname_not_in_dictionary
    model_identifiers = np.full(
        row_count, f"census-name-dictionary-{year}", dtype=object
    )
    model_revisions = np.full(row_count, dictionary_revision, dtype=object)
    reference_populations = np.full(
        row_count,
        f"U.S. Census {year} surnames and 2020 first names",
        dtype=object,
    )
    if uses_neural_fallback.any():
        model_file, vocabulary_file, labels_file = get_census_surname_model_files(year)
        model_bundle = resolve_model_bundle(model_file, vocabulary_file, labels_file)
        hybrid_revision = artifact_revision(
            first_name_table_path,
            surname_table_path,
            *model_bundle.revision_files,
        )
        model_identifiers[uses_neural_fallback] = f"census-name-hybrid-{year}"
        model_revisions[uses_neural_fallback] = hybrid_revision
        reference_populations[uses_neural_fallback] = (
            "U.S. Census surname LSTM and 2020 first-name table"
        )
    calibration_statuses = np.full(row_count, "not-applicable-dictionary", dtype=object)
    calibration_reference = np.full(row_count, pd.NA, dtype=object)
    if neural_model_result is not None:
        calibration_statuses[surname_not_in_dictionary] = neural_model_result[
            "calibration_status"
        ].to_numpy()
        calibration_reference[surname_not_in_dictionary] = neural_model_result[
            "calibration_reference"
        ].to_numpy()
    add_inference_metadata(
        result,
        target="race-ethnicity",
        input_scope="full-name",
        scored=scored_rows,
        script_supported=script_supported,
        abstained=~scored_rows,
        abstention_reasons=abstention_reasons,
        label_column="race",
        model_id=model_identifiers,
        model_revision=model_revisions,
        reference_population=reference_populations,
        calibration_reference=calibration_reference,
        calibration_status=calibration_statuses,
    )
    return result


def estimate_voter_file_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    target_prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from six-state voter-file name frequencies.

    Five categories (white, black, hispanic, asian, other) with self-reported
    race from six Southern-state voter files. First+last combined via naive
    Bayes. The ``evidence_basis`` column records which name dictionaries
    contributed to each estimate.

    Args:
        data: Input DataFrame.
        surname_column: Last-name column.
        first_name_column: First-name column.
        target_prior: Optional target class distribution; reweights posteriors from
            the six-state voter-population prior.

    Returns:
        DataFrame with the five probability columns, ``race``, and ``evidence_basis``.
    """
    if surname_column not in data.columns or first_name_column not in data.columns:
        raise ValueError(
            "surname_column and first_name_column must exist in the DataFrame"
        )

    result = data.copy()
    script_supported, abstention_reasons = combined_name_support(
        cast(pd.Series, result[surname_column]),
        cast(pd.Series, result[first_name_column]),
    )
    normalized_surnames = _normalize_names(cast(pd.Series, result[surname_column]))
    normalized_first_names = _normalize_names(
        cast(pd.Series, result[first_name_column])
    )

    surname_table = _NameTables.voter_file_name_probabilities("last")
    first_name_table = _NameTables.voter_file_name_probabilities("first")
    reference_prior = _NameTables.voter_file_population_marginal()

    surname_probabilities = surname_table.reindex(normalized_surnames).to_numpy()
    first_name_probabilities = first_name_table.reindex(
        normalized_first_names
    ).to_numpy()
    surname_in_dictionary = ~np.isnan(surname_probabilities[:, 0])
    first_name_in_dictionary = ~np.isnan(first_name_probabilities[:, 0])

    row_count = len(result)
    category_probabilities = np.full((row_count, len(VOTER_FILE_CATEGORIES)), np.nan)
    evidence_basis = np.empty(row_count, dtype=object)

    both_names_in_dictionary = surname_in_dictionary & first_name_in_dictionary
    category_probabilities[both_names_in_dictionary] = _combine_name_probabilities(
        surname_probabilities[both_names_in_dictionary],
        first_name_probabilities[both_names_in_dictionary],
        reference_prior,
    )
    evidence_basis[both_names_in_dictionary] = "first-name-and-surname-dictionaries"
    only_surname_in_dictionary = surname_in_dictionary & ~first_name_in_dictionary
    category_probabilities[only_surname_in_dictionary] = surname_probabilities[
        only_surname_in_dictionary
    ]
    evidence_basis[only_surname_in_dictionary] = "surname-dictionary"
    only_first_name_in_dictionary = first_name_in_dictionary & ~surname_in_dictionary
    category_probabilities[only_first_name_in_dictionary] = first_name_probabilities[
        only_first_name_in_dictionary
    ]
    evidence_basis[only_first_name_in_dictionary] = "first-name-dictionary"
    evidence_basis[~surname_in_dictionary & ~first_name_in_dictionary] = "none"

    if target_prior is not None:
        dictionary_matched_rows = surname_in_dictionary | first_name_in_dictionary
        if dictionary_matched_rows.any():
            category_probabilities[dictionary_matched_rows] = (
                adjust_probabilities_for_prior(
                    category_probabilities[dictionary_matched_rows],
                    VOTER_FILE_CATEGORIES,
                    target_prior,
                    dict(zip(VOTER_FILE_CATEGORIES, reference_prior, strict=True)),
                )
            )

    category_probabilities[~script_supported] = np.nan
    evidence_basis[~script_supported] = "none"
    scored_rows = script_supported & (surname_in_dictionary | first_name_in_dictionary)
    abstention_reasons[script_supported & ~scored_rows] = "out-of-dictionary"

    result = rename_conflicting_input_columns(
        result, {*VOTER_FILE_CATEGORIES, "race", "evidence_basis"}
    )
    for category_index, category in enumerate(VOTER_FILE_CATEGORIES):
        result[category] = category_probabilities[:, category_index]
    predicted_categories = np.full(row_count, None, dtype=object)
    if scored_rows.any():
        predicted_categories[scored_rows] = [
            VOTER_FILE_CATEGORIES[category_index]
            for category_index in np.nanargmax(
                category_probabilities[scored_rows], axis=1
            )
        ]
    result["race"] = predicted_categories
    result["evidence_basis"] = evidence_basis

    add_inference_metadata(
        result,
        target="race-ethnicity",
        input_scope="full-name",
        scored=scored_rows,
        script_supported=script_supported,
        abstained=~scored_rows,
        abstention_reasons=abstention_reasons,
        label_column="race",
        model_id="six-state-voter-file-name-dictionary",
        model_revision=artifact_revision(
            Path(VOTER_FILE_FIRST_NAME_TABLE),
            Path(VOTER_FILE_LAST_NAME_TABLE),
            Path(VOTER_FILE_REFERENCE_STATS),
        ),
        reference_population="AL/FL/GA/LA/NC/SC voter files",
        calibration_status="not-applicable-dictionary",
    )

    evidence_counts = pd.Series(evidence_basis).value_counts().to_dict()
    logger.info(f"Evidence counts: {evidence_counts}")
    return result
