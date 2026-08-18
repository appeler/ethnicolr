"""Stable public API for name-pattern estimation."""

from __future__ import annotations

import pandas as pd

from .census_surname import lookup_census_surname as _lookup_census_surname
from .census_surname_model import estimate_census_surname as _estimate_census_surname
from .florida_voter import (
    estimate_florida_voter_full_name as _estimate_florida_voter_full_name,
)
from .florida_voter import (
    estimate_florida_voter_surname as _estimate_florida_voter_surname,
)
from .name_dictionaries import (
    estimate_census_full_name as _estimate_census_full_name,
)
from .name_dictionaries import (
    estimate_voter_file_full_name as _estimate_voter_file_full_name,
)
from .name_dictionaries import (
    lookup_census_first_name as _lookup_census_first_name,
)
from .north_carolina_voter_full_name import (
    estimate_north_carolina_voter_full_name as _estimate_north_carolina_full_name,
)
from .wikipedia_full_name import (
    estimate_wikipedia_full_name as _estimate_wikipedia_full_name,
)
from .wikipedia_origin import estimate_wikipedia_origin as _estimate_wikipedia_origin
from .wikipedia_surname import (
    estimate_wikipedia_surname as _estimate_wikipedia_surname,
)


def lookup_census_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    year: int = 2020,
    uncertainty_level: float | None = None,
) -> pd.DataFrame:
    """Look up Census race/ethnicity proportions for a surname."""
    return _lookup_census_surname(
        data,
        surname_column,
        year=year,
        uncertainty_level=uncertainty_level,
    )


def lookup_census_first_name(
    data: pd.DataFrame,
    first_name_column: str,
    *,
    year: int = 2020,
    uncertainty_level: float | None = None,
) -> pd.DataFrame:
    """Look up Census race/ethnicity proportions for a first name."""
    return _lookup_census_first_name(
        data,
        first_name_column,
        year=year,
        uncertainty_level=uncertainty_level,
    )


def estimate_census_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    year: int = 2020,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a surname using a Census-trained model."""
    return _estimate_census_surname(
        data,
        surname_column,
        year=year,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_census_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    year: int = 2020,
    target_prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from Census first- and surname evidence."""
    return _estimate_census_full_name(
        data,
        surname_column,
        first_name_column,
        year=year,
        target_prior=target_prior,
    )


def estimate_voter_file_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    target_prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from six-state voter-file name frequencies."""
    return _estimate_voter_file_full_name(
        data,
        surname_column,
        first_name_column,
        target_prior=target_prior,
    )


def estimate_florida_voter_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a surname using a Florida voter model."""
    return _estimate_florida_voter_surname(
        data,
        surname_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_florida_voter_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a full name using a Florida voter model."""
    return _estimate_florida_voter_full_name(
        data,
        surname_column,
        first_name_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_north_carolina_voter_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a full name using North Carolina voters."""
    return _estimate_north_carolina_full_name(
        data,
        surname_column,
        first_name_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_wikipedia_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a surname using Wikipedia biographies."""
    return _estimate_wikipedia_surname(
        data,
        surname_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_wikipedia_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity from a full name using Wikipedia biographies."""
    return _estimate_wikipedia_full_name(
        data,
        surname_column,
        first_name_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_wikipedia_origin(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    uncertainty_level: float | None = None,
    mc_iterations: int = 100,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate national origin from a full name using Wikipedia biographies."""
    return _estimate_wikipedia_origin(
        data,
        surname_column,
        first_name_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
