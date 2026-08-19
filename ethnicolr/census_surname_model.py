"""Neural surname estimators trained on U.S. Census data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import NeuralNameModel

if TYPE_CHECKING:
    import pandas as pd

CENSUS_MODEL_YEARS = (2000, 2010, 2020)
CENSUS_NGRAM_SIZE = 2
CENSUS_MAX_SEQUENCE_LENGTH = 20


def get_census_surname_model_files(year: int) -> tuple[str, str, str]:
    """Return the weight, vocabulary, and label file names for a Census year."""
    if year not in CENSUS_MODEL_YEARS:
        raise ValueError("year must be 2000, 2010, or 2020")

    model_directory = "models/census/lstm"
    return (
        f"{model_directory}/census{year}_ln_lstm_pytorch.pt",
        f"{model_directory}/census{year}_ln_vocab_pytorch.json",
        f"{model_directory}/census{year}_labels_pytorch.json",
    )


def estimate_census_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    year: int = 2020,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from surnames using a Census model."""
    model_file, vocabulary_file, labels_file = get_census_surname_model_files(year)
    return NeuralNameModel.estimate_names(
        data=data,
        name_column=surname_column,
        vocabulary_file=vocabulary_file,
        labels_file=labels_file,
        model_file=model_file,
        ngram_size=CENSUS_NGRAM_SIZE,
        max_sequence_length=CENSUS_MAX_SEQUENCE_LENGTH,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target="race-ethnicity",
        input_scope="last-name",
        label_column="race",
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
