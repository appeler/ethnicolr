"""Name-pattern estimators trained on 2022 Florida voter records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import FullNameModel, SurnameModel

if TYPE_CHECKING:
    import pandas as pd


class FloridaVoterSurnameModel(SurnameModel):
    """Five-category model for Florida voter surnames."""

    MODEL_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_lstm_pt.pt"
    VOCABULARY_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_vocab_pt.json"
    LABELS_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 20


class FloridaVoterFullNameModel(FullNameModel):
    """Five-category model for full names in Florida voter records."""

    MODEL_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_lstm_pt.pt"
    VOCABULARY_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_vocab_pt.json"
    LABELS_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 25


def estimate_florida_voter_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from surnames in 2022 Florida voters."""
    return FloridaVoterSurnameModel.estimate(
        data,
        surname_column,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )


def estimate_florida_voter_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from full names in 2022 Florida voters."""
    return FloridaVoterFullNameModel.estimate(
        data,
        surname_column,
        first_name_column,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
