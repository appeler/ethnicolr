"""Name-pattern estimators trained on 2022 Florida voter records."""

from __future__ import annotations

import pandas as pd

from .inference import prepare_full_name_data
from .neural_name_model import NeuralNameModel


class FloridaVoterSurnameModel(NeuralNameModel):
    """Five-category model for Florida voter surnames."""

    MODEL_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_lstm_pt.pt"
    VOCABULARY_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_vocab_pt.json"
    LABELS_FILE = "models/fl_voter_reg/lstm/fl_ln_five_cat_2022_labels_pt.json"
    NGRAM_SIZE = 2
    MAX_SEQUENCE_LENGTH = 20

    @classmethod
    def estimate(
        cls,
        data: pd.DataFrame,
        surname_column: str,
        *,
        mc_iterations: int = 100,
        uncertainty_level: float | None = None,
        target_prior: dict[str, float] | None = None,
        conformal_coverage: float | None = None,
    ) -> pd.DataFrame:
        """Estimate race/ethnicity patterns from surnames."""
        if surname_column not in data.columns:
            raise ValueError(f"Surname column {surname_column!r} does not exist.")

        return cls.estimate_names(
            data=data.copy(),
            name_column=surname_column,
            vocabulary_file=cls.VOCABULARY_FILE,
            labels_file=cls.LABELS_FILE,
            model_file=cls.MODEL_FILE,
            ngram_size=cls.NGRAM_SIZE,
            max_sequence_length=cls.MAX_SEQUENCE_LENGTH,
            mc_iterations=mc_iterations,
            uncertainty_level=uncertainty_level,
            target="race-ethnicity",
            input_scope="last-name",
            target_prior=target_prior,
            conformal_coverage=conformal_coverage,
        )


class FloridaVoterFullNameModel(NeuralNameModel):
    """Five-category model for full names in Florida voter records."""

    MODEL_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_lstm_pt.pt"
    VOCABULARY_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_vocab_pt.json"
    LABELS_FILE = "models/fl_voter_reg/lstm/fl_name_five_cat_2022_labels_pt.json"
    NGRAM_SIZE = 2
    MAX_SEQUENCE_LENGTH = 20

    @classmethod
    def estimate(
        cls,
        data: pd.DataFrame,
        surname_column: str,
        first_name_column: str,
        *,
        mc_iterations: int = 100,
        uncertainty_level: float | None = None,
        target_prior: dict[str, float] | None = None,
        conformal_coverage: float | None = None,
    ) -> pd.DataFrame:
        """Estimate race/ethnicity patterns from full names."""
        result, full_name_column = prepare_full_name_data(
            data, surname_column, first_name_column
        )

        result = cls.estimate_names(
            data=result,
            name_column=full_name_column,
            vocabulary_file=cls.VOCABULARY_FILE,
            labels_file=cls.LABELS_FILE,
            model_file=cls.MODEL_FILE,
            ngram_size=cls.NGRAM_SIZE,
            max_sequence_length=cls.MAX_SEQUENCE_LENGTH,
            mc_iterations=mc_iterations,
            uncertainty_level=uncertainty_level,
            target="race-ethnicity",
            input_scope="full-name",
            target_prior=target_prior,
            conformal_coverage=conformal_coverage,
        )
        return result.drop(columns=[full_name_column])


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
