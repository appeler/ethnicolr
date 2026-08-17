"""Full-name estimator trained on North Carolina voter records."""

from __future__ import annotations

import pandas as pd

from .inference import prepare_full_name_data
from .neural_name_model import NeuralNameModel


class NorthCarolinaVoterFullNameModel(NeuralNameModel):
    """Twelve-category model for full names in North Carolina voter records."""

    MODEL_FILE = "models/nc_voter_reg/lstm/nc_name_lstm_pt.pt"
    VOCABULARY_FILE = "models/nc_voter_reg/lstm/nc_name_vocab_pt.json"
    LABELS_FILE = "models/nc_voter_reg/lstm/nc_name_labels_pt.json"
    NGRAM_SIZE = 2
    MAX_SEQUENCE_LENGTH = 25

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


def estimate_north_carolina_voter_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from North Carolina voter names."""
    return NorthCarolinaVoterFullNameModel.estimate(
        data,
        surname_column,
        first_name_column,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
