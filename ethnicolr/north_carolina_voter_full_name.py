"""Full-name estimator trained on North Carolina voter records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import FullNameModel

if TYPE_CHECKING:
    import pandas as pd


class NorthCarolinaVoterFullNameModel(FullNameModel):
    """Twelve-category model for full names in North Carolina voter records."""

    MODEL_FILE = "models/nc_voter_reg/lstm/nc_name_lstm_pt.pt"
    VOCABULARY_FILE = "models/nc_voter_reg/lstm/nc_name_vocab_pt.json"
    LABELS_FILE = "models/nc_voter_reg/lstm/nc_name_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 25


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
