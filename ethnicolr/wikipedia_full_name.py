"""Full-name estimator trained on Wikipedia and Wikidata biographies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import FullNameModel

if TYPE_CHECKING:
    import pandas as pd


class WikipediaFullNameModel(FullNameModel):
    """Race/ethnicity model for full names in Wikipedia biographies."""

    MODEL_FILE = "models/wiki/lstm/wiki_name_lstm_pt.pt"
    VOCABULARY_FILE = "models/wiki/lstm/wiki_name_vocab_pt.json"
    LABELS_FILE = "models/wiki/lstm/wiki_name_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 25


def estimate_wikipedia_full_name(
    data: pd.DataFrame,
    surname_column: str,
    first_name_column: str,
    *,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from full Wikipedia names."""
    return WikipediaFullNameModel.estimate(
        data,
        surname_column,
        first_name_column,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
