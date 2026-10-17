"""Surname estimator trained on Wikipedia and Wikidata biographies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import SurnameModel

if TYPE_CHECKING:
    import pandas as pd


class WikipediaSurnameModel(SurnameModel):
    """Race/ethnicity model for surnames in Wikipedia biographies."""

    MODEL_FILE = "models/wiki/lstm/wiki_ln_lstm_pt.pt"
    VOCABULARY_FILE = "models/wiki/lstm/wiki_ln_vocab_pt.json"
    LABELS_FILE = "models/wiki/lstm/wiki_ln_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 20


def estimate_wikipedia_surname(
    data: pd.DataFrame,
    surname_column: str,
    *,
    mc_iterations: int = 100,
    uncertainty_level: float | None = None,
    target_prior: dict[str, float] | None = None,
    conformal_coverage: float | None = None,
) -> pd.DataFrame:
    """Estimate race/ethnicity patterns from Wikipedia surnames."""
    return WikipediaSurnameModel.estimate(
        data,
        surname_column,
        mc_iterations=mc_iterations,
        uncertainty_level=uncertainty_level,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
