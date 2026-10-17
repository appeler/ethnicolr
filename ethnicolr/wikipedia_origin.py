"""Country-of-origin estimator trained on Wikipedia and Wikidata biographies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .neural_name_model import FullNameModel

if TYPE_CHECKING:
    import pandas as pd


class WikipediaOriginModel(FullNameModel):
    """Country-of-origin model for full names in Wikipedia biographies."""

    MODEL_FILE = "models/wiki/lstm/wiki_origin_lstm_pt.pt"
    VOCABULARY_FILE = "models/wiki/lstm/wiki_origin_vocab_pt.json"
    LABELS_FILE = "models/wiki/lstm/wiki_origin_labels_pt.json"
    MAX_SEQUENCE_LENGTH = 25
    TARGET = "country-origin"
    LABEL_COLUMN = "origin"


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
    """Estimate country-of-origin patterns from full Wikipedia names."""
    return WikipediaOriginModel.estimate(
        data,
        surname_column,
        first_name_column,
        uncertainty_level=uncertainty_level,
        mc_iterations=mc_iterations,
        target_prior=target_prior,
        conformal_coverage=conformal_coverage,
    )
