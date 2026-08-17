"""Surname estimator trained on Wikipedia and Wikidata biographies."""

from __future__ import annotations

import pandas as pd

from .neural_name_model import NeuralNameModel


class WikipediaSurnameModel(NeuralNameModel):
    """Race/ethnicity model for surnames in Wikipedia biographies."""

    MODEL_FILE = "models/wiki/lstm/wiki_ln_lstm_pt.pt"
    VOCABULARY_FILE = "models/wiki/lstm/wiki_ln_vocab_pt.json"
    LABELS_FILE = "models/wiki/lstm/wiki_ln_labels_pt.json"
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
            data=data,
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
            label_column="race",
            target_prior=target_prior,
            conformal_coverage=conformal_coverage,
        )


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
