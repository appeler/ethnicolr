"""Country-of-origin estimator trained on Wikipedia and Wikidata biographies."""

from __future__ import annotations

import pandas as pd

from .inference import prepare_full_name_data
from .neural_name_model import NeuralNameModel


class WikipediaOriginModel(NeuralNameModel):
    """Country-of-origin model for full names in Wikipedia biographies."""

    MODEL_FILE = "models/wiki/lstm/wiki_origin_lstm_pt.pt"
    VOCABULARY_FILE = "models/wiki/lstm/wiki_origin_vocab_pt.json"
    LABELS_FILE = "models/wiki/lstm/wiki_origin_labels_pt.json"
    NGRAM_SIZE = 2
    MAX_SEQUENCE_LENGTH = 25

    @classmethod
    def estimate(
        cls,
        data: pd.DataFrame,
        surname_column: str,
        first_name_column: str,
        *,
        uncertainty_level: float | None = None,
        mc_iterations: int = 100,
        target_prior: dict[str, float] | None = None,
        conformal_coverage: float | None = None,
    ) -> pd.DataFrame:
        """Estimate country-of-origin patterns from full names."""
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
            target="country-origin",
            input_scope="full-name",
            target_prior=target_prior,
            conformal_coverage=conformal_coverage,
        )
        result.drop(columns=[full_name_column], inplace=True)
        renamed_columns = {"race": "origin"}
        if "race_set" in result.columns:
            renamed_columns["race_set"] = "origin_set"
        return result.rename(columns=renamed_columns)


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
