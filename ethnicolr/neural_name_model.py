#!/usr/bin/env python

import json
import logging
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inference import (
    add_inference_metadata,
    rename_conflicting_input_columns,
    validate_inference_options,
)
from .model_artifacts import resolve_model_bundle
from .model_metadata import load_class_labels, load_vocabulary
from .torch_utils import (
    adjust_probabilities_for_prior,
    artifact_revision,
    build_conformal_prediction_sets,
    load_character_ngram_model,
    name_support_reason,
    pad_name_sequences,
    select_inference_device,
    validate_mc_dropout,
)

logger = logging.getLogger(__name__)


class NeuralNameModel:
    """Shared loading, encoding, and inference for character n-gram models."""

    _model_cache: dict[str, dict] = {}

    @classmethod
    def _load_resources(
        cls, vocabulary_file: str, labels_file: str, model_file: str
    ) -> dict:
        """Load and cache vocabulary, class labels, and model weights."""
        model_bundle = resolve_model_bundle(model_file, vocabulary_file, labels_file)
        cache_key = str(model_bundle.model_weight)
        model_resources = cls._model_cache.get(cache_key)
        if model_resources is None:
            vocabulary_path = model_bundle.vocabulary
            labels_path = model_bundle.labels
            model_path = model_bundle.model_weight
            vocabulary = load_vocabulary(vocabulary_path)
            class_labels = load_class_labels(labels_path)
            device = select_inference_device()
            logger.info(f"Loading model {model_file} on {device}")
            model = load_character_ngram_model(
                model_path=model_path,
                vocabulary_size=len(vocabulary),
                category_count=len(class_labels),
                device=device,
            )
            model_statistics = (
                json.loads(model_bundle.statistics.read_text())
                if model_bundle.statistics
                else None
            )
            calibration_status = (
                model_statistics.get("calibration_status", "validated-legacy")
                if model_statistics
                else "unavailable"
            )
            if calibration_status == "validated-legacy":
                calibration_status = "legacy-unaudited"
            model_resources = {
                "vocabulary_index": {
                    token: index for index, token in enumerate(vocabulary)
                },
                "class_labels": class_labels,
                "model": model,
                "device": device,
                "model_statistics": model_statistics,
                "temperature": (
                    model_statistics["temperature"]
                    if model_statistics
                    and not calibration_status.startswith("invalid-")
                    else 1.0
                ),
                "calibration_status": calibration_status,
                "model_revision": artifact_revision(*model_bundle.revision_files),
            }
            cls._model_cache[cache_key] = model_resources
        return model_resources

    @staticmethod
    def validate_name_column(data: pd.DataFrame, name_column: str) -> pd.DataFrame:
        """Validate a name column without dropping null or duplicate rows."""
        if name_column not in data.columns:
            raise ValueError(f"Column {name_column!r} does not exist.")

        if (data.columns == name_column).sum() > 1:
            raise ValueError(f"Duplicate column name {name_column!r}.")
        null_rows = np.asarray(data[name_column].isna(), dtype=bool)
        null_count = int(null_rows.sum())
        if null_count > 0:
            logger.info(f"Preserving {null_count} null rows in {name_column!r}")

        if data.empty or null_rows.all():
            logger.warning("The name column has no usable values.")

        duplicate_count = int(data.duplicated(subset=[name_column]).sum())
        if duplicate_count > 0:
            logger.info(
                f"Preserving {duplicate_count} duplicate rows based on {name_column!r}"
            )

        return data.copy()

    @staticmethod
    def generate_ngrams(sequence, size: int = 1):
        """Generate overlapping n-grams from a sequence."""

        def tokens_from_offset(offset: int):
            return (
                token for position, token in enumerate(sequence) if position >= offset
            )

        shifted_tokens = (tokens_from_offset(offset) for offset in range(size))
        return zip(*shifted_tokens, strict=False)

    @staticmethod
    def generate_ngram_range(sequence, ngram_range: tuple[int, int] = (1, 2)):
        """Generate overlapping n-grams for a half-open range of sizes."""
        return chain(
            *(
                NeuralNameModel.generate_ngrams(sequence, size)
                for size in range(*ngram_range)
            )
        )

    @classmethod
    def encode_ngrams(
        cls,
        vocabulary_index: dict[str, int],
        text: str,
        ngram_size: int | tuple[int, int],
    ) -> list[int]:
        """Map character n-grams to indices, using zero for unknowns."""
        if isinstance(ngram_size, tuple):
            ngram_iterator = cls.generate_ngram_range(text, ngram_size)
        else:
            ngram_iterator = zip(
                *[text[offset:] for offset in range(ngram_size)], strict=False
            )

        return [vocabulary_index.get("".join(ngram), 0) for ngram in ngram_iterator]

    @classmethod
    def estimate_names(
        cls,
        data: pd.DataFrame,
        name_column: str,
        vocabulary_file: str,
        labels_file: str,
        model_file: str,
        ngram_size,
        max_sequence_length: int,
        mc_iterations: int,
        uncertainty_level: float | None,
        target: str,
        input_scope: str,
        label_column: str,
        target_prior: dict[str, float] | None = None,
        conformal_coverage: float | None = None,
    ) -> pd.DataFrame:
        """Run a neural name model and append probabilities and metadata.

        Args:
            data: Input DataFrame containing names to predict.
            name_column: Column name containing the names to predict on.
            vocabulary_file: Path to vocabulary metadata used for n-gram mapping.
            labels_file: Path to ordered class-label metadata.
            model_file: Path to the trained LSTM state dictionary.
            ngram_size: N-gram size (int) or range (tuple) for feature extraction.
            max_sequence_length: Maximum sequence length for padding/truncation.
            mc_iterations: Number of Monte Carlo iterations for MC-dropout ranges.
            uncertainty_level: Optional MC-dropout range level. ``None`` returns
                point estimates.
            label_column: Name of the returned predicted-label column.

        Returns:
            DataFrame with original data plus prediction columns:
            - 'race': Predicted race/ethnicity category
            - Probability columns for each race/ethnicity
            - MC-dropout range bounds (if uncertainty_level is not None)

        Raises:
            FileNotFoundError: If model files don't exist.
            ValueError: If required columns are missing.
        """
        data = cls.validate_name_column(data, name_column)
        validate_inference_options(
            uncertainty_level=uncertainty_level,
            target_prior=target_prior,
            conformal_coverage=conformal_coverage,
        )
        validate_mc_dropout(uncertainty_level, mc_iterations)

        data[name_column] = (
            data[name_column].fillna("").astype(str).str.strip().str.title()
        )

        model_resources = cls._load_resources(vocabulary_file, labels_file, model_file)
        model = model_resources["model"]
        class_labels = model_resources["class_labels"]
        model_statistics = model_resources["model_statistics"]
        temperature = model_resources["temperature"]
        calibration_status = model_resources["calibration_status"]

        if (
            target_prior is not None or conformal_coverage is not None
        ) and model_statistics is None:
            raise ValueError(
                "this model has no calibration stats file; run "
                "scripts/model-training/calibrate_model.py"
            )
        if conformal_coverage is not None:
            if calibration_status.startswith("invalid-"):
                raise ValueError(
                    "conformal_coverage is unavailable because this model's calibration "
                    f"artifact is {calibration_status}; retrain and recalibrate it"
                )
            available_coverages = (
                model_statistics["conformal_quantiles"] if model_statistics else {}
            )
            coverage_key = f"{conformal_coverage:.2f}"
            if coverage_key not in available_coverages:
                raise ValueError(
                    "conformal_coverage must be one of "
                    f"{sorted(available_coverages)}, got {conformal_coverage}"
                )

        # Vectorize input
        logger.debug(f"Vectorizing {len(data)} names using {ngram_size}-grams")
        encoded_names = [
            cls.encode_ngrams(model_resources["vocabulary_index"], name, ngram_size)
            for name in data[name_column]
        ]
        support_reasons = np.array(
            [name_support_reason(name) for name in data[name_column]], dtype=object
        )
        script_supported = np.fromiter(
            (reason is None for reason in support_reasons), dtype=bool
        )
        has_known_features = np.array(
            [any(token != 0 for token in sequence) for sequence in encoded_names],
            dtype=bool,
        )
        scored_rows = script_supported & has_known_features
        padded_name_sequences = pad_name_sequences(encoded_names, max_sequence_length)
        input_tensor = torch.from_numpy(padded_name_sequences[scored_rows]).to(
            model_resources["device"]
        )
        output_columns = set(class_labels) | {label_column}
        if conformal_coverage is not None:
            output_columns.add(f"{label_column}_set")
        if uncertainty_level is not None:
            for class_label in class_labels:
                output_columns.update(
                    {
                        f"{class_label}_mc_mean",
                        f"{class_label}_mc_std",
                        f"{class_label}_mc_lower",
                        f"{class_label}_mc_upper",
                    }
                )
        result_input = rename_conflicting_input_columns(data, output_columns)

        if uncertainty_level is None:
            probabilities = np.full((len(data), len(class_labels)), np.nan, dtype=float)
            if scored_rows.any():
                with torch.no_grad():
                    scored_probabilities = (
                        torch.softmax(model(input_tensor) / temperature, dim=1)
                        .cpu()
                        .numpy()
                    )
                if target_prior is not None:
                    scored_probabilities = adjust_probabilities_for_prior(
                        scored_probabilities,
                        class_labels,
                        target_prior,
                        model_statistics["train_class_distribution"],
                    )
                probabilities[scored_rows] = scored_probabilities
            probability_frame = pd.DataFrame(probabilities, columns=class_labels)
            probability_frame[label_column] = pd.Series(
                None, index=probability_frame.index, dtype="object"
            )
            probability_frame.loc[scored_rows, label_column] = probability_frame.loc[
                scored_rows, class_labels
            ].idxmax(axis=1)
            if conformal_coverage is not None:
                conformal_quantile = model_statistics["conformal_quantiles"][
                    f"{conformal_coverage:.2f}"
                ]
                prediction_sets = pd.Series(
                    None, index=probability_frame.index, dtype="object"
                )
                prediction_sets.loc[scored_rows] = build_conformal_prediction_sets(
                    probabilities[scored_rows], class_labels, conformal_quantile
                )
                probability_frame[f"{label_column}_set"] = prediction_sets
            probability_frame.index = data.index
            result = pd.concat([result_input, probability_frame], axis=1)

        else:
            lower_percentile = (0.5 - uncertainty_level / 2) * 100
            upper_percentile = (0.5 + uncertainty_level / 2) * 100

            logger.info(
                f"Generating {mc_iterations} MC-dropout samples "
                f"[{lower_percentile:.1f}%, {upper_percentile:.1f}%]"
            )

            uncertainty_summary = pd.DataFrame(index=data.index)
            for class_label in class_labels:
                uncertainty_summary[f"{class_label}_mc_mean"] = np.nan
                uncertainty_summary[f"{class_label}_mc_std"] = np.nan
                uncertainty_summary[f"{class_label}_mc_lower"] = np.nan
                uncertainty_summary[f"{class_label}_mc_upper"] = np.nan
                uncertainty_summary[class_label] = np.nan
            uncertainty_summary[label_column] = pd.Series(
                None, index=data.index, dtype="object"
            )
            if scored_rows.any():
                model.train()
                try:
                    with torch.no_grad():
                        sampled_probabilities = np.stack(
                            [
                                torch.softmax(model(input_tensor) / temperature, dim=1)
                                .cpu()
                                .numpy()
                                for _ in range(mc_iterations)
                            ],
                            axis=0,
                        )
                finally:
                    model.eval()

                mean_probabilities = sampled_probabilities.mean(axis=0)
                probability_standard_deviation = sampled_probabilities.std(
                    axis=0, ddof=1
                )
                lower_probabilities = np.percentile(
                    sampled_probabilities, lower_percentile, axis=0
                )
                upper_probabilities = np.percentile(
                    sampled_probabilities, upper_percentile, axis=0
                )
                for class_index, class_label in enumerate(class_labels):
                    uncertainty_summary.loc[scored_rows, f"{class_label}_mc_mean"] = (
                        mean_probabilities[:, class_index]
                    )
                    uncertainty_summary.loc[scored_rows, f"{class_label}_mc_std"] = (
                        probability_standard_deviation[:, class_index]
                    )
                    uncertainty_summary.loc[scored_rows, f"{class_label}_mc_lower"] = (
                        lower_probabilities[:, class_index]
                    )
                    uncertainty_summary.loc[scored_rows, f"{class_label}_mc_upper"] = (
                        upper_probabilities[:, class_index]
                    )
                    uncertainty_summary.loc[scored_rows, class_label] = (
                        mean_probabilities[:, class_index]
                    )
                uncertainty_summary.loc[scored_rows, label_column] = (
                    pd.DataFrame(mean_probabilities, columns=class_labels)
                    .idxmax(axis=1)
                    .to_numpy()
                )
            result = pd.concat([result_input, uncertainty_summary], axis=1)

        abstention_reasons = support_reasons.copy()
        abstention_reasons[script_supported & ~has_known_features] = "out-of-vocabulary"
        reference_population = (
            model_statistics["calibration_weighting"] if model_statistics else pd.NA
        )
        if uncertainty_level is not None:
            uncertainty_method = "mc-dropout"
            reported_uncertainty_level = uncertainty_level
        elif conformal_coverage is not None:
            uncertainty_method = "split-conformal"
            reported_uncertainty_level = conformal_coverage
        else:
            uncertainty_method = None
            reported_uncertainty_level = None
        add_inference_metadata(
            result,
            script_supported=script_supported,
            abstained=~scored_rows,
            abstention_reasons=abstention_reasons,
            model_id=(
                model_statistics["model"] if model_statistics else Path(model_file).stem
            ),
            model_revision=model_resources["model_revision"],
            reference_population=reference_population,
            calibration_reference=reference_population,
            calibration_status=calibration_status,
            uncertainty_method=uncertainty_method,
            uncertainty_level=reported_uncertainty_level,
            target=target,
            input_scope=input_scope,
            scored=scored_rows,
            label_column=label_column,
        )

        return result
