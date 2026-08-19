"""Shared PyTorch utilities for ethnicolr models."""

from __future__ import annotations

import hashlib
import math
import os
import unicodedata
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path


def name_support_reason(value: str) -> str | None:
    """Return why a normalized name cannot be scored, if it is unsupported."""
    if not value.strip():
        return "missing-name"
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return "no-letters"
    if not all(
        unicodedata.name(character, "").startswith("LATIN ") for character in letters
    ):
        return "unsupported-script"
    return None


def validate_mc_dropout(uncertainty_level: float | None, mc_iterations: int) -> None:
    """Validate an optional Monte Carlo dropout uncertainty request."""
    if uncertainty_level is None:
        return
    if isinstance(uncertainty_level, bool) or not isinstance(
        uncertainty_level, (int, float)
    ):
        raise TypeError("uncertainty_level must be a finite number")
    if not math.isfinite(float(uncertainty_level)) or not 0 < uncertainty_level < 1:
        raise ValueError("uncertainty_level must be strictly between 0 and 1")
    if isinstance(mc_iterations, bool) or not isinstance(mc_iterations, int):
        raise TypeError("mc_iterations must be an integer")
    if mc_iterations < 2:
        raise ValueError("mc_iterations must be at least 2")


@cache
def artifact_revision(*paths: Path) -> str:
    """Return an immutable SHA-256 revision for an ordered artifact bundle."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class CharacterNgramLSTM(nn.Module):
    """Character n-gram LSTM shared by all ethnicolr prediction models."""

    def __init__(
        self,
        vocabulary_size: int,
        category_count: int,
        embedding_dimension: int = 32,
        hidden_dimension: int = 128,
        dropout: float = 0.2,
    ):
        """Build the embedding, LSTM, dropout, and output layers."""
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=0
        )
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dimension, category_count)

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """Return category logits for a batch of encoded name sequences."""
        embedded = self.embedding(input_sequences)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        return self.output_layer(hidden)


def load_character_ngram_model(
    model_path: Path,
    vocabulary_size: int,
    category_count: int,
    device: torch.device,
) -> CharacterNgramLSTM:
    """Load a character n-gram model from a packaged state dictionary."""
    model = CharacterNgramLSTM(
        vocabulary_size=vocabulary_size, category_count=category_count
    )
    model_state = torch.load(model_path, map_location=device, weights_only=True)
    model_state = {
        (
            key.replace("fc.", "output_layer.", 1) if key.startswith("fc.") else key
        ): value
        for key, value in model_state.items()
    }
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def pad_name_sequences(
    sequences: list[list[int]], max_sequence_length: int
) -> np.ndarray:
    """Pre-pad integer sequences with zeros to a fixed length.

    Sequences longer than the maximum keep their first tokens.
    """
    padded_sequences = np.zeros((len(sequences), max_sequence_length), dtype=np.int64)
    for sequence_index, sequence in enumerate(sequences):
        if len(sequence) > max_sequence_length:
            padded_sequences[sequence_index] = sequence[:max_sequence_length]
        elif sequence:
            padded_sequences[sequence_index, -len(sequence) :] = sequence
    return padded_sequences


def adjust_probabilities_for_prior(
    category_probabilities: np.ndarray,
    categories: list[str],
    target_prior: dict[str, float],
    train_distribution: dict[str, float],
) -> np.ndarray:
    """Reweight calibrated probabilities to a target class distribution.

    Bayes adjustment: p_adj(y|x) ∝ p(y|x) · π_target(y) / π_train(y). This is
    the correction needed when a model was trained on class-balanced data, and
    with geographic margins as the target it is the name-likelihood step of
    BISG-style pipelines.
    """
    missing_categories = set(categories) - set(target_prior)
    if missing_categories:
        raise ValueError(
            f"target_prior is missing classes: {sorted(missing_categories)}"
        )
    target_distribution = np.array(
        [target_prior[category] for category in categories], dtype=float
    )
    if (target_distribution < 0).any() or target_distribution.sum() <= 0:
        raise ValueError("target_prior probabilities must be non-negative and sum > 0")
    target_distribution = target_distribution / target_distribution.sum()
    training_distribution = np.array(
        [train_distribution[category] for category in categories], dtype=float
    )
    adjusted_probabilities = category_probabilities * (
        target_distribution / training_distribution
    )
    return adjusted_probabilities / adjusted_probabilities.sum(axis=1, keepdims=True)


def build_conformal_prediction_sets(
    category_probabilities: np.ndarray,
    categories: list[str],
    conformal_quantile: float,
) -> list[list[str]]:
    """Build adaptive prediction sets.

    Returns the smallest class sets whose cumulative calibrated probability
    mass reaches the conformal quantile.
    """
    descending_category_indices = np.argsort(-category_probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(
        category_probabilities, descending_category_indices, axis=1
    )
    cumulative_probabilities = np.cumsum(sorted_probabilities, axis=1)
    prediction_set_sizes = (cumulative_probabilities < conformal_quantile).sum(
        axis=1
    ) + 1
    return [
        [
            categories[category_index]
            for category_index in descending_category_indices[
                row_index, : prediction_set_sizes[row_index]
            ]
        ]
        for row_index in range(category_probabilities.shape[0])
    ]


def select_inference_device() -> torch.device:
    """Select the torch device for model inference.

    The ``ETHNICOLR_DEVICE`` environment variable (``cpu``, ``cuda``, ``mps``)
    takes precedence. Otherwise CUDA is used when available, else CPU.

    MPS is never auto-selected: virtualized Apple Silicon environments (such as
    GitHub Actions macOS runners) advertise MPS but produce incorrect LSTM
    output on it, so it is opt-in only.
    """
    override = os.environ.get("ETHNICOLR_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
