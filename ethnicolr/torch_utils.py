"""Shared PyTorch utilities for ethnicolr models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class NameLSTM(nn.Module):
    """Character n-gram LSTM shared by all ethnicolr prediction models."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)


def pad_sequences(sequences: list[list[int]], maxlen: int) -> np.ndarray:
    """Pre-pad integer sequences with zeros to a fixed length.

    Sequences longer than ``maxlen`` keep their first ``maxlen`` tokens.
    """
    padded = np.zeros((len(sequences), maxlen), dtype=np.int64)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[:maxlen]
        elif len(seq) > 0:
            padded[i, -len(seq) :] = seq
    return padded


def load_model_stats(model_path: Path) -> dict | None:
    """Load the calibration/conformal stats JSON shipped next to a model.

    Produced by ``scripts/model-training/calibrate_model.py``. Returns None
    when the model has no stats file.
    """
    name = model_path.name
    for suffix, stats_suffix in (
        ("_lstm_pt.pt", "_stats_pt.json"),
        ("_lstm_pytorch.pt", "_stats_pytorch.json"),
    ):
        if name.endswith(suffix):
            stats_path = model_path.with_name(name.replace(suffix, stats_suffix))
            if stats_path.exists():
                return json.loads(stats_path.read_text())
    return None


def apply_prior(
    probs: np.ndarray,
    classes: list[str],
    prior: dict[str, float],
    train_distribution: dict[str, float],
) -> np.ndarray:
    """Reweight calibrated probabilities to a target class distribution.

    Bayes adjustment: p_adj(y|x) ∝ p(y|x) · π_target(y) / π_train(y). This is
    the correction needed when a model was trained on class-balanced data, and
    with geographic margins as the target it is the name-likelihood step of
    BISG-style pipelines.
    """
    missing = set(classes) - set(prior)
    if missing:
        raise ValueError(f"prior is missing classes: {sorted(missing)}")
    target = np.array([prior[c] for c in classes], dtype=float)
    if (target < 0).any() or target.sum() <= 0:
        raise ValueError("prior probabilities must be non-negative and sum > 0")
    target = target / target.sum()
    train = np.array([train_distribution[c] for c in classes], dtype=float)
    adjusted = probs * (target / train)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def conformal_sets(
    probs: np.ndarray, classes: list[str], qhat: float
) -> list[list[str]]:
    """Adaptive prediction sets: smallest class sets with cumulative
    calibrated probability mass reaching the conformal quantile ``qhat``."""
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    sizes = (cum < qhat).sum(axis=1) + 1
    return [[classes[j] for j in order[i, : sizes[i]]] for i in range(probs.shape[0])]


def get_device() -> torch.device:
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
