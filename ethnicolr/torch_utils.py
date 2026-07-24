"""Shared PyTorch utilities for ethnicolr models."""

from __future__ import annotations

import os

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
