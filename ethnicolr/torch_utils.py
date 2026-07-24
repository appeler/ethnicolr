"""Shared PyTorch utilities for ethnicolr models."""

from __future__ import annotations

import os

import torch


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
