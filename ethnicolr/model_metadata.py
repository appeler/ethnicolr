"""Read and write validated model metadata artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

MODEL_METADATA_SCHEMA_VERSION = 1


def _load_ordered_strings(
    path: Path, *, artifact_type: str, values_key: str
) -> list[str]:
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"cannot read model metadata {path}: {error}") from error

    if not isinstance(document, dict):
        raise ValueError(f"model metadata {path} must be a JSON object")
    if document.get("schema_version") != MODEL_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"model metadata {path} has unsupported schema_version "
            f"{document.get('schema_version')!r}"
        )
    if document.get("artifact_type") != artifact_type:
        raise ValueError(
            f"model metadata {path} must have artifact_type={artifact_type!r}"
        )

    values = document.get(values_key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"model metadata {path} must contain a non-empty {values_key}")
    if any(not isinstance(value, str) or value == "" for value in values):
        raise ValueError(
            f"model metadata {path} {values_key} must be non-empty strings"
        )
    if len(values) != len(set(values)):
        raise ValueError(f"model metadata {path} {values_key} must be unique")
    return values


def _write_ordered_strings(
    path: Path, *, artifact_type: str, values_key: str, values: list[str]
) -> None:
    document = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        values_key: values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_vocabulary(path: Path) -> list[str]:
    """Load an ordered vocabulary and validate its schema."""
    tokens = _load_ordered_strings(
        path, artifact_type="vocabulary", values_key="tokens"
    )
    if tokens[0] != "UNK":
        raise ValueError(f"model vocabulary {path} must begin with the UNK token")
    return tokens


def write_vocabulary(path: Path, tokens: list[str]) -> None:
    """Write an ordered vocabulary using the model metadata schema."""
    if not tokens or tokens[0] != "UNK":
        raise ValueError("model vocabulary must begin with the UNK token")
    _write_ordered_strings(
        path, artifact_type="vocabulary", values_key="tokens", values=tokens
    )


def load_class_labels(path: Path) -> list[str]:
    """Load ordered class labels and validate their schema."""
    return _load_ordered_strings(
        path, artifact_type="class-labels", values_key="labels"
    )


def write_class_labels(path: Path, labels: list[str]) -> None:
    """Write ordered class labels using the model metadata schema."""
    _write_ordered_strings(
        path, artifact_type="class-labels", values_key="labels", values=labels
    )
