"""Resolve pinned neural model weights from Hugging Face."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from huggingface_hub import hf_hub_download

HUGGING_FACE_REPOSITORY = "gojiberries/ethnicolr"
HUGGING_FACE_REVISION = "0b89b64ea7b8ace75917ab3681e0358e08ed6a5c"
MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "ETHNICOLR_MODEL_DIR"
MODEL_CACHE_ENVIRONMENT_VARIABLE = "ETHNICOLR_MODEL_CACHE"


@dataclass(frozen=True)
class ModelBundlePaths:
    """Resolved local paths for one neural model bundle."""

    model_weight: Path
    vocabulary: Path
    labels: Path
    statistics: Path | None
    training_manifest: Path | None

    @property
    def revision_files(self) -> tuple[Path, ...]:
        """Return every artifact that defines the model revision."""
        optional_files = tuple(
            path
            for path in (self.statistics, self.training_manifest)
            if path is not None
        )
        return (self.model_weight, self.vocabulary, self.labels, *optional_files)


def _resolve_packaged_file(file_name: str) -> Path:
    path = Path(file_name)
    return path if path.is_absolute() else Path(str(files("ethnicolr"))) / path


def _associated_file(
    model_file: str, suffix_pairs: tuple[tuple[str, str], ...]
) -> str | None:
    model_path = Path(model_file)
    for model_suffix, associated_suffix in suffix_pairs:
        if model_path.name.endswith(model_suffix):
            return str(
                model_path.with_name(
                    model_path.name.replace(model_suffix, associated_suffix)
                )
            )
    return None


def resolve_model_weight(file_name: str) -> Path:
    """Return a local path for a pinned model weight file.

    An absolute path is used directly by training and calibration tools. Set
    ``ETHNICOLR_MODEL_DIR`` to use a local mirror whose layout matches the Hub
    repository. Otherwise the requested file is downloaded through the Hugging
    Face cache at the package's pinned commit.
    """
    requested_path = Path(file_name)
    if requested_path.is_absolute():
        return requested_path

    repository_file = requested_path.as_posix().removeprefix("models/")
    local_model_directory = os.environ.get(MODEL_DIRECTORY_ENVIRONMENT_VARIABLE)
    if local_model_directory:
        local_path = Path(local_model_directory) / repository_file
        if not local_path.is_file():
            raise FileNotFoundError(
                f"Model weight not found in {MODEL_DIRECTORY_ENVIRONMENT_VARIABLE}: "
                f"{local_path}"
            )
        return local_path

    cache_directory = os.environ.get(MODEL_CACHE_ENVIRONMENT_VARIABLE)
    downloaded_path = hf_hub_download(
        repo_id=HUGGING_FACE_REPOSITORY,
        filename=repository_file,
        revision=HUGGING_FACE_REVISION,
        cache_dir=cache_directory,
        library_name="ethnicolr",
        library_version=version("ethnicolr"),
    )
    return Path(downloaded_path)


def resolve_model_bundle(
    model_file: str, vocabulary_file: str, labels_file: str
) -> ModelBundlePaths:
    """Resolve the weight and packaged metadata paths for one model."""
    statistics_file = _associated_file(
        model_file,
        (
            ("_lstm_pt.pt", "_stats_pt.json"),
            ("_lstm_pytorch.pt", "_stats_pytorch.json"),
        ),
    )
    training_manifest_file = _associated_file(
        model_file,
        (
            ("_lstm_pt.pt", "_training_pt.json"),
            ("_lstm_pytorch.pt", "_training_pytorch.json"),
        ),
    )
    statistics_path = (
        _resolve_packaged_file(statistics_file) if statistics_file else None
    )
    training_manifest_path = (
        _resolve_packaged_file(training_manifest_file)
        if training_manifest_file
        else None
    )
    return ModelBundlePaths(
        model_weight=resolve_model_weight(model_file),
        vocabulary=_resolve_packaged_file(vocabulary_file),
        labels=_resolve_packaged_file(labels_file),
        statistics=(
            statistics_path
            if statistics_path is not None and statistics_path.is_file()
            else None
        ),
        training_manifest=(
            training_manifest_path
            if training_manifest_path is not None and training_manifest_path.is_file()
            else None
        ),
    )
