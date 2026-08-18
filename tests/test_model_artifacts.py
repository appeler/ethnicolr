"""Tests for pinned Hugging Face model resolution."""

from pathlib import Path

import pytest

from ethnicolr import model_artifacts


def test_local_model_directory_override(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "wiki" / "lstm" / "model.pt"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    monkeypatch.setenv("ETHNICOLR_MODEL_DIR", str(tmp_path))

    resolved_path = model_artifacts.resolve_model_weight("models/wiki/lstm/model.pt")

    assert resolved_path == model_path


def test_missing_local_model_weight_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ETHNICOLR_MODEL_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="ETHNICOLR_MODEL_DIR"):
        model_artifacts.resolve_model_weight("models/wiki/lstm/missing.pt")


def test_hugging_face_download_uses_full_pinned_revision(
    tmp_path: Path, monkeypatch
) -> None:
    downloaded_path = tmp_path / "model.pt"
    downloaded_path.touch()
    captured_arguments: dict[str, object] = {}

    def fake_download(**arguments: object) -> str:
        captured_arguments.update(arguments)
        return str(downloaded_path)

    monkeypatch.delenv("ETHNICOLR_MODEL_DIR", raising=False)
    monkeypatch.setattr(model_artifacts, "hf_hub_download", fake_download)

    resolved_path = model_artifacts.resolve_model_weight("models/wiki/lstm/model.pt")

    assert resolved_path == downloaded_path
    assert captured_arguments["repo_id"] == "gojiberries/ethnicolr"
    assert captured_arguments["revision"] == model_artifacts.HUGGING_FACE_REVISION
    assert len(captured_arguments["revision"]) == 40
    assert captured_arguments["filename"] == "wiki/lstm/model.pt"
