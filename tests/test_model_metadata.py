"""Tests for typed, ordered model metadata artifacts."""

import json

import pytest

from ethnicolr.model_metadata import (
    load_class_labels,
    load_vocabulary,
    write_class_labels,
    write_vocabulary,
)


def test_vocabulary_round_trip_preserves_order_unicode_and_whitespace(tmp_path) -> None:
    path = tmp_path / "vocabulary.json"
    tokens = ["UNK", "an", " a", "a ", "é"]

    write_vocabulary(path, tokens)

    assert load_vocabulary(path) == tokens


def test_class_labels_round_trip(tmp_path) -> None:
    path = tmp_path / "labels.json"
    labels = ["Asian,IndianSubContinent", "GreaterEuropean,Jewish"]

    write_class_labels(path, labels)

    assert load_class_labels(path) == labels


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {"schema_version": 2, "artifact_type": "vocabulary", "tokens": ["UNK"]},
            "schema_version",
        ),
        (
            {"schema_version": 1, "artifact_type": "class-labels", "tokens": ["UNK"]},
            "artifact_type",
        ),
        (
            {
                "schema_version": 1,
                "artifact_type": "vocabulary",
                "tokens": ["UNK", None],
            },
            "non-empty strings",
        ),
        (
            {
                "schema_version": 1,
                "artifact_type": "vocabulary",
                "tokens": ["UNK", "an", "an"],
            },
            "unique",
        ),
    ],
)
def test_invalid_vocabulary_metadata_is_rejected(tmp_path, document, message) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_vocabulary(path)
