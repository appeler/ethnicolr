#!/usr/bin/env python3
"""Convert legacy model vocabulary and label CSVs to validated JSON metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ethnicolr.model_metadata import (  # noqa: E402
    load_class_labels,
    load_vocabulary,
    write_class_labels,
    write_vocabulary,
)


def _read_string_column(path: Path, column: str) -> list[str]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if table.columns.tolist() != [column]:
        raise ValueError(f"{path} must contain only the {column!r} column")
    return table[column].tolist()


def main() -> None:
    models_root = REPOSITORY_ROOT / "ethnicolr" / "models"
    vocabulary_paths = sorted(models_root.rglob("*vocab*.csv"))
    label_paths = sorted(models_root.rglob("*race*.csv"))

    for source_path in vocabulary_paths:
        target_path = source_path.with_suffix(".json")
        source_values = _read_string_column(source_path, "vocab")
        write_vocabulary(target_path, source_values)
        if load_vocabulary(target_path) != source_values:
            raise RuntimeError(f"conversion changed vocabulary values in {source_path}")
        print(f"wrote {target_path.relative_to(REPOSITORY_ROOT)}")

    for source_path in label_paths:
        target_name = source_path.name.replace("_race_", "_labels_")
        target_path = source_path.with_name(target_name).with_suffix(".json")
        source_values = _read_string_column(source_path, "race")
        write_class_labels(target_path, source_values)
        if load_class_labels(target_path) != source_values:
            raise RuntimeError(f"conversion changed class labels in {source_path}")
        print(f"wrote {target_path.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
