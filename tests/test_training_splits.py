"""Regression tests for source-disjoint model evaluation splits."""

import sys
from pathlib import Path

import pandas as pd

TRAINING_DIR = Path(__file__).parents[1] / "scripts" / "model-training"
sys.path.insert(0, str(TRAINING_DIR))

from train_name_lstm import ModelConfig, prepare_data_partitions  # noqa: E402


def test_balancing_happens_after_source_disjoint_split(tmp_path: Path) -> None:
    categories = [
        "asian",
        "hispanic",
        "nh_black",
        "nh_white",
        "other",
    ]
    rows = []
    for category in categories:
        rows.extend(
            {
                "name_first": f"first{index}{category}",
                "name_last": f"last{index}{category}",
                "race": category,
            }
            for index in range(40)
        )
    path = tmp_path / "fl.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    config = ModelConfig("florida", "name", 25, 1, "unused", path.name, 5)

    train, validation, test = prepare_data_partitions(
        config, path, seed=42, source_row_limit=100
    )

    assert len(train) == 70
    assert len(validation) == 10
    assert len(test) == 20
    assert train["race"].value_counts().nunique() == 1
    assert set(train["__source_row"]).isdisjoint(validation["__source_row"])
    assert set(train["__source_row"]).isdisjoint(test["__source_row"])
    assert set(validation["__source_row"]).isdisjoint(test["__source_row"])
