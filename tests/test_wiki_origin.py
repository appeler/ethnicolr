"""Tests for the name-to-country-of-origin model."""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ethnicolr import estimate_wikipedia_origin

ORIGIN_STATISTICS_PATH = (
    Path(str(files("ethnicolr")))
    / "models"
    / "wiki"
    / "lstm"
    / "wiki_origin_stats_pt.json"
)
ORIGIN_STATISTICS = json.loads(ORIGIN_STATISTICS_PATH.read_text())
ORIGIN_CLASSES = ORIGIN_STATISTICS["classes"]


@pytest.fixture
def origin_names():
    return pd.DataFrame(
        {
            "last": [
                "tanaka",
                "kowalski",
                "nguyen",
                "rossi",
                "andersson",
                "papadopoulos",
            ],
            "first": ["yuki", "piotr", "minh", "marco", "lars", "nikos"],
            "expected": ["Japan", "Poland", "Vietnam", "Italy", "Sweden", "Greece"],
        }
    )


class TestWikiOrigin:
    def test_structure(self, origin_names):
        result = estimate_wikipedia_origin(origin_names, "last", "first")
        assert "origin" in result.columns
        assert len(result) == len(origin_names)
        assert len(ORIGIN_CLASSES) >= 50
        assert np.allclose(result[ORIGIN_CLASSES].sum(axis=1), 1.0, atol=1e-4)

    def test_distinctive_names_in_top3(self, origin_names):
        result = estimate_wikipedia_origin(origin_names, "last", "first")
        hits = 0
        for i, row in result.iterrows():
            top3 = row[ORIGIN_CLASSES].astype(float).nlargest(3).index
            if origin_names.loc[i, "expected"] in top3:
                hits += 1
        assert hits >= 4, f"only {hits}/6 distinctive names had origin in top 3"

    def test_coverage_sets(self, origin_names):
        result = estimate_wikipedia_origin(
            origin_names, "last", "first", conformal_coverage=0.9
        )
        assert "origin_set" in result.columns
        for _, row in result.iterrows():
            assert isinstance(row["origin_set"], list)
            assert row["origin"] in row["origin_set"]

    def test_prior_shifts(self, origin_names):
        classes = ORIGIN_CLASSES
        prior = {
            c: (0.5 if c == "Japan" else 0.5 / (len(classes) - 1)) for c in classes
        }
        base = estimate_wikipedia_origin(origin_names, "last", "first")
        adjusted = estimate_wikipedia_origin(
            origin_names, "last", "first", target_prior=prior
        )
        assert (adjusted["Japan"] >= base["Japan"] - 1e-9).all()

    def test_missing_column_raises(self, origin_names):
        with pytest.raises(ValueError, match="does not exist"):
            estimate_wikipedia_origin(origin_names, "nope", "first")

    def test_output_column_collision_preserves_input_values(self, origin_names):
        data = origin_names.head(2).copy()
        data["origin"] = ["observed-a", "observed-b"]

        result = estimate_wikipedia_origin(data, "last", "first")

        assert result.columns.is_unique
        assert result["input_origin"].tolist() == data["origin"].tolist()
