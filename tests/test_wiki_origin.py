"""
Tests for the name -> country-of-origin model.
"""

import numpy as np
import pandas as pd
import pytest

from ethnicolr import pred_wiki_origin


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
        result = pred_wiki_origin(origin_names, "last", "first")
        assert "origin" in result.columns
        assert len(result) == len(origin_names)
        prob_cols = [
            c for c in result.columns if c not in origin_names.columns and c != "origin"
        ]
        assert len(prob_cols) >= 50
        assert np.allclose(result[prob_cols].sum(axis=1), 1.0, atol=1e-4)

    def test_distinctive_names_in_top3(self, origin_names):
        result = pred_wiki_origin(origin_names, "last", "first")
        prob_cols = [
            c for c in result.columns if c not in origin_names.columns and c != "origin"
        ]
        hits = 0
        for i, row in result.iterrows():
            top3 = row[prob_cols].astype(float).nlargest(3).index
            if origin_names.loc[i, "expected"] in top3:
                hits += 1
        assert hits >= 4, f"only {hits}/6 distinctive names had origin in top 3"

    def test_coverage_sets(self, origin_names):
        result = pred_wiki_origin(origin_names, "last", "first", coverage=0.9)
        assert "origin_set" in result.columns
        for _, row in result.iterrows():
            assert isinstance(row["origin_set"], list)
            assert row["origin"] in row["origin_set"]

    def test_prior_shifts(self, origin_names):
        from importlib.resources import files
        from pathlib import Path

        from ethnicolr.torch_utils import load_model_stats

        stats = load_model_stats(
            Path(str(files("ethnicolr")))
            / "models"
            / "wiki"
            / "lstm"
            / "wiki_origin_lstm_pt.pt"
        )
        classes = stats["classes"]
        prior = {
            c: (0.5 if c == "Japan" else 0.5 / (len(classes) - 1)) for c in classes
        }
        base = pred_wiki_origin(origin_names, "last", "first")
        adjusted = pred_wiki_origin(origin_names, "last", "first", prior=prior)
        assert (adjusted["Japan"] >= base["Japan"] - 1e-9).all()

    def test_missing_column_raises(self, origin_names):
        with pytest.raises(ValueError, match="must exist"):
            pred_wiki_origin(origin_names, "nope", "first")
