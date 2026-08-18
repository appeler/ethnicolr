"""Tests for schema-validated runtime tables."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ethnicolr.census_surname import CENSUS_SURNAME_FILES
from ethnicolr.name_dictionaries import (
    CENSUS_FIRST_NAME_2020_FILE,
    VOTER_FILE_FIRST_NAME_TABLE,
    VOTER_FILE_LAST_NAME_TABLE,
)
from ethnicolr.runtime_tables import (
    CENSUS_FIRST_NAME_SCHEMA,
    CENSUS_SURNAME_SCHEMA,
    NAME_RACE_PROBABILITY_SCHEMA,
    RACE_PERCENTAGE_COLUMNS,
    read_runtime_table,
)


@pytest.mark.parametrize("path", CENSUS_SURNAME_FILES.values())
def test_census_surname_tables_have_declared_schema(path: str) -> None:
    frame = read_runtime_table(path, CENSUS_SURNAME_SCHEMA)

    assert not frame["name"].isna().any()
    assert not frame["name"].duplicated().any()
    assert frame["rank"].dtype == "int64"
    assert frame["count"].dtype == "int64"


def test_census_suppression_markers_are_distributed() -> None:
    frame = read_runtime_table(
        CENSUS_SURNAME_FILES[2000], CENSUS_SURNAME_SCHEMA
    ).set_index("name")
    yu_percentages = frame.loc["YU", RACE_PERCENTAGE_COLUMNS]

    assert yu_percentages.sum() == pytest.approx(100)
    assert yu_percentages["pctblack"] == pytest.approx(0.045)
    assert yu_percentages["pctaian"] == pytest.approx(0.045)


def test_first_name_table_has_declared_nullable_rank() -> None:
    frame = read_runtime_table(CENSUS_FIRST_NAME_2020_FILE, CENSUS_FIRST_NAME_SCHEMA)

    assert frame.loc[frame["name"] == "ALL OTHER NAMES", "rank"].isna().all()
    assert frame["count"].dtype == "int64"


@pytest.mark.parametrize(
    "path", [VOTER_FILE_FIRST_NAME_TABLE, VOTER_FILE_LAST_NAME_TABLE]
)
def test_voter_file_name_tables_have_declared_schema(path: str) -> None:
    frame = read_runtime_table(path, NAME_RACE_PROBABILITY_SCHEMA)

    assert not frame["name"].isna().any()
    assert not frame["name"].duplicated().any()


def test_runtime_reader_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "wrong.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"name": ["SMITH"]})), path)

    with pytest.raises(ValueError, match="schema"):
        read_runtime_table(path, CENSUS_SURNAME_SCHEMA)
