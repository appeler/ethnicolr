"""Schemas and validated readers for runtime Parquet tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

RACE_PERCENTAGE_COLUMNS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]


def _field(name: str, data_type: pa.DataType, *, nullable: bool = False) -> pa.Field:
    return pa.field(name, data_type, nullable=nullable)


def _schema(table_name: str, fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(
        fields,
        metadata={
            b"ethnicolr.schema_version": b"1",
            b"ethnicolr.table": table_name.encode("utf-8"),
        },
    )


CENSUS_SURNAME_SCHEMA = _schema(
    "census-surname",
    [
        _field("name", pa.string()),
        _field("rank", pa.int64()),
        _field("count", pa.int64()),
        _field("prop100k", pa.float64()),
        _field("cum_prop100k", pa.float64()),
        *[_field(column, pa.float64()) for column in RACE_PERCENTAGE_COLUMNS],
    ],
)

CENSUS_FIRST_NAME_SCHEMA = _schema(
    "census-first-name",
    [
        _field("name", pa.string()),
        _field("rank", pa.int64(), nullable=True),
        _field("count", pa.int64()),
        _field("prop100k", pa.float64()),
        _field("cum_prop100k", pa.float64()),
        *[_field(column, pa.float64()) for column in RACE_PERCENTAGE_COLUMNS],
        _field("pctmale", pa.float64()),
        _field("pctfemale", pa.float64()),
    ],
)

NAME_RACE_PROBABILITY_SCHEMA = _schema(
    "name-race-probability",
    [
        _field("name", pa.string()),
        _field("white", pa.float64()),
        _field("black", pa.float64()),
        _field("hispanic", pa.float64()),
        _field("asian", pa.float64()),
        _field("other", pa.float64()),
    ],
)


def read_runtime_table(
    path: str | Path, expected_schema: pa.Schema, *, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read a runtime table and reject schema drift before inference."""
    table = pq.read_table(path)
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise ValueError(
            f"runtime table {path} has schema {table.schema}; "
            f"expected {expected_schema}"
        )
    if columns is not None:
        table = table.select(columns)
    return table.to_pandas()
