#!/usr/bin/env python3
"""Normalize legacy runtime CSV tables to schema-validated Parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ethnicolr.runtime_tables import (  # noqa: E402
    CENSUS_FIRST_NAME_SCHEMA,
    CENSUS_SURNAME_SCHEMA,
    NAME_RACE_PROBABILITY_SCHEMA,
    RACE_PERCENTAGE_COLUMNS,
    read_runtime_table,
)


def _normalize_frame(source_path: Path, schema: pa.Schema) -> pd.DataFrame:
    frame = pd.read_csv(source_path).dropna(subset=["name"]).reset_index(drop=True)
    frame = frame[schema.names].copy()
    if schema.metadata[b"ethnicolr.table"] == b"census-surname":
        suppressed = frame[RACE_PERCENTAGE_COLUMNS].eq("(S)")
        numeric_percentages = frame[RACE_PERCENTAGE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        suppressed_count = suppressed.sum(axis=1)
        distributed_percentage = (100 - numeric_percentages.sum(axis=1)).clip(
            lower=0
        ) / suppressed_count.replace(0, pd.NA)
        frame[RACE_PERCENTAGE_COLUMNS] = numeric_percentages.mask(
            suppressed, distributed_percentage, axis=0
        )
    frame["name"] = frame["name"].astype("string")
    for field in schema:
        if pa.types.is_int64(field.type):
            pandas_type = "Int64" if field.nullable else "int64"
            frame[field.name] = frame[field.name].astype(pandas_type)
        elif pa.types.is_float64(field.type):
            frame[field.name] = frame[field.name].astype("float64")
    if frame["name"].duplicated().any():
        raise ValueError(f"{source_path} contains duplicate names")
    return frame


def _convert(source_path: Path, target_path: Path, schema: pa.Schema) -> None:
    source_frame = _normalize_frame(source_path, schema)
    table = pa.Table.from_pandas(
        source_frame, schema=schema, preserve_index=False
    ).replace_schema_metadata(schema.metadata)
    pq.write_table(table, target_path, compression="zstd")
    converted_frame = read_runtime_table(target_path, schema)
    assert_frame_equal(source_frame, converted_frame, check_dtype=False)
    print(f"wrote {target_path.relative_to(REPOSITORY_ROOT)}")


def main() -> None:
    source_root = REPOSITORY_ROOT / "scripts" / "data-acquisition" / "source-tables"
    package_data = REPOSITORY_ROOT / "ethnicolr" / "data"

    for year in (2000, 2010, 2020):
        _convert(
            source_root / "census" / f"census_{year}.csv",
            package_data / "census" / f"census_{year}.parquet",
            CENSUS_SURNAME_SCHEMA,
        )

    _convert(
        source_root / "census" / "census_2020_first_names.csv",
        package_data / "census" / "census_2020_first_names.parquet",
        CENSUS_FIRST_NAME_SCHEMA,
    )

    for name_scope in ("first", "last"):
        _convert(
            source_root / "rosenman" / f"{name_scope}_name_race.csv.gz",
            package_data / "rosenman" / f"{name_scope}_name_race.parquet",
            NAME_RACE_PROBABILITY_SCHEMA,
        )


if __name__ == "__main__":
    main()
