#!/usr/bin/env python3
"""Build the census 2020 first-name table from the Census Bureau release.

Downloads the April 2026 "Frequently Occurring First Names in the 2020 Census"
Excel files (race/Hispanic-origin counts and sex counts), converts the
noise-infused category counts to percentages, and writes
ethnicolr/data/census/census_2020_first_names.csv in the same schema as the
surname files (plus pctmale/pctfemale).

Counts carry the Census Bureau's disclosure-avoidance noise (±3 per cell at
95% probability), so tiny percentages for rare names are approximate.

Usage:
    python fetch_census_2020_names.py
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://www2.census.gov/topics/genealogy/2020surnames"
RACE_URL = f"{BASE}/Names2020_FirstNames_RaceHispanic.xlsx"
SEX_URL = f"{BASE}/Names2020_FirstNames_Sex.xlsx"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = REPO_ROOT / "ethnicolr" / "data" / "census" / "census_2020_first_names.csv"

RACE_COLS = ["pctwhite", "pctblack", "pctaian", "pctapi", "pct2prace", "pcthispanic"]


def fetch_sheet(url: str, columns: list[str]) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    df = pd.read_excel(io.BytesIO(data), skiprows=2, header=None)
    df = df.iloc[:, : len(columns)]
    df.columns = columns
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip().str.upper()
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["count"])
    return df


def main() -> None:
    race = fetch_sheet(
        RACE_URL,
        [
            "name",
            "rank",
            "count",
            "prop100k",
            "cum_prop100k",
            "n_white",
            "n_black",
            "n_aian",
            "n_api",
            "n_2prace",
            "n_hispanic",
        ],
    )
    sex = fetch_sheet(
        SEX_URL,
        ["name", "rank", "count", "prop100k", "cum_prop100k", "n_male", "n_female"],
    )

    for col in ["n_white", "n_black", "n_aian", "n_api", "n_2prace", "n_hispanic"]:
        pct = "pct" + col.removeprefix("n_")
        race[pct] = (
            pd.to_numeric(race[col], errors="coerce").clip(lower=0)
            / pd.to_numeric(race["count"], errors="coerce")
            * 100
        ).round(2)

    sex_counts = sex.set_index("name")[["n_male", "n_female"]]
    sex_total = sex.set_index("name")["count"]
    merged = race.set_index("name")
    merged["pctmale"] = ((sex_counts["n_male"].clip(lower=0) / sex_total) * 100).round(
        2
    )
    merged["pctfemale"] = (
        (sex_counts["n_female"].clip(lower=0) / sex_total) * 100
    ).round(2)
    merged = merged.reset_index()

    out_cols = [
        "name",
        "rank",
        "count",
        "prop100k",
        "cum_prop100k",
        *RACE_COLS,
        "pctmale",
        "pctfemale",
    ]
    merged["prop100k"] = merged["prop100k"].round(2)
    merged["cum_prop100k"] = merged["cum_prop100k"].round(2)
    merged[out_cols].to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(merged):,} first names")
    print(merged[out_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
