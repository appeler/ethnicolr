#!/usr/bin/env python3
"""Build NC voter registration training data from the raw Dataverse archive.

Reads ncvoter_Statewide.zip (tab-delimited, latin-1) and writes last_name,
first_name, race_code, ethnic_code to a gzipped CSV for the model trainer.

Usage:
    python prepare_nc_data.py --archive ../raw/ncvoter_Statewide.zip \
        --out ../raw/nc_voter_name_race.csv.gz
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(
        args.archive,
        delimiter="\t",
        usecols=["last_name", "first_name", "race_code", "ethnic_code"],
        encoding="latin-1",
        low_memory=False,
    )
    df = df.dropna(subset=["last_name", "first_name"])

    print(f"{len(df):,} rows")
    print(df.groupby(["ethnic_code", "race_code"]).size())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, compression="gzip", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
