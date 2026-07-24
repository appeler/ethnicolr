#!/usr/bin/env python3
"""Build FL voter registration training data from the raw Dataverse archive.

Extracts the per-county voter detail files (tab-delimited, no header) and
writes name_last, name_first, race to a gzipped CSV for the model trainer.

The archives are .7z; extraction requires 7z (brew install p7zip).

Column positions (2 = last name, 4 = first name, 20 = race code) hold for the
2017 extract. For 2022, verify against the layout PDF (Dataverse file 6378554)
before trusting the output.

Usage:
    python prepare_fl_data.py --archive ../raw/20170207_VoterDetail.7z \
        --out ../raw/fl_reg_name_race.csv.gz
    python prepare_fl_data.py --archive ../raw/20220621_VoterDetail_2.7z \
        --out ../raw/fl_reg_name_race_2022.csv.gz
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

RACE_MAP = {
    1: "native_indian",
    2: "asian",
    3: "nh_black",
    4: "hispanic",
    5: "nh_white",
    6: "other",
    7: "multi_racial",
    9: "unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--last-col", type=int, default=2, help="0-based column index of last name"
    )
    parser.add_argument(
        "--first-col", type=int, default=4, help="0-based column index of first name"
    )
    parser.add_argument(
        "--race-col", type=int, default=20, help="0-based column index of race code"
    )
    args = parser.parse_args()

    usecols = [args.last_col, args.first_col, args.race_col]

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["7z", "x", str(args.archive), f"-o{tmp}", "-y"],
            check=True,
            capture_output=True,
        )
        # The 2022 archive nests a zip inside the 7z
        for nested in Path(tmp).rglob("*.zip"):
            subprocess.run(
                ["7z", "x", str(nested), f"-o{tmp}", "-y"],
                check=True,
                capture_output=True,
            )
        txt_files = sorted(Path(tmp).rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt voter files found in {args.archive}")
        print(f"Extracted {len(txt_files)} county files")

        frames = []
        for fn in txt_files:
            df = pd.read_csv(
                fn,
                delimiter="\t",
                header=None,
                usecols=usecols,
                encoding="latin-1",
                on_bad_lines="skip",
                low_memory=False,
            )
            frames.append(df)
        adf = pd.concat(frames, ignore_index=True)

    adf = adf[usecols]
    adf.columns = ["name_last", "name_first", "race"]
    adf["race"] = pd.to_numeric(adf["race"], errors="coerce")
    adf = adf.dropna(subset=["race"])
    adf["race"] = adf["race"].astype(int).map(RACE_MAP)
    adf = adf.dropna(subset=["race"])

    print(f"{len(adf):,} rows")
    print(adf["race"].value_counts())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    adf.to_csv(args.out, compression="gzip", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
