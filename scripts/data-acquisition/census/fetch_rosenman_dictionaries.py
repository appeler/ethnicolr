#!/usr/bin/env python3
"""Fetch the Rosenman-Olivella-Imai name-race dictionaries.

Downloads the CC0 dictionaries of P(race | name) for first names and surnames
built from six Southern-state voter files (Rosenman, Olivella & Imai 2023,
Scientific Data; doi:10.7910/DVN/YL2OXB) and writes them as package data:

    ethnicolr/data/rosenman/first_name_race.csv.gz
    ethnicolr/data/rosenman/last_name_race.csv.gz
    ethnicolr/data/rosenman/rosenman_stats.json

The stats file records the implied voter-population race marginal, recovered
via Bayes' rule from the paired P(race|name) and P(name|race) matrices:
P(r)/P(r') = [P(r|n)/P(n|r)] / [P(r'|n)/P(n|r')] for any name n; we take the
weighted median over the most common surnames for stability. This marginal is
the pi_train used by the `prior=` reweighting API.

Anonymous download; no Dataverse token needed (files are unrestricted, CC0).

Usage:
    python fetch_rosenman_dictionaries.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

BASE = "https://dataverse.harvard.edu/api/access/datafile"
FILES = {
    "first_nameRaceProbs": 7084378,
    "last_nameRaceProbs": 7084373,
    "last_raceNameProbs": 7084380,
}
RACES = {
    "whi": "white",
    "bla": "black",
    "his": "hispanic",
    "asi": "asian",
    "oth": "other",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "ethnicolr" / "data" / "rosenman"


def fetch_rdata(name: str, file_id: int, cache: Path) -> pd.DataFrame:
    path = cache / f"{name}.rData"
    if not path.exists():
        subprocess.run(
            [
                "curl",
                "-sL",
                "--fail",
                "--retry",
                "3",
                "-o",
                str(path),
                f"{BASE}/{file_id}",
            ],
            check=True,
        )
    result = pyreadr.read_r(str(path))
    return next(iter(result.values()))


def implied_marginal(
    race_given_name: pd.DataFrame, name_given_race: pd.DataFrame
) -> dict[str, float]:
    """Recover P(race) from P(race|name) and P(name|race) via Bayes' rule."""
    ngr = name_given_race.copy()
    ngr.columns = ["name"] + [c.split(".")[0] for c in ngr.columns[1:]]
    merged = race_given_name.merge(ngr, on="name", suffixes=("_rgn", "_ngr"))

    race_keys = list(RACES)
    rgn = merged[[f"{r}_rgn" for r in race_keys]].to_numpy()
    ngr_arr = merged[[f"{r}_ngr" for r in race_keys]].to_numpy()

    valid = (rgn > 0).all(axis=1) & (ngr_arr > 0).all(axis=1)
    ratios = rgn[valid] / ngr_arr[valid]
    # Weight toward common names: P(name) proportional to ngr under any race
    weights = ngr_arr[valid].sum(axis=1)
    top = np.argsort(-weights)[:1000]
    marginal = np.median(ratios[top] / ratios[top].sum(axis=1, keepdims=True), axis=0)
    marginal = marginal / marginal.sum()
    return {RACES[r]: float(p) for r, p in zip(race_keys, marginal, strict=True)}


def main() -> None:
    cache = Path(__file__).parent.parent / "raw" / "rosenman"
    cache.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    for key in ("first_nameRaceProbs", "last_nameRaceProbs"):
        df = fetch_rdata(key, FILES[key], cache)
        df = df.rename(columns=RACES)
        df["name"] = df["name"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates(subset=["name"])
        out_name = (
            "first_name_race.csv.gz"
            if key.startswith("first")
            else "last_name_race.csv.gz"
        )
        df.to_csv(OUT_DIR / out_name, index=False, compression="gzip")
        frames[key] = df
        print(f"Wrote {OUT_DIR / out_name}: {len(df):,} names")

    last_rgn_raw = fetch_rdata("last_nameRaceProbs", FILES["last_nameRaceProbs"], cache)
    last_ngr = fetch_rdata("last_raceNameProbs", FILES["last_raceNameProbs"], cache)
    marginal = implied_marginal(last_rgn_raw, last_ngr)
    stats = {
        "source": "Rosenman, Olivella & Imai (2023), doi:10.7910/DVN/YL2OXB (CC0)",
        "population": "registered voters in AL, FL, GA, LA, NC, SC",
        "voter_population_marginal": marginal,
        "n_first_names": len(frames["first_nameRaceProbs"]),
        "n_last_names": len(frames["last_nameRaceProbs"]),
    }
    (OUT_DIR / "rosenman_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("Implied voter-population marginal:", marginal)


if __name__ == "__main__":
    main()
