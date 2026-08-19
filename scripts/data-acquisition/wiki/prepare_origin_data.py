#!/usr/bin/env python3
"""Build the name -> country-of-origin training dataset from fetched Wikidata.

Uses mappings/country_to_origin.csv: modern countries map to themselves,
historical states to their unambiguous successor, melting-pot citizenships
(US/CA/AU/NZ) and ambiguous unions (Soviet Union, ...) are dropped. People
whose citizenships map to different origin countries are dropped. Name
building and Latin-script filtering are shared with prepare_wiki_data.py.

Output: ../raw/wiki_origin.csv.gz with columns name_last, name_first, race
(the origin-country label — column named "race" so the standard trainer
loader applies unchanged). Classes with fewer than --min-rows rows are
dropped and logged.

Usage:
    python prepare_origin_data.py
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from prepare_wiki_data import PAREN_RE, is_latin, read_people, split_label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "wikidata",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "wiki_origin.csv.gz",
    )
    parser.add_argument(
        "--mappings", type=Path, default=Path(__file__).parent / "mappings"
    )
    parser.add_argument("--min-rows", type=int, default=3000)
    args = parser.parse_args()

    with (args.mappings / "country_to_origin.csv").open() as fh:
        origin_map = {
            row["qid"]: row["origin"]
            for row in csv.DictReader(fh)
            if row["origin"] != "DROP"
        }

    stats = Counter()
    person_origins: dict[str, set[str]] = {}
    person_rows: dict[str, tuple[str, str, str]] = {}

    for path in sorted(args.raw_dir.glob("country_*.csv")):
        origin = origin_map.get(path.stem.removeprefix("country_"))
        if origin is None:
            continue
        for row in read_people(path).itertuples(index=False):
            person_origins.setdefault(row.person, set()).add(origin)
            person_rows.setdefault(
                row.person, (row.label, row.family_name, row.given_name)
            )

    records = []
    for person, origins in person_origins.items():
        if len(origins) > 1:
            stats["dropped_conflicting_origin"] += 1
            continue
        label, fam, giv = person_rows[person]
        if fam and giv:
            name_last, name_first = fam, giv
        else:
            parts = split_label(label)
            if parts is None:
                stats["dropped_unsplittable_name"] += 1
                continue
            name_last, name_first = parts
        name_last = PAREN_RE.sub(" ", name_last).strip()
        name_first = PAREN_RE.sub(" ", name_first).strip()
        if not name_last or not name_first:
            stats["dropped_empty_name"] += 1
            continue
        if not (is_latin(name_last) and is_latin(name_first)):
            stats["dropped_non_latin"] += 1
            continue
        records.append((name_last, name_first, next(iter(origins))))
        stats["kept"] += 1

    df = pd.DataFrame(records, columns=["name_last", "name_first", "race"])
    df = df.drop_duplicates(subset=["name_last", "name_first", "race"])

    sizes = df["race"].value_counts()
    small = sizes[sizes < args.min_rows]
    if len(small):
        print(f"Dropping {len(small)} classes under {args.min_rows} rows:")
        print(small.to_string())
        df = df[~df["race"].isin(small.index)]

    print("Processing stats:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count:,}")
    print(f"\n{df['race'].nunique()} origin classes, {len(df):,} rows")
    print(df["race"].value_counts().head(15).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
