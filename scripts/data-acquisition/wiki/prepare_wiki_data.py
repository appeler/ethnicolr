#!/usr/bin/env python3
"""Build the merged wiki training dataset from fetched Wikidata people.

Labeling policy:
- An ethnic-group statement (P172) mapped in ethnic_group_to_category.csv wins.
- Otherwise the person's citizenships (from the per-country files) must all map
  to one category; people whose citizenships map to conflicting categories are
  dropped. Melting-pot countries are never fetched (DROP in the mapping), so
  their people only enter via P172.
- Name = family-name/given-name labels where present, else the English label
  split on its last space. Names must be mostly-Latin after NFKD normalization.
- The result is merged with the 2009-era ethnicolr/data/wiki/wiki_name_race.csv
  (kept in full) and deduplicated on (name_last, name_first, race).

Usage:
    python prepare_wiki_data.py
    python prepare_wiki_data.py --out ../raw/wiki_name_race_2026.csv.gz
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OLD_DATA = REPO_ROOT / "ethnicolr" / "data" / "wiki" / "wiki_name_race.csv"

PAREN_RE = re.compile(r"\s*\(.*?\)\s*")
SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def load_mapping(path: Path, key: str) -> dict[str, str]:
    with open(path) as fh:
        return {
            row["qid"]: row["category"]
            for row in csv.DictReader(fh)
            if row["category"] != "DROP"
        }


def is_latin(text: str) -> bool:
    """True if every alphabetic character is Latin-script."""
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if not name.startswith("LATIN"):
                return False
    return True


def split_label(label: str) -> tuple[str, str] | None:
    label = PAREN_RE.sub(" ", label).strip().strip(",")
    tokens = [t for t in label.split() if t.lower().strip(".,") not in SUFFIXES]
    if len(tokens) < 2:
        return None
    return tokens[-1], " ".join(tokens[:-1])


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def read_people(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["person"] = df["person"].map(qid)
    return df


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
        default=Path(__file__).parent.parent / "raw" / "wiki_name_race_2026.csv.gz",
    )
    parser.add_argument(
        "--mappings", type=Path, default=Path(__file__).parent / "mappings"
    )
    args = parser.parse_args()

    country_map = load_mapping(args.mappings / "country_to_category.csv", "country")
    ethnic_map = load_mapping(
        args.mappings / "ethnic_group_to_category.csv", "ethnic_group"
    )

    stats = Counter()

    # person QID -> set of citizenship categories
    person_country_cats: dict[str, set[str]] = {}
    # person QID -> (label, family_name, given_name)
    person_rows: dict[str, tuple[str, str, str]] = {}
    # person QID -> set of ethnic-group QIDs
    person_eth: dict[str, set[str]] = {}

    country_files = sorted(args.raw_dir.glob("country_*.csv"))
    if not country_files:
        sys.exit(f"No fetched data in {args.raw_dir}; run fetch_wikidata_people.py")

    for path in country_files:
        cat = country_map.get(path.stem.removeprefix("country_"))
        if cat is None:
            continue
        df = read_people(path)
        for row in df.itertuples(index=False):
            person_country_cats.setdefault(row.person, set()).add(cat)
            person_rows.setdefault(
                row.person, (row.label, row.family_name, row.given_name)
            )

    p172_path = args.raw_dir / "p172_people.csv"
    if p172_path.exists():
        for row in read_people(p172_path).itertuples(index=False):
            person_rows.setdefault(
                row.person, (row.label, row.family_name, row.given_name)
            )

    edges_path = args.raw_dir / "p172_edges.csv"
    if edges_path.exists():
        with open(edges_path) as fh:
            for edge in csv.DictReader(fh):
                person_eth.setdefault(qid(edge["person"]), set()).add(
                    qid(edge["ethnic_group"])
                )

    records = []
    for person, (label, fam, giv) in person_rows.items():
        # 1) P172 override
        eth_cats = {
            ethnic_map[e] for e in person_eth.get(person, ()) if e in ethnic_map
        }
        if len(eth_cats) == 1:
            category, source = next(iter(eth_cats)), "wikidata_p172"
        elif len(eth_cats) > 1:
            stats["dropped_conflicting_ethnicity"] += 1
            continue
        else:
            # 2) citizenship consensus
            cats = person_country_cats.get(person, set())
            if not cats:
                stats["dropped_no_signal"] += 1
                continue
            if len(cats) > 1:
                stats["dropped_conflicting_citizenship"] += 1
                continue
            category, source = next(iter(cats)), "wikidata_p27"

        # Build name
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

        records.append((name_last, name_first, category, source))
        stats[f"kept_{source}"] += 1

    new_df = pd.DataFrame(
        records, columns=["name_last", "name_first", "race", "source"]
    )
    new_df = new_df.drop_duplicates(subset=["name_last", "name_first", "race"])

    old_df = pd.read_csv(OLD_DATA, dtype=str, keep_default_na=False)
    old_df = old_df[["name_last", "name_first", "race"]]
    old_df["source"] = "wiki2009"

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["name_last", "name_first", "race"])

    print("Processing stats:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count:,}")
    print("\nPer-category counts (merged):")
    print(merged.groupby(["race", "source"]).size().unstack(fill_value=0).to_string())
    print(f"\nTotal rows: {len(merged):,} ({len(old_df):,} old, {len(new_df):,} new)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, compression="gzip")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
