#!/usr/bin/env python3
"""Fetch people (name + citizenship + ethnic group) from Wikidata via SPARQL.

Downloads one CSV per mapped country (from mappings/country_to_category.csv)
plus one CSV for everyone with an ethnic-group statement (P172), into
<out-dir>/wikidata/. Results are ordered by person QID, so re-runs are
deterministic against the same Wikidata snapshot.

Uses the QLever Wikidata endpoint by default (fast full-corpus queries, no
auth). Pass --endpoint to use another SPARQL service.

Usage:
    python fetch_wikidata_people.py            # fetch everything (~1 GB)
    python fetch_wikidata_people.py --only Q142 Q30
"""

import argparse
import csv
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://qlever.dev/api/wikidata"
PAGE_SIZE = 500_000

PREFIXES = (
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
    "PREFIX wd: <http://www.wikidata.org/entity/> "
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
)

PERSON_FIELDS = """?p ?pLabel (SAMPLE(?famLabel_) AS ?famLabel)
  (SAMPLE(?givLabel_) AS ?givLabel)"""

PERSON_BODY = """
  ?p wdt:P31 wd:Q5 .
  {anchor}
  ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel) = "en")
  OPTIONAL {{ ?p wdt:P734 ?fam . ?fam rdfs:label ?famLabel_ . FILTER(LANG(?famLabel_) = "en") }}
  OPTIONAL {{ ?p wdt:P735 ?giv . ?giv rdfs:label ?givLabel_ . FILTER(LANG(?givLabel_) = "en") }}
"""

EDGES_QUERY = (
    f"{PREFIXES} SELECT ?p ?eth WHERE "
    "{ ?p wdt:P31 wd:Q5 . ?p wdt:P172 ?eth . } ORDER BY ?p ?eth"
)


def build_query(anchor: str, offset: int) -> str:
    body = PERSON_BODY.format(anchor=anchor)
    return (
        f"{PREFIXES} SELECT {PERSON_FIELDS} WHERE {{ {body} }} "
        f"GROUP BY ?p ?pLabel ORDER BY ?p LIMIT {PAGE_SIZE} OFFSET {offset}"
    )


def run_query(endpoint: str, query: str, retries: int = 3) -> list[list[str]]:
    data = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(
        endpoint, data=data, headers={"Accept": "text/csv"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                text = response.read().decode("utf-8", errors="replace")
            rows = list(csv.reader(text.splitlines()))
            if not rows or not rows[0] or rows[0][0].lstrip("?") != "p":
                raise ValueError(f"Unexpected response header: {rows[:1]}")
            return rows[1:]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  retry in {wait}s after: {exc}", file=sys.stderr)
            time.sleep(wait)
    return []


def fetch_anchor(endpoint: str, anchor: str, out_path: Path) -> int:
    total = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["person", "label", "family_name", "given_name"])
        offset = 0
        while True:
            rows = run_query(endpoint, build_query(anchor, offset))
            writer.writerows(rows)
            total += len(rows)
            if len(rows) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "raw" / "wikidata",
    )
    parser.add_argument(
        "--mappings", type=Path, default=Path(__file__).parent / "mappings"
    )
    parser.add_argument("--only", nargs="*", help="Fetch only these country QIDs")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.mappings / "country_to_category.csv") as fh:
        countries = [
            row
            for row in csv.DictReader(fh)
            if row["category"] != "DROP" and (not args.only or row["qid"] in args.only)
        ]

    # Everyone with an ethnic-group statement, regardless of citizenship,
    # plus the person -> ethnic-group edge list used for label overrides
    if not args.only:
        out = args.out_dir / "p172_people.csv"
        if not (args.skip_existing and out.exists()):
            n = fetch_anchor(args.endpoint, "?p wdt:P172 ?anyeth .", out)
            print(f"p172_people: {n:,} rows")
        out = args.out_dir / "p172_edges.csv"
        if not (args.skip_existing and out.exists()):
            rows = run_query(args.endpoint, EDGES_QUERY)
            with open(out, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["person", "ethnic_group"])
                writer.writerows(rows)
            print(f"p172_edges: {len(rows):,} rows")

    for i, row in enumerate(countries, 1):
        out = args.out_dir / f"country_{row['qid']}.csv"
        if args.skip_existing and out.exists():
            continue
        n = fetch_anchor(args.endpoint, f"?p wdt:P27 wd:{row['qid']} .", out)
        print(f"[{i}/{len(countries)}] {row['country']} ({row['qid']}): {n:,} rows")


if __name__ == "__main__":
    main()
