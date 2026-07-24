#!/usr/bin/env python3
"""Download a restricted file from Harvard Dataverse.

The FL and NC voter registration datasets used to train ethnicolr models are
restricted; downloading requires a Dataverse account with access and an API
token (https://dataverse.harvard.edu/ -> account -> API Token), passed via the
DATAVERSE_API_TOKEN environment variable.

Known file ids:
    FL 2017 voter detail:  3015861  (20170207_VoterDetail.7z, 414 MB)
    FL 2022 voter detail:  6378553  (20220621_VoterDetail_2.7z, 759 MB)
    FL 2022 layout PDF:    6378554
    NC statewide voters:   3724157  (ncvoter_Statewide.zip, 488 MB)

Usage:
    export DATAVERSE_API_TOKEN=...
    python download_dataverse.py --file-id 3015861 --out raw/20170207_VoterDetail.7z
    python download_dataverse.py --file-id 3724157 --out raw/ncvoter_Statewide.zip
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = "https://dataverse.harvard.edu/api/access/datafile"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-id", type=int)
    group.add_argument("--persistent-id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    token = os.environ.get("DATAVERSE_API_TOKEN")
    if not token:
        print(
            "DATAVERSE_API_TOKEN is not set. Create a token at "
            "https://dataverse.harvard.edu/dataverseuser.xhtml?selectTab=apiTokenTab\n"
            "Alternatively download the file manually in a browser and place it "
            f"at {args.out}",
            file=sys.stderr,
        )
        return 1

    if args.file_id:
        url = f"{BASE}/{args.file_id}"
    else:
        url = f"{BASE}/:persistentId?persistentId={args.persistent_id}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {args.out}")
    # curl follows the 303 redirect to pre-signed S3 storage and (unlike
    # urllib) drops the custom auth header on the cross-host redirect,
    # which S3 requires. The header goes through stdin config so the token
    # never appears in process arguments.
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--config",
            "-",
            "-o",
            str(args.out),
            url,
        ],
        input=f'header = "X-Dataverse-key: {token}"\n',
        text=True,
    )
    if result.returncode != 0:
        print(
            "Download failed. Your token may lack access to this restricted "
            "file; request access on the dataset page or download manually "
            "in a browser.",
            file=sys.stderr,
        )
        return 1
    print(f"Done: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
