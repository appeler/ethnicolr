#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Census Last Name Data Module.

Enriches input data with demographic percentages from U.S. Census (2000 or 2010).
"""

import sys
import os
import logging
from typing import List, Optional
import pandas as pd
import importlib.resources as resources
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CENSUS2000 = str(resources.files("ethnicolr") / "data/census/census_2000.csv")
CENSUS2010 = str(resources.files("ethnicolr") / "data/census/census_2010.csv")
CENSUS_COLS = [
    'pctwhite', 'pctblack', 'pctapi', 'pctaian',
    'pct2prace', 'pcthispanic'
]

class CensusLnData:
    """Handles Census Last Name demographic enrichment."""

    census_df = None
    census_year = None

    @classmethod
    def census_ln(cls,
                  df: pd.DataFrame,
                  lname_col: str,
                  year: int = 2000) -> pd.DataFrame:
        if year not in [2000, 2010]:
            raise ValueError("Census year must be either 2000 or 2010")

        df = EthnicolrModelClass.test_and_norm_df(df, lname_col)

        # Ensure temporary column doesn't conflict
        temp_col = "__ethnicolr_temp_lname"
        while temp_col in df.columns:
            temp_col += "_"

        df[temp_col] = df[lname_col].fillna("").astype(str).str.strip().str.upper()

        if cls.census_df is None or cls.census_year != year:
            census_file = CENSUS2000 if year == 2000 else CENSUS2010
            logger.info(f"Loading Census {year} data from {census_file}...")

            try:
                census_df = pd.read_csv(
                    census_file,
                    usecols=['name'] + CENSUS_COLS
                ).dropna(subset=['name'])

                census_df.columns = [temp_col] + CENSUS_COLS
                cls.census_df = census_df
                cls.census_year = year

                logger.info(f"Loaded {len(cls.census_df)} last names from Census {year}")
            except Exception as e:
                logger.error(f"Failed to load Census data: {e}")
                raise

        logger.info(f"Merging demographic data for {len(df)} records...")
        start_cols = set(df.columns)

        rdf = pd.merge(df, cls.census_df, how='left', on=temp_col)

        if temp_col in df.columns:
            df.drop(columns=[temp_col], inplace=True)
        if temp_col in rdf.columns:
            rdf.drop(columns=[temp_col], inplace=True)

        matched = rdf.dropna(subset=[CENSUS_COLS[0]]).shape[0]
        logger.info(f"Matched {matched} of {len(rdf)} rows ({matched / len(rdf) * 100:.1f}%)")

        new_cols = set(rdf.columns) - start_cols
        logger.info(f"Added columns: {', '.join(sorted(new_cols))}")

        return rdf


# Backwards compatibility alias
census_ln = CensusLnData.census_ln


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Append Census demographic data by last name",
            default_out="census-output.csv",
            default_year=2010,
            year_choices=[2000, 2010]
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = census_ln(df, args.last, args.year)

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"📦 Output written: {args.output} ({len(rdf)} rows)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
