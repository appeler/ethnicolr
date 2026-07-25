#!/usr/bin/env python
"""
Census Last Name Data Module.

Enriches input data with demographic percentages from U.S. Census (2000, 2010, or 2020).
"""

import importlib.resources as resources
import logging
import os
import sys

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CENSUS2000 = str(resources.files("ethnicolr") / "data/census/census_2000.csv")
CENSUS2010 = str(resources.files("ethnicolr") / "data/census/census_2010.csv")
CENSUS2020 = str(resources.files("ethnicolr") / "data/census/census_2020.csv")
CENSUS_COLS = ["pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]


class CensusLnData:
    """Census last name demographic data enrichment.

    Provides functionality to enrich DataFrames with U.S. Census demographic
    percentages based on last names. Supports 2000, 2010, and 2020 Census data.
    """

    census_df = None
    census_year = None

    @classmethod
    def census_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        year: int = 2000,
        conf_int: float | None = None,
    ) -> pd.DataFrame:
        """Append Census demographic percentages by last name.

        Merges input DataFrame with U.S. Census surname data to add
        demographic percentage columns for race/ethnicity categories.

        Args:
            df: Input DataFrame containing last names.
            lname_col: Column name containing last names.
            year: Census year (2000, 2010, or 2020).
            conf_int: Optional confidence level (e.g. 0.95) to add exact
                Wilson score bounds (``<col>_lb``/``<col>_ub``) computed from
                the published name counts.

        Returns:
            DataFrame with original data plus Census demographic columns:
            - 'pctwhite': Percentage white
            - 'pctblack': Percentage black
            - 'pctapi': Percentage Asian/Pacific Islander
            - 'pctaian': Percentage American Indian/Alaska Native
            - 'pct2prace': Percentage two or more races
            - 'pcthispanic': Percentage Hispanic

        Raises:
            ValueError: If year not in [2000, 2010, 2020] or column missing.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'surname': ['Smith', 'Garcia']})
            >>> result = CensusLnData.census_ln(df, 'surname', 2010)
            >>> print(result[['surname', 'pctwhite', 'pcthispanic']].head())
              surname  pctwhite  pcthispanic
            0   Smith      70.9          2.3
            1  Garcia       3.7         92.8
        """
        if year not in [2000, 2010, 2020]:
            raise ValueError("Census year must be 2000, 2010, or 2020")

        df = EthnicolrModelClass.test_and_norm_df(df, lname_col)

        # Ensure temporary column doesn't conflict
        temp_col = "__ethnicolr_temp_lname"
        while temp_col in df.columns:
            temp_col += "_"

        df[temp_col] = df[lname_col].fillna("").astype(str).str.strip().str.upper()

        if cls.census_df is None or cls.census_year != year:
            census_files = {2000: CENSUS2000, 2010: CENSUS2010, 2020: CENSUS2020}
            census_file = census_files[year]
            logger.info(f"Loading Census {year} data from {census_file}...")

            try:
                columns_to_use: list[str] = ["name", "count"] + CENSUS_COLS
                census_df = pd.read_csv(
                    census_file,
                    usecols=columns_to_use,  # type: ignore
                ).dropna(subset=["name"])
                census_df = census_df[columns_to_use]
                for col in CENSUS_COLS + ["count"]:
                    census_df[col] = pd.to_numeric(census_df[col], errors="coerce")
                census_df.columns = [temp_col, "__census_count"] + CENSUS_COLS
                cls.census_df = census_df
                cls.census_year = year

                logger.info(
                    f"Loaded {len(cls.census_df)} last names from Census {year}"
                )
            except Exception as e:
                logger.error(f"Failed to load Census data: {e}")
                raise

        logger.info(f"Merging demographic data for {len(df)} records...")
        start_cols = set(df.columns)

        rdf = pd.merge(df, cls.census_df, how="left", on=temp_col)

        if conf_int is not None:
            if not 0 < conf_int < 1:
                raise ValueError("conf_int must be between 0 and 1")
            from .dict_models import wilson_interval

            counts = rdf["__census_count"].to_numpy(dtype=float)
            for col in CENSUS_COLS:
                p = rdf[col].to_numpy(dtype=float) / 100
                lb, ub = wilson_interval(p, counts, conf_int)
                rdf[f"{col}_lb"] = (lb * 100).round(2)
                rdf[f"{col}_ub"] = (ub * 100).round(2)
        rdf.drop(columns=["__census_count"], inplace=True, errors="ignore")

        if temp_col in df.columns:
            df.drop(columns=[temp_col], inplace=True)
        if temp_col in rdf.columns:
            rdf.drop(columns=[temp_col], inplace=True)

        matched = rdf.dropna(subset=[CENSUS_COLS[0]]).shape[0]
        logger.info(
            f"Matched {matched} of {len(rdf)} rows ({matched / len(rdf) * 100:.1f}%)"
        )

        new_cols = set(rdf.columns) - start_cols
        logger.info(f"Added columns: {', '.join(sorted(new_cols))}")

        return rdf


# Backwards compatibility alias
census_ln = CensusLnData.census_ln


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for Census demographic data enrichment.

    Provides CLI access to Census-based demographic percentage lookup
    by last name. Processes CSV files and outputs enriched data.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code: 0 success, 1 general error, 2 missing files, 3 invalid data.
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Append Census demographic data by last name",
            default_out="census-output.csv",
            default_year=2020,
            year_choices=[2000, 2010, 2020],
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = census_ln(df, args.last, args.year)

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"Output written: {args.output} ({len(rdf)} rows)")

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
