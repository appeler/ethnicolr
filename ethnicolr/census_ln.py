#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Census Last Name Data Module.

This module provides tools to enrich data with demographic information
from the U.S. Census based on last names.
"""
import sys
import logging
from typing import List, Optional, Union
import pandas as pd
from pkg_resources import resource_filename
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CENSUS2000 = resource_filename(__name__, "data/census/census_2000.csv")
CENSUS2010 = resource_filename(__name__, "data/census/census_2010.csv")
CENSUS_COLS = [
    'pctwhite', 'pctblack', 'pctapi', 'pctaian', 
    'pct2prace', 'pcthispanic'
]

class CensusLnData:
    """Class for handling Census Last Name demographic data."""
    
    census_df = None
    census_year = None
    
    @classmethod
    def census_ln(cls, 
                  df: pd.DataFrame,
                  lname_col: str, 
                  year: int = 2000) -> pd.DataFrame:
        """
        Append Census demographic columns to the input DataFrame based on last name.
        
        This method matches last names with Census data and adds demographic 
        percentage columns for various racial and ethnic groups.
        
        Args:
            df (pd.DataFrame): Pandas DataFrame containing a last name column
            lname_col (str): Column name for the last name
            year (int): Census data year to use (2000 or 2010, default: 2000)
            
        Returns:
            pd.DataFrame: Original DataFrame with appended Census demographic columns:
                - pctwhite: Percentage of White population with this last name
                - pctblack: Percentage of Black population with this last name
                - pctapi: Percentage of Asian/Pacific Islander population with this last name
                - pctaian: Percentage of American Indian/Alaska Native population with this last name
                - pct2prace: Percentage of mixed-race population with this last name
                - pcthispanic: Percentage of Hispanic population with this last name
        
        Raises:
            ValueError: If year is not 2000 or 2010
        """
        if year not in [2000, 2010]:
            raise ValueError("Census year must be either 2000 or 2010")
            
        # Validate and normalize input DataFrame
        df = EthnicolrModelClass.test_and_norm_df(df, lname_col)
        
        # Create a temporary normalized last name column
        df['__last_name'] = df[lname_col].str.strip().str.upper()
        
        # Load census data if not already loaded or if year changed
        if cls.census_df is None or cls.census_year != year:
            logger.info(f"Loading Census {year} data...")
            
            census_file = CENSUS2000 if year == 2000 else CENSUS2010
            try:
                cls.census_df = pd.read_csv(
                    census_file, 
                    usecols=['name'] + CENSUS_COLS
                )
                # Clean up census data
                cls.census_df.dropna(subset=['name'], inplace=True)
                cls.census_df.columns = ['__last_name'] + CENSUS_COLS
                cls.census_year = year
                
                logger.info(f"Loaded {len(cls.census_df)} last names from Census {year}")
            except Exception as e:
                logger.error(f"Failed to load Census data: {e}")
                raise
        
        # Merge input data with census data
        logger.info(f"Merging demographic data for {len(df)} records...")
        start_cols = len(df.columns)
        
        rdf = pd.merge(df, cls.census_df, how='left', on='__last_name')
        match_count = rdf.dropna(subset=CENSUS_COLS[0:1]).shape[0]
        
        # Clean up temporary column
        del df['__last_name']
        del rdf['__last_name']
        
        logger.info(f"Matched {match_count} of {len(df)} records ({match_count/len(df)*100:.1f}%)")
        logger.info(f"Added {len(rdf.columns) - start_cols} demographic columns")
        
        return rdf

# Function alias for backward compatibility
census_ln = CensusLnData.census_ln

def main(argv: Optional[List[str]] = None) -> int:
    """
    Command-line interface for census_ln function.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
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
        df = pd.read_csv(args.input)
        logger.info(f"Input file contains {len(df)} records")
        
        rdf = census_ln(df, args.last, args.year)
        
        logger.info(f"Saving output to file: {args.output}")
        rdf.to_csv(args.output, index=False)
        logger.info("Processing complete")
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
