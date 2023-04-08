#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd

from pkg_resources import resource_filename

from .utils import test_and_norm_df

CENSUS2000 = resource_filename(__name__, "data/census/census_2000.csv")
CENSUS2010 = resource_filename(__name__, "data/census/census_2010.csv")

CENSUS_COLS = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace',
               'pcthispanic']


class CensusLnData():
    census_df = None

    @classmethod
    def census_ln(cls, 
                  df: pd.DataFrame,
                  lname_col: str, 
                  year: int=2000) -> pd.DataFrame:
        """Appends columns from Census data to the input DataFrame
        based on the last name.

        Removes extra space. Checks if the name is the Census data.  If it is,
        outputs data from that row.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            year (int): The year of Census data to be used. (2000 or 2010)
                (default is 2000)

        Returns:
            DataFrame: Pandas DataFrame with additional columns 'pctwhite',
                'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic'

        """

        df = test_and_norm_df(df, lname_col)
        
        df['__last_name'] = df[lname_col].str.strip().str.upper()

        if cls.census_df is None or cls.census_year != year:
            if year == 2000:
                cls.census_df = pd.read_csv(CENSUS2000, usecols=['name']
                                            + CENSUS_COLS)
            elif year == 2010:
                cls.census_df = pd.read_csv(CENSUS2010, usecols=['name']
                                            + CENSUS_COLS)

            cls.census_df.drop(cls.census_df[cls.census_df.name.isnull()]
                               .index, inplace=True)

            cls.census_df.columns = ['__last_name'] + CENSUS_COLS
            cls.census_year = year

        rdf = pd.merge(df, cls.census_df, how='left', on='__last_name')

        del df['__last_name']
        del rdf['__last_name']

        return rdf


census_ln = CensusLnData.census_ln


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Appends Census columns by last name", 
                default_out = "census-output.csv", 
                default_year = 2010, 
                year_choices = [2000, 2010])

    df = pd.read_csv(args.input)

    rdf = census_ln(df, args.last, args.year)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
