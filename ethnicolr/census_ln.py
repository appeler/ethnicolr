#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd

from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns

CENSUS2000 = resource_filename(__name__, "data/census/census_2000.csv")
CENSUS2010 = resource_filename(__name__, "data/census/census_2010.csv")

CENSUS_COLS = ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace',
               'pcthispanic']


def census_ln(df, namecol, year=2000):
    """Appends additional columns from Census data to the input DataFrame
    based on the last name.

    Removes extra space. Checks if the name is the Census data.  If it is,
    outputs data from that row.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame containing the last name
            column.
        namecol (str or int): Column's name or location of the name in
            DataFrame.
        year (int): The year of Census data to be used. (2000 or 2010)
            (default is 2000)

    Returns:
        DataFrame: Pandas DataFrame with additional columns 'pctwhite',
            'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic'

    """

    if namecol not in df.columns:
        print("No column `{0!s}` in the DataFrame".format(namecol))
        return df

    df['__last_name'] = df[namecol].str.strip()
    df['__last_name'] = df['__last_name'].str.upper()

    if year == 2000:
        census_df = pd.read_csv(CENSUS2000, usecols=['name'] + CENSUS_COLS)
    elif year == 2010:
        census_df = pd.read_csv(CENSUS2010, usecols=['name'] + CENSUS_COLS)

    census_df.columns = ['__last_name'] + CENSUS_COLS

    rdf = pd.merge(df, census_df, how='left', on='__last_name')

    del rdf['__last_name']

    return rdf


def main(argv=sys.argv[1:]):
    title = 'Appends Census columns by last name'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-y', '--year', type=int, default=2000,
                        choices=[2000, 2010],
                        help='Year of Census data (default=2000)')
    parser.add_argument('-o', '--output', default='census-output.csv',
                        help='Output file with Census data columns')
    parser.add_argument('-l', '--last', required=True,
                        help='Name or index location of column contains '
                             'the last name')

    args = parser.parse_args(argv)

    print(args)

    if not args.last.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.last = int(args.last)

    if not column_exists(df, args.last):
        return -1

    rdf = census_ln(df, args.last, args.year)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
