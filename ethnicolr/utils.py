# -*- coding: utf-8 -*-

import argparse

def arg_parser(argv, title: str, default_out: str, default_year: int, year_choices: list, first: bool = False):

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output',
                        default=default_out,
                        help='Output file with prediction data')
    if first:
        parser.add_argument('-f', '--first', required=True,
                        help='Column name for the column with the first name')
    parser.add_argument('-l', '--last', required=True,
                        help='Column name for the column with the last name')
    parser.add_argument('-i', '--iter', default=100, type=int,
                        help='Number of iterations to measure uncertainty')
    parser.add_argument('-c', '--conf', default=1.0, type=float,
                        help='Confidence interval of Predictions')
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=default_year,
        choices=year_choices,
        help=f"Year of data (default={default_year})",
    )
    args = parser.parse_args(argv)

    print(args)

    return(args)
