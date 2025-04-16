# -*- coding: utf-8 -*-
import argparse
import os
import sys

def arg_parser(argv, title: str, default_out: str, default_year: int, year_choices: list, first: bool = False):
    parser = argparse.ArgumentParser(
        description=title,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input",
        help="Input CSV file path (with name columns)"
    )

    parser.add_argument(
        "-o", "--output",
        default=default_out,
        help="Output file path to save predictions"
    )

    if first:
        parser.add_argument(
            "-f", "--first",
            required=True,
            help="Column name for first name"
        )

    parser.add_argument(
        "-l", "--last",
        required=True,
        help="Column name for last name"
    )

    parser.add_argument(
        "-i", "--iter",
        type=int,
        default=100,
        help="Number of sampling iterations for confidence interval estimation"
    )

    parser.add_argument(
        "-c", "--conf",
        type=float,
        default=1.0,
        help="Confidence level (between 0 and 1)"
    )

    parser.add_argument(
        "-y", "--year",
        type=int,
        choices=year_choices,
        default=default_year,
        help=f"Year of model (must be one of: {year_choices})"
    )

    args = parser.parse_args(argv)

    # 🔒 Additional sanity checks
    if not os.path.isfile(args.input):
        sys.exit(f"❌ Input file not found: {args.input}")
    
    if not (0 < args.conf <= 1):
        sys.exit("❌ --conf must be a float between 0 and 1.")
    
    if args.iter <= 0:
        sys.exit("❌ --iter must be a positive integer.")

    print("✅ Parsed arguments:")
    for k, v in vars(args).items():
        print(f"   {k}: {v}")

    return args
