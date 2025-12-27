from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def arg_parser(
    argv: list[str],
    title: str,
    default_out: str,
    default_year: int,
    year_choices: list[int],
    first: bool = False,
) -> argparse.Namespace:
    """Parse command-line arguments for ethnicolr CLI tools.

    Creates a standardized argument parser for all ethnicolr prediction modules.
    Handles input/output files, column specifications, model parameters, and
    performs validation on argument values.

    Args:
        argv: Command-line arguments to parse, typically sys.argv[1:].
        title: Description text for the argument parser.
        default_out: Default output filename for predictions.
        default_year: Default model year to use if not specified.
        year_choices: Valid years for model selection.
        first: Whether to include first name column argument.

    Returns:
        Parsed arguments namespace containing validated input parameters.

    Raises:
        SystemExit: If input file doesn't exist, confidence interval is invalid,
            or iteration count is non-positive.

    Example:
        >>> args = arg_parser(
        ...     ['input.csv', '-l', 'lastname', '-o', 'output.csv'],
        ...     'Test prediction tool',
        ...     'default-output.csv',
        ...     2010,
        ...     [2000, 2010]
        ... )
        >>> print(args.input)
        'input.csv'
    """
    parser = argparse.ArgumentParser(
        description=title, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", help="Input CSV file path (with name columns)")

    parser.add_argument(
        "-o",
        "--output",
        default=default_out,
        help="Output file path to save predictions",
    )

    if first:
        parser.add_argument(
            "-f", "--first", required=True, help="Column name for first name"
        )

    parser.add_argument("-l", "--last", required=True, help="Column name for last name")

    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        help="Number of sampling iterations for confidence interval estimation",
    )

    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=1.0,
        help="Confidence level (between 0 and 1)",
    )

    parser.add_argument(
        "-y",
        "--year",
        type=int,
        choices=year_choices,
        default=default_year,
        help=f"Year of model (must be one of: {year_choices})",
    )

    args = parser.parse_args(argv)

    # Additional input validation
    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")

    if not (0 < args.conf <= 1):
        sys.exit("ERROR: --conf must be a float between 0 and 1.")

    if args.iter <= 0:
        sys.exit("ERROR: --iter must be a positive integer.")

    logger.info("Parsed arguments:")
    for k, v in vars(args).items():
        logger.info(f"   {k}: {v}")

    return args
