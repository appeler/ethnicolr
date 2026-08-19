"""Modern CLI for ethnicolr using Click framework.

Provides user-friendly commands for race/ethnicity estimation with improved
help, progress indicators, and better error handling.
"""

from pathlib import Path

import click
import pandas as pd

from .api import (
    estimate_census_surname,
    estimate_florida_voter_surname,
    estimate_wikipedia_surname,
)

CLI_MODEL_CATALOG = {
    "census-surname": {
        "estimator": "estimate_census_surname",
        "source": "U.S. Census surname tables",
        "input": "surname",
        "target": "race/ethnicity",
        "variants": "2000, 2010, 2020",
    },
    "florida-voter-surname": {
        "estimator": "estimate_florida_voter_surname",
        "source": "Florida voter registration records",
        "input": "surname",
        "target": "race/ethnicity",
        "variants": "2022 five-category model",
    },
    "wikipedia-surname": {
        "estimator": "estimate_wikipedia_surname",
        "source": "Wikipedia biographies",
        "input": "surname",
        "target": "race/ethnicity",
        "variants": "bundled model",
    },
}

# ASCII symbols for cross-platform compatibility
CHECK = "[OK]"
CROSS = "[FAIL]"


# Custom click types for better validation
class CSVFile(click.Path):
    """Click type for CSV file validation."""

    def __init__(self):
        """Require an existing, readable path."""
        super().__init__(exists=True, readable=True, path_type=Path)

    def convert(self, value, param, ctx):
        """Convert the value to a path and require a .csv suffix."""
        path = super().convert(value, param, ctx)
        if not str(path).lower().endswith(".csv"):
            self.fail(f"File must be a CSV file: {path}", param, ctx)
        return path


class OutputPath(click.Path):
    """Click type for output file validation."""

    def __init__(self):
        """Require a writable output path."""
        super().__init__(writable=True, path_type=Path)

    def convert(self, value, param, ctx):
        """Convert the value to a path, creating its parent directory."""
        path = super().convert(value, param, ctx)
        # Create parent directory if it doesn't exist
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--debug", "-d", is_flag=True, help="Enable debug output (implies --verbose)"
)
@click.version_option()
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool):
    """Ethnicolr: Estimate race/ethnicity from names using machine learning.

    This tool provides multiple models trained on different datasets
    for estimating race and ethnicity from first and last names.
    """
    # Ensure ctx.obj exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug

    if debug or verbose:
        import logging

        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )


@cli.group()
def estimate():
    """Estimate race/ethnicity from name patterns using supported models."""


@estimate.command("census-surname")
@click.argument("input_file", type=CSVFile())
@click.option(
    "-l",
    "--last-column",
    "last_col",
    required=True,
    help="Column name containing last names",
)
@click.option(
    "-o",
    "--output",
    type=OutputPath(),
    help="Output CSV file (default: census-estimates.csv)",
)
@click.option(
    "-y",
    "--year",
    type=click.Choice(["2000", "2010", "2020"]),
    default="2020",
    show_default=True,
    help="Census year for model",
)
@click.option(
    "-u",
    "--uncertainty-level",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
    show_default=True,
    help="MC-dropout range level (0.0-1.0)",
)
@click.option(
    "-m",
    "--mc-iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for MC-dropout ranges",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def estimate_census_surname_command(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    year: str,
    uncertainty_level: float | None,
    mc_iterations: int,
    overwrite: bool,
):
    """Estimate race/ethnicity using Census LSTM model.

    Uses LSTM neural networks trained on U.S. Census data to predict
    race/ethnicity probabilities from last names.

    Examples:

    \b
        # Basic estimate
        ethnicolr estimate census-surname data.csv -l surname

    \b
        # With MC-dropout ranges
        ethnicolr estimate census-surname data.csv -l surname -u 0.95 -m 200

    \b
        # Specify output file and Census year
        ethnicolr estimate census-surname data.csv -l surname -o results.csv -y 2000
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("census-estimates.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            input_data = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in input_data.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: "
                    f"{', '.join(input_data.columns)}"
                )
            bar.update(1)

            # Run estimate
            click.echo(f"Running Census {year} estimate on {len(input_data)} rows...")
            result = estimate_census_surname(
                input_data,
                last_col,
                year=int(year),
                uncertainty_level=uncertainty_level,
                mc_iterations=mc_iterations,
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        estimated_count = result.dropna(subset=["race"]).shape[0]
        success_rate = estimated_count / len(result) * 100

        click.echo(click.style(f"{CHECK} Estimate completed successfully!", fg="green"))
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Estimated: {estimated_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample estimates
        if verbose and len(result) > 0:
            click.echo("\nSample estimates:")
            sample = result.head(3)[
                [last_col, "race"]
                + [
                    col
                    for col in result.columns
                    if col in ["white", "black", "api", "hispanic"]
                ]
            ]
            click.echo(sample.to_string(index=False))

    except FileNotFoundError as e:
        raise click.ClickException(f"Model files not found: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Estimate failed: {e}") from e


@estimate.command("florida-voter-surname")
@click.argument("input_file", type=CSVFile())
@click.option(
    "-l",
    "--last-column",
    "last_col",
    required=True,
    help="Column name containing last names",
)
@click.option(
    "-o",
    "--output",
    type=OutputPath(),
    help="Output CSV file (default: florida-estimates.csv)",
)
@click.option(
    "-u",
    "--uncertainty-level",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
    show_default=True,
    help="MC-dropout range level (0.0-1.0)",
)
@click.option(
    "-m",
    "--mc-iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for MC-dropout ranges",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def estimate_florida_voter_surname_command(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    uncertainty_level: float | None,
    mc_iterations: int,
    overwrite: bool,
):
    """Estimate race/ethnicity using Florida voter registration LSTM model.

    Uses LSTM neural networks trained on Florida voter registration data
    to predict race/ethnicity probabilities from last names. Estimates
    5 categories: asian, hispanic, nh_black, nh_white, other.

    Examples:

    \b
        # Basic estimate
        ethnicolr estimate florida-voter-surname data.csv -l surname

    \b
        # With MC-dropout ranges
        ethnicolr estimate florida-voter-surname data.csv -l surname -u 0.95 -m 200

    \b
        # Specify output file
        ethnicolr estimate florida-voter-surname data.csv -l surname -o results.csv
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("florida-estimates.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            input_data = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in input_data.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: "
                    f"{', '.join(input_data.columns)}"
                )
            bar.update(1)

            # Run estimate
            click.echo(f"Running Florida estimate on {len(input_data)} rows...")
            result = estimate_florida_voter_surname(
                input_data,
                last_col,
                uncertainty_level=uncertainty_level,
                mc_iterations=mc_iterations,
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        estimated_count = result.dropna(subset=["race"]).shape[0]
        success_rate = estimated_count / len(result) * 100

        click.echo(click.style(f"{CHECK} Estimate completed successfully!", fg="green"))
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Estimated: {estimated_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample estimates
        if verbose and len(result) > 0:
            click.echo("\nSample estimates:")
            sample = result.head(3)[
                [last_col, "race"]
                + [
                    col
                    for col in result.columns
                    if col in ["asian", "hispanic", "nh_black", "nh_white"]
                ]
            ]
            click.echo(sample.to_string(index=False))

    except FileNotFoundError as e:
        raise click.ClickException(f"Model files not found: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Estimate failed: {e}") from e


@estimate.command("wikipedia-surname")
@click.argument("input_file", type=CSVFile())
@click.option(
    "-l",
    "--last-column",
    "last_col",
    required=True,
    help="Column name containing last names",
)
@click.option(
    "-o",
    "--output",
    type=OutputPath(),
    help="Output CSV file (default: wiki-estimates.csv)",
)
@click.option(
    "-u",
    "--uncertainty-level",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=None,
    show_default=True,
    help="MC-dropout range level (0.0-1.0)",
)
@click.option(
    "-m",
    "--mc-iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for MC-dropout ranges",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def estimate_wikipedia_surname_command(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    uncertainty_level: float | None,
    mc_iterations: int,
    overwrite: bool,
):
    """Estimate race/ethnicity using Wikipedia LSTM model.

    Uses LSTM neural networks trained on Wikipedia person data to predict
    detailed ethnic categories from last names. Provides 13 ethnic categories
    with hierarchical naming (e.g., "GreaterEuropean,WestEuropean,Germanic").

    Examples:

    \b
        # Basic estimate
        ethnicolr estimate wikipedia-surname data.csv -l surname

    \b
        # With MC-dropout ranges
        ethnicolr estimate wikipedia-surname data.csv -l surname -u 0.95 -m 200

    \b
        # Specify output file
        ethnicolr estimate wikipedia-surname data.csv -l surname -o results.csv
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("wiki-estimates.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            input_data = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in input_data.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: "
                    f"{', '.join(input_data.columns)}"
                )
            bar.update(1)

            # Run estimate
            click.echo(f"Running Wikipedia estimate on {len(input_data)} rows...")
            result = estimate_wikipedia_surname(
                input_data,
                last_col,
                uncertainty_level=uncertainty_level,
                mc_iterations=mc_iterations,
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        estimated_count = result.dropna(subset=["race"]).shape[0]
        success_rate = estimated_count / len(result) * 100

        click.echo(click.style(f"{CHECK} Estimate completed successfully!", fg="green"))
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Estimated: {estimated_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample estimates
        if verbose and len(result) > 0:
            click.echo("\nSample estimates:")
            # Show just the main race column for readability since wiki has
            # very detailed categories
            sample = result.head(3)[[last_col, "race"]]
            click.echo(sample.to_string(index=False))

    except FileNotFoundError as e:
        raise click.ClickException(f"Model files not found: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Estimate failed: {e}") from e


@cli.group()
def models():
    """Inspect bundled estimate models (list, info)."""


@models.command("list")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed information")
def list_models(detailed: bool):
    """List models available through the command-line interface."""
    click.echo(click.style("Available CLI Estimate Models", bold=True))
    click.echo("=" * 40)
    for model_name, details in CLI_MODEL_CATALOG.items():
        click.echo(f"\n{click.style(model_name, fg='blue', bold=True)}")
        click.echo(f"  Estimator: {details['estimator']}")
        if detailed:
            for field in ("source", "input", "target", "variants"):
                click.echo(f"  {field.title()}: {details[field]}")


@models.command("info")
@click.argument(
    "model_type",
    type=click.Choice(sorted(CLI_MODEL_CATALOG)),
)
def model_info(model_type: str):
    """Show detailed information about a specific model type."""
    click.echo(f"Information for {model_type.upper()} model:")

    for field, value in CLI_MODEL_CATALOG[model_type].items():
        click.echo(f"  {field.title()}: {value}")


@cli.command()
@click.argument("input_file", type=CSVFile())
@click.option("-l", "--last-column", "last_col", required=True)
@click.option("-o", "--output", type=OutputPath())
@click.option(
    "--model",
    type=click.Choice(sorted(CLI_MODEL_CATALOG)),
    default="census-surname",
    show_default=True,
)
@click.pass_context
def quick_estimate(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    model: str,
):
    """Quick estimate using the best available model.

    Automatically selects appropriate model based on available data
    and provides fast estimates with minimal configuration.
    """
    # Default output
    if output is None:
        output = Path(f"{input_file.stem}-estimates.csv")

    click.echo(f"Running quick estimate with {model} model...")

    # Delegate to specific model command
    if model == "census-surname":
        ctx.invoke(
            estimate_census_surname_command,
            input_file=input_file,
            last_col=last_col,
            output=output,
            year="2020",
            uncertainty_level=None,
            mc_iterations=100,
            overwrite=True,
        )
    elif model == "florida-voter-surname":
        ctx.invoke(
            estimate_florida_voter_surname_command,
            input_file=input_file,
            last_col=last_col,
            output=output,
            uncertainty_level=None,
            mc_iterations=100,
            overwrite=True,
        )
    elif model == "wikipedia-surname":
        ctx.invoke(
            estimate_wikipedia_surname_command,
            input_file=input_file,
            last_col=last_col,
            output=output,
            uncertainty_level=None,
            mc_iterations=100,
            overwrite=True,
        )
    else:
        click.echo(f"Quick estimate for {model} model not yet implemented")
        click.echo(
            "Available: ethnicolr estimate census-surname, "
            "ethnicolr estimate florida-voter-surname, "
            "ethnicolr estimate wikipedia-surname"
        )


if __name__ == "__main__":
    cli()
