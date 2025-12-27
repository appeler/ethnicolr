#!/usr/bin/env python
"""
Modern CLI for ethnicolr using Click framework.

Provides user-friendly commands for race/ethnicity prediction with improved
help, progress indicators, and better error handling.
"""

from pathlib import Path

import click
import pandas as pd

from .download import (
    DownloadError,
    ModelNotAvailableError,
    download_model,
    get_installed_models,
    list_available_models,
)
from .model_base import ModelRegistry
from .pred_census_ln import CensusLnModel
from .pred_fl_reg_ln import FloridaRegLnModel
from .pred_wiki_ln import WikiLnModel

# ASCII symbols for cross-platform compatibility
CHECK = "[OK]"
CROSS = "[FAIL]"


# Custom click types for better validation
class CSVFile(click.Path):
    """Click type for CSV file validation."""

    def __init__(self):
        super().__init__(exists=True, readable=True, path_type=Path)

    def convert(self, value, param, ctx):
        path = super().convert(value, param, ctx)
        if not str(path).lower().endswith(".csv"):
            self.fail(f"File must be a CSV file: {path}", param, ctx)
        return path


class OutputPath(click.Path):
    """Click type for output file validation."""

    def __init__(self):
        super().__init__(writable=True, path_type=Path)

    def convert(self, value, param, ctx):
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
    """
    Ethnicolr: Predict race/ethnicity from names using machine learning.

    This tool provides multiple models trained on different datasets
    for predicting race and ethnicity from first and last names.
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
def predict():
    """Predict race/ethnicity from names using various models."""
    pass


@predict.command("census")
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
    help="Output CSV file (default: census-predictions.csv)",
)
@click.option(
    "-y",
    "--year",
    type=click.Choice(["2000", "2010"]),
    default="2010",
    show_default=True,
    help="Census year for model",
)
@click.option(
    "-c",
    "--confidence",
    type=click.FloatRange(0.0, 1.0),
    default=1.0,
    show_default=True,
    help="Confidence interval level (0.0-1.0)",
)
@click.option(
    "-i",
    "--iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for confidence intervals",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def predict_census(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    year: str,
    confidence: float,
    iterations: int,
    overwrite: bool,
):
    """
    Predict race/ethnicity using Census LSTM model.

    Uses LSTM neural networks trained on U.S. Census data to predict
    race/ethnicity probabilities from last names.

    Examples:

    \b
        # Basic prediction
        ethnicolr predict census data.csv -l surname

        # With confidence intervals
        ethnicolr predict census data.csv -l surname -c 0.95 -i 200

        # Specify output file and Census year
        ethnicolr predict census data.csv -l surname -o results.csv -y 2000
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("census-predictions.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in df.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: {', '.join(df.columns)}"
                )
            bar.update(1)

            # Run prediction
            click.echo(f"Running Census {year} prediction on {len(df)} rows...")
            result = CensusLnModel.predict_with_confidence(  # type: ignore
                df, last_col, year=int(year), conf_int=confidence, num_iter=iterations
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        predicted_count = result.dropna(subset=["race"]).shape[0]
        success_rate = predicted_count / len(result) * 100

        click.echo(
            click.style(f"{CHECK} Prediction completed successfully!", fg="green")
        )
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Predicted: {predicted_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample predictions
        if verbose and len(result) > 0:
            click.echo("\nSample predictions:")
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
        raise click.ClickException(f"Prediction failed: {e}") from e


@predict.command("florida")
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
    help="Output CSV file (default: florida-predictions.csv)",
)
@click.option(
    "-c",
    "--confidence",
    type=click.FloatRange(0.0, 1.0),
    default=1.0,
    show_default=True,
    help="Confidence interval level (0.0-1.0)",
)
@click.option(
    "-i",
    "--iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for confidence intervals",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def predict_florida(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    confidence: float,
    iterations: int,
    overwrite: bool,
):
    """
    Predict race/ethnicity using Florida voter registration LSTM model.

    Uses LSTM neural networks trained on Florida voter registration data
    to predict race/ethnicity probabilities from last names. Predicts
    5 categories: asian, hispanic, nh_black, nh_white, other.

    Examples:

    \\b
        # Basic prediction
        ethnicolr predict florida data.csv -l surname

        # With confidence intervals
        ethnicolr predict florida data.csv -l surname -c 0.95 -i 200

        # Specify output file
        ethnicolr predict florida data.csv -l surname -o results.csv
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("florida-predictions.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in df.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: {', '.join(df.columns)}"
                )
            bar.update(1)

            # Run prediction
            click.echo(f"Running Florida prediction on {len(df)} rows...")
            result = FloridaRegLnModel.predict_with_confidence(  # type: ignore
                df, last_col, conf_int=confidence, num_iter=iterations
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        predicted_count = result.dropna(subset=["race"]).shape[0]
        success_rate = predicted_count / len(result) * 100

        click.echo(
            click.style(f"{CHECK} Prediction completed successfully!", fg="green")
        )
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Predicted: {predicted_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample predictions
        if verbose and len(result) > 0:
            click.echo("\nSample predictions:")
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
        raise click.ClickException(f"Prediction failed: {e}") from e


@predict.command("wiki")
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
    help="Output CSV file (default: wiki-predictions.csv)",
)
@click.option(
    "-c",
    "--confidence",
    type=click.FloatRange(0.0, 1.0),
    default=1.0,
    show_default=True,
    help="Confidence interval level (0.0-1.0)",
)
@click.option(
    "-i",
    "--iterations",
    type=click.IntRange(10, 1000),
    default=100,
    show_default=True,
    help="Monte Carlo iterations for confidence intervals",
)
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.pass_context
def predict_wiki(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    output: Path | None,
    confidence: float,
    iterations: int,
    overwrite: bool,
):
    """
    Predict race/ethnicity using Wikipedia LSTM model.

    Uses LSTM neural networks trained on Wikipedia person data to predict
    detailed ethnic categories from last names. Provides 13 ethnic categories
    with hierarchical naming (e.g., "GreaterEuropean,WestEuropean,Germanic").

    Examples:

    \\b
        # Basic prediction
        ethnicolr predict wiki data.csv -l surname

        # With confidence intervals
        ethnicolr predict wiki data.csv -l surname -c 0.95 -i 200

        # Specify output file
        ethnicolr predict wiki data.csv -l surname -o results.csv
    """
    verbose = ctx.obj.get("verbose", False)

    # Default output filename
    if output is None:
        output = Path("wiki-predictions.csv")

    # Check if output file exists
    if output.exists() and not overwrite:
        click.confirm(f"Output file {output} exists. Overwrite?", abort=True)

    try:
        with click.progressbar(length=4, label="Loading data") as bar:
            # Load input data
            click.echo(f"Reading input file: {input_file}")
            df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
            bar.update(1)

            # Validate column exists
            if last_col not in df.columns:
                raise click.ClickException(
                    f"Column '{last_col}' not found. Available columns: {', '.join(df.columns)}"
                )
            bar.update(1)

            # Run prediction
            click.echo(f"Running Wikipedia prediction on {len(df)} rows...")
            result = WikiLnModel.predict_with_confidence(  # type: ignore
                df, last_col, conf_int=confidence, num_iter=iterations
            )
            bar.update(1)

            # Save results
            click.echo(f"Saving results to: {output}")
            result.to_csv(output, index=False)
            bar.update(1)

        # Success message
        predicted_count = result.dropna(subset=["race"]).shape[0]
        success_rate = predicted_count / len(result) * 100

        click.echo(
            click.style(f"{CHECK} Prediction completed successfully!", fg="green")
        )
        click.echo(f"  Processed: {len(result)} rows")
        click.echo(f"  Predicted: {predicted_count} rows ({success_rate:.1f}%)")
        click.echo(f"  Output: {output}")

        # Show sample predictions
        if verbose and len(result) > 0:
            click.echo("\nSample predictions:")
            # Show just the main race column for readability since wiki has very detailed categories
            sample = result.head(3)[[last_col, "race"]]
            click.echo(sample.to_string(index=False))

    except FileNotFoundError as e:
        raise click.ClickException(f"Model files not found: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Prediction failed: {e}") from e


@cli.group()
def models():
    """Manage prediction models (download, list, info)."""
    pass


@models.command("list")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed information")
def list_models(detailed: bool):
    """List available prediction models."""
    click.echo(click.style("Available Prediction Models", bold=True))
    click.echo("=" * 40)

    # Get registered models
    available_models = ModelRegistry.get_available_models()

    if not available_models:
        click.echo("No models registered. Try importing model modules first.")
        return

    for model_type, model_class in available_models.items():
        click.echo(f"\n{click.style(model_type.value.upper(), fg='blue', bold=True)}")
        click.echo(f"  Class: {model_class.__name__}")

        if detailed:
            # Show supported categories
            try:
                categories = model_class.get_supported_categories()
                click.echo(f"  Categories: {', '.join(categories)}")
            except Exception:
                click.echo("  Categories: Unknown")

            # Show model documentation
            if model_class.__doc__:
                doc_lines = model_class.__doc__.strip().split("\n")
                click.echo(f"  Description: {doc_lines[0]}")


@models.command("download")
@click.argument("model_type", type=click.Choice(["census", "wiki", "florida", "nc"]))
@click.option("--year", type=str, help="Specific model year to download")
@click.option("--force", is_flag=True, help="Force redownload existing files")
def download_models(model_type: str, year: str | None, force: bool):
    """
    Download prediction model files.

    Downloads pre-trained models and vocabulary files for the specified
    model type. Models are required for prediction but are not included
    in the base package due to size constraints.

    Examples:

    \b
        # Download all Census models
        ethnicolr models download census

        # Download specific year
        ethnicolr models download census --year 2010

        # Force redownload
        ethnicolr models download census --force
    """
    try:
        click.echo(f"Downloading {model_type} models...")

        # Show available years if none specified
        if not year:
            available = list_available_models()
            years_available = available.get(model_type, [])
            if years_available:
                click.echo(f"Available years: {', '.join(years_available)}")

        # Perform download
        downloaded_files = download_model(model_type=model_type, year=year, force=force)

        if downloaded_files:
            click.echo(
                click.style(f"{CHECK} Download completed successfully!", fg="green")
            )
            click.echo(f"Downloaded {len(downloaded_files)} files for {model_type}")

            if year:
                click.echo(f"Model ready: ethnicolr predict {model_type}")
            else:
                click.echo(f"Models ready: ethnicolr predict {model_type}")
        else:
            click.echo(
                click.style("No files downloaded (may already exist)", fg="yellow")
            )
            click.echo("Use --force to redownload existing files")

    except ModelNotAvailableError as e:
        raise click.ClickException(f"Model not available: {e}") from e
    except DownloadError as e:
        raise click.ClickException(f"Download failed: {e}") from e
    except Exception as e:
        raise click.ClickException(f"Unexpected error: {e}") from e


@models.command("status")
def model_status():
    """Show status of installed and available models."""
    try:
        available = list_available_models()
        installed = get_installed_models()

        click.echo(click.style("Model Status", bold=True))
        click.echo("=" * 40)

        for model_type in available.keys():
            click.echo(f"\n{click.style(model_type.upper(), fg='blue', bold=True)}")

            available_years = available[model_type]
            installed_years = installed.get(model_type, [])

            for year in available_years:
                if year in installed_years:
                    status = click.style(f"{CHECK} Installed", fg="green")
                else:
                    status = click.style(f"{CROSS} Not installed", fg="yellow")
                click.echo(f"  {year}: {status}")

            if not available_years:
                click.echo("  No versions available")

        # Summary
        total_available = sum(len(years) for years in available.values())
        total_installed = sum(len(years) for years in installed.values())

        click.echo(f"\nSummary: {total_installed}/{total_available} models installed")

        if total_installed < total_available:
            click.echo("\nTo download models: ethnicolr models download <model_type>")

    except Exception as e:
        raise click.ClickException(f"Failed to check model status: {e}") from e


@models.command("info")
@click.argument("model_type", type=click.Choice(["census", "wiki", "florida", "nc"]))
def model_info(model_type: str):
    """Show detailed information about a specific model type."""
    click.echo(f"Information for {model_type.upper()} model:")

    # Model-specific information
    info_map = {
        "census": {
            "name": "U.S. Census LSTM Model",
            "description": "LSTM trained on U.S. Census surname data",
            "categories": ["white", "black", "api", "hispanic"],
            "years": ["2000", "2010"],
            "input": "Last names only",
            "accuracy": "75-85% on census-representative data",
        },
        "wiki": {
            "name": "Wikipedia LSTM Model",
            "description": "LSTM trained on Wikipedia person names",
            "categories": ["13 detailed ethnic categories"],
            "years": ["2017"],
            "input": "First and last names",
            "accuracy": "60-75% on diverse international names",
        },
        "florida": {
            "name": "Florida Voter Registration Model",
            "description": "LSTM trained on Florida voter registration data",
            "categories": ["asian", "hispanic", "nh_black", "nh_white", "other"],
            "years": ["2017", "2022"],
            "input": "First and/or last names",
            "accuracy": "70-80% on Florida-representative data",
        },
        "nc": {
            "name": "North Carolina Voter Registration Model",
            "description": "LSTM trained on NC voter registration data",
            "categories": ["12 Hispanic/Latino + race combinations"],
            "years": ["2017"],
            "input": "First and last names",
            "accuracy": "65-75% on NC-representative data",
        },
    }

    info = info_map.get(model_type, {})

    for key, value in info.items():
        if isinstance(value, list):
            value = ", ".join(value)
        click.echo(f"  {key.title()}: {value}")


@cli.command()
@click.argument("input_file", type=CSVFile())
@click.option("-l", "--last-column", "last_col", required=True)
@click.option(
    "-f",
    "--first-column",
    "first_col",
    help="Column name for first names (if available)",
)
@click.option("-o", "--output", type=OutputPath())
@click.option(
    "--model",
    type=click.Choice(["census", "wiki", "florida", "nc"]),
    default="census",
    show_default=True,
)
@click.pass_context
def quick_predict(
    ctx: click.Context,
    input_file: Path,
    last_col: str,
    first_col: str | None,
    output: Path | None,
    model: str,
):
    """
    Quick prediction using the best available model.

    Automatically selects appropriate model based on available data
    and provides fast predictions with minimal configuration.
    """
    # Default output
    if output is None:
        output = Path(f"{input_file.stem}-predictions.csv")

    click.echo(f"Running quick prediction with {model} model...")

    # Delegate to specific model command
    if model == "census":
        ctx.invoke(
            predict_census,
            input_file=input_file,
            last_col=last_col,
            output=output,
            year="2010",
            confidence=1.0,
            iterations=100,
            overwrite=True,
        )
    elif model == "florida":
        ctx.invoke(
            predict_florida,
            input_file=input_file,
            last_col=last_col,
            output=output,
            confidence=1.0,
            iterations=100,
            overwrite=True,
        )
    elif model == "wiki":
        ctx.invoke(
            predict_wiki,
            input_file=input_file,
            last_col=last_col,
            output=output,
            confidence=1.0,
            iterations=100,
            overwrite=True,
        )
    else:
        click.echo(f"Quick predict for {model} model not yet implemented")
        click.echo(
            "Available: ethnicolr predict census, ethnicolr predict florida, ethnicolr predict wiki"
        )


if __name__ == "__main__":
    cli()
