#!/usr/bin/env python
"""
Model download utilities for ethnicolr.

Handles downloading pre-trained models and vocabulary files from GitHub releases
or other distribution endpoints. Provides progress tracking, integrity checks,
and proper error handling for model file management.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve

import click

logger = logging.getLogger(__name__)

# GitHub release configuration
GITHUB_REPO = "appeler/ethnicolr"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/download"

# Model file configurations
MODEL_CONFIGS = {
    "census": {
        "2000": [
            ("census2000_ln_lstm.h5", "models/census/lstm/"),
            ("census2000_ln_vocab.csv", "models/census/lstm/"),
            ("census2000_race.csv", "models/census/lstm/"),
        ],
        "2010": [
            ("census2010_ln_lstm.h5", "models/census/lstm/"),
            ("census2010_ln_vocab.csv", "models/census/lstm/"),
            ("census2010_race.csv", "models/census/lstm/"),
        ],
    },
    "florida": {
        "2017": [
            ("fl_all_ln_lstm.h5", "models/fl_voter_reg/lstm/"),
            ("fl_all_ln_vocab.csv", "models/fl_voter_reg/lstm/"),
            ("fl_ln_race.csv", "models/fl_voter_reg/lstm/"),
        ],
    },
    "wiki": {
        "2017": [
            ("wiki_name_lstm.h5", "models/wiki/lstm/"),
            ("wiki_name_vocab.csv", "models/wiki/lstm/"),
            ("wiki_name_race.csv", "models/wiki/lstm/"),
            ("wiki_ln_lstm.h5", "models/wiki/lstm/"),
            ("wiki_ln_vocab.csv", "models/wiki/lstm/"),
            ("wiki_ln_race.csv", "models/wiki/lstm/"),
        ],
    },
    "nc": {
        "2017": [
            ("nc_name_lstm.h5", "models/nc_voter_reg/lstm/"),
            ("nc_name_vocab.csv", "models/nc_voter_reg/lstm/"),
            ("nc_name_race.csv", "models/nc_voter_reg/lstm/"),
        ],
    },
}


class DownloadError(Exception):
    """Base exception for download-related errors."""

    pass


class ModelNotAvailableError(DownloadError):
    """Raised when requested model is not available for download."""

    pass


def get_package_models_dir() -> Path:
    """Get the models directory within the installed package."""
    try:
        from importlib import resources

        package_root = resources.files("ethnicolr")
        return Path(str(package_root))
    except ImportError:
        # Fallback for older Python versions
        import ethnicolr

        return Path(ethnicolr.__file__).parent


def verify_file_integrity(file_path: Path, expected_size: int | None = None) -> bool:
    """
    Verify downloaded file integrity.

    Args:
        file_path: Path to file to verify.
        expected_size: Expected file size in bytes (optional).

    Returns:
        True if file appears valid, False otherwise.
    """
    if not file_path.exists():
        return False

    # Check file size if provided
    if expected_size and file_path.stat().st_size != expected_size:
        logger.warning(f"File size mismatch for {file_path}")
        return False

    # Basic sanity checks
    if file_path.stat().st_size == 0:
        logger.warning(f"Downloaded file is empty: {file_path}")
        return False

    # For .h5 files, check HDF5 magic number
    if file_path.suffix == ".h5":
        try:
            with open(file_path, "rb") as f:
                magic = f.read(8)
                if not magic.startswith(b"\x89HDF"):
                    logger.warning(f"Invalid HDF5 file: {file_path}")
                    return False
        except OSError:
            return False

    return True


def download_file_with_progress(
    url: str, dest_path: Path, desc: str = "Downloading"
) -> None:
    """
    Download file with progress bar.

    Args:
        url: URL to download from.
        dest_path: Destination file path.
        desc: Description for progress bar.

    Raises:
        DownloadError: If download fails.
    """
    try:
        # Create destination directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size for progress tracking
        try:
            with urlopen(url) as response:
                total_size = int(response.headers.get("content-length", 0))
        except (URLError, HTTPError):
            total_size = 0

        # Download with progress bar
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

            pbar_ref: dict[str, Any] = {"pbar": None}

            def progress_hook(block_num, block_size, total_size):
                if pbar_ref["pbar"] is not None:
                    downloaded = min(block_num * block_size, total_size)
                    pbar_ref["pbar"].update(downloaded - pbar_ref["pbar"].n)

            # Only show progress bar if we know the size
            if total_size > 0:
                with click.progressbar(length=total_size, label=desc) as pbar:
                    pbar_ref["pbar"] = pbar
                    urlretrieve(url, tmp_path, progress_hook)
            else:
                click.echo(f"{desc}...")
                urlretrieve(url, tmp_path)

        # Verify download
        if not verify_file_integrity(tmp_path):
            tmp_path.unlink()
            raise DownloadError(f"Downloaded file failed integrity check: {url}")

        # Move to final location
        shutil.move(str(tmp_path), str(dest_path))
        logger.info(f"Downloaded: {dest_path}")

    except (URLError, HTTPError) as e:
        raise DownloadError(f"Failed to download {url}: {e}") from e
    except OSError as e:
        raise DownloadError(f"Failed to save {dest_path}: {e}") from e


def get_release_tag(model_type: str, year: str) -> str:
    """Get GitHub release tag for model type and year."""
    # For now, use a simple mapping. In the future, this could
    # query the GitHub API to find the latest appropriate release
    tag_mapping = {
        ("census", "2000"): "v2.0.0",
        ("census", "2010"): "v2.0.0",
        ("florida", "2017"): "v2.0.0",
        ("wiki", "2017"): "v2.0.0",
        ("nc", "2017"): "v2.0.0",
    }

    key = (model_type, year)
    if key not in tag_mapping:
        raise ModelNotAvailableError(f"No release found for {model_type} {year}")

    return tag_mapping[key]


def download_model(
    model_type: str,
    year: str | None = None,
    force: bool = False,
    base_url: str | None = None,
) -> list[Path]:
    """
    Download model files for specified type and year.

    Args:
        model_type: Type of model ('census', 'florida', 'wiki', 'nc').
        year: Model year to download (downloads all if None).
        force: Force redownload even if files exist.
        base_url: Base URL for downloads (uses GitHub releases if None).

    Returns:
        List of downloaded file paths.

    Raises:
        ModelNotAvailableError: If model type/year combination not available.
        DownloadError: If download fails.
    """
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ModelNotAvailableError(
            f"Model type '{model_type}' not available. Available: {available}"
        )

    model_config = MODEL_CONFIGS[model_type]
    years_to_download = [year] if year else list(model_config.keys())

    # Validate years
    for y in years_to_download:
        if y not in model_config:
            available = ", ".join(model_config.keys())
            raise ModelNotAvailableError(
                f"Year '{y}' not available for {model_type}. Available: {available}"
            )

    base_url = base_url or GITHUB_RELEASES_URL
    package_dir = get_package_models_dir()
    downloaded_files = []

    for y in years_to_download:
        release_tag = get_release_tag(model_type, y)
        files = model_config[y]

        for filename, subdir in files:
            # Check if file already exists
            dest_path = package_dir / subdir / filename

            if dest_path.exists() and not force:
                click.echo(f"✓ Already exists: {dest_path.relative_to(package_dir)}")
                downloaded_files.append(dest_path)
                continue

            # Download file
            url = f"{base_url}/{release_tag}/{filename}"
            desc = f"Downloading {model_type} {y} - {filename}"

            try:
                download_file_with_progress(url, dest_path, desc)
                downloaded_files.append(dest_path)
                click.echo(f"✓ Downloaded: {dest_path.relative_to(package_dir)}")

            except DownloadError as e:
                click.echo(f"✗ Failed: {filename} - {e}")
                # Continue with other files rather than aborting
                logger.error(f"Failed to download {filename}: {e}")

    return downloaded_files


def list_available_models() -> dict[str, list[str]]:
    """Get dictionary of available model types and their years."""
    return {
        model_type: list(years.keys()) for model_type, years in MODEL_CONFIGS.items()
    }


def check_model_availability(model_type: str, year: str | None = None) -> bool:
    """Check if a model is available for download."""
    if model_type not in MODEL_CONFIGS:
        return False

    if year is None:
        return True

    return year in MODEL_CONFIGS[model_type]


def get_installed_models() -> dict[str, list[str]]:
    """Get dictionary of currently installed models."""
    package_dir = get_package_models_dir()
    installed = {}

    for model_type, years in MODEL_CONFIGS.items():
        installed[model_type] = []

        for year, files in years.items():
            # Check if all required files exist for this model/year
            all_files_exist = True
            for filename, subdir in files:
                file_path = package_dir / subdir / filename
                if not file_path.exists():
                    all_files_exist = False
                    break

            if all_files_exist:
                installed[model_type].append(year)

    return installed
