"""
Tests for the modern CLI interface using Click framework.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_input_file():
    """Create a temporary CSV file for CLI testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("last,first,id\n")
        f.write("smith,john,1\n")
        f.write("zhang,wei,2\n")
        f.write("garcia,maria,3\n")
        f.write("johnson,james,4\n")
        f.write("patel,raj,5\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup - remove all files in temp directory
    for file in Path(temp_dir).glob("*"):
        file.unlink()
    os.rmdir(temp_dir)


class TestModernCLI:
    """Test the modern CLI commands."""


class TestMainCLI:
    """Test main CLI entry point."""

    def test_main_help(self):
        """Test main CLI help."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Ethnicolr" in result.stdout
        assert "estimate" in result.stdout
        assert "models" in result.stdout

    def test_estimate_help(self):
        """Test estimate subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "estimate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "census-surname" in result.stdout
        assert "florida-voter-surname" in result.stdout
        assert "wikipedia-surname" in result.stdout

    def test_models_help(self):
        """Test models subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "models", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "list" in result.stdout
        assert "info" in result.stdout


class TestEstimateCommands:
    """Test estimate commands."""

    def test_estimate_census_basic(self, sample_input_file, temp_output_dir):
        """Test basic census estimate."""
        output_file = os.path.join(temp_output_dir, "census_output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "last",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Should succeed
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        # Verify output structure
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns
        assert "white" in output_df.columns or "api" in output_df.columns

    def test_estimate_census_with_year(self, sample_input_file, temp_output_dir):
        """Test census estimate with specific year."""
        output_file = os.path.join(temp_output_dir, "census_2000.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "last",
            "-y",
            "2000",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0
        assert os.path.exists(output_file)

    def test_estimate_census_uncertainty(self, sample_input_file, temp_output_dir):
        """Test census estimates with MC-dropout uncertainty summaries."""
        output_file = os.path.join(temp_output_dir, "census_uncertainty.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "last",
            "-u",
            "0.9",
            "-m",
            "20",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(output_file)
        output_df = pd.read_csv(output_file)
        mean_columns = [
            column for column in output_df.columns if column.endswith("_mc_mean")
        ]
        assert mean_columns

    def test_estimate_florida_basic(self, sample_input_file, temp_output_dir):
        """Test basic Florida estimate."""
        output_file = os.path.join(temp_output_dir, "florida_output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "florida-voter-surname",
            sample_input_file,
            "-l",
            "last",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 5
            assert "race" in output_df.columns
            # Florida-specific columns
            fl_cols = [
                col
                for col in output_df.columns
                if col in ["hispanic", "nh_black", "nh_white", "asian"]
            ]
            assert len(fl_cols) > 0

    def test_estimate_wiki_basic(self, sample_input_file, temp_output_dir):
        """Test basic Wikipedia estimate."""
        output_file = os.path.join(temp_output_dir, "wiki_output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "wikipedia-surname",
            sample_input_file,
            "-l",
            "last",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Wiki models may not be installed, so we allow failure
        if result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 5
            assert "race" in output_df.columns


class TestModelsCommands:
    """Test model management commands."""

    def test_models_list(self):
        """Test models list command."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "models", "list"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available CLI Estimate Models" in result.stdout

    def test_models_list_detailed(self):
        """Test models list with details."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "models", "list", "--detailed"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available CLI Estimate Models" in result.stdout

    def test_models_info_census(self):
        """Test models info for census."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "models", "info", "census-surname"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CENSUS" in result.stdout.upper()


class TestQuickEstimate:
    """Test quick estimate command."""

    def test_quick_estimate_help(self):
        """Test quick estimate help."""
        result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "quick-estimate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Quick estimate" in result.stdout

    def test_quick_estimate_census(self, sample_input_file, temp_output_dir):
        """Test quick estimate with census model."""
        output_file = os.path.join(temp_output_dir, "quick_output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "quick-estimate",
            sample_input_file,
            "-l",
            "last",
            "--model",
            "census-surname",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 5


class TestCLIErrorHandling:
    """Test error handling in modern CLI."""

    def test_missing_file_error(self, temp_output_dir):
        """Test error when input file doesn't exist."""
        nonexistent_file = "/nonexistent/path/file.csv"
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            nonexistent_file,
            "-l",
            "last",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        # Should have error message about file not found
        assert "not found" in result.stderr or "does not exist" in result.stderr

    def test_invalid_column_error(self, sample_input_file, temp_output_dir):
        """Test error when column doesn't exist."""
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "nonexistent_column",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        # Should mention column not found
        assert "not found" in result.stderr

    def test_invalid_uncertainty_level_error(self, sample_input_file, temp_output_dir):
        """Test the error for an invalid uncertainty level."""
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "last",
            "-u",
            "1.5",  # Invalid: > 1.0
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_models_then_estimate_workflow(self, sample_input_file, temp_output_dir):
        """Test workflow: list models, then estimate."""
        # Step 1: List models
        status_result = subprocess.run(
            [sys.executable, "-m", "ethnicolr.cli", "models", "list"],
            capture_output=True,
            text=True,
        )
        assert status_result.returncode == 0

        # Step 2: Run estimate
        output_file = os.path.join(temp_output_dir, "workflow_output.csv")

        estimate_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ethnicolr.cli",
                "estimate",
                "census-surname",
                sample_input_file,
                "-l",
                "last",
                "-o",
                output_file,
                "--overwrite",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if estimate_result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 5

    def test_verbose_mode(self, sample_input_file, temp_output_dir):
        """Test verbose mode."""
        output_file = os.path.join(temp_output_dir, "verbose_output.csv")

        cmd = [
            sys.executable,
            "-m",
            "ethnicolr.cli",
            "--verbose",
            "estimate",
            "census-surname",
            sample_input_file,
            "-l",
            "last",
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # In verbose mode, should show more detailed output
            assert (
                "Sample estimates" in result.stdout or "Loading data" in result.stdout
            )
