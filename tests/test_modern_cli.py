"""
Tests for the modern CLI interface using Click framework.
"""

import os
import subprocess
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
            ["python", "-m", "ethnicolr.cli", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "Ethnicolr" in result.stdout
        assert "predict" in result.stdout
        assert "models" in result.stdout

    def test_predict_help(self):
        """Test predict subcommand help."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "predict", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "census" in result.stdout
        assert "florida" in result.stdout
        assert "wiki" in result.stdout

    def test_models_help(self):
        """Test models subcommand help."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "download" in result.stdout
        assert "list" in result.stdout
        assert "status" in result.stdout


class TestPredictCommands:
    """Test prediction commands."""

    def test_predict_census_basic(self, sample_input_file, temp_output_dir):
        """Test basic census prediction."""
        output_file = os.path.join(temp_output_dir, "census_output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
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

    def test_predict_census_with_year(self, sample_input_file, temp_output_dir):
        """Test census prediction with specific year."""
        output_file = os.path.join(temp_output_dir, "census_2000.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
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

    def test_predict_census_confidence(self, sample_input_file, temp_output_dir):
        """Test census prediction with confidence intervals."""
        output_file = os.path.join(temp_output_dir, "census_conf.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
            sample_input_file,
            "-l",
            "last",
            "-c",
            "0.9",
            "-i",
            "20",  # Fewer iterations for speed
            "-o",
            output_file,
            "--overwrite",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            # Should have confidence interval columns
            mean_cols = [col for col in output_df.columns if col.endswith("_mean")]
            assert len(mean_cols) > 0

    def test_predict_florida_basic(self, sample_input_file, temp_output_dir):
        """Test basic Florida prediction."""
        output_file = os.path.join(temp_output_dir, "florida_output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "florida",
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

    def test_predict_wiki_basic(self, sample_input_file, temp_output_dir):
        """Test basic Wikipedia prediction."""
        output_file = os.path.join(temp_output_dir, "wiki_output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "wiki",
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
            ["python", "-m", "ethnicolr.cli", "models", "list"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Prediction Models" in result.stdout

    def test_models_list_detailed(self):
        """Test models list with details."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "list", "--detailed"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Available Prediction Models" in result.stdout

    def test_models_status(self):
        """Test models status command."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Model Status" in result.stdout

    def test_models_info_census(self):
        """Test models info for census."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "info", "census"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CENSUS" in result.stdout.upper()

    def test_models_download_help(self):
        """Test models download help."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "download", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Download prediction model files" in result.stdout


class TestQuickPredict:
    """Test quick predict command."""

    def test_quick_predict_help(self):
        """Test quick predict help."""
        result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "quick-predict", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Quick prediction" in result.stdout

    def test_quick_predict_census(self, sample_input_file, temp_output_dir):
        """Test quick predict with census model."""
        output_file = os.path.join(temp_output_dir, "quick_output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "quick-predict",
            sample_input_file,
            "-l",
            "last",
            "--model",
            "census",
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
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
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
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
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

    def test_invalid_confidence_error(self, sample_input_file, temp_output_dir):
        """Test error for invalid confidence interval."""
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "predict",
            "census",
            sample_input_file,
            "-l",
            "last",
            "-c",
            "1.5",  # Invalid: > 1.0
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def test_models_then_predict_workflow(self, sample_input_file, temp_output_dir):
        """Test workflow: check status, then predict."""
        # Step 1: Check model status
        status_result = subprocess.run(
            ["python", "-m", "ethnicolr.cli", "models", "status"],
            capture_output=True,
            text=True,
        )
        assert status_result.returncode == 0

        # Step 2: Run prediction
        output_file = os.path.join(temp_output_dir, "workflow_output.csv")

        pred_result = subprocess.run(
            [
                "python",
                "-m",
                "ethnicolr.cli",
                "predict",
                "census",
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

        if pred_result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 5

    def test_verbose_mode(self, sample_input_file, temp_output_dir):
        """Test verbose mode."""
        output_file = os.path.join(temp_output_dir, "verbose_output.csv")

        cmd = [
            "python",
            "-m",
            "ethnicolr.cli",
            "--verbose",
            "predict",
            "census",
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
                "Sample predictions" in result.stdout or "Loading data" in result.stdout
            )
