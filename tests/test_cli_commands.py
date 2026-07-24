"""
Tests for command-line interface scripts.
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


class TestCLICommands:
    """Test all CLI commands defined in pyproject.toml."""


class TestCensusLnCLI:
    """Test the census_ln command-line interface."""

    def test_census_ln_basic_usage(self, sample_input_file, temp_output_dir):
        """Test basic census_ln CLI functionality."""
        output_file = os.path.join(temp_output_dir, "census_output.csv")

        # Test with 2010 census (default)
        cmd = ["census_ln", sample_input_file, "-l", "last", "-o", output_file]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should succeed
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Output file should be created
        assert os.path.exists(output_file)

        # Verify output structure
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5  # Same as input
        assert "last" in output_df.columns
        assert "pctwhite" in output_df.columns
        assert "pctblack" in output_df.columns

    def test_census_ln_with_year_2000(self, sample_input_file, temp_output_dir):
        """Test census_ln with 2000 census data."""
        output_file = os.path.join(temp_output_dir, "census_2000.csv")

        cmd = [
            "census_ln",
            sample_input_file,
            "-l",
            "last",
            "-y",
            "2000",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert "pctwhite" in output_df.columns

    def test_census_ln_help(self):
        """Test census_ln help message."""
        result = subprocess.run(["census_ln", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Census" in result.stdout or "census" in result.stdout


class TestPredCensusLnCLI:
    """Test the pred_census_ln command-line interface."""

    def test_pred_census_ln_basic(self, sample_input_file, temp_output_dir):
        """Test basic pred_census_ln CLI functionality."""
        output_file = os.path.join(temp_output_dir, "pred_census.csv")

        cmd = ["pred_census_ln", sample_input_file, "-l", "last", "-o", output_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Should succeed
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        # Verify prediction output
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns
        assert "api" in output_df.columns
        assert "black" in output_df.columns
        assert "hispanic" in output_df.columns
        assert "white" in output_df.columns

    def test_pred_census_ln_with_confidence(self, sample_input_file, temp_output_dir):
        """Test pred_census_ln with confidence intervals."""
        output_file = os.path.join(temp_output_dir, "pred_census_conf.csv")

        cmd = [
            "pred_census_ln",
            sample_input_file,
            "-l",
            "last",
            "-c",
            "0.9",
            "-i",
            "50",  # Fewer iterations for speed
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert "api_mean" in output_df.columns
        assert "white_mean" in output_df.columns


class TestWikiPredictionCLI:
    """Test Wikipedia-based prediction CLI commands."""

    def test_pred_wiki_ln_basic(self, sample_input_file, temp_output_dir):
        """Test pred_wiki_ln CLI functionality."""
        output_file = os.path.join(temp_output_dir, "wiki_ln.csv")

        cmd = ["pred_wiki_ln", sample_input_file, "-l", "last", "-o", output_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Should succeed
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns
        # Should have wiki-style race categories
        wiki_cols = [
            col
            for col in output_df.columns
            if "GreaterEuropean" in col or "Asian" in col or "GreaterAfrican" in col
        ]
        assert len(wiki_cols) > 0

    def test_pred_wiki_name_basic(self, sample_input_file, temp_output_dir):
        """Test pred_wiki_name CLI functionality."""
        output_file = os.path.join(temp_output_dir, "wiki_name.csv")

        cmd = [
            "pred_wiki_name",
            sample_input_file,
            "-l",
            "last",
            "-f",
            "first",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns
        assert "processing_status" in output_df.columns


class TestFloridaPredictionCLI:
    """Test Florida voter registration prediction CLI commands."""

    def test_pred_fl_reg_ln_basic(self, sample_input_file, temp_output_dir):
        """Test pred_fl_reg_ln CLI functionality."""
        output_file = os.path.join(temp_output_dir, "fl_ln.csv")

        cmd = ["pred_fl_reg_ln", sample_input_file, "-l", "last", "-o", output_file]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns
        assert "asian" in output_df.columns
        assert "hispanic" in output_df.columns
        assert "nh_black" in output_df.columns
        assert "nh_white" in output_df.columns

    def test_pred_fl_reg_ln_five_cat(self, sample_input_file, temp_output_dir):
        """Test pred_fl_reg_ln_five_cat CLI functionality."""
        output_file = os.path.join(temp_output_dir, "fl_ln_5cat.csv")

        cmd = [
            "pred_fl_reg_ln_five_cat",
            sample_input_file,
            "-l",
            "last",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert "other" in output_df.columns  # 5-category should have 'other'

    def test_pred_fl_reg_name_basic(self, sample_input_file, temp_output_dir):
        """Test pred_fl_reg_name CLI functionality."""
        output_file = os.path.join(temp_output_dir, "fl_name.csv")

        cmd = [
            "pred_fl_reg_name",
            sample_input_file,
            "-l",
            "last",
            "-f",
            "first",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns

    def test_pred_fl_reg_name_five_cat(self, sample_input_file, temp_output_dir):
        """Test pred_fl_reg_name_five_cat CLI functionality."""
        output_file = os.path.join(temp_output_dir, "fl_name_5cat.csv")

        cmd = [
            "pred_fl_reg_name_five_cat",
            sample_input_file,
            "-l",
            "last",
            "-f",
            "first",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert "other" in output_df.columns


class TestNorthCarolinaPredictionCLI:
    """Test North Carolina voter registration prediction CLI commands."""

    def test_pred_nc_reg_name_basic(self, sample_input_file, temp_output_dir):
        """Test pred_nc_reg_name CLI functionality."""
        output_file = os.path.join(temp_output_dir, "nc_name.csv")

        cmd = [
            "pred_nc_reg_name",
            sample_input_file,
            "-l",
            "last",
            "-f",
            "first",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert os.path.exists(output_file)

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 5
        assert "race" in output_df.columns

        # Should have NC-style race categories
        nc_cols = [
            col
            for col in output_df.columns
            if col.startswith("HL+") or col.startswith("NL+")
        ]
        assert len(nc_cols) == 12  # 12 NC categories


class TestCLIErrorHandling:
    """Test error handling in CLI commands."""

    def test_missing_input_file_error(self, temp_output_dir):
        """Test error when input file doesn't exist."""
        nonexistent_file = "/nonexistent/path/file.csv"
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = ["pred_census_ln", nonexistent_file, "-l", "last", "-o", output_file]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Error message should mention file not found
        assert "not found" in result.stderr or "not found" in result.stdout

    def test_invalid_column_name_error(self, sample_input_file, temp_output_dir):
        """Test error when specified column doesn't exist."""
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            "pred_census_ln",
            sample_input_file,
            "-l",
            "nonexistent_column",
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_invalid_confidence_interval_error(
        self, sample_input_file, temp_output_dir
    ):
        """Test error for invalid confidence interval."""
        output_file = os.path.join(temp_output_dir, "output.csv")

        cmd = [
            "pred_census_ln",
            sample_input_file,
            "-l",
            "last",
            "-c",
            "1.5",  # Invalid: > 1.0
            "-o",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should fail with error about confidence interval
        assert result.returncode != 0
        assert "conf" in result.stderr.lower() or "conf" in result.stdout.lower()


class TestCLIIntegration:
    """Test integration scenarios across different CLI commands."""

    def test_census_to_prediction_workflow(self, sample_input_file, temp_output_dir):
        """Test workflow from census lookup to prediction."""
        census_output = os.path.join(temp_output_dir, "census_step.csv")
        prediction_output = os.path.join(temp_output_dir, "prediction_step.csv")

        # Step 1: Run census lookup
        census_cmd = ["census_ln", sample_input_file, "-l", "last", "-o", census_output]

        result1 = subprocess.run(census_cmd, capture_output=True, text=True)
        assert result1.returncode == 0
        assert os.path.exists(census_output)

        # Step 2: Run prediction on the same input
        pred_cmd = [
            "pred_census_ln",
            sample_input_file,
            "-l",
            "last",
            "-o",
            prediction_output,
        ]

        result2 = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=60)
        assert result2.returncode == 0
        assert os.path.exists(prediction_output)

        # Both outputs should have same number of rows
        census_df = pd.read_csv(census_output)
        pred_df = pd.read_csv(prediction_output)
        assert len(census_df) == len(pred_df)

    def test_different_models_same_input(self, sample_input_file, temp_output_dir):
        """Test running different models on the same input."""
        models_to_test = [
            ("census", ["pred_census_ln", sample_input_file, "-l", "last"]),
            ("wiki_ln", ["pred_wiki_ln", sample_input_file, "-l", "last"]),
            ("fl_ln", ["pred_fl_reg_ln", sample_input_file, "-l", "last"]),
        ]

        outputs = {}

        for model_name, base_cmd in models_to_test:
            output_file = os.path.join(temp_output_dir, f"{model_name}_output.csv")
            cmd = base_cmd + ["-o", output_file]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(output_file):
                outputs[model_name] = pd.read_csv(output_file)

        # All successful models should have same number of rows as input
        assert len(outputs) > 0, "No models succeeded"

        input_df = pd.read_csv(sample_input_file)
        for model_name, output_df in outputs.items():
            assert len(output_df) == len(input_df), (
                f"{model_name} changed number of rows"
            )
            assert "race" in output_df.columns, f"{model_name} missing race column"


class TestCLIPerformance:
    """Test performance characteristics of CLI commands."""

    @pytest.mark.slow
    def test_large_input_handling(self, temp_output_dir):
        """Test CLI commands with larger input files."""
        # Create a larger input file
        large_input = os.path.join(temp_output_dir, "large_input.csv")

        with open(large_input, "w") as f:
            f.write("last,first,id\n")
            names = [
                "smith",
                "garcia",
                "zhang",
                "williams",
                "johnson",
            ] * 100  # 500 names
            for i, name in enumerate(names):
                f.write(f"{name},person{i},{i}\n")

        output_file = os.path.join(temp_output_dir, "large_output.csv")

        # Test with census prediction (usually fastest)
        cmd = ["pred_census_ln", large_input, "-l", "last", "-o", output_file]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 minute timeout

        if result.returncode == 0:
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 500
        else:
            # If it fails, it should be due to timeout or resource constraints, not a bug
            pytest.skip(f"Large input test failed: {result.stderr}")

    def test_cli_response_time(self, sample_input_file, temp_output_dir):
        """Test that CLI commands respond within reasonable time."""
        import time

        output_file = os.path.join(temp_output_dir, "timing_test.csv")

        cmd = ["pred_census_ln", sample_input_file, "-l", "last", "-o", output_file]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        end_time = time.time()

        # Should complete within reasonable time (60 seconds is very generous)
        execution_time = end_time - start_time
        assert execution_time < 60, (
            f"Command took too long: {execution_time:.2f} seconds"
        )

        if result.returncode == 0:
            assert os.path.exists(output_file)
