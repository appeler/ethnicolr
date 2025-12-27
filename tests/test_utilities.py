"""
Tests for utility functions in ethnicolr.utils module.
"""

import logging
import os
import tempfile
from unittest.mock import patch

import pytest

from ethnicolr.utils import arg_parser


class TestArgParser:
    """Test the command-line argument parser utility."""

    @pytest.fixture
    def temp_input_file(self):
        """Create a temporary input file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("last,first\nsmith,john\ngarcia,maria\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_basic_argument_parsing(self, temp_input_file):
        """Test basic argument parsing functionality."""
        argv = [temp_input_file, "-l", "lastname", "-o", "output.csv"]

        with patch("builtins.print"):  # Suppress output during test
            args = arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="default.csv",
                default_year=2010,
                year_choices=[2000, 2010],
                first=False,
            )

        assert args.input == temp_input_file
        assert args.output == "output.csv"
        assert args.last == "lastname"
        assert args.year == 2010  # default
        assert args.conf == 1.0  # default
        assert args.iter == 100  # default

    def test_argument_parsing_with_first_name(self, temp_input_file):
        """Test argument parsing when first name is required."""
        argv = [
            temp_input_file,
            "-l",
            "lastname",
            "-f",
            "firstname",
            "-o",
            "output.csv",
        ]

        with patch("builtins.print"):
            args = arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="default.csv",
                default_year=2010,
                year_choices=[2000, 2010],
                first=True,
            )

        assert args.input == temp_input_file
        assert args.last == "lastname"
        assert args.first == "firstname"

    def test_all_optional_arguments(self, temp_input_file):
        """Test parsing with all optional arguments specified."""
        argv = [
            temp_input_file,
            "-l",
            "surname",
            "-o",
            "predictions.csv",
            "-y",
            "2000",
            "-c",
            "0.9",
            "-i",
            "50",
        ]

        with patch("builtins.print"):
            args = arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="default.csv",
                default_year=2010,
                year_choices=[2000, 2010],
                first=False,
            )

        assert args.input == temp_input_file
        assert args.last == "surname"
        assert args.output == "predictions.csv"
        assert args.year == 2000
        assert args.conf == 0.9
        assert args.iter == 50

    def test_default_values(self, temp_input_file):
        """Test that default values are applied correctly."""
        argv = [temp_input_file, "-l", "lastname"]

        with patch("builtins.print"):
            args = arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="custom_default.csv",
                default_year=2000,
                year_choices=[2000, 2010],
                first=False,
            )

        assert args.output == "custom_default.csv"
        assert args.year == 2000
        assert args.conf == 1.0
        assert args.iter == 100

    def test_missing_input_file_error(self):
        """Test error when input file doesn't exist."""
        nonexistent_file = "/nonexistent/path/file.csv"
        argv = [nonexistent_file, "-l", "lastname"]

        with pytest.raises(SystemExit) as excinfo:
            with patch("builtins.print"):
                arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

        # Should exit with error message about file not found
        assert excinfo.type is SystemExit

    def test_invalid_confidence_interval_error(self, temp_input_file):
        """Test error for invalid confidence interval values."""
        test_cases = [
            [temp_input_file, "-l", "lastname", "-c", "0"],  # Zero
            [temp_input_file, "-l", "lastname", "-c", "-0.5"],  # Negative
            [temp_input_file, "-l", "lastname", "-c", "1.5"],  # Greater than 1
            [temp_input_file, "-l", "lastname", "-c", "2.0"],  # Much greater than 1
        ]

        for argv in test_cases:
            with pytest.raises(SystemExit):
                with patch("builtins.print"):
                    arg_parser(
                        argv=argv,
                        title="Test Tool",
                        default_out="default.csv",
                        default_year=2010,
                        year_choices=[2000, 2010],
                        first=False,
                    )

    def test_valid_confidence_interval_values(self, temp_input_file):
        """Test that valid confidence interval values are accepted."""
        valid_values = ["0.01", "0.5", "0.9", "0.95", "0.99", "1.0"]

        for conf_val in valid_values:
            argv = [temp_input_file, "-l", "lastname", "-c", conf_val]

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

            assert args.conf == float(conf_val)

    def test_invalid_iteration_count_error(self, temp_input_file):
        """Test error for invalid iteration counts."""
        invalid_iterations = ["0", "-1", "-10"]

        for iter_val in invalid_iterations:
            argv = [temp_input_file, "-l", "lastname", "-i", iter_val]

            with pytest.raises(SystemExit):
                with patch("builtins.print"):
                    arg_parser(
                        argv=argv,
                        title="Test Tool",
                        default_out="default.csv",
                        default_year=2010,
                        year_choices=[2000, 2010],
                        first=False,
                    )

    def test_valid_iteration_counts(self, temp_input_file):
        """Test that valid iteration counts are accepted."""
        valid_iterations = ["1", "10", "50", "100", "500", "1000"]

        for iter_val in valid_iterations:
            argv = [temp_input_file, "-l", "lastname", "-i", iter_val]

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

            assert args.iter == int(iter_val)

    def test_invalid_year_choice_error(self, temp_input_file):
        """Test error for invalid year choices."""
        argv = [temp_input_file, "-l", "lastname", "-y", "2020"]  # Not in choices

        with pytest.raises(SystemExit):
            with patch("builtins.print"):
                arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

    def test_valid_year_choices(self, temp_input_file):
        """Test that valid year choices are accepted."""
        year_choices = [2000, 2010, 2020]

        for year in year_choices:
            argv = [temp_input_file, "-l", "lastname", "-y", str(year)]

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=year_choices,
                    first=False,
                )

            assert args.year == year

    def test_missing_required_last_name_error(self, temp_input_file):
        """Test error when required last name argument is missing."""
        argv = [temp_input_file]  # Missing -l lastname

        with pytest.raises(SystemExit):
            with patch("builtins.print"):
                arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

    def test_missing_required_first_name_error(self, temp_input_file):
        """Test error when required first name argument is missing."""
        argv = [
            temp_input_file,
            "-l",
            "lastname",
        ]  # Missing -f firstname when first=True

        with pytest.raises(SystemExit):
            with patch("builtins.print"):
                arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=True,  # Requires first name
                )

    def test_help_message_generation(self, temp_input_file):
        """Test that help message can be generated without errors."""
        argv = [temp_input_file, "-l", "lastname", "--help"]

        with pytest.raises(SystemExit) as excinfo:
            with patch("builtins.print"):
                arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

        # Help should exit with code 0
        assert excinfo.value.code == 0

    def test_argument_types(self, temp_input_file):
        """Test that arguments are parsed to correct types."""
        argv = [
            temp_input_file,
            "-l",
            "lastname",
            "-y",
            "2010",
            "-c",
            "0.95",
            "-i",
            "200",
        ]

        with patch("builtins.print"):
            args = arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="default.csv",
                default_year=2000,
                year_choices=[2000, 2010],
                first=False,
            )

        # Check types
        assert isinstance(args.input, str)
        assert isinstance(args.output, str)
        assert isinstance(args.last, str)
        assert isinstance(args.year, int)
        assert isinstance(args.conf, float)
        assert isinstance(args.iter, int)

    def test_diverse_year_choices(self, temp_input_file):
        """Test argument parser with diverse year choice configurations."""
        test_configs = [
            ([2000], 2000),  # Single choice
            ([1990, 2000, 2010], 2000),  # Multiple choices
            ([2020], 2020),  # Modern year
        ]

        for year_choices, default_year in test_configs:
            argv = [temp_input_file, "-l", "lastname"]

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=default_year,
                    year_choices=year_choices,
                    first=False,
                )

            assert args.year == default_year
            assert args.year in year_choices

    def test_output_messages_format(self, temp_input_file, capsys):
        """Test that output messages are properly formatted."""
        argv = [temp_input_file, "-l", "lastname", "-o", "test.csv"]

        # Configure logging to output to stdout for capture
        logger = logging.getLogger("ethnicolr.utils")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            arg_parser(
                argv=argv,
                title="Test Tool",
                default_out="default.csv",
                default_year=2010,
                year_choices=[2000, 2010],
                first=False,
            )

            captured = capsys.readouterr()

            # Should contain success message and argument listing
            assert "INFO: Parsed arguments:" in captured.err
            assert "input:" in captured.err
            assert "output:" in captured.err
            assert "last:" in captured.err
        finally:
            # Clean up logging configuration
            logger.removeHandler(handler)
            logger.setLevel(logging.WARNING)


class TestArgParserEdgeCases:
    """Test edge cases and boundary conditions for arg_parser."""

    def test_very_long_file_paths(self):
        """Test handling of very long file paths."""
        # Create a temporary file with a reasonable path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("last,first\nsmith,john\n")
            temp_path = f.name

        try:
            argv = [
                temp_path,
                "-l",
                "a" * 100,
                "-o",
                "b" * 100,
            ]  # Long column/file names

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

            assert args.last == "a" * 100
            assert args.output == "b" * 100

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_special_characters_in_arguments(self):
        """Test handling of special characters in arguments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("last,first\nsmith,john\n")
            temp_path = f.name

        try:
            argv = [
                temp_path,
                "-l",
                "surname-with-dashes_and_underscores",
                "-o",
                "output-file_name.csv",
            ]

            with patch("builtins.print"):
                args = arg_parser(
                    argv=argv,
                    title="Test Tool",
                    default_out="default.csv",
                    default_year=2010,
                    year_choices=[2000, 2010],
                    first=False,
                )

            assert args.last == "surname-with-dashes_and_underscores"
            assert args.output == "output-file_name.csv"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
