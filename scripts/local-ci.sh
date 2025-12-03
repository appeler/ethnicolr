#!/bin/bash
# Local CI script to check tests and linting before commits
# This mimics the GitHub Actions CI pipeline locally

set -e  # Exit on any error

echo "🔧 Local CI: Installing dependencies..."
uv sync --group test --group inference --group dev

echo "✅ Local CI: Verifying TensorFlow installation..."
uv run python -c "import tensorflow; print('✓ TensorFlow version:', tensorflow.__version__)"
uv run python -c "import pandas; print('✓ Pandas version:', pandas.__version__)"
uv run python -c "import numpy; print('✓ NumPy version:', numpy.__version__)"

echo "🔍 Local CI: Running linting checks..."
uv run ruff check .

echo "🔍 Local CI: Verifying ruff formatting is clean..."
uv run ruff format --check .

echo "🧪 Local CI: Running key validation tests..."
# Run representative tests that cover major functionality and edge cases
uv run pytest tests/test_wiki_prediction.py::TestWikiErrorHandling::test_empty_names -v
uv run pytest tests/test_north_carolina_prediction.py::TestNorthCarolinaErrorHandling::test_nc_empty_dataframe -v
uv run pytest tests/test_florida_prediction.py::TestFloridaErrorHandling::test_empty_dataframe -v
uv run pytest tests/test_census_prediction.py::TestCensusPrediction::test_basic_prediction -v
uv run pytest tests/test_cli_commands.py::TestCensusLnCLI::test_census_ln_basic_usage -v

echo "✅ Local CI checks passed!"
echo ""
echo "💡 To run full test suite: uv run pytest"
echo "💡 To check specific tests: uv run pytest tests/test_<module>.py"
