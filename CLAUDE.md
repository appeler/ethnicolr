# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About ethnicolr

ethnicolr is a Python package that predicts race and ethnicity from names using machine learning models trained on US Census data, Florida voter registration data, and Wikipedia data. The package provides both command-line utilities and Python APIs for race/ethnicity prediction.

## Development Commands

### Testing
- Run all tests: `pytest`
- Run tests with coverage: `pytest --cov=ethnicolr`
- Run specific test file: `pytest tests/test_010_census_ln.py`

### Code Quality
- Format code: `ruff format .`
- Fix import sorting and basic issues: `ruff check . --fix`
- Lint code (check only): `ruff check .`
- All quality checks: `ruff format . && ruff check . --fix && ruff check .`

### Installation
- Install package in development mode: `pip install -e .`
- Install with optional dependencies: `pip install -e .[dev,test]`
- For macOS: `pip install -e .[macos]` or `uv sync --group macos`
- For Linux: `pip install -e .[linux]` or `uv sync --group linux`
- For Windows: `pip install -e .[windows]` or `uv sync --group windows`

### Documentation
- Build documentation locally: `pip install -e .[docs]` then `cd docs && make html`
- View built docs: Open `docs/build/html/index.html` in browser
- Documentation is automatically deployed to GitHub Pages on pushes to main
- Live documentation: https://appeler.github.io/ethnicolr/
- Documentation configuration reads metadata from `pyproject.toml` automatically

## Package Architecture

### Core Modules
- `ethnicolr/__init__.py` - Main package imports and public API
- `ethnicolr/utils.py` - Common argument parsing utilities for CLI tools
- `ethnicolr/census_ln.py` - Census data lookup by last name
- `ethnicolr/pred_*.py` - Various prediction models:
  - `pred_census_ln.py` - Census-based last name predictions
  - `pred_wiki_*.py` - Wikipedia-based predictions (name/last name)
  - `pred_fl_reg_*.py` - Florida voter registration based predictions
  - `pred_nc_reg_name.py` - North Carolina voter registration predictions

### Data and Models
- `ethnicolr/data/` - Training datasets (census, Wikipedia, voter registration)
- `ethnicolr/models/` - Pre-trained LSTM models stored as .h5 files and vocabulary CSV files
- Models are organized by source: `census/`, `wiki/`, `fl_voter_reg/`, `nc_voter_reg/`

### Command Line Interface
The package provides CLI commands defined in pyproject.toml:
- `census_ln` - Append census race probabilities by last name
- `pred_census_ln` - Predict race using census LSTM model
- `pred_wiki_name` / `pred_wiki_ln` - Wikipedia-based predictions
- `pred_fl_reg_name` / `pred_fl_reg_ln` - Florida voter registration predictions
- `pred_fl_reg_*_five_cat` - 5-category Florida models
- `pred_nc_reg_name` - North Carolina predictions
- `ethnicolr_download_models` - Download model files

### Testing Structure
- Tests are in `tests/` with descriptive numeric prefixes
- Each major module has corresponding test files
- Tests use unittest framework with pandas DataFrame fixtures

## Key Dependencies

- **TensorFlow/Keras**: For LSTM model inference (version 2.13.x)
- **pandas**: Data manipulation and CSV I/O
- **numpy**: Numerical operations
- Models work with standard TensorFlow installations across all platforms (Windows, macOS, Linux)

## Important Notes

- Models work best with clean alphabetic names (remove titles, punctuation, non-ASCII)
- The package supports confidence intervals and uncertainty estimation via Monte Carlo sampling
- Different models predict at different granularities (4-category vs 13-category ethnicity)
- Census models predict: white, black, Asian, Hispanic
- Wikipedia models predict detailed ethnic categories
- Florida/NC models include regional variations

## Recent Improvements (2024)

### Enhanced Name Processing
- **Wikipedia models now preserve all input names** - no more silent dropping of problematic names
- Added normalization tracking columns: `name_normalized`, `name_normalized_clean`, `processing_status`
- Better handling of accented characters, punctuation, and special characters
- Improved logging shows exactly which names are skipped and why
- Expected improvement from 60-80% to 85-95% success rates for diverse name datasets

### New Output Columns
- `processing_status`: Shows if name was "processed", "skipped_empty_original", or "skipped_empty_after_normalization"
- `__name`: Full name used for processing
- Normalization tracking helps debug problematic names

This addresses issues with Canadian/international datasets containing accented names, titles, and special characters.

## Known Issues

### Protobuf Warnings with TensorFlow
You may see protobuf version warnings when using TensorFlow:
```
UserWarning: Protobuf gencode version 5.28.3 is exactly one major version older than the runtime version 6.31.1
```

This is a TensorFlow compatibility issue and does not affect functionality. To suppress these warnings:
```bash
export TF_CPP_MIN_LOG_LEVEL=3
```

Or in Python:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
```

## Model File Locations

Pre-trained models are stored in `ethnicolr/models/*/lstm/` directories:
- `.h5` files contain the neural network weights
- `.csv` files contain vocabulary mappings
- Models are loaded dynamically based on the prediction function used
