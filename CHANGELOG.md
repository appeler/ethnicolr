# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-07-25

A statistical-rigor release: calibrated probabilities and formal uncertainty
across all models, first names in the census pipeline, academic name
dictionaries, and a global name-origin model. (#127, #128, #129)

### Added
- **Calibration + conformal layer** (#127). Every model ships a stats file
  with a fitted temperature (probabilities are now measured-calibrated, not
  assumed), the training class distribution, and split-conformal quantiles.
  - `prior=` on every prediction function reweights probabilities to a target
    population (`p_adj ∝ p·π_target/π_train`) — the base-rate fix for the
    class-balanced models and the name-likelihood step for BISG pipelines.
  - `coverage=` adds a conformal prediction set (`race_set`/`origin_set`) with
    empirically verified marginal coverage at 0.80/0.90/0.95.
  - Model cards and a statistical-principles guide document reference
    populations, calibration, weighting, and the conformal guarantee.
- **Census 2020 first names + dictionary estimators** (#128):
  - `census_fn` — first-name lookup against the Census 2020 first-name file
    (53,616 names; first such release since 1990).
  - `pred_census_name` — six-category first+last posterior via naive Bayes
    with LSTM fallback and a `basis` column ("Tyrone Smith" → ~90% Black).
  - `pred_voter_name` — five-category posterior from the CC0
    Rosenman-Olivella-Imai voter-file dictionaries (338k surnames).
  - `conf_int=` on `census_ln`/`census_fn` for exact Wilson score intervals.
- **`pred_wiki_origin`** (#129) — name → country-of-origin over 90 countries,
  trained on 3.6M Wikidata people (62% top-1 / 81% top-3; chance ≈ 1.1%).

### Changed
- NC calibration re-inflates deduplicated names to person frequency, so its
  guarantees describe a random person rather than a random unique name.

## [1.0.0] - 2026-07-25

Complete migration from TensorFlow to PyTorch, all models retrained, and a
refreshed Wikipedia/Wikidata training corpus. (#121, #122, #123, #124, #125)

### Added
- Reproducible Wikidata data pipeline (`scripts/data-acquisition/wiki/`):
  fetches ~4M people from the public QLever SPARQL endpoint and labels them
  via auditable country/ethnic-group mapping tables. The wiki models now
  train on 3.69M rows (25x the 2009-era dataset).
- Scripted acquisition for the Florida and North Carolina voter data
  (`scripts/data-acquisition/`, Dataverse token via `DATAVERSE_API_TOKEN`).
- One parameterized trainer (`scripts/model-training/train_name_lstm.py`)
  replacing the Keras notebooks; reports top-2/top-3 accuracy.
- `ETHNICOLR_DEVICE` environment variable (`cpu`/`cuda`/`mps`) for device
  selection; CUDA auto-selected when available, CPU otherwise.
- Python 3.13 support (`requires-python >=3.11,<3.14`, numpy >= 1.26 without
  the 2.0 cap).

### Changed
- Inference engine rewritten on PyTorch; models ship as `.pt` state dicts.
- All nine LSTM models retrained from their original data sources. Held-out
  accuracy: wiki_ln 0.78 (was 0.67), wiki_name 0.86 (was 0.71), fl_ln 0.81,
  fl_name 0.84, FL five-cat 0.59-0.63 (balanced), nc_name 0.57 (12-class).
- Monte Carlo dropout confidence intervals now work for every model
  (the old NC model shipped with dropout 0.0, so its intervals were
  degenerate zero-width).

### Fixed
- macOS mispredictions in CI: MPS is never auto-selected (virtualized Apple
  Silicon environments advertise MPS but return incorrect LSTM output).
- Florida five-category models silently reused whichever year variant
  (2017/2022) loaded first; the model cache is now keyed by model path.
- Vocabulary CSVs are quoted so n-grams with meaningful trailing spaces
  survive; a pre-commit hook had been stripping them.

### Removed
- TensorFlow, tensorflow-intel, and protobuf dependencies (the `inference`
  extra is gone; the base install is all you need).
- Legacy `.h5` models, Keras-era vocab files, and the Keras training
  notebooks.
- The stale model-download machinery: `ethnicolr/download.py`, the
  `ethnicolr models download`/`status` commands, and the broken
  `ethnicolr_download_models` entry point. All model files ship with the
  package.

## [0.22.0] - 2025-05-28

### Added
- **2020 Census surname data support** for `census_ln` function
  - 156,619 surnames with race/ethnicity percentages
  - Use with `census_ln(df, 'name', year=2020)`
  - CLI default year changed from 2010 to 2020

### Changed
- `census_ln` now accepts `year` parameter of 2000, 2010, or 2020

## [0.21.1] - 2024-12-27

### Changed
- Documentation improvements and deduplication
- Removed lazy loading for simpler imports
- Added pydoclint and pyright type checking

## [0.21.0] - 2024-12-03

### Changed
- Documentation consolidation
- TensorFlow compatibility fixes for Windows
- CI/CD improvements

## [0.20.0] - 2024-11-30

### Changed
- Windows TensorFlow support via tensorflow-intel
- CI workflow enhancements
- Dependency management improvements

## [0.19.0] - 2024-11-27

### Added
- Comprehensive Google-style docstrings throughout the codebase
- Enhanced documentation for all public APIs with Args/Returns/Raises sections
- Usage examples for main prediction functions
- Better error handling and validation in all modules

### Changed
- **BREAKING**: Dropped Python 3.10 support, now requires Python 3.11+
- Modernized codebase with latest Python 3.11+ features
- Updated from `pkg_resources` to `importlib.resources` for Python 3.11+ compatibility
- Migrated to UV build system from hatchling
- Standardized all dependency management to use UV
- Enhanced CI/CD workflows with cross-platform support (Windows, macOS, Linux)
- Updated GitHub Actions workflows to use UV consistently
- Improved type hints with `from __future__ import annotations`

### Fixed
- TensorFlow compatibility issues with `typing_extensions`
- Lazy imports implementation to avoid loading TensorFlow unnecessarily
- MyPy type checking errors in prediction modules
- Cross-platform dependency resolution and testing
- CI workflow branch triggers (master vs main)
- Model loading compatibility with newer Keras versions

### Technical
- Pinned exact working dependency versions: TensorFlow 2.16.2-2.17.0 + typing_extensions 4.4.0
- Added platform-specific dependency groups for reliable installation
- Enhanced name processing with better normalization tracking
- Improved logging and status reporting throughout prediction pipeline
- Updated ruff linting configuration for Python 3.11+

## [0.18.4] - Previous Release
- Previous functionality with Python 3.10+ support
