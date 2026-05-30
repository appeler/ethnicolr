# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
