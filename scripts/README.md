# Scripts Directory

This directory contains scripts and notebooks for data acquisition and model training for the ethnicolr package.

## Structure

### data-acquisition/
Contains scripts for collecting and processing raw data from various sources:

- **census/**: Census surname data processing
  - `census.R`: R script for processing Census surname data
  - `*.pdf`, `*.xlsx`: Raw Census documentation and data files
  - **hispanic_ln/**: Hispanic surname data processing
    - `hispanic_ln.ipynb`: Jupyter notebook for processing Census Hispanic surnames list
    - Raw CSV and PDF files from Census Appendix E

- **fl_voter_reg/**: Florida voter registration data processing
  - `fl_voter_name_race.ipynb`: Notebook for processing Florida voter data

- **nc_voter_reg/**: North Carolina voter registration data processing
  - `nc_voter_name_race_ethic.ipynb`: Notebook for processing NC voter data

- **wiki/**: Wikipedia data processing
  - `wikilabels-name-race.py`: Python script for processing Wikipedia name-race data
  - `WikiLabels.tar.gz`: Raw Wikipedia data archive
  - Documentation PDF files

### model-training/
Contains Jupyter notebooks for training LSTM models organized by data source:

- **census/**: Census-based model training notebooks
- **florida/**: Florida voter registration model training notebooks
- **north_carolina/**: North Carolina voter registration model training notebooks
- **wikipedia/**: Wikipedia-based model training notebooks

## Usage

These scripts are development tools and are not part of the main ethnicolr package. They are used to:

1. **Data Acquisition**: Download and process raw data from various sources
2. **Model Training**: Train new LSTM models when new data becomes available
3. **Research**: Explore new data sources and model architectures

## Note

The processed data files and trained models from these scripts are stored in the main `ethnicolr/` package directories:
- Runtime data: `ethnicolr/data/`
- Trained models: `ethnicolr/models/`
