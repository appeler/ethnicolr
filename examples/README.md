# Examples Directory

This directory contains example files and tutorial notebooks for using the ethnicolr package.

## Files

### Input Data Example
- **`input-with-header.csv`** - Sample CSV file (100 rows) with `last_name` and `first_name` columns
  - Demonstrates proper input format for ethnicolr functions
  - Contains diverse names from different ethnic backgrounds
  - Used in documentation examples and tutorials

### Jupyter Notebooks
- **`ethnicolr_app_contrib2000.ipynb`** - Tutorial using 2000 Census data
- **`ethnicolr_app_contrib2010.ipynb`** - Tutorial using 2010 Census data
- **`ethnicolr_app_contrib20xx-census_ln.ipynb`** - Census last name predictions
- **`ethnicolr_app_contrib20xx-fl_reg.ipynb`** - Florida voter registration models
- **`ethnicolr_app_contrib20xx.ipynb`** - General usage examples

## Quick Start

### Download the Sample File
```bash
# Download sample input file
curl -O https://raw.githubusercontent.com/appeler/ethnicolr/master/examples/input-with-header.csv
```

### Census Data Lookup
```bash
# Append 2010 census demographics by last name
census_ln -y 2010 -o output-census2010.csv -l last_name input-with-header.csv
```

### Machine Learning Predictions
```bash
# Predict race/ethnicity using Wikipedia model
pred_wiki_name -o output-wiki-pred-race.csv -l last_name -f first_name input-with-header.csv

# Predict using Florida voter registration model
pred_fl_reg_name -o output-fl-pred-race.csv -l last_name -f first_name input-with-header.csv
```

### Python API Usage
```python
import pandas as pd
import ethnicolr

# Load sample data
df = pd.read_csv('input-with-header.csv')

# Census data lookup
result = ethnicolr.census_ln(df, 'last_name', year=2010)

# Machine learning prediction
result = ethnicolr.pred_wiki_name(df, 'last_name', 'first_name')
```

## Expected Output Columns

### Census Lookup (`census_ln`)
- `pctwhite`, `pctblack`, `pctapi`, `pctaian`, `pct2prace`, `pcthispanic`

### Wikipedia Model (`pred_wiki_name`)
- Detailed ethnic categories (e.g., `GreaterEuropean,WestEuropean,Hispanic`)
- Race prediction and confidence scores

### Florida Models (`pred_fl_reg_*`)
- 4-category: `white`, `black`, `asian`, `hispanic`
- 5-category: includes `other` category

See the Jupyter notebooks for detailed tutorials and explanations of each model's capabilities and output format.
