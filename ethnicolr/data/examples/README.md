# Example Input Files

This directory contains example input files demonstrating the expected format for ethnicolr functions.

## Files

### input-with-header.csv
Sample CSV file with column headers showing the expected format for name data:
- Contains `last_name` and `first_name` columns
- Used in documentation examples
- Shows proper CSV format with comma separation
- Includes variety of surnames from different ethnic backgrounds

## Usage

This file is referenced in:
- README.md examples
- Documentation (docs/source/ethnicolr.rst)
- Streamlit web application
- Command-line tool examples

Example usage:
```bash
# Use with census lookup
census_ln -y 2010 -o output-census2010.csv -l last_name input-with-header.csv

# Use with Wikipedia prediction
pred_wiki_name -o output-wiki-pred-race.csv -l last_name -f first_name input-with-header.csv
```
