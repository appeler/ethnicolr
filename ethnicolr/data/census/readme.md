## Census Last Name Data

The Census Bureau provides frequency of all surnames occurring 100 or more times for the [2000](http://www.census.gov/topics/population/genealogy/data/2000_surnames.html) and [2010](http://www.census.gov/topics/population/genealogy/data/2010_surnames.html) census. Technical details of how the 2000 and 2010 data were collected can be found in the data acquisition scripts (`../../scripts/data-acquisition/census/`).

In the census data, for names with a count of 1--4, the counts are suppressed and replaced with the '(S)'. We replaced with '(S)' within a row by equally dividing the remaining percentage (100 minus the rest) across all the '(S).' For details, see the R script in the data acquisition directory.

## Files

- `census_2000.csv`: Processed 2000 Census surname data with demographic percentages
- `census_2010.csv`: Processed 2010 Census surname data with demographic percentages
- `census_hispanic_ln.csv`: List of 12,498 Hispanic surnames from Census Appendix E

## Data Processing

The raw census data and processing scripts have been moved to `../../scripts/data-acquisition/census/` to separate development tools from runtime package data.
