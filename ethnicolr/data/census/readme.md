## Census Last Name Data

The Census Bureau provides frequency of all surnames occurring 100 or more times for the [2000](http://www.census.gov/topics/population/genealogy/data/2000_surnames.html) and [2010](http://www.census.gov/topics/population/genealogy/data/2010_surnames.html) census. Technical details of how the 2000 and 2010 data were collected can be found [here (pdf)](census_2000.pdf) and [here (pdf)](census_2000.pdf) respectively. 

In the census data, for names with a count of 1--4, the counts are suppressed and replaced with the '(S)'. We replaced with '(S)' within a row by equally dividing the remaining percentage (100 minus the rest) across all the '(S).' For details, see the [R script](census.R).

