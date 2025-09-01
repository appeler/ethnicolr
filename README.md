## ethnicolr: Predict Race and Ethnicity From Name

![PyPI Authenicated](https://notarypy.soodoku.workers.dev/badge/ethnicolr/0.18.3/ethnicolr-0.18.3-py3-none-any.whl)
![Test Badge](https://github.com/appeler/ethnicolr/workflows/test/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/ethnicolr.svg)](https://pypi.python.org/pypi/ethnicolr)
[![Anaconda version](https://anaconda.org/soodoku/ethnicolr/badges/version.svg)](https://anaconda.org/soodoku/ethnicolr/)
[![PePy Downloads](https://static.pepy.tech/badge/ethnicolr)](https://www.pepy.tech/projects/ethnicolr)

We exploit the US census data, the Florida voting registration data, and
the Wikipedia data collected by Skiena and colleagues to predict race
and ethnicity based on first and last name or just the last name. The
granularity at which we predict the race depends on the dataset. For
instance, Skiena et al.\' Wikipedia data is at the ethnic group level,
while the census data we use in the model (the raw data has additional
categories of Native Americans and Bi-racial) merely categorizes between
Non-Hispanic Whites, Non-Hispanic Blacks, Asians, and Hispanics.

### New Package With New Models in Pytorch

[https://github.com/appeler/ethnicolr2](https://github.com/appeler/ethnicolr2)

### Streamlit App

[https://ethnicolr.streamlit.app/](https://ethnicolr.streamlit.app/)

### Caveats and Notes

If you picked a person at random with the last name \'Smith\' in the US
in 2010 and asked us to guess this person\'s race (as measured by the
census), the best guess would be based on what is available from the
aggregated Census file. It is the Bayes Optimal Solution. So what good
are last-name-only predictive models for? A few things\-\--if you want
to impute race and ethnicity for last names that are not in the census
file, infer the race and ethnicity in different years than when the
census was conducted (if some assumptions hold), infer the race of
people in different countries (if some assumptions hold), etc. The
biggest benefit comes in cases where both the first name and last name
are known.

### Install

We strongly recommend installing ethnicolr inside a Python virtual
environment (see [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

```bash
pip install ethnicolr
```

Notes:

> - The models are run and verified on TensorFlow 2.x using Python 3.10
>   through 3.12
> - If you install on Windows, Theano installation typically needs
>   admin. privileges on the shell.

### Jupyter Quickstart

```bash
pip install ethnicolr jupyter
ethnicolr_download_models
jupyter notebook ethnicolr/examples
```

Open one of the example notebooks and run the cells to see the package in
action.

## General API

To see the available command line options for any function, please type
in [`<function-name>`]` `[`--help`]

```python
# census_ln --help
usage: census_ln [-h] [-y {2000,2010}] [-o OUTPUT] -l LAST input

Appends Census columns by last name

positional arguments:
  input                 Input file

optional arguments:
  -h, --help            show this help message and exit
  -y {2000,2010}, --year {2000,2010}
                        Year of Census data (default=2000)
  -o OUTPUT, --output OUTPUT
                        Output file with Census data columns
  -l LAST, --last LAST  Name of the column containing the last name
```

### Cleaning Names

The prediction models work best when first and last names contain only
alphabetic characters. Before calling the CLI or Python APIs, strip out
titles (e.g., *Dr*, *Hon.*), middle names, suffixes, punctuation and
non\-ASCII characters. The `pred_wiki_name` command automatically
normalizes names by removing diacritics and extraneous characters. If
the tool still skips entries, check that the first and last name columns
are not empty after cleaning.

## Examples

To append census data from 2010 to a [file with column header in the
first row](ethnicolr/data/input-with-header.csv),
specify the column name carrying last names using the [`-l`] option, keeping the rest the same:

```bash
census_ln -y 2010 -o output-census2010.csv -l last_name input-with-header.csv
```

To predict race/ethnicity using [Wikipedia full name
model](ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb), specify the column name of last name and first name by using
[`-l`] and [`-f`]
flags respectively.

```bash
pred_wiki_name -o output-wiki-pred-race.csv -l last_name -f first_name input-with-header.csv
```

## Functions

We expose several functions, each of which either takes a pandas DataFrame
or a CSV.

- **census_ln(df, lname_col, year=2000)**
  - What it does:
    - Removes extra space
    - For names in the [census file](ethnicolr/data/census), it appends relevant data of what probability the name
      provided is of a certain race/ethnicity

> -----------------------------------------------------------------------------
>   Parameters    
>   ------------ ----------------------------------------------------------------
>                **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
>                contains the names of the individual to be inferred
>
>                **lname_col** : *{string}* name of the column containing the
>                last name
>
>                **Year** : *{2000, 2010}, default=2000* year of census to use
>   -----------------------------------------------------------------------------

- Output: Appends the following columns to the pandas DataFrame or CSV:
  pctwhite, pctblack, pctapi, pctaian, pct2prace, pcthispanic. See
  [here](https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/census/census_2000.pdf) for what the column names mean.

  ``` literal-block
  >>> import pandas as pd

  >>> from ethnicolr import census_ln, pred_census_ln

  >>> names = [{'name': 'smith'},
  ...         {'name': 'zhang'},
  ...         {'name': 'jackson'}]

  >>> df = pd.DataFrame(names)

  >>> df
        name
  0    smith
  1    zhang
  2  jackson

  >>> census_ln(df, 'name')
        name pctwhite pctblack pctapi pctaian pct2prace pcthispanic
  0    smith    73.35    22.22   0.40    0.85      1.63        1.56
  1    zhang     0.61     0.09  98.16    0.02      0.96        0.16
  2  jackson    41.93    53.02   0.31    1.04      2.18        1.53
  ```

- **pred_census_ln(df, lname_col, year=2000, num_iter=100,
  conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [last name census 2000
      model](ethnicolr/models/ethnicolr_keras_lstm_census2000_ln.ipynb) or [last name census 2010
      model](ethnicolr/models/ethnicolr_keras_lstm_census2010_ln.ipynb) to predict race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **namecol** : *{string}* name of the column containing the last
                 name

                 **year** : *{2000, 2010}, default=2000* year of census to use

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty in model

                 **conf_int** : *float, default=1.0* confidence interval in
                 predicted class
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (white, black, asian, or hispanic), api (percentage chance
    asian), black, hispanic, white. For each race it will provide the
    mean, standard error, lower & upper bound of confidence interval

  *(Using the same dataframe from example above)*

  ```python
  >>> census_ln(df, 'name')
        name pctwhite pctblack pctapi pctaian pct2prace pcthispanic
  0    smith    73.35    22.22   0.40    0.85      1.63        1.56
  1    zhang     0.61     0.09  98.16    0.02      0.96        0.16
  2  jackson    41.93    53.02   0.31    1.04      2.18        1.53

  >>> census_ln(df, 'name', 2010)
        name   race pctwhite pctblack pctapi pctaian pct2prace pcthispanic
  0    smith  white     70.9    23.11    0.5    0.89      2.19         2.4
  1    zhang    api     0.99     0.16  98.06    0.02      0.62        0.15
  2  jackson  black    39.89    53.04   0.39    1.06      3.12         2.5

  >>> pred_census_ln(df, 'name')
        name   race       api     black  hispanic     white
  0    smith  white  0.002019  0.247235  0.014485  0.736260
  1    zhang    api  0.997807  0.000149  0.000470  0.001574
  2  jackson  black  0.002797  0.528193  0.014605  0.454405
  ```

- **pred_wiki_ln( df, lname_col, num_iter=100, conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [last name wiki
      model](ethnicolr/models/ethnicolr_keras_lstm_wiki_ln.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **lname_col** : *{string}* name of the column containing the
                 last name

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty in model

                 **conf_int** : *float, default=1.0* confidence interval in
                 predicted class
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (categorical variable \-\-- category with the highest
    probability). For each race it will provide the mean, standard
    error, lower & upper bound of confidence interval

  ``` literal-block
  "Asian,GreaterEastAsian,EastAsian",
  "Asian,GreaterEastAsian,Japanese", "Asian,IndianSubContinent",
  "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
  "GreaterEuropean,British","GreaterEuropean,EastEuropean",
  "GreaterEuropean,Jewish","GreaterEuropean,WestEuropean,French",
  "GreaterEuropean,WestEuropean,Germanic","GreaterEuropean,WestEuropean,Hispanic",
  "GreaterEuropean,WestEuropean,Italian","GreaterEuropean,WestEuropean,Nordic".
  ```

  ```python
  >>> import pandas as pd

  >>> names = [
  ...             {"last": "smith", "first": "john", "true_race": "GreaterEuropean,British"},
  ...             {
  ...                 "last": "zhang",
  ...                 "first": "simon",
  ...                 "true_race": "Asian,GreaterEastAsian,EastAsian",
  ...             },
  ...         ]
  >>> df = pd.DataFrame(names)

  >>> from ethnicolr import pred_wiki_ln, pred_wiki_name

  >>> odf = pred_wiki_ln(df,'last', conf_int=0.9)
  ['Asian,GreaterEastAsian,EastAsian', 'Asian,GreaterEastAsian,Japanese', 'Asian,IndianSubContinent', 'GreaterAfrican,Africans', 'GreaterAfrican,Muslim', 'GreaterEuropean,British', 'GreaterEuropean,EastEuropean', 'GreaterEuropean,Jewish', 'GreaterEuropean,WestEuropean,French', 'GreaterEuropean,WestEuropean,Germanic', 'GreaterEuropean,WestEuropean,Hispanic', 'GreaterEuropean,WestEuropean,Italian', 'GreaterEuropean,WestEuropean,Nordic']

  >>> odf
     last  first                         true_race  ...  GreaterEuropean,WestEuropean,Nordic_lb  GreaterEuropean,WestEuropean,Nordic_ub                              race
  0  Smith   john           GreaterEuropean,British                               0.016103  ...                                 0.014135                                0.007382                                0.048828           GreaterEuropean,British
  1  Zhang  simon  Asian,GreaterEastAsian,EastAsian                               0.863391  ...                                 0.017452                                0.001844                                0.027252  Asian,GreaterEastAsian,EastAsian

  [2 rows x 56 columns]

  >>> odf.iloc[0, :8]
  last                                                       Smith
  first                                                       john
  true_race                                GreaterEuropean,British
  Asian,GreaterEastAsian,EastAsian_mean                   0.016103
  Asian,GreaterEastAsian,EastAsian_std                    0.009735
  Asian,GreaterEastAsian,EastAsian_lb                     0.005873
  Asian,GreaterEastAsian,EastAsian_ub                     0.034637
  Asian,GreaterEastAsian,Japanese_mean                    0.003814
  Name: 0, dtype: object
  ```

- **pred_wiki_name(df, namecol, num_iter=100, conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [full name wiki
      model](ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **namecol** : *{string}* name of the column containing the
                 name.

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty of predictions

                 **conf_int** : *float, default=1.0* confidence interval
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (categorical variable\-\--category with the highest
    probability), \"Asian,GreaterEastAsian,EastAsian\",
    \"Asian,GreaterEastAsian,Japanese\", \"Asian,IndianSubContinent\",
    \"GreaterAfrican,Africans\", \"GreaterAfrican,Muslim\",
    \"GreaterEuropean,British\",\"GreaterEuropean,EastEuropean\",
    \"GreaterEuropean,Jewish\",\"GreaterEuropean,WestEuropean,French\",
    \"GreaterEuropean,WestEuropean,Germanic\",\"GreaterEuropean,WestEuropean,Hispanic\",
    \"GreaterEuropean,WestEuropean,Italian\",\"GreaterEuropean,WestEuropean,Nordic\".
    For each race it will provide the mean, standard error, lower &
    upper bound of confidence interval

  *(Using the same dataframe from example above)*

  ``` literal-block
  >>> odf = pred_wiki_name(df,'last', 'first', conf_int=0.9)
  ['Asian,GreaterEastAsian,EastAsian', 'Asian,GreaterEastAsian,Japanese', 'Asian,IndianSubContinent', 'GreaterAfrican,Africans', 'GreaterAfrican,Muslim', 'GreaterEuropean,British', 'GreaterEuropean,EastEuropean', 'GreaterEuropean,Jewish', 'GreaterEuropean,WestEuropean,French', 'GreaterEuropean,WestEuropean,Germanic', 'GreaterEuropean,WestEuropean,Hispanic', 'GreaterEuropean,WestEuropean,Italian', 'GreaterEuropean,WestEuropean,Nordic']

  >>> odf
     last  first                         true_race       __name  Asian,GreaterEastAsian,EastAsian_mean  ...  GreaterEuropean,WestEuropean,Nordic_mean  GreaterEuropean,WestEuropean,Nordic_std  GreaterEuropean,WestEuropean,Nordic_lb  GreaterEuropean,WestEuropean,Nordic_ub                              race
  0  Smith   john           GreaterEuropean,British   Smith John                               0.004111  ...                                  0.006246                                 0.004760                                0.001048                                0.016288           GreaterEuropean,British
  1  Zhang  simon  Asian,GreaterEastAsian,EastAsian  Zhang Simon                               0.944203  ...                                  0.000793                                 0.002557                                0.000019                                0.002470  Asian,GreaterEastAsian,EastAsian

  [2 rows x 57 columns]

  >>> odf.iloc[0,:8]
  last                                                       Smith
  first                                                       john
  true_race                                GreaterEuropean,British
  __name                                                Smith John
  Asian,GreaterEastAsian,EastAsian_mean                   0.004111
  Asian,GreaterEastAsian,EastAsian_std                    0.002929
  Asian,GreaterEastAsian,EastAsian_lb                     0.001356
  Asian,GreaterEastAsian,EastAsian_ub                     0.010571
  Name: 0, dtype: object
  ```

- **pred_fl_reg_ln(df, lname_col, num_iter=100, conf_int=1.0)**

  - What does it do?:
    - Removes extra space, if there.
    - Uses the [last name FL registration
      model](ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **lname_col** : *{string}* name of the column containing the
                 last name

                 **num_iter** : *int, default=100* number of iterations to
                 calculate the uncertainty

                 **conf_int** : *float, default=1.0* confidence interval
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (white, black, asian, or Hispanic), asian (percentage
    chance Asian), Hispanic, nh_black, nh_white. For each race, it will
    provide the mean, standard error, lower & upper bound of confidence
    interval

  ```python
  >>> import pandas as pd

  >>> names = [
  ...             {"last": "sawyer", "first": "john", "true_race": "nh_white"},
  ...             {"last": "torres", "first": "raul", "true_race": "hispanic"},
  ...         ]

  >>> df = pd.DataFrame(names)

  >>> from ethnicolr import pred_fl_reg_ln, pred_fl_reg_name, pred_fl_reg_ln_five_cat, pred_fl_reg_name_five_cat

  >>> odf = pred_fl_reg_ln(df, 'last', conf_int=0.9)
  ['asian', 'hispanic', 'nh_black', 'nh_white']

  >>> odf
     last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  hispanic_lb  hispanic_ub  nh_black_mean  nh_black_std  nh_black_lb  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
  0  Sawyer  john  nh_white    0.009859   0.006819  0.005338  0.019673       0.021488      0.004602     0.014802     0.030148       0.180929      0.052784     0.105756     0.270238       0.787724      0.051082     0.705290     0.860286  nh_white
  1  Torres  raul  hispanic    0.006463   0.001985  0.003915  0.010146       0.878119      0.021998     0.839274     0.909151       0.013118      0.005002     0.007364     0.021633       0.102300      0.017828     0.075911     0.130929  hispanic

  [2 rows x 20 columns]

  >>> odf.iloc[0]
  last               Sawyer
  first                john
  true_race        nh_white
  asian_mean       0.009859
  asian_std        0.006819
  asian_lb         0.005338
  asian_ub         0.019673
  hispanic_mean    0.021488
  hispanic_std     0.004602
  hispanic_lb      0.014802
  hispanic_ub      0.030148
  nh_black_mean    0.180929
  nh_black_std     0.052784
  nh_black_lb      0.105756
  nh_black_ub      0.270238
  nh_white_mean    0.787724
  nh_white_std     0.051082
  nh_white_lb       0.70529
  nh_white_ub      0.860286
  race             nh_white
  Name: 0, dtype: object
  ```

- **pred_fl_reg_name(df, lname_col, num_iter=100, conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [full name FL
      model](ethnicolr/models/ethnicolr_keras_lstm_fl_voter_name.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **namecol** : *{list}* name of the column containing the name.

                 **num_iter** : *int, default=100* number of iterations to
                 calculate the uncertainty

                 **conf_int** : *float, default=1.0* confidence interval in
                 predicted class
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (white, black, asian, or Hispanic), asian (percentage
    chance Asian), Hispanic, nh_black, nh_white. For each race, it will
    provide the mean, standard error, lower & upper bound of confidence
    interval

  *(Using the same dataframe from example above)*

  ``` literal-block
  >>> odf = pred_fl_reg_name(df, 'last', 'first', conf_int=0.9)
  ['asian', 'hispanic', 'nh_black', 'nh_white']

  >>> odf
     last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  hispanic_lb  hispanic_ub  nh_black_mean  nh_black_std  nh_black_lb  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
  0  Sawyer  john  nh_white    0.001534   0.000850  0.000636  0.002691       0.006818      0.002557     0.003684     0.011660       0.028068      0.015095     0.011488     0.055149       0.963581      0.015738     0.935445     0.983224  nh_white
  1  Torres  raul  hispanic    0.005791   0.002906  0.002446  0.011748       0.890561      0.029581     0.841328     0.937706       0.011397      0.004682     0.005829     0.020796       0.092251      0.026675     0.049868     0.139210  hispanic

  >>> odf.iloc[1]
  last               Torres
  first                raul
  true_race        hispanic
  asian_mean       0.005791
  asian_std        0.002906
  asian_lb         0.002446
  asian_ub         0.011748
  hispanic_mean    0.890561
  hispanic_std     0.029581
  hispanic_lb      0.841328
  hispanic_ub      0.937706
  nh_black_mean    0.011397
  nh_black_std     0.004682
  nh_black_lb      0.005829
  nh_black_ub      0.020796
  nh_white_mean    0.092251
  nh_white_std     0.026675
  nh_white_lb      0.049868
  nh_white_ub       0.13921
  race             hispanic
  Name: 1, dtype: object
  ```

- **pred_fl_reg_ln_five_cat(df, namecol, num_iter=100, conf_int=1.0)**

  - What does it do?:
    - Removes extra space, if there.
    - Uses the [last name FL registration
      model](ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_five_cat.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **lname_col** : *{string, list, int}* name of location of the
                 column containing the last name

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty

                 **conf_int** : *float, default=1.0* confidence interval
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (white, black, asian, Hispanic or other), asian
    (percentage chance Asian), hispanic, nh_black, nh_white, other. For
    each race, it will provide the mean, standard error, lower & upper
    bound of confidence interval

  *(Using the same dataframe from example above)*

  ```python
  >>> odf = pred_fl_reg_ln_five_cat(df,'last')
  ['asian', 'hispanic', 'nh_black', 'nh_white', 'other']

  >>> odf
     last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  ...  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub  other_mean  other_std  other_lb  other_ub      race
  0  Sawyer  john  nh_white    0.100038   0.020539  0.073266  0.143334       0.044263      0.013077  ...       0.376639      0.048289     0.296989     0.452834    0.248466   0.021040  0.219721  0.283785  nh_white
  1  Torres  raul  hispanic    0.062390   0.021863  0.033837  0.103737       0.774414      0.043238  ...       0.030393      0.009591     0.019713     0.046483    0.117761   0.019524  0.089418  0.150615  hispanic

  [2 rows x 24 columns]

  >>> odf.iloc[0]
  last               Sawyer
  first                john
  true_race        nh_white
  asian_mean       0.100038
  asian_std        0.020539
  asian_lb         0.073266
  asian_ub         0.143334
  hispanic_mean    0.044263
  hispanic_std     0.013077
  hispanic_lb       0.02476
  hispanic_ub      0.067965
  nh_black_mean    0.230593
  nh_black_std     0.063948
  nh_black_lb      0.130577
  nh_black_ub      0.343513
  nh_white_mean    0.376639
  nh_white_std     0.048289
  nh_white_lb      0.296989
  nh_white_ub      0.452834
  other_mean       0.248466
  other_std         0.02104
  other_lb         0.219721
  other_ub         0.283785
  race             nh_white
  Name: 0, dtype: object
  ```

- **pred_fl_reg_name_five_cat(df, namecol, num_iter=100, conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [full name FL
      model](ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_five_cat.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **namecol** : *{string, list}* string or list of the name or
                 location of the column containing the first name, last name.

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty

                 **conf_int** : *float, default=1.0* confidence interval
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race (white, black, asian, Hispanic, or other), asian
    (percentage chance Asian), hispanic, nh_black, nh_white, other. For
    each race, it will provide the mean, standard error, lower & upper
    bound of confidence interval

  *(Using the same dataframe from example above)*

  ```python
  >>> odf = pred_fl_reg_name_five_cat(df, 'last','first')
  ['asian', 'hispanic', 'nh_black', 'nh_white', 'other']

  >>> odf
     last first true_race  asian_mean  asian_std  asian_lb  asian_ub  hispanic_mean  hispanic_std  ...  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub  other_mean  other_std  other_lb  other_ub      race
  0  Sawyer  john  nh_white    0.039310   0.011657  0.025982  0.059719       0.019737      0.005813  ...       0.650306      0.059327     0.553913     0.733201    0.192242   0.021004  0.160185  0.226063  nh_white
  1  Torres  raul  hispanic    0.020086   0.011765  0.008240  0.041741       0.899110      0.042237  ...       0.019073      0.009901     0.010166     0.040081    0.055774   0.017897  0.036245  0.088741  hispanic

  [2 rows x 24 columns]

  >>> odf.iloc[1]
  last               Torres
  first                raul
  true_race        hispanic
  asian_mean       0.020086
  asian_std        0.011765
  asian_lb          0.00824
  asian_ub         0.041741
  hispanic_mean     0.89911
  hispanic_std     0.042237
  hispanic_lb      0.823799
  hispanic_ub      0.937612
  nh_black_mean    0.005956
  nh_black_std     0.006528
  nh_black_lb      0.002686
  nh_black_ub      0.010134
  nh_white_mean    0.019073
  nh_white_std     0.009901
  nh_white_lb      0.010166
  nh_white_ub      0.040081
  other_mean       0.055774
  other_std        0.017897
  other_lb         0.036245
  other_ub         0.088741
  race             hispanic
  Name: 1, dtype: object
  ```

- **pred_nc_reg_name(df, namecol, num_iter=100, conf_int=1.0)**

  - What it does:
    - Removes extra space.
    - Uses the [full name NC
      model](ethnicolr/models/ethnicolr_keras_lstm_nc_12_cat_model.ipynb) to predict the race and ethnicity.

    ----------------------------------------------------------------------------
    Parameters    
    ------------ ---------------------------------------------------------------
                 **df** : *{DataFrame, csv}* Pandas dataframe of CSV file
                 contains the names of the individual to be inferred

                 **namecol** : *{string, list}* string or list of the name or
                 location of the column containing the first name and last name.

                 **num_iter** : *int, default=100* number of iterations to
                 calculate uncertainty

                 **conf_int** : *float, default=1.0* confidence interval
    ----------------------------------------------------------------------------

  - Output: Appends the following columns to the pandas DataFrame or
    CSV: race + ethnicity. The codebook is
    [here](https://github.com/appeler/nc_race_ethnicity). For each race, it will provide the mean, standard error,
    lower & upper bound of confidence interval

  ```python
  >>> import pandas as pd

  >>> names = [
  ...             {"last": "hernandez", "first": "hector", "true_race": "HL+O"},
  ...             {"last": "zhang", "first": "simon", "true_race": "NL+A"},
  ...         ]

  >>> df = pd.DataFrame(names)

  >>> from ethnicolr import pred_nc_reg_name

  >>> odf = pred_nc_reg_name(df, 'last','first', conf_int=0.9)
  ['HL+A', 'HL+B', 'HL+I', 'HL+M', 'HL+O', 'HL+W', 'NL+A', 'NL+B', 'NL+I', 'NL+M', 'NL+O', 'NL+W']

  >>> odf
        last   first true_race            __name     HL+A_mean  HL+A_std       HL+A_lb       HL+A_ub     HL+B_mean  HL+B_std       HL+B_lb       HL+B_ub  HL+I_mean  ...     NL+M_mean  NL+M_std       NL+M_lb       NL+M_ub  NL+O_mean  NL+O_std   NL+O_lb   NL+O_ub  NL+W_mean  NL+W_std   NL+W_lb   NL+W_ub  race
  0  hernandez  hector      HL+O  Hernandez Hector  2.727371e-13       0.0  2.727372e-13  2.727372e-13  6.542178e-04       0.0  6.542183e-04  6.542183e-04   0.000032  ...  7.863581e-06       0.0  7.863589e-06  7.863589e-06   0.184513       0.0  0.184514  0.184514   0.001256       0.0  0.001256  0.001256  HL+O
  1      zhang   simon      NL+A       Zhang Simon  1.985421e-06       0.0  1.985423e-06  1.985423e-06  8.708256e-09       0.0  8.708265e-09  8.708265e-09   0.000049  ...  1.446786e-07       0.0  1.446784e-07  1.446784e-07   0.003238       0.0  0.003238  0.003238   0.000154       0.0  0.000154  0.000154  NL+A

  [2 rows x 53 columns]

  >>> odf.iloc[0]
  last                hernandez
  first                  hector
  true_race                HL+O
  __name       Hernandez Hector
  HL+A_mean                 0.0
  HL+A_std                  0.0
  HL+A_lb                   0.0
  HL+A_ub                   0.0
  HL+B_mean            0.000654
  HL+B_std                  0.0
  HL+B_lb              0.000654
  HL+B_ub              0.000654
  HL+I_mean            0.000032
  HL+I_std                  0.0
  HL+I_lb              0.000032
  HL+I_ub              0.000032
  HL+M_mean            0.000541
  HL+M_std                  0.0
  HL+M_lb              0.000541
  HL+M_ub              0.000541
  HL+O_mean             0.58944
  HL+O_std                  0.0
  HL+O_lb               0.58944
  HL+O_ub               0.58944
  HL+W_mean            0.221309
  HL+W_std                  0.0
  HL+W_lb              0.221309
  HL+W_ub              0.221309
  NL+A_mean            0.000044
  NL+A_std                  0.0
  NL+A_lb              0.000044
  NL+A_ub              0.000044
  NL+B_mean            0.002199
  NL+B_std                  0.0
  NL+B_lb              0.002199
  NL+B_ub              0.002199
  NL+I_mean            0.000004
  NL+I_std                  0.0
  NL+I_lb              0.000004
  NL+I_ub              0.000004
  NL+M_mean            0.000008
  NL+M_std                  0.0
  NL+M_lb              0.000008
  NL+M_ub              0.000008
  NL+O_mean            0.184513
  NL+O_std                  0.0
  NL+O_lb              0.184514
  NL+O_ub              0.184514
  NL+W_mean            0.001256
  NL+W_std                  0.0
  NL+W_lb              0.001256
  NL+W_ub              0.001256
  race                     HL+O
  Name: 0, dtype: object
  ```

### Application

To illustrate how the package can be used, we impute the race of the
campaign contributors recorded by FEC for the years 2000 and 2010 and
tally campaign contributions by race.

- [Contrib 2000/2010 using
  census_ln](ethnicolr/examples/ethnicolr_app_contrib20xx-census_ln.ipynb)
- [Contrib 2000/2010 using
  pred_census_ln](ethnicolr/examples/ethnicolr_app_contrib20xx.ipynb)
- [Contrib 2000/2010 using
  pred_fl_reg_name](ethnicolr/examples/ethnicolr_app_contrib20xx-fl_reg.ipynb)
  
Data on race of all the people in the [DIME
data](https://data.stanford.edu/dime) is posted
[here](http://dx.doi.org/10.7910/DVN/M5K7VR). The
underlying Python scripts are posted
[here](https://github.com/appeler/dime_race)
# Data

In particular, we utilize the last-name\--race data from the [2000
census](http://www.census.gov/topics/population/genealogy/data/2000_surnames.html) and [2010
census](http://www.census.gov/topics/population/genealogy/data/2010_surnames.html), the [Wikipedia data](ethnicolr/data/wiki/) collected by Skiena and colleagues, and the Florida voter
registration data from early 2017.

- [Census](ethnicolr/data/census/)
- [The Wikipedia dataset](ethnicolr/data/wiki/)
- [Florida voter registration
  database](http://dx.doi.org/10.7910/DVN/UBIG3F)

### Evaluation

1.  SCAN Health Plan, a Medicare Advantage plan that serves over 200,000
    members throughout California used the software to better assess
    racial disparities of health among the people they serve. They only
    had racial data on about 47% of their members, so they used it to learn
    the race of the remaining 53%. On the data they had labels for, they
    found .9 AUC and 83% accuracy for the last name model.
    
3.  Evaluation on NC Data:
    [https://github.com/appeler/nc_race_ethnicity](https://github.com/appeler/nc_race_ethnicity)
    
### Authors

Suriyan Laohaprapanon and Gaurav Sood

### Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the [Contributor Code of
Conduct](http://contributor-covenant.org/version/1/0/0/)


## License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).


## 🔗 Adjacent Repositories

- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
- [appeler/ethnicolor](https://github.com/appeler/ethnicolor) — Race and Ethnicity based on name using data from census, voter reg. files, etc.
- [appeler/instate](https://github.com/appeler/instate) — instate: predict the state of residence from last name using the indian electoral rolls
- [appeler/search_names](https://github.com/appeler/search_names) — Search a long list of names (patterns) in a large text corpus systematically and quickly
- [appeler/nc_race_ethnicity](https://github.com/appeler/nc_race_ethnicity) — Evaluation of some of the ethnicolr models on the NC Voter Registration Data + New Models Based on NC Voter Registration Data.
