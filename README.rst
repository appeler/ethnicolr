ethnicolr: Predict Race and Ethnicity From Name
----------------------------------------------------

.. image:: https://travis-ci.com/appeler/ethnicolr.svg?branch=master
    :target: https://travis-ci.com/appeler/ethnicolr
.. image:: https://ci.appveyor.com/api/projects/status/u9fe72hn8nnhmaxt?svg=true
    :target: https://ci.appveyor.com/project/soodoku/ethnicolr-m6u1p
.. image:: https://img.shields.io/pypi/v/ethnicolr.svg
    :target: https://pypi.python.org/pypi/ethnicolr
.. image:: https://anaconda.org/soodoku/ethnicolr/badges/version.svg
    :target: https://anaconda.org/soodoku/ethnicolr/
.. image:: https://pepy.tech/badge/ethnicolr
    :target: https://pepy.tech/project/ethnicolr

We exploit the US census data, the Florida voting registration data, and 
the Wikipedia data collected by Skiena and colleagues, to predict race
and ethnicity based on first and last name or just the last name. The granularity 
at which we predict the race depends on the dataset. For instance, 
Skiena et al.' Wikipedia data is at the ethnic group level, while the 
census data we use in the model (the raw data has additional categories of 
Native Americans and Bi-racial) merely categorizes between Non-Hispanic Whites, 
Non-Hispanic Blacks, Asians, and Hispanics.

DIME Race
-----------
Data on race of all the people in the `DIME data <https://data.stanford.edu/dime>`__ 
is posted `here <http://dx.doi.org/10.7910/DVN/M5K7VR>`__ The underlying python scripts 
are posted `here <https://github.com/appeler/dime_race>`__ 

Caveats and Notes
-----------------------

If you picked a random individual with last name 'Smith' from the US in 2010  
and asked us to guess this person's race (measured as crudely as by the census),
the best guess would be based on what is available from the aggregated Census file. 
It is the Bayes Optimal Solution. So what good are last name only predictive models
for? A few things---if you want to impute ethnicity at a more granular level,
guess the race of people in different years (than when the census was conducted 
if some assumptions hold), guess the race of people in different countries (again if some 
assumptions hold), when names are slightly different (again with some assumptions), etc. 
The big benefit comes from when both the first name and last name is known.

Install
----------

We strongly recommend installing `ethnicolor` inside a Python virtual environment
(see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`__)

::

    pip install ethnicolr

Or ::
   
   conda install -c soodoku ethnicolr 

Notes:
 - The models are run and verified on Tensorflow 2x using Python 3.7 and 3.8 and lower will work. tf1x has been deprecated.
 - If you are installing on Windows, Theano installation typically needs admin. privileges on the shell.

General API
------------------

To see the available command line options for any function, please type in 
``<function-name> --help``

::

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
     -l LAST, --last LAST  Name or index location of column contains the last
                           name


Examples
----------

To append census data from 2010 to a `file without column headers <ethnicolr/data/input-without-header.csv>`__ and the first column carries the last name, use ``-l 0``

::

   census_ln -y 2010 -o output-census2010.csv -l 0 input-without-header.csv

To append census data from 2010 to a `file with column header in the first row <ethnicolr/data/input-with-header.csv>`__, specify the column name carrying last names using the ``-l`` option, keeping the rest the same:

::

   census_ln -y 2010 -o output-census2010.csv -l last_name input-with-header.csv   


To predict race/ethnicity using `Wikipedia full name model <ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb>`__, if the input file doesn't have any column headers, you must using ``-l`` and ``-f`` to specify the index of column carrying the last name and first name respectively (first column has index 0).

::

   pred_wiki_name -o output-wiki-pred-race.csv -l 0 -f 1 input-without-header.csv


And to predict race/ethnicity using `Wikipedia full name model <ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb>`__ for a file with column headers, you can specify the column name of last name and first name by using ``-l`` and ``-f`` flags respectively.

::

   pred_wiki_name -o output-wiki-pred-race.csv -l last_name -f first_name input-with-header.csv


Functions
----------

We expose 6 functions, each of which either take a pandas DataFrame or a CSV. If the CSV doesn't have a header,
we make some assumptions about where the data is

-  **census\_ln**

   -  Input: pandas DataFrame or CSV and a string or list of the name or
      location of the column containing the last name.

   -  What it does:

      -  Removes extra space.
      -  For names in the `census file <ethnicolr/data/census>`__, it appends relevant data.

   -  Options:

      -  year: 2000 or 2010
      -  if no year is given, data from the 2000 census is appended

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      pctwhite, pctblack, pctapi, pctaian, pct2prace, pcthispanic. See 
      `here <https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/census/census_2000.pdf>`__ for what the column names mean.

-  **pred\_census\_ln**

   -  Input: pandas DataFrame or CSV and string or list of the name or
      location of the column containing the last name.

   -  What it does:

      -  Removes extra space.
      -  Uses the `last name census 2000
         model <ethnicolr/models/ethnicolr_keras_lstm_census2000_ln.ipynb>`__
         or `last name census 2010
         model <ethnicolr/models/ethnicolr_keras_lstm_census2010_ln.ipynb>`__
         to predict the race and ethnicity.

   -  Options:

      -  year: 2000 or 2010

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), api (percentage chance asian),
      black, hispanic, white. 

-  **pred\_wiki\_ln**

   -  Input: pandas DataFrame or CSV and string or list of the name or
      location of the column containing the last name.

   -  What it does:

      -  Removes extra space.
      -  Uses the `last name wiki model <ethnicolr/models/ethnicolr_keras_lstm_wiki_ln.ipynb>`__
         to predict the race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (categorical variable --- category with the highest probability), 
      "Asian,GreaterEastAsian,EastAsian", "Asian,GreaterEastAsian,Japanese", 
      "Asian,IndianSubContinent", "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
      "GreaterEuropean,British","GreaterEuropean,EastEuropean", 
      "GreaterEuropean,Jewish","GreaterEuropean,WestEuropean,French",
      "GreaterEuropean,WestEuropean,Germanic","GreaterEuropean,WestEuropean,Hispanic",
      "GreaterEuropean,WestEuropean,Italian","GreaterEuropean,WestEuropean,Nordic"

-  **pred\_wiki\_name**

   -  Input: pandas DataFrame or CSV and string or list containing the name or
      location of the column containing the first name, last name, middle
      name, and suffix, if there. The first name and last name columns are
      required. If no middle name of suffix columns are there, it is
      assumed that there are no middle names or suffixes.

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name wiki
         model <ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb>`__ to predict the
         race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (categorical variable---category with the highest probability), 
      "Asian,GreaterEastAsian,EastAsian", "Asian,GreaterEastAsian,Japanese", 
      "Asian,IndianSubContinent", "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
      "GreaterEuropean,British","GreaterEuropean,EastEuropean", 
      "GreaterEuropean,Jewish","GreaterEuropean,WestEuropean,French",
      "GreaterEuropean,WestEuropean,Germanic","GreaterEuropean,WestEuropean,Hispanic",
      "GreaterEuropean,WestEuropean,Italian","GreaterEuropean,WestEuropean,Nordic"

-  **pred\_fl\_reg\_ln**

   -  Input: pandas DataFrame or CSV and string or list of the name or location
      of the column containing the last name.

   -  What it does?:

      -  Removes extra space, if there.
      -  Uses the `last name FL registration
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln.ipynb>`__ to predict the race
         and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance Asian),
      hispanic, nh_black, nh_white

-  **pred\_fl\_reg\_name**

   -  Input: pandas DataFrame or CSV and string or list containing the name or
      location of the column containing the first name, last name, middle
      name, and suffix, if there. The first name and last name columns are
      required. If no middle name of suffix columns are there, it is
      assumed that there are no middle names or suffixes.

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name FL
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_name.ipynb>`__ to predict the
         race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance Asian),
      hispanic, nh_black, nh_white

-  **pred\_fl\_reg\_ln\_five\_cat**

   -  Input: pandas DataFrame or CSV and string or list of the name or location
      of the column containing the last name.

   -  What it does?:

      -  Removes extra space, if there.
      -  Uses the `last name FL registration
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_5classes_100k.ipynb>`__ to predict the race
         and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, hispanic or other), asian (percentage chance Asian),
      hispanic, nh_black, nh_white, other

-  **pred\_fl\_reg\_name\_five\_cat**

   -  Input: pandas DataFrame or CSV and string or list containing the name or
      location of the column containing the first name, last name, middle
      name, and suffix, if there. The first name and last name columns are
      required. If no middle name of suffix columns are there, it is
      assumed that there are no middle names or suffixes.

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name FL
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_5classes_100k.ipynb>`__ to predict the
         race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, hispanic, or other), asian (percentage chance Asian),
      hispanic, nh_black, nh_white, other

-  **pred\_nc\_reg\_name**

   -  Input: pandas DataFrame or CSV and string or list containing the name or
      location of the column containing the first name, last name, middle
      name, and suffix, if there. The first name and last name columns are
      required. If no middle name of suffix columns are there, it is
      assumed that there are no middle names or suffixes.

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name NC
         model <ethnicolr/models/ethnicolr_keras_lstm_nc_12_cat_model.ipynb>`__ to predict the
         race and ethnicity.

   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race + ethnicity. The codebook is `here <https://github.com/appeler/nc_race_ethnicity>`__


Using ethnicolr
----------------

::

   >>> import pandas as pd

   >>> from ethnicolr import census_ln, pred_census_ln
   Using TensorFlow backend.

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

   >>> help(pred_census_ln)
   Help on function pred_census_ln in module ethnicolr.pred_census_ln:

   pred_census_ln(df, namecol, year=2000)
       Predict the race/ethnicity by the last name using Census model.

       Using the Census last name model to predict the race/ethnicity of the input
       DataFrame.

       Args:
           df (:obj:`DataFrame`): Pandas DataFrame containing the last name
               column.
           namecol (str or int): Column's name or location of the name in
               DataFrame.
           year (int): The year of Census model to be used. (2000 or 2010)
               (default is 2000)

       Returns:
           DataFrame: Pandas DataFrame with additional columns:
               - `race` the predict result
               - `black`, `api`, `white`, `hispanic` are the prediction
                   probability.

Application
--------------

To illustrate how the package can be used, we impute the race of the campaign contributors recorded by FEC for the years 2000 and 2010 and tally campaign contributions by race.

- `Contrib 2000/2010 using census_ln <ethnicolr/examples/ethnicolr_app_contrib20xx-census_ln.ipynb>`__
- `Contrib 2000/2010 using pred_census_ln <ethnicolr/examples/ethnicolr_app_contrib20xx.ipynb>`__
- `Contrib 2000/2010 using pred_fl_reg_name <ethnicolr/examples/ethnicolr_app_contrib20xx-fl_reg.ipynb>`__

Data on race of all the people in the `DIME data <https://data.stanford.edu/dime>`__ is posted `here <http://dx.doi.org/10.7910/DVN/M5K7VR>`__ The underlying python scripts are posted `here <https://github.com/appeler/dime_race>`__ 

Data
----------

In particular, we utilize the last-name--race data from the `2000
census <http://www.census.gov/topics/population/genealogy/data/2000_surnames.html>`__
and `2010
census <http://www.census.gov/topics/population/genealogy/data/2010_surnames.html>`__,
the `Wikipedia data <ethnicolr/data/wiki/>`__ collected by Skiena and colleagues,
and the Florida voter registration data from early 2017.

-  `Census <ethnicolr/data/census/>`__
-  `The Wikipedia dataset <ethnicolr/data/wiki/>`__
-  `Florida voter registration database <http://dx.doi.org/10.7910/DVN/UBIG3F>`__

Evaluation
------------------------------------------
1. SCAN Health Plan, a Medicare Advantage plan that serves over 200,000 members throughout California used the software to better assess racial disparities of health among the people they serve. They only had racial data on about 47% of their members so used it to learn the race of the remaining 53%. On the data they had labels for, they found .9 AUC and 83% accuracy for the last name model.

2. Evaluation on NC Data: https://github.com/appeler/nc_race_ethnicity

Authors
----------

Suriyan Laohaprapanon and Gaurav Sood

Contributor Code of Conduct
---------------------------------

The project welcomes contributions from everyone! In fact, it depends on
it. To maintain this welcoming atmosphere, and to collaborate in a fun
and productive way, we expect contributors to the project to abide by
the `Contributor Code of
Conduct <http://contributor-covenant.org/version/1/0/0/>`__.

License
----------

The package is released under the `MIT
License <https://opensource.org/licenses/MIT>`__.
