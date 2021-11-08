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

Or 

::
   
   conda install -c soodoku ethnicolr 

Notes:
 - The models are run and verified on TensorFlow 2.x using Python 3.7 and 3.8 and lower will work. TensorFlow 1.x has been deprecated.
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

We expose 6 functions, each of which either take a pandas DataFrame or a
CSV. If the CSV doesn't have a header, we make some assumptions about
where the data is:

- **census\_ln(df, namecol, year=2000)**

  -  What it does:

     - Removes extra space
     - For names in the `census file <ethnicolr/data/census>`__, it appends 
       relevant data of what probability the name provided is of a certain race/ethnicity


 +------------+--------------------------------------------------------------------------------------------------------------------------+
 | Parameters |                                                                                                                          |
 +============+==========================================================================================================================+
 |            | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred             |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name        |
 +------------+--------------------------------------------------------------------------------------------------------------------------+
 |            | **Year** : *{2000, 2010}, default=2000* year of census to use                                                            |
 +------------+--------------------------------------------------------------------------------------------------------------------------+


-  Output: Appends the following columns to the pandas DataFrame or CSV: 
   pctwhite, pctblack, pctapi, pctaian, pct2prace, pcthispanic. 
   See `here <https://github.com/appeler/ethnicolr/blob/master/ethnicolr/data/census/census_2000.pdf>`__ 
   for what the column names mean.

   ::

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


-  **pred\_census\_ln(df, namecol, year=2000, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `last name census 2000 
         model <ethnicolr/models/ethnicolr_keras_lstm_census2000_ln.ipynb>`__ or 
         `last name census 2010 model <ethnicolr/models/ethnicolr_keras_lstm_census2010_ln.ipynb>`__ 
         to predict the race and ethnicity.


   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name   |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **year** : *{2000, 2010}, default=2000* year of census to use                                                       |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                           |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                         |
   +--------------+---------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), api (percentage chance
      asian), black, hispanic, white. For each race it will provide the
      mean, standard error, lower & upper bound of confidence interval

   *(Using the same dataframe from example above)*
   ::

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


-  **pred\_wiki\_ln( df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `last name wiki
         model <ethnicolr/models/ethnicolr_keras_lstm_wiki_ln.ipynb>`__ to
         predict the race and ethnicity.


   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name   |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                           |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                         |
   +--------------+---------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (categorical variable --- category with the highest
      probability), "Asian,GreaterEastAsian,EastAsian",
      "Asian,GreaterEastAsian,Japanese", "Asian,IndianSubContinent",
      "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
      "GreaterEuropean,British","GreaterEuropean,EastEuropean",
      "GreaterEuropean,Jewish","GreaterEuropean,WestEuropean,French",
      "GreaterEuropean,WestEuropean,Germanic","GreaterEuropean,WestEuropean,Hispanic",
      "GreaterEuropean,WestEuropean,Italian","GreaterEuropean,WestEuropean,Nordic".
      For each race it will provide the mean, standard error, lower & upper
      bound of confidence interval

   ::

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

      >>> odf = pred_wiki_ln(df,'last')
      ['Asian,GreaterEastAsian,EastAsian', 'Asian,GreaterEastAsian,Japanese', 'Asian,IndianSubContinent', 'GreaterAfrican,Africans', 'GreaterAfrican,Muslim', 'GreaterEuropean,British', 'GreaterEuropean,EastEuropean', 'GreaterEuropean,Jewish', 'GreaterEuropean,WestEuropean,French', 'GreaterEuropean,WestEuropean,Germanic', 'GreaterEuropean,WestEuropean,Hispanic', 'GreaterEuropean,WestEuropean,Italian', 'GreaterEuropean,WestEuropean,Nordic']
      
      >>> odf
         last  first  ... GreaterEuropean,WestEuropean,Nordic_ub                              race
      0  Smith   john  ...                               0.004559           GreaterEuropean,British
      1  Zhang  simon  ...                               0.004076  Asian,GreaterEastAsian,EastAsian

      [2 rows x 57 columns]

      >>> odf.iloc[0,:8]
      last                                                       Smith
      first                                                       john
      true_race                                GreaterEuropean,British
      rowindex                                                       0
      Asian,GreaterEastAsian,EastAsian_mean                   0.004554
      Asian,GreaterEastAsian,EastAsian_std                    0.003358
      Asian,GreaterEastAsian,EastAsian_lb                     0.000535
      Asian,GreaterEastAsian,EastAsian_ub                     0.000705
      Name: 0, dtype: object



-  **pred\_wiki\_name(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name wiki
         model <ethnicolr/models/ethnicolr_keras_lstm_wiki_name.ipynb>`__
         to predict the race and ethnicity.

   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                                                                                                                                                                                                                            |
   +==============+============================================================================================================================================================================================================================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred                                                                                                                                                                                                               |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list}* string or list of the name or location of the column containing the first name, last name, middle name, and suffix, if there. The first name and last name columns are required. If no middle name of suffix columns are there, it is assumed that there are no middle names or suffixes.   |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                                                                                                                                                                                                                                  |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                                                                                                                                                                                                                                |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+



   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (categorical variable---category with the highest probability),
      "Asian,GreaterEastAsian,EastAsian",
      "Asian,GreaterEastAsian,Japanese", "Asian,IndianSubContinent",
      "GreaterAfrican,Africans", "GreaterAfrican,Muslim",
      "GreaterEuropean,British","GreaterEuropean,EastEuropean",
      "GreaterEuropean,Jewish","GreaterEuropean,WestEuropean,French",
      "GreaterEuropean,WestEuropean,Germanic","GreaterEuropean,WestEuropean,Hispanic",
      "GreaterEuropean,WestEuropean,Italian","GreaterEuropean,WestEuropean,Nordic".
      For each race it will provide the mean, standard error, lower & upper
      bound of confidence interval

   *(Using the same dataframe from example above)*
   ::

      >>> odf = pred_wiki_name(df, 'last', 'first')
      ['Asian,GreaterEastAsian,EastAsian', 'Asian,GreaterEastAsian,Japanese', 'Asian,IndianSubContinent', 'GreaterAfrican,Africans', 'GreaterAfrican,Muslim', 'GreaterEuropean,British', 'GreaterEuropean,EastEuropean', 'GreaterEuropean,Jewish', 'GreaterEuropean,WestEuropean,French', 'GreaterEuropean,WestEuropean,Germanic', 'GreaterEuropean,WestEuropean,Hispanic', 'GreaterEuropean,WestEuropean,Italian', 'GreaterEuropean,WestEuropean,Nordic']
      
      >>> odf
         last  first  ... GreaterEuropean,WestEuropean,Nordic_ub                              race
      0  Smith   john  ...                               0.000236           GreaterEuropean,British
      1  Zhang  simon  ...                               0.000021  Asian,GreaterEastAsian,EastAsian

      [2 rows x 58 columns]
      
      >>> odf.iloc[1,:8]
      last                                                                Zhang
      first                                                               simon
      true_race                                Asian,GreaterEastAsian,EastAsian
      rowindex                                                                1
      __name                                                        Zhang Simon
      Asian,GreaterEastAsian,EastAsian_mean                            0.890619
      Asian,GreaterEastAsian,EastAsian_std                             0.119097
      Asian,GreaterEastAsian,EastAsian_lb                              0.391496
      Name: 1, dtype: object


-  **pred\_fl\_reg\_ln(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does?:

      -  Removes extra space, if there.
      -  Uses the `last name FL registration
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln.ipynb>`__
         to predict the race and ethnicity.

   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name   |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                           |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                         |
   +--------------+---------------------------------------------------------------------------------------------------------------------+



   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance
      Asian), hispanic, nh\_black, nh\_white. For each race it will provide
      the mean, standard error, lower & upper bound of confidence interval

   ::

      >>> import pandas as pd

      >>> names = [
      ...             {"last": "sawyer", "first": "john", "true_race": "nh_white"},
      ...             {"last": "torres", "first": "raul", "true_race": "hispanic"},
      ...         ]
      
      >>> df = pd.DataFrame(names)

      >>> from ethnicolr import pred_fl_reg_ln, pred_fl_reg_name, pred_fl_reg_ln_five_cat, pred_fl_reg_name_five_cat

      >>> odf = pred_fl_reg_ln(df, 'last')
      ['asian', 'hispanic', 'nh_black', 'nh_white']

      >>> odf
         last first true_race  rowindex  asian_mean  asian_std  ...  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
      0  Sawyer  john  nh_white         0    0.004004   0.004483  ...     0.015442       0.908452      0.035121     0.722879     0.804443  nh_white
      1  Torres  raul  hispanic         1    0.005882   0.002249  ...     0.005305       0.182575      0.072142     0.074511     0.090856  hispanic

      [2 rows x 21 columns]
      
      >>> odf.iloc[0]
      last               Sawyer
      first                john
      true_race        nh_white
      rowindex                0
      asian_mean       0.004004
      asian_std        0.004483
      asian_lb         0.000899
      asian_ub          0.00103
      hispanic_mean    0.034227
      hispanic_std      0.01294
      hispanic_lb      0.017406
      hispanic_ub      0.017625
      nh_black_mean    0.053317
      nh_black_std     0.028634
      nh_black_lb      0.010537
      nh_black_ub      0.015442
      nh_white_mean    0.908452
      nh_white_std     0.035121
      nh_white_lb      0.722879
      nh_white_ub      0.804443
      race             nh_white
      Name: 0, dtype: object


-  **pred\_fl\_reg\_name(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name FL
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_name.ipynb>`__
         to predict the race and ethnicity.

   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                                                                                                                                                                                                                            |
   +==============+============================================================================================================================================================================================================================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred                                                                                                                                                                                                               |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list}* string or list of the name or location of the column containing the first name, last name, middle name, and suffix, if there. The first name and last name columns are required. If no middle name of suffix columns are there, it is assumed that there are no middle names or suffixes.   |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                                                                                                                                                                                                                                  |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                                                                                                                                                                                                                                |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, or hispanic), asian (percentage chance
      Asian), hispanic, nh\_black, nh\_white. For each race it will provide
      the mean, standard error, lower & upper bound of confidence interval

   
   *(Using the same dataframe from example above)*
   ::

      >>> odf = pred_fl_reg_name(df, 'last', 'first')
      ['asian', 'hispanic', 'nh_black', 'nh_white']

      >>> odf
         last first true_race  rowindex       __name  asian_mean  ...  nh_black_ub  nh_white_mean  nh_white_std  nh_white_lb  nh_white_ub      race
      0  Sawyer  john  nh_white         0  Sawyer John    0.001196  ...     0.005450       0.971152      0.015757     0.915592     0.918630  nh_white
      1  Torres  raul  hispanic         1  Torres Raul    0.004770  ...     0.000885       0.066303      0.028486     0.022593     0.024143  hispanic

      [2 rows x 22 columns]
      
      >>> odf.iloc[1]
      last                  Torres
      first                   raul
      true_race           hispanic
      rowindex                   1
      __name           Torres Raul
      asian_mean           0.00477
      asian_std           0.002943
      asian_lb            0.000904
      asian_ub            0.001056
      hispanic_mean         0.9251
      hispanic_std        0.032224
      hispanic_lb         0.829494
      hispanic_ub           0.8385
      nh_black_mean       0.003826
      nh_black_std        0.002735
      nh_black_lb         0.000838
      nh_black_ub         0.000885
      nh_white_mean       0.066303
      nh_white_std        0.028486
      nh_white_lb         0.022593
      nh_white_ub         0.024143
      race                hispanic
      Name: 1, dtype: object


-  **pred\_fl\_reg\_ln\_five\_cat(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does?:

      -  Removes extra space, if there.
      -  Uses the `last name FL registration
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_five_cat.ipynb>`__
         to predict the race and ethnicity.

   +--------------+---------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                     |
   +==============+=====================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred        |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list, int}* string or list of the name or location of the column containing the last name   |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                           |
   +--------------+---------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                         |
   +--------------+---------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, hispanic or other), asian (percentage
      chance Asian), hispanic, nh\_black, nh\_white, other. For each race
      it will provide the mean, standard error, lower & upper bound of
      confidence interval

   *(Using the same dataframe from example above)*
   ::

      >>> odf = pred_fl_reg_ln_five_cat(df,'last')
      ['asian', 'hispanic', 'nh_black', 'nh_white', 'other']

      >>> odf
         last first true_race  rowindex       __name  asian_mean  asian_std  ...  nh_white_lb  nh_white_ub  other_mean  other_std  other_lb  other_ub      race
      0  Sawyer  john  nh_white         0  Sawyer John    0.142867   0.046145  ...     0.203204     0.221313    0.235889   0.023794  0.192840  0.193671  nh_white
      1  Torres  raul  hispanic         1  Torres Raul    0.101397   0.028399  ...     0.090068     0.100212    0.238645   0.034070  0.136617  0.145928  hispanic

      [2 rows x 26 columns]

      >>> odf.iloc[0]
      last                  Sawyer
      first                   john
      true_race           nh_white
      rowindex                   0
      __name           Sawyer John
      asian_mean          0.142867
      asian_std           0.046145
      asian_lb            0.067382
      asian_ub            0.073285
      hispanic_mean       0.068199
      hispanic_std        0.020641
      hispanic_lb          0.02565
      hispanic_ub         0.030017
      nh_black_mean       0.239793
      nh_black_std        0.076287
      nh_black_lb         0.084239
      nh_black_ub         0.085626
      nh_white_mean       0.313252
      nh_white_std        0.046173
      nh_white_lb         0.203204
      nh_white_ub         0.221313
      other_mean          0.235889
      other_std           0.023794
      other_lb             0.19284
      other_ub            0.193671
      race                nh_white
      Name: 0, dtype: object




-  **pred\_fl\_reg\_name\_five\_cat(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name FL
         model <ethnicolr/models/ethnicolr_keras_lstm_fl_voter_ln_five_cat.ipynb>`__
         to predict the race and ethnicity.

   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                                                                                                                                                                                                                            |
   +==============+============================================================================================================================================================================================================================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred                                                                                                                                                                                                               |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list}* string or list of the name or location of the column containing the first name, last name, middle name, and suffix, if there. The first name and last name columns are required. If no middle name of suffix columns are there, it is assumed that there are no middle names or suffixes.   |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                                                                                                                                                                                                                                  |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                                                                                                                                                                                                                                |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race (white, black, asian, hispanic, or other), asian (percentage
      chance Asian), hispanic, nh\_black, nh\_white, other. For each race
      it will provide the mean, standard error, lower & upper bound of
      confidence interval

   *(Using the same dataframe from example above)*
   ::

      >>> odf = pred_fl_reg_name_five_cat(df, 'last','first')
      ['asian', 'hispanic', 'nh_black', 'nh_white', 'other']

      >>> odf
         last first true_race  rowindex       __name  asian_mean  asian_std  ...  nh_white_lb  nh_white_ub  other_mean  other_std  other_lb  other_ub      race
      0  Sawyer  john  nh_white         0  Sawyer John    0.194250   0.120314  ...     0.126987     0.167742    0.259069   0.030386  0.142455  0.177375  nh_white
      1  Torres  raul  hispanic         1  Torres Raul    0.081465   0.038318  ...     0.019312     0.020782    0.158614   0.039180  0.081994  0.083105  hispanic

      [2 rows x 26 columns]

      >>> odf.iloc[1]
      last                  Torres
      first                   raul
      true_race           hispanic
      rowindex                   1
      __name           Torres Raul
      asian_mean          0.081465
      asian_std           0.038318
      asian_lb            0.032789
      asian_ub            0.034667
      hispanic_mean       0.646059
      hispanic_std        0.144663
      hispanic_lb         0.188246
      hispanic_ub         0.219772
      nh_black_mean       0.037737
      nh_black_std        0.045439
      nh_black_lb         0.006477
      nh_black_ub         0.006603
      nh_white_mean       0.076125
      nh_white_std        0.059213
      nh_white_lb         0.019312
      nh_white_ub         0.020782
      other_mean          0.158614
      other_std            0.03918
      other_lb            0.081994
      other_ub            0.083105
      race                hispanic
      Name: 1, dtype: object



-  **pred\_nc\_reg\_name(df, namecol, num\_iter=100, conf\_int=0.9)**

   -  What it does:

      -  Removes extra space.
      -  Uses the `full name NC
         model <ethnicolr/models/ethnicolr_keras_lstm_nc_12_cat_model.ipynb>`__
         to predict the race and ethnicity.

   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Parameters   |                                                                                                                                                                                                                                                                                                                            |
   +==============+============================================================================================================================================================================================================================================================================================================================+
   |              | **df** : *{DataFrame, csv}* Pandas dataframe of CSV file contains the names of the individual to be inferred                                                                                                                                                                                                               |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **namecol** : *{string, list}* string or list of the name or location of the column containing the first name, last name, middle name, and suffix, if there. The first name and last name columns are required. If no middle name of suffix columns are there, it is assumed that there are no middle names or suffixes.   |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **num\_iter** : *int, default=100* number of iterations to calculate uncertainty in model                                                                                                                                                                                                                                  |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   |              | **conf\_int** : *float, default=0.9* confidence interval in predicted class                                                                                                                                                                                                                                                |
   +--------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


   -  Output: Appends the following columns to the pandas DataFrame or CSV:
      race + ethnicity. The codebook is
      `here <https://github.com/appeler/nc_race_ethnicity>`__. For each
      race it will provide the mean, standard error, lower & upper bound of
      confidence interval

   ::

      >>> import pandas as pd

      >>> names = [
      ...             {"last": "hernandez", "first": "hector", "true_race": "HL+O"},
      ...             {"last": "zhang", "first": "simon", "true_race": "NL+A"},
      ...         ]

      >>> df = pd.DataFrame(names)

      >>> from ethnicolr import pred_nc_reg_name

      >>> odf = pred_nc_reg_name(df, 'last','first')
      ['HL+A', 'HL+B', 'HL+I', 'HL+M', 'HL+O', 'HL+W', 'NL+A', 'NL+B', 'NL+I', 'NL+M', 'NL+O', 'NL+W']
      
      >>> odf
            last   first true_race            __name  rowindex  HL+A_mean  HL+A_std       HL+A_lb       HL+A_ub  HL+B_mean  ...   NL+M_ub  NL+O_mean  NL+O_std   NL+O_lb   NL+O_ub  NL+W_mean  NL+W_std   NL+W_lb   NL+W_ub  race
      0  hernandez  hector      HL+O  Hernandez Hector         0   0.000054  0.000354  5.833132e-10  4.291366e-09   0.009606  ...  0.000416   0.090123  0.036310  0.000705  0.003757   0.021228  0.021222  0.000368  0.001230  HL+O
      1      zhang   simon      NL+A       Zhang Simon         1   0.000603  0.002808  1.988648e-07  2.766486e-07   0.000026  ...  0.000086   0.125159  0.042818  0.050547  0.057208   0.003149  0.005437  0.000210  0.000225  NL+A

      [2 rows x 54 columns]
      
      >>> odf.iloc[0]
      last                hernandez
      first                  hector
      true_race                HL+O
      __name       Hernandez Hector
      rowindex                    0
      HL+A_mean            0.000054
      HL+A_std             0.000354
      HL+A_lb                   0.0
      HL+A_ub                   0.0
      HL+B_mean            0.009606
      HL+B_std             0.040739
      HL+B_lb                   0.0
      HL+B_ub              0.000003
      HL+I_mean            0.001605
      HL+I_std             0.004569
      HL+I_lb                   0.0
      HL+I_ub                   0.0
      HL+M_mean            0.147628
      HL+M_std             0.215733
      HL+M_lb              0.001253
      HL+M_ub              0.001297
      HL+O_mean             0.36902
      HL+O_std             0.132249
      HL+O_lb              0.002289
      HL+O_ub              0.019187
      HL+W_mean            0.264246
      HL+W_std             0.090536
      HL+W_lb              0.001782
      HL+W_ub              0.015628
      NL+A_mean            0.012004
      NL+A_std             0.010873
      NL+A_lb              0.000121
      NL+A_ub              0.000281
      NL+B_mean            0.010891
      NL+B_std              0.01404
      NL+B_lb              0.000094
      NL+B_ub              0.000383
      NL+I_mean            0.005182
      NL+I_std             0.008259
      NL+I_lb              0.000009
      NL+I_ub              0.000068
      NL+M_mean            0.068412
      NL+M_std              0.08564
      NL+M_lb              0.000172
      NL+M_ub              0.000416
      NL+O_mean            0.090123
      NL+O_std              0.03631
      NL+O_lb              0.000705
      NL+O_ub              0.003757
      NL+W_mean            0.021228
      NL+W_std             0.021222
      NL+W_lb              0.000368
      NL+W_ub               0.00123
      race                     HL+O
      Name: 0, dtype: object


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

Suriyan Laohaprapanon, Gaurav Sood and Bashar Naji

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