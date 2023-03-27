#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import test_and_norm_df, transform_and_pred, arg_parser

MODELFN = "models/fl_voter_reg/lstm/fl_all_fullname_lstm_5_cat{0:s}.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_fullname_vocab_5_cat{0:s}.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_name_five_cat_race{0:s}.csv"

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegNameFiveCatModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_name(cls, 
                         df: pd.DataFrame,
                         lname_col: str,
                         fname_col: str,
                         num_iter: int=100,
                         conf_int: float=1.0, 
                         year: int=2022) -> pd.DataFrame:
        """Predict the race/ethnicity of the full name using the Florida voter registration
        5 category model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            fname_col (str or int): Column name for the first name.
            num_iter (int): Number of iterations do calculate the confidence interval. Default is 100.
            conf_int (float): What confidence interval? Default is 1, which means just the point estimate.
            year (int): the year of the model. Default = 2022. 

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        if lname_col not in df.columns:
            raise Exception(f"The {lname_col} column doesn't exist in the dataframe.")
        if fname_col not in df.columns:
            raise Exception(f"The {fname_col} column doesn't exist in the dataframe.")

        df['__name'] = (df[lname_col].str.strip()
                        + ' ' + df[fname_col].str.strip()).str.title()

        df = test_and_norm_df(df, '__name')

        year = '_2022' if year == 2022 else ''
        VOCAB = resource_filename(__name__, VOCABFN.format(year))
        MODEL = resource_filename(__name__, MODELFN.format(year))
        RACE = resource_filename(__name__, RACEFN.format(year))

        rdf = transform_and_pred(df=df,
                                 newnamecol='__name',
                                 cls=cls,
                                 VOCAB=VOCAB,
                                 RACE=RACE,
                                 MODEL=MODEL,
                                 NGRAMS=NGRAMS,
                                 maxlen=FEATURE_LEN,
                                 num_iter=num_iter,
                                 conf_int=conf_int)

        del rdf['__name']
        return rdf


pred_fl_reg_name_five_cat = FloridaRegNameFiveCatModel.pred_fl_reg_name


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using Florida registration model (Five Cat)", 
                default_out = "fl-pred-name-five-cat-output.csv", 
                default_year = 2022, 
                year_choices = [2017, 2022],
                first = True)

    df = pd.read_csv(args.input)

    rdf = pred_fl_reg_name_five_cat(df = df, 
                                   lname_col = args.last, 
                                   fname_col = args.first, 
                                   num_iter = args.iter, 
                                   conf_int = args.conf,
                                   year = args.year)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
