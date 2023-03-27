#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, transform_and_pred, arg_parser

MODELFN = "models/fl_voter_reg/lstm/fl_all_name_lstm.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_name_vocab.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_name_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 25


class FloridaRegNameModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_name(cls, df: pd.DataFrame, lname_col: str, fname_col: str, num_iter: int=100,
                         conf_int: float=1.0) -> pd.DataFrame:
        """Predict the race/ethnicity by the full name using the Florida voter
        registration data model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            fname_col (str or int): Column name for the first name.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

       
        if not column_exists(df, lname_col):
            return df
        if not column_exists(df, fname_col):
            return df

        df['__name'] = (df[lname_col].str.strip()
                        + ' ' + df[fname_col].str.strip()).str.title()

        df.dropna(subset=['__name'])
        if df.shape[0] == 0:
            del df['__name']
            return df

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


pred_fl_reg_name = FloridaRegNameModel.pred_fl_reg_name


def main(argv=sys.argv[1:]):
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using Florida registration model", 
                default_out = "fl-pred-name-output.csv", 
                default_year = 2017, 
                year_choices = [2017],
                first = True)

    df = pd.read_csv(args.input)
   
    if not column_exists(df, args.last):
        return -1
    if not column_exists(df, args.first):
        return -1

    rdf = pred_fl_reg_name(df, args.last, args.first, args.iter, args.conf)

    print(f"Saving output to file: `{args.output}`")
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
