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

MODELFN = "models/nc_voter_reg/lstm/nc_voter_name_lstm_oversample.h5"
VOCABFN = "models/nc_voter_reg/lstm/nc_voter_name_vocab_oversample.csv"
RACEFN = "models/nc_voter_reg/lstm/nc_name_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = (2, 3)
FEATURE_LEN = 25


class NCRegNameModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_nc_reg_name(cls, df: pd.DataFrame, lname_col: str, fname_col: str, num_iter: int=100,
                         conf_int: float=1.0) -> pd.DataFrame:
        """Predict the race+ethnicity by the full name using the
        North Carolina 12 category voter model.

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

        df.dropna(subset=['__name'], inplace = True)
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

        return rdf


pred_nc_reg_name = NCRegNameModel.pred_nc_reg_name


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using NC 12 category voter registration model", 
                default_out = "fl-pred-name-output.csv", 
                default_year = 2017, 
                year_choices = [2017],
                first = True)

    df = pd.read_csv(args.input)
   
    rdf = pred_nc_reg_name(df = df, 
                           lname_col = args.last, 
                           fname_col = args.first, 
                           num_iter = args.iter, 
                           conf_int = args.conf)


    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
