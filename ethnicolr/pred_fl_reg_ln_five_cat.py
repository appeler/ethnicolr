#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, transform_and_pred, arg_parser

MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_5_cat{0:s}.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab_5_cat{0:s}.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_ln_five_cat_race{0:s}.csv"

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegLnFiveCatModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_ln(cls, df: pd.DataFrame, lname_col: str, num_iter: int=100, conf_int: float=1.0, year: int=2022) -> pd.DataFrame:

        """Predict the race/ethnicity of the last name using the Florida voter
        registration data model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            lname_col (str): Column name for the last name.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        df = test_and_norm_df(df)

        year = '_2022' if year == 2022 else ''
        VOCAB = resource_filename(__name__, VOCABFN.format(year))
        MODEL = resource_filename(__name__, MODELFN.format(year))
        RACE = resource_filename(__name__, RACEFN.format(year))

        rdf = transform_and_pred(df=df,
                                 newnamecol=lname_col,
                                 cls=cls,
                                 VOCAB=VOCAB,
                                 RACE=RACE,
                                 MODEL=MODEL,
                                 NGRAMS=NGRAMS,
                                 maxlen=FEATURE_LEN,
                                 num_iter=num_iter,
                                 conf_int=conf_int)

        return rdf


pred_fl_reg_ln_five_cat = FloridaRegLnFiveCatModel.pred_fl_reg_ln


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by last name using the Florida registration 5 cat. model", 
                default_out = "fl-pred-ln-five-cat-output.csv", 
                default_year = 2022, 
                year_choices = [2017, 2022])

    df = pd.read_csv(args.input)

    rdf = pred_fl_reg_ln_five_cat(df, args.last, args.iter, args.conf,
                                  args.year)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
