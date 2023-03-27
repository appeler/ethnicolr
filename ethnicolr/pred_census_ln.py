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

MODELFN = "models/census/lstm/census{0:d}_ln_lstm.h5"
VOCABFN = "models/census/lstm/census{0:d}_ln_vocab.csv"
RACEFN = "models/census/lstm/census{0:d}_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20

class CensusLnModel:
    vocab = None
    race = None
    model = None
    model_year = None

    @classmethod
    def pred_census_ln(cls, 
                       df: pd.DataFrame,
                       lname_col: str,
                       year: int=2010,
                       num_iter: int=100,
                       conf_int: float=1.0) -> pd.DataFrame:
        """Predict the race/ethnicity of the last name using the Census model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            lname_col (str): Column name for the last name.
            year (int): The year of Census model to be used. 2000 or 2010.
                Default is 2010.
            num_iter (int): Number of iterations do calculate the confidence interval. Default is 100.
            conf_int (float): What confidence interval? Default is 1, which means just the point estimate.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` contains predicted race/ethnicity
                - `black`, `api`, `white`, `hispanic` contain the prediction
                    probabilities.

        """

        if not column_exists(df, lname_col):
            return df

        df.dropna(subset=[lname_col], inplace = True)
        if df.shape[0] == 0:
            return df

        VOCAB = resource_filename(__name__, VOCABFN.format(year))
        MODEL = resource_filename(__name__, MODELFN.format(year))
        RACE = resource_filename(__name__, RACEFN.format(year))

        rdf = transform_and_pred(
            df=df,
            newnamecol=lname_col,
            cls=cls,
            VOCAB=VOCAB,
            RACE=RACE,
            MODEL=MODEL,
            NGRAMS=NGRAMS,
            maxlen=FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int,
        )

        return rdf

pred_census_ln = CensusLnModel.pred_census_ln

def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by last name using Census last name model", 
                default_out = "census-pred-ln-output.csv", 
                default_year = 2010, 
                year_choices = [2000, 2010])

    df = pd.read_csv(args.input)

    rdf = pred_census_ln(df, args.last, args.year, args.iter, args.conf)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)

if __name__ == "__main__":
    sys.exit(main())
