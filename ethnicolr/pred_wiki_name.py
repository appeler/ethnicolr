#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from pkg_resources import resource_filename

from .utils import test_and_norm_df, transform_and_pred, arg_parser

MODELFN = "models/wiki/lstm/wiki_name_lstm.h5"
VOCABFN = "models/wiki/lstm/wiki_name_vocab.csv"
RACEFN = "models/wiki/lstm/wiki_name_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 25


class WikiNameModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_wiki_name(cls, 
                       df: pd.DataFrame,
                       lname_col: str, 
                       fname_col: str, 
                       num_iter: int=100,
                       conf_int: float=1.0) -> pd.DataFrame:
        """Predict the race/ethnicity by the full name using the Wikipedia model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            fname_col (str or int): Column name for the first name.
            num_iter (int): Number of iterations do calculate the confidence interval. Default is 100.
            conf_int (float): What confidence interval? Default is 1, which means just the point estimate.
            
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


pred_wiki_name = WikiNameModel.pred_wiki_name


def main(argv=sys.argv[1:]):
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using Wiki model", 
                default_out = "wiki-pred-name-output.csv", 
                default_year = 2017, 
                year_choices = [2017],
                first = True)

    df = pd.read_csv(args.input)
   
    rdf = pred_wiki_name(df, args.last, args.first, args.iter, args.conf)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
