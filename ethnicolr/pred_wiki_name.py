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
    def pred_wiki_name(cls, df: pd.DataFrame, lname_col: str, fname_col: str, num_iter: int=100,
                       conf_int: float=1.0):
        """Predict the race/ethnicity by the full name using Wiki model.

        Using the Wiki full name model to predict the race/ethnicity of
        the input DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                and first name column.
            lname_col (str or int): Column's name or location of the last name
                in DataFrame.
            fname_col (str or int): Column's name or location of the first name
                in DataFrame.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        if lname_col not in df.columns:
            print(f"No column `{lname_col}` in the DataFrame")
            return df
        if fname_col not in df.columns:
            print(f"No column `{fname_col}` in the DataFrame")
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

        return rdf


pred_wiki_name = WikiNameModel.pred_wiki_name


def main(argv=sys.argv[1:]):
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using Wiki model", 
                default_out: "wiki-pred-name-output.csv", 
                default_year = 2017, 
                year_choices = [2017],
                first = True)

    if not args.last.isdigit() and not args.first.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.last = int(args.last)
        args.first = int(args.first)

    if not column_exists(df, args.last):
        return -1
    if not column_exists(df, args.first):
        return -1

    rdf = pred_wiki_name(df, args.last, args.first, args.iter, args.conf)

    print(f"Saving output to file: `{args.output}`")
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
