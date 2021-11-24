#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, transform_and_pred

MODELFN = "models/fl_voter_reg/lstm/fl_all_fullname_lstm_5_cat.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_fullname_vocab_5_cat.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_name_five_cat_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegNameFiveCatModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_name(cls, df, lname_col, fname_col, num_iter=100, conf_int=0.9):
        """Predict the race/ethnicity by the full name using Florida voter model.

        Using the Florida voter full name model to predict the race/ethnicity of
        the input DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name and
                first name column.
            lname_col (str or int): Column's name or location of the last name in
                DataFrame.
            fname_col (str or int): Column's name or location of the first name in
                DataFrame.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        if lname_col not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(lname_col))
            return df
        if fname_col not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(fname_col))
            return df

        df['__name'] = (df[lname_col] + ' ' + df[fname_col]).str.title()

        df.dropna(subset=['__name'])
        if df.shape[0] == 0:
            del df['__name']
            return df

        rdf = transform_and_pred(df = df, 
                                newnamecol = '__name', 
                                cls = cls, 
                                VOCAB = VOCAB,
                                RACE = RACE,
                                MODEL = MODEL,
                                NGRAMS = NGRAMS,
                                maxlen=FEATURE_LEN,
                                num_iter=num_iter, 
                                conf_int=conf_int)

        return rdf


pred_fl_reg_name_five_cat = FloridaRegNameFiveCatModel.pred_fl_reg_name


def main(argv=sys.argv[1:]):
    title = 'Predict Race/Ethnicity by name using Florida registration model (Five Cat)'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='fl-pred-name-five-cat-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-f', '--first', required=True,
                        help='Name or index location of column contains '
                             'the first name')
    parser.add_argument('-l', '--last', required=True,
                        help='Name or index location of column contains '
                             'the last name')
    parser.add_argument('-i', '--iter', default=100, type=int,
                        help='Number of iterations to measure uncertainty')
    parser.add_argument('-c', '--conf', default=0.9, type=float,
                         help='Confidence interval of Predictions')

    args = parser.parse_args(argv)

    print(args)

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

    rdf = pred_fl_reg_name_five_cat(df, args.last, args.first, args.iter,
                                    args.conf)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
