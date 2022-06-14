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

MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_ln_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegLnModel:
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_ln(cls, df, namecol, num_iter=100, conf_int=1):
        """Predict the race/ethnicity by the last name using Florida voter
        model.

        Using the Florida voter last name model to predict the race/ethnicity
        oF the input DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            namecol (str or int): Column's name or location of the name in
                DataFrame.

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        if namecol not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(namecol))
            return df

        df.dropna(subset=[namecol])
        if df.shape[0] == 0:
            return df
        
        rdf = transform_and_pred(
                df=df,
                newnamecol=namecol,
                cls=cls,
                VOCAB=VOCAB,
                RACE=RACE,
                MODEL=MODEL,
                NGRAMS=NGRAMS,
                maxlen=FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int
            )

        return rdf

pred_fl_reg_ln = FloridaRegLnModel.pred_fl_reg_ln


def main(argv=sys.argv[1:]):
    title = "Predict Race/Ethnicity by name using Florida registration model"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    parser.add_argument(
        "-o",
        "--output",
        default="fl-pred-ln-output.csv",
        help="Output file with prediction data",
    )
    parser.add_argument(
        "-l",
        "--last",
        required=True,
        help="Name or index location of column contains " "the last name",
    )
    parser.add_argument('-i', '--iter', default=100, type=int,
                        help='Number of iterations to measure uncertainty')
    parser.add_argument('-c', '--conf', default=1.0, type=float,
                        help='Confidence interval of Predictions')

    args = parser.parse_args(argv)

    print(args)

    if not args.last.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.last = int(args.last)

    if not column_exists(df, args.last):
        return -1

    rdf = pred_fl_reg_ln(df, args.last, args.iter, args.conf)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
