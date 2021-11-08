#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, transform_and_pred

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
    def pred_census_ln(cls, df, namecol, year=2000, num_iter=100, conf_int=0.9):
        """Predict the race/ethnicity by the last name using Census model.

        Using the Census last name model to predict the race/ethnicity of the input
        DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            namecol (str or int): Column's name or location of the name in
                DataFrame.
            year (int): The year of Census model to be used. (2000 or 2010)
                (default is 2000)

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - `black`, `api`, `white`, `hispanic` are the prediction
                    probability.

        """

        if namecol not in df.columns:
            print("No column `{0!s}` in the DataFrame".format(namecol))
            return df

        df.dropna(subset=[namecol])
        if df.shape[0] == 0:
            return df

        VOCAB = resource_filename(__name__, VOCABFN.format(year))
        MODEL = resource_filename(__name__, MODELFN.format(year))
        RACE = resource_filename(__name__, RACEFN.format(year))

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
            conf_int=conf_int,
        )

        return rdf


pred_census_ln = CensusLnModel.pred_census_ln


def main(argv=sys.argv[1:]):
    title = "Predict Race/Ethnicity by last name using Census model"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("input", default=None, help="Input file")
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=2000,
        choices=[2000, 2010],
        help="Year of Census data (default=2000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="census-pred-ln-output.csv",
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
    parser.add_argument('-c', '--conf', default=0.9, type=float,
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

    rdf = pred_census_ln(df, args.last, args.year, args.iter, args.conf)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
