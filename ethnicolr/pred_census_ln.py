#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence

from pkg_resources import resource_filename

from .utils import column_exists, find_ngrams, fixup_columns

MODELFN = "models/census/lstm/census{0:d}_ln_lstm.h5"
VOCABFN = "models/census/lstm/census{0:d}_ln_vocab.csv"
RACEFN = "models/census/lstm/census{0:d}_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20


def pred_census_ln(df, namecol, year=2000):
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

    df['__last_name'] = df[namecol].str.strip()
    df['__last_name'] = df['__last_name'].str.title()

    #  sort n-gram by freq (highest -> lowest)
    vdf = pd.read_csv(VOCAB.format(year))
    vocab = vdf.vocab.tolist()

    rdf = pd.read_csv(RACE.format(year))
    race = rdf.race.tolist()

    model = load_model(MODEL.format(year))

    # build X from index of n-gram sequence
    X = np.array(df.__last_name.apply(lambda c: find_ngrams(vocab, c, NGRAMS)))
    X = sequence.pad_sequences(X, maxlen=FEATURE_LEN)

    df['__pred'] = model.predict_classes(X, verbose=2)

    df['race'] = df.__pred.apply(lambda c: race[c])

    # take out temporary working columns
    del df['__pred']
    del df['__last_name']

    proba = model.predict_proba(X, verbose=2)

    pdf = pd.DataFrame(proba, columns=race)
    pdf.set_index(df.index, inplace=True)

    rdf = pd.concat([df, pdf], axis=1)

    return rdf


def main(argv=sys.argv[1:]):
    title = 'Predict Race/Ethnicity by last name using Census model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-y', '--year', type=int, default=2000,
                        choices=[2000, 2010],
                        help='Year of Census data (default=2000)')
    parser.add_argument('-o', '--output', default='census-pred-ln-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-l', '--last', required=True,
                        help='Name or index location of column contains '
                             'the last name')

    args = parser.parse_args(argv)

    print(args)

    if not args.last.isdigit():
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(args.input, header=None)
        args.last = int(args.last)

    if not column_exists(df, args.last):
        return -1

    rdf = pred_census_ln(df, args.last, args.year)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
