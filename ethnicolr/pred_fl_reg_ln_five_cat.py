#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import column_exists, find_ngrams, fixup_columns

MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_5_cat.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab_5_cat.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_race_five_cat.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegLnFiveCatModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_ln(cls, df, namecol):
        """Predict the race/ethnicity by the last name using Florida voter model.

        Using the Florida voter last name model to predict the race/ethnicity of
        the input DataFrame.

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

        nn = df[namecol].notnull()
        if df[nn].shape[0] == 0:
            return df

        df['__last_name'] = df[namecol].str.strip()
        df['__last_name'] = df['__last_name'].str.title()

        if cls.model is None:
            #  sort n-gram by freq (highest -> lowest)
            vdf = pd.read_csv(VOCAB)
            cls.vocab = vdf.vocab.tolist()

            rdf = pd.read_csv(RACE)
            cls.race = rdf.race.tolist()

            cls.model = load_model(MODEL)

        # build X from index of n-gram sequence
        X = np.array(df[nn]['__last_name'].apply(lambda c:
                                                 find_ngrams(cls.vocab,
                                                             c, NGRAMS)))
        X = sequence.pad_sequences(X, maxlen=FEATURE_LEN)

        df.loc[nn, '__pred'] = cls.model.predict_classes(X, verbose=2)

        df.loc[nn, 'race'] = df[nn]['__pred'].apply(lambda c:
                                                    cls.race[int(c)])

        # take out temporary working columns
        del df['__pred']
        del df['__last_name']

        proba = cls.model.predict_proba(X, verbose=2)

        pdf = pd.DataFrame(proba, columns=cls.race)
        pdf.set_index(df[nn].index, inplace=True)

        rdf = pd.concat([df, pdf], axis=1)

        return rdf


pred_fl_reg_ln_five_cat = FloridaRegLnFiveCatModel.pred_fl_reg_ln


def main(argv=sys.argv[1:]):
    title = 'Predict Race/Ethnicity by last name using the Florida registration 5 cat. model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='fl-pred-ln-five-cat-output.csv',
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

    rdf = pred_fl_reg_ln_five_cat(df, args.last)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())
