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

MODELFN = "models/wiki/lstm/wiki_name_lstm.h5"
VOCABFN = "models/wiki/lstm/wiki_name_vocab.csv"
RACEFN = "models/wiki/lstm/wiki_race.csv"

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
    def pred_wiki_name(cls, df, lname_col, fname_col):
        """Predict the race/ethnicity by the full name using Wiki model.

        Using the Wiki full name model to predict the race/ethnicity of the input
        DataFrame.

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

        nn = df['__name'].notnull()
        if df[nn].shape[0] == 0:
            del df['__name']
            return df

        if cls.model is None:
            #  sort n-gram by freq (highest -> lowest)
            vdf = pd.read_csv(VOCAB)
            cls.vocab = vdf.vocab.tolist()

            rdf = pd.read_csv(RACE)
            cls.race = rdf.race.tolist()

            cls.model = load_model(MODEL)

        # build X from index of n-gram sequence
        X = np.array(df[nn]['__name'].apply(lambda c:
                                            find_ngrams(cls.vocab,
                                                        c, NGRAMS)))
        X = sequence.pad_sequences(X, maxlen=FEATURE_LEN)

        df.loc[nn, '__pred'] = cls.model.predict_classes(X, verbose=2)

        df.loc[nn, 'race'] = df[nn]['__pred'].apply(lambda c:
                                                    cls.race[int(c)])

        # take out temporary working columns
        del df['__pred']
        del df['__name']

        proba = cls.model.predict_proba(X, verbose=2)

        pdf = pd.DataFrame(proba, columns=cls.race)
        pdf.set_index(df[nn].index, inplace=True)

        rdf = pd.concat([df, pdf], axis=1)

        return rdf


pred_wiki_name = WikiNameModel.pred_wiki_name


def main(argv=sys.argv[1:]):
    title = 'Predict Race/Ethnicity by name using Wiki model'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='wiki-pred-name-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-f', '--first', required=True,
                        help='Name or index location of column contains '
                             'the first name')
    parser.add_argument('-l', '--last', required=True,
                        help='Name or index location of column contains '
                             'the last name')

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

    rdf = pred_wiki_name(df, args.last, args.first)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
