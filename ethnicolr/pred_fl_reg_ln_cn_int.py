#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from ethnicolr.utils import column_exists, find_ngrams, fixup_columns, transform_and_pred

MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_uncrtn.h5"
VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab.csv"
RACEFN = "models/fl_voter_reg/lstm/fl_ln_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20


class FloridaRegLnCnIntModel():
    vocab = None
    race = None
    model = None

    @classmethod
    def pred_fl_reg_ln_cn_int(cls, df, namecol, num_iter=100, conf_int=0.9):
        """Predict the race/ethnicity by the last name using Florida voter model.

        Using the Florida voter last name model to predict the race/ethnicity of
        the input DataFrame.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            namecol (str or int): Column's name or location of the name in
                DataFrame.
            num_iter (int): Number of iterations/Predictions for each entry
                that determines model uncertainity
            conf_int (float): confidence interval of the prediction made

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - probability of the class predicted by the model
                - standard error of the prediction made by the model
                - confidence interval as list [lower quantile, upper quantile]
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

        # define the quantile ranges for the confidence interval
        low_quantile = 0.5 - (conf_int / 2)
        high_quantile = 0.5 + (conf_int / 2)

        # get num_iter predictions to be able to measure model uncertainty
        proba = []
        for _ in range(num_iter):
            proba.append(cls.model.predict(X, verbose=2))

        # creating arrays for the confidence interval analysis:
        #   1 - mean of the predictions
        #   2 - std deviation of the predictions
        #   3 - lower quantile of the predictions
        #   4 - upper quantile of the predictions
        proba = np.array(proba)
        mean_arr = proba.mean(axis=0).reshape(-1, len(cls.race))
        std_arr = proba.std(axis=0).reshape(-1, len(cls.race))
        pct_low_arr = np.quantile(proba, low_quantile, axis=0).reshape(-1, len(cls.race))
        pct_high_arr = np.quantile(proba, high_quantile, axis=0).reshape(-1, len(cls.race))

        df.loc[nn, '__pred'] = np.argmax(mean_arr, axis=-1)

        df.loc[nn, 'race'] = df[nn]['__pred'].apply(lambda c:
                                                    cls.race[int(c)])
        stats = np.zeros((df.shape[0], 4))
        conf_int = []

        # selecting the statistics of the chosen class
        for i in range(df.shape[0]):
            select_class = np.argmax(mean_arr[i], axis=-1)
            stats[i, 0] = mean_arr[i, select_class]
            stats[i, 1] = std_arr[i, select_class]
            stats[i, 2] = pct_low_arr[i, select_class]
            stats[i, 3] = pct_high_arr[i, select_class]
            conf_int.append(np.array([stats[i, 2], stats[i, 3]]).tolist())

        df['proba'] = stats[:, 0]
        df['std_err'] = stats[:, 1]
        df['conf_int'] = conf_int

        # take out temporary working columns
        del df['__pred']
        del df['__last_name']

        pdf = pd.DataFrame(mean_arr, columns=cls.race)
        pdf.set_index(df[nn].index, inplace=True)

        rdf = pd.concat([df, pdf], axis=1)

        return rdf


pred_fl_reg_ln_cn_int = FloridaRegLnCnIntModel.pred_fl_reg_ln_cn_int


def main(argv=sys.argv[1:]):
    title = 'Predict Race/Ethnicity by name using Florida registration model with Confidence Interval'
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', default='fl-pred-ln-conf-output.csv',
                        help='Output file with prediction data')
    parser.add_argument('-i', '--iter', default=100, type=int,
                        help='Number of iterations to measure uncertainty')
    parser.add_argument('-c', '--conf', default=0.9, type=float,
                         help='Confidence interval of Predictions')
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

    rdf = pred_fl_reg_ln_cn_int(df, args.last, args.iter, args.conf)

    print("Saving output to file: `{0:s}`".format(args.output))
    rdf.columns = fixup_columns(rdf.columns)
    rdf.to_csv(args.output, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
