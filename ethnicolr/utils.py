# -*- coding: utf-8 -*-

from re import VERBOSE
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename
from itertools import chain


def isstring(s: str) -> bool:
    # if we use Python 3
    if sys.version_info[0] >= 3:
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def column_exists(df: pd.DataFrame, col: str) -> bool:
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        print(f"Column `{col}` not found in the input file")
        return False
    else:
        return True


def fixup_columns(cols: list) -> list:
    """Replace index location column to name with `col` prefix

    Args:
        cols (list): List of original columns

    Returns:
        list: List of column names

    """
    out_cols = []
    for col in cols:
        if type(col) == int:
            out_cols.append(f"col{col}")
        else:
            out_cols.append(col)
    return out_cols


def n_grams(seq, n:int=1):
    """Returns an itirator over the n-grams given a listTokens"""
    shiftToken = lambda i: (el for j,el in enumerate(seq) if j>=i)
    shiftedTokens = (shiftToken(i) for i in range(n))
    tupleNGrams = zip(*shiftedTokens)
    return tupleNGrams


def range_ngrams(listTokens, ngramRange=(1,2)):
    """Returns an itirator over all n-grams for n in range(ngramRange)
       given a listTokens.
    """

    ngrams = (ngramRange[0], ngramRange[1] + 1)
    return chain(*(n_grams(listTokens, i) for i in range(*ngramRange)))


def find_ngrams(vocab, text: str, n) -> list:
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int or tuple): N-grams or tuple of range N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    if not isstring(text):
        return wi

    if type(n) is tuple:
        a = range_ngrams(text, n)
    else:
        a = zip(*[text[i:] for i in range(n)])

    for i in a:
        w = "".join(i)
        try:
            idx = vocab.index(w)
        except Exception as e:
            idx = 0
        wi.append(idx)
    return wi


def transform_and_pred(
    df: pd.DataFrame, newnamecol: str, cls, VOCAB, RACE, MODEL, NGRAMS, maxlen: int, num_iter: int, conf_int: float
) -> pd.DataFrame:

    df[newnamecol] = df[newnamecol].str.strip().str.title()
    df["rowindex"] = df.index

    if cls.model is None:
        vdf = pd.read_csv(VOCAB)
        cls.vocab = vdf.vocab.tolist()

        rdf = pd.read_csv(RACE)
        cls.race = rdf.race.tolist()

        cls.model = load_model(MODEL)

    # build X from index of n-gram sequence
    X = np.array(df[newnamecol].apply(lambda c: find_ngrams(cls.vocab,
                                                            c, NGRAMS)))
    X = sequence.pad_sequences(X, maxlen=maxlen)

    if conf_int == 1:
        # Predict
        proba = cls.model(X, training=False).numpy()
        pdf = pd.DataFrame(proba, columns=cls.race)
        pdf["__race"] = np.argmax(proba, axis=-1)
        pdf["race"] = pdf["__race"].apply(lambda c: cls.race[int(c)])
        del pdf["__race"]
        final_df = pd.concat([df.reset_index(drop=True),
                              pdf.reset_index(drop=True)], axis=1 )
    else:
        # define the quantile ranges for the confidence interval
        lower_perc = (0.5 - (conf_int / 2)) * 100
        upper_perc = (0.5 + (conf_int / 2)) * 100

        # Predict
        pdf = pd.DataFrame()

        for _ in range(num_iter):
            pdf = pd.concat([pdf, pd.DataFrame(cls.model(X, training=True))])
        print(cls.race)
        pdf.columns = cls.race
        pdf["rowindex"] = pdf.index

        res = (
            pdf.groupby("rowindex")
            .agg(
                [
                    np.mean,
                    np.std,
                    lambda x: np.percentile(x, q=lower_perc),
                    lambda x: np.percentile(x, q=upper_perc),
                ]
            )
            .reset_index()
        )
        res.columns = [f"{i}_{j}" for i, j in res.columns]
        res.columns = res.columns.str.replace("<lambda_0>", "lb")
        res.columns = res.columns.str.replace("<lambda_1>", "ub")
        res.columns = res.columns.str.replace("rowindex_", "rowindex")

        means = list(filter(lambda x: "_mean" in x, res.columns))
        res["race"] = res[means].idxmax(axis=1).str.replace("_mean", "")

        for suffix in ["_lb", "ub"]:
            conv_filt = list(filter(lambda x: suffix in x, res.columns))
            res[conv_filt] = res[conv_filt].to_numpy().astype(float)

        final_df = df.merge(res, on="rowindex", how="left")

    del final_df['rowindex']
    del df['rowindex']

    return final_df

def arg_parser(argv, title: str, default_out: str, default_year: int, year_choices: list, first: bool = False):

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('input', default=None,
                        help='Input file')
    parser.add_argument('-o', '--output',
                        default=default_out,
                        help='Output file with prediction data')
    if first:
        parser.add_argument('-f', '--first', required=True,
                        help='Column name for the column with the first name')
    parser.add_argument('-l', '--last', required=True,
                        help='Column name for the column with the last name')
    parser.add_argument('-i', '--iter', default=100, type=int,
                        help='Number of iterations to measure uncertainty')
    parser.add_argument('-c', '--conf', default=1.0, type=float,
                        help='Confidence interval of Predictions')
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=default_year,
        choices=year_choices,
        help=f"Year of data (default={default_year})",
    )
    args = parser.parse_args(argv)

    print(args)

    return(args)

