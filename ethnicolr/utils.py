# -*- coding: utf-8 -*-

from re import VERBOSE
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename


def isstring(s):
    # if we use Python 3
    if sys.version_info[0] >= 3:
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def column_exists(df, col):
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        print("The specify column `{0!s}` not found in the input file".format(col))
        return False
    else:
        return True


def fixup_columns(cols):
    """Replace index location column to name with `col` prefix

    Args:
        cols (list): List of original columns

    Returns:
        list: List of column names

    """
    out_cols = []
    for col in cols:
        if type(col) == int:
            out_cols.append("col{:d}".format(col))
        else:
            out_cols.append(col)
    return out_cols


def find_ngrams(vocab, text, n):
    """Find and return list of the index of n-grams in the vocabulary list.

    Generate the n-grams of the specific text, find them in the vocabulary list
    and return the list of index have been found.

    Args:
        vocab (:obj:`list`): Vocabulary list.
        text (str): Input text
        n (int): N-grams

    Returns:
        list: List of the index of n-grams in the vocabulary list.

    """

    wi = []

    if not isstring(text):
        return wi

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
    df, newnamecol, cls, VOCAB, RACE, MODEL, NGRAMS, maxlen, num_iter, conf_int
):

    df[newnamecol] = df[newnamecol].str.strip().str.title()
    df["rowindex"] = df.index

    if cls.model is None:
        vdf = pd.read_csv(VOCAB)
        cls.vocab = vdf.vocab.tolist()

        rdf = pd.read_csv(RACE)
        cls.race = rdf.race.tolist()

        cls.model = load_model(MODEL)

    # build X from index of n-gram sequence
    X = np.array(df[newnamecol].apply(lambda c: find_ngrams(cls.vocab, c, NGRAMS)))
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # define the quantile ranges for the confidence interval
    lower_perc = 0.5 - (conf_int / 2)
    upper_perc = 0.5 + (conf_int / 2)

    # Predict
    pdf = pd.DataFrame()

    for _ in range(num_iter):
        pdf = pdf.append(pd.DataFrame(cls.model(X, training=True)))
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

    return final_df
