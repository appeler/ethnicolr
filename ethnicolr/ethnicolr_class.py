#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename
from itertools import chain

class EthnicolrModelClass:
    vocab = None
    race = None
    model = None
    model_year = None

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Handles cases like:
            - column doesn't exist, nukes missing rows

        """
        if col and (col not in df.columns):
            raise Exception(f"The column {col} doesn't exist in the dataframe.")

        df.dropna(subset=[col], inplace = True)
        if df.shape[0] == 0:
            raise Exception("The name column has no non-NaN values.")

        df.drop_duplicates(subset = [col], inplace = True)

        return df

    @staticmethod
    def n_grams(seq, n:int=1):
        """Returns an iterator over the n-grams given a listTokens"""
        shiftToken = lambda i: (el for j, el in enumerate(seq) if j >= i)
        shiftedTokens = (shiftToken(i) for i in range(n))
        tupleNGrams = zip(*shiftedTokens)
        return tupleNGrams

    @staticmethod
    def range_ngrams(listTokens, ngramRange=(1, 2)):
        """Returns an iterator over all n-grams for n in range(ngramRange)
          given a listTokens.
        """
        ngrams = (ngramRange[0], ngramRange[1] + 1)
        return chain(*(EthnicolrModelClass.n_grams(listTokens, i) for i in range(*ngramRange)))


    @staticmethod
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

        if type(n) is tuple:
            a = EthnicolrModelClass.range_ngrams(text, n)
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


    @classmethod
    def transform_and_pred(cls,
        df: pd.DataFrame, newnamecol: str, vocab_fn: str , race_fn: str, model_fn: str, ngrams, maxlen: int, num_iter: int, conf_int: float
    ) -> pd.DataFrame:

        VOCAB = resource_filename(__name__, vocab_fn)
        MODEL = resource_filename(__name__, model_fn)
        RACE = resource_filename(__name__, race_fn)

        df = EthnicolrModelClass.test_and_norm_df(df, newnamecol)

        df[newnamecol] = df[newnamecol].str.strip().str.title()
        df["rowindex"] = df.index

        if cls.model is None:
            vdf = pd.read_csv(VOCAB)
            cls.vocab = vdf.vocab.tolist()

            rdf = pd.read_csv(RACE)
            cls.race = rdf.race.tolist()

            cls.model = load_model(MODEL)

        # build X from index of n-gram sequence
        X = np.array(df[newnamecol].apply(lambda c: EthnicolrModelClass.find_ngrams(cls.vocab, c, ngrams)))
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
