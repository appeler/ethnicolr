#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from itertools import chain
from pathlib import Path
from importlib.resources import files

logger = logging.getLogger(__name__)


class EthnicolrModelClass:
    vocab = None
    race = None
    model = None
    model_year = None

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Validates the presence of the column and removes rows with NaNs or duplicates.
        """
        if col not in df.columns:
            raise ValueError(f"The column '{col}' does not exist in the DataFrame.")

        df = df.dropna(subset=[col])
        if df.empty:
            raise ValueError("The name column has no non-NaN values.")
        df = df.drop_duplicates(subset=[col])
        return df

    @staticmethod
    def n_grams(seq, n: int = 1):
        """Returns an iterator over n-grams given a sequence"""
        shiftToken = lambda i: (el for j, el in enumerate(seq) if j >= i)
        shiftedTokens = (shiftToken(i) for i in range(n))
        return zip(*shiftedTokens)

    @staticmethod
    def range_ngrams(seq, ngramRange=(1, 2)):
        """Returns iterator over all n-grams for n in range"""
        return chain(*(EthnicolrModelClass.n_grams(seq, i) for i in range(*ngramRange)))

    @staticmethod
    def find_ngrams(vocab, text: str, n) -> list:
        """
        Generate n-grams from a string and return their indices in the vocabulary.
        """
        if isinstance(n, tuple):
            ngram_iter = EthnicolrModelClass.range_ngrams(text, n)
        else:
            ngram_iter = zip(*[text[i:] for i in range(n)])

        return [vocab.index("".join(gram)) if "".join(gram) in vocab else 0 for gram in ngram_iter]

    @classmethod
    def transform_and_pred(
        cls,
        df: pd.DataFrame,
        newnamecol: str,
        vocab_fn: str,
        race_fn: str,
        model_fn: str,
        ngrams,
        maxlen: int,
        num_iter: int,
        conf_int: float
    ) -> pd.DataFrame:
        # Load resources
        vocab_path = files("ethnicolr") / vocab_fn
        model_path = files("ethnicolr") / model_fn
        race_path = files("ethnicolr") / race_fn

        df = df.copy()
        df = cls.test_and_norm_df(df, newnamecol)
        df[newnamecol] = df[newnamecol].astype(str).str.strip().str.title()
        df["__rowindex"] = np.arange(len(df))

        # Load model, vocab, and race label set once
        if cls.model is None:
            cls.vocab = pd.read_csv(vocab_path).vocab.tolist()
            cls.race = pd.read_csv(race_path).race.tolist()
            cls.model = load_model(model_path)

        # Vectorize input
        X = [cls.find_ngrams(cls.vocab, name, ngrams) for name in df[newnamecol]]
        X = sequence.pad_sequences(X, maxlen=maxlen)

        if conf_int == 1:
            proba = cls.model(X, training=False).numpy()
            proba_df = pd.DataFrame(proba, columns=cls.race)
            proba_df["race"] = proba_df.idxmax(axis=1)
            final_df = pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

        else:
            lower_perc = (0.5 - conf_int / 2) * 100
            upper_perc = (0.5 + conf_int / 2) * 100

            logger.info(f"Generating {num_iter} samples for CI [{lower_perc:.1f}%, {upper_perc:.1f}%]")

            all_preds = [cls.model(X, training=True).numpy() for _ in range(num_iter)]
            stacked = np.vstack(all_preds)
            pdf = pd.DataFrame(stacked, columns=cls.race)
            pdf["__rowindex"] = np.tile(df["__rowindex"].values, num_iter)

            agg = {
                col: ["mean", "std",
                      lambda x: np.percentile(x, q=lower_perc),
                      lambda x: np.percentile(x, q=upper_perc)]
                for col in cls.race
            }

            summary = pdf.groupby("__rowindex").agg(agg).reset_index()

            # Flatten column names
            summary.columns = ['_'.join(filter(None, map(str, col))) for col in summary.columns]
            summary.columns = summary.columns \
                .str.replace("<lambda_0>", "lb") \
                .str.replace("<lambda_1>", "ub")

            # Choose race with highest mean
            means = [col for col in summary.columns if col.endswith("_mean")]
            summary["race"] = summary[means].idxmax(axis=1).str.replace("_mean", "")

            # Convert CI columns to float
            for suffix in ["_lb", "_ub"]:
                target = [col for col in summary.columns if col.endswith(suffix)]
                summary[target] = summary[target].astype(float)

            # Align rowindex column name for join
            summary.rename(columns={"__rowindex_": "__rowindex"}, inplace=True)

            final_df = df.merge(summary, on="__rowindex", how="left")

        # Clean up
        final_df.drop(columns=["__rowindex"], inplace=True, errors="ignore")
        return final_df.reset_index(drop=True)


