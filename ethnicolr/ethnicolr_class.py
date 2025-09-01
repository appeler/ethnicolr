#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from importlib.resources import files
from itertools import chain
import warnings

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import LSTM as _KerasLSTM

logger = logging.getLogger(__name__)


class _CompatLSTM(_KerasLSTM):
    """Backward-compatible LSTM layer.

    Older versions of Keras exposed a ``time_major`` argument on ``LSTM``
    layers. The pre-trained models distributed with *ethnicolr* were saved
    with this argument set to ``False``. Keras 3 removed the keyword which
    causes deserialization to fail with ``ValueError``. This shim accepts the
    deprecated argument but simply ignores it, allowing legacy models to load
    on newer Keras versions."""

    def __init__(self, *args, time_major=False, **kwargs):
        # ``time_major`` is unused but preserved for compatibility.
        super().__init__(*args, **kwargs)


class EthnicolrModelClass:
    vocab = None
    race = None
    model = None
    model_year = None

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Validate presence of ``col`` and drop rows with NaNs while preserving duplicates.

        Earlier implementations removed duplicate values which could silently drop
        rows. This caused a mismatch between the number of input rows and the
        predictions returned. The function now keeps duplicates but logs how many
        were found so callers remain informed.
        """
        if col not in df.columns:
            raise ValueError(f"The column '{col}' does not exist in the DataFrame.")

        original_length = len(df)

        # Track NaN removals
        nan_mask = df[col].isna()
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.info(f"Removing {nan_count} rows with NaN values in column '{col}'")
            df = df.dropna(subset=[col])

        if df.empty:
            raise ValueError("The name column has no non-NaN values.")

        # Log duplicates but keep them so prediction results align with inputs
        dup_count = df.duplicated(subset=[col]).sum()
        if dup_count > 0:
            logger.info(
                f"Preserving {dup_count} duplicate rows based on column '{col}'"
            )

        final_length = len(df)
        logger.info(
            f"Data filtering summary: {original_length} → {final_length} rows (kept {final_length/original_length*100:.1f}%)"
        )

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

        return [
            vocab.index("".join(gram)) if "".join(gram) in vocab else 0
            for gram in ngram_iter
        ]

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
        conf_int: float,
    ) -> pd.DataFrame:
        # Load resources
        vocab_path = files("ethnicolr") / vocab_fn
        model_path = files("ethnicolr") / model_fn
        race_path = files("ethnicolr") / race_fn

        df = df.copy()
        df = cls.test_and_norm_df(df, newnamecol)
        df[newnamecol] = df[newnamecol].astype(str).str.strip().str.title()
        rowindex_added = "__rowindex" not in df.columns
        if rowindex_added:
            df["__rowindex"] = np.arange(len(df))

        # Load model, vocab, and race label set once
        if cls.model is None:
            cls.vocab = pd.read_csv(vocab_path).vocab.tolist()
            cls.race = pd.read_csv(race_path).race.tolist()
            # ``time_major`` argument was removed in Keras 3. Models bundled with
            # ethnicolr were trained with ``time_major=False`` which causes
            # deserialization errors on newer Keras versions. We register a
            # compatibility LSTM that simply ignores the argument.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Argument `input_length` is deprecated",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Do not pass an `input_shape`/`input_dim` argument",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Argument `decay` is no longer supported",
                    category=UserWarning,
                )
                cls.model = load_model(
                    model_path,
                    custom_objects={"LSTM": _CompatLSTM},
                    compile=False,
                )

        # Vectorize input
        X = [cls.find_ngrams(cls.vocab, name, ngrams) for name in df[newnamecol]]
        X = sequence.pad_sequences(X, maxlen=maxlen)

        if conf_int == 1:
            proba = cls.model(X, training=False).numpy()
            proba_df = pd.DataFrame(proba, columns=cls.race)
            proba_df["race"] = proba_df.idxmax(axis=1)
            final_df = pd.concat(
                [df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1
            )

        else:
            lower_perc = (0.5 - conf_int / 2) * 100
            upper_perc = (0.5 + conf_int / 2) * 100

            logger.info(
                f"Generating {num_iter} samples for CI [{lower_perc:.1f}%, {upper_perc:.1f}%]"
            )

            all_preds = [cls.model(X, training=True).numpy() for _ in range(num_iter)]
            stacked = np.vstack(all_preds)
            pdf = pd.DataFrame(stacked, columns=cls.race)
            pdf["__rowindex"] = np.tile(df["__rowindex"].values, num_iter)

            agg = {
                col: [
                    "mean",
                    "std",
                    lambda x: np.percentile(x, q=lower_perc),
                    lambda x: np.percentile(x, q=upper_perc),
                ]
                for col in cls.race
            }

            summary = pdf.groupby("__rowindex").agg(agg).reset_index()

            # Flatten column names
            summary.columns = [
                "_".join(filter(None, map(str, col))) for col in summary.columns
            ]
            summary.columns = summary.columns.str.replace(
                "<lambda_0>", "lb"
            ).str.replace("<lambda_1>", "ub")

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
        if rowindex_added:
            final_df.drop(columns=["__rowindex"], inplace=True, errors="ignore")
        return final_df.reset_index(drop=True)
