#!/usr/bin/env python
"""Dictionary-based race/ethnicity estimators.

These estimators use exact conditional frequencies from public name tables —
no neural network involved where the name is in-dictionary:

- ``census_fn``: first-name lookup against the Census 2020 first-name file
  (53,616 names covering ~94% of the enumerated population).
- ``pred_census_name``: six-category posterior from first+last name via naive
  Bayes over the census first-name and surname tables, with the census LSTM
  as fallback for out-of-dictionary surnames.
- ``pred_voter_name``: five-category posterior from the Rosenman-Olivella-
  Imai voter-file dictionaries (CC0; AL/FL/GA/LA/NC/SC voters).

The first+last combination assumes conditional independence of first and last
name given race (naive Bayes) — an explicit, documented approximation; see
docs/source/statistical_principles.md.
"""

from __future__ import annotations

import importlib.resources as resources
import json
import logging
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from .census_ln import CENSUS2000, CENSUS2010, CENSUS2020
from .ethnicolr_class import EthnicolrModelClass
from .torch_utils import apply_prior
from .utils import arg_parser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DATA = resources.files("ethnicolr") / "data"
CENSUS_FIRST_2020 = str(_DATA / "census/census_2020_first_names.csv")
ROSENMAN_FIRST = str(_DATA / "rosenman/first_name_race.csv.gz")
ROSENMAN_LAST = str(_DATA / "rosenman/last_name_race.csv.gz")
ROSENMAN_STATS = str(_DATA / "rosenman/rosenman_stats.json")

CENSUS_PCT_COLS = [
    "pctwhite",
    "pctblack",
    "pctapi",
    "pctaian",
    "pct2prace",
    "pcthispanic",
]
CENSUS_CATS = ["white", "black", "api", "aian", "2prace", "hispanic"]
LSTM_CATS = ["api", "black", "hispanic", "white"]
VOTER_CATS = ["white", "black", "hispanic", "asian", "other"]

_EPS = 1e-8


class _Tables:
    """Lazy caches for the dictionary tables."""

    _census_first: pd.DataFrame | None = None
    _census_last: dict[int, pd.DataFrame] = {}
    _census_marginal: np.ndarray | None = None
    _rosenman: dict[str, pd.DataFrame] = {}
    _rosenman_marginal: np.ndarray | None = None

    @classmethod
    def census_first(cls) -> pd.DataFrame:
        if cls._census_first is None:
            df = pd.read_csv(CENSUS_FIRST_2020).dropna(subset=["name"])
            for col in CENSUS_PCT_COLS + ["count"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            cls._census_first = df.set_index("name")
        return cls._census_first

    @classmethod
    def census_last(cls, year: int) -> pd.DataFrame:
        if year not in cls._census_last:
            path = {2000: CENSUS2000, 2010: CENSUS2010, 2020: CENSUS2020}[year]
            df = pd.read_csv(path, usecols=["name", *CENSUS_PCT_COLS]).dropna(
                subset=["name"]
            )
            for col in CENSUS_PCT_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            cls._census_last[year] = df.set_index("name")
        return cls._census_last[year]

    @classmethod
    def census_marginal(cls) -> np.ndarray:
        """US person-level race marginal implied by the first-name table."""
        if cls._census_marginal is None:
            df = cls.census_first()
            weights = (
                df[CENSUS_PCT_COLS].fillna(0).to_numpy()
                * df["count"].to_numpy()[:, None]
            )
            marginal = weights.sum(axis=0)
            cls._census_marginal = marginal / marginal.sum()
        return cls._census_marginal

    @classmethod
    def rosenman(cls, which: str) -> pd.DataFrame:
        if which not in cls._rosenman:
            path = ROSENMAN_FIRST if which == "first" else ROSENMAN_LAST
            df = pd.read_csv(path).dropna(subset=["name"])
            cls._rosenman[which] = df.set_index("name")[VOTER_CATS]
        return cls._rosenman[which]

    @classmethod
    def rosenman_marginal(cls) -> np.ndarray:
        if cls._rosenman_marginal is None:
            stats = json.loads(open(ROSENMAN_STATS).read())
            cls._rosenman_marginal = np.array(
                [stats["voter_population_marginal"][c] for c in VOTER_CATS]
            )
        return cls._rosenman_marginal


def wilson_interval(
    p: np.ndarray, n: np.ndarray, conf: float
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (vectorized)."""
    z = NormalDist().inv_cdf((1 + conf) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def _norm_names(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def census_fn(
    df: pd.DataFrame,
    fname_col: str,
    year: int = 2020,
    conf_int: float | None = None,
) -> pd.DataFrame:
    """Append Census 2020 demographic percentages by first name.

    Mirrors :func:`ethnicolr.census_ln` for first names, using the Census
    Bureau's 2020 first-name file. With ``conf_int``, adds exact Wilson
    score bounds (``<col>_lb``/``<col>_ub``) computed from the published
    name counts (sampling uncertainty only; the counts also carry the
    Bureau's small disclosure-avoidance noise).

    Args:
        df: Input DataFrame containing first names.
        fname_col: Column name containing first names.
        year: Data year (only 2020 is available).
        conf_int: Optional confidence level for Wilson bounds, e.g. 0.95.

    Returns:
        DataFrame with original data plus pctwhite/pctblack/pctapi/pctaian/
        pct2prace/pcthispanic columns (NaN for names not in the file).
    """
    if year != 2020:
        raise ValueError("First-name data is only available for 2020")
    df = EthnicolrModelClass.test_and_norm_df(df, fname_col)

    table = _Tables.census_first()
    keys = _norm_names(df[fname_col])
    matched = table.reindex(keys)

    rdf = df.copy()
    for col in CENSUS_PCT_COLS:
        rdf[col] = matched[col].to_numpy()

    if conf_int is not None:
        if not 0 < conf_int < 1:
            raise ValueError("conf_int must be between 0 and 1")
        counts = matched["count"].to_numpy()
        for col in CENSUS_PCT_COLS:
            p = rdf[col].to_numpy() / 100
            lb, ub = wilson_interval(p, counts, conf_int)
            rdf[f"{col}_lb"] = (lb * 100).round(2)
            rdf[f"{col}_ub"] = (ub * 100).round(2)

    matched_n = int(matched[CENSUS_PCT_COLS[0]].notna().sum())
    logger.info(f"Matched {matched_n} of {len(rdf)} first names")
    return rdf


def _bayes_combine(
    p_last: np.ndarray, p_first: np.ndarray, marginal: np.ndarray
) -> np.ndarray:
    """Naive-Bayes posterior: p(r|f,l) ∝ p(r|l)·p(r|f)/π(r)."""
    post = (
        np.maximum(p_last, _EPS)
        * np.maximum(p_first, _EPS)
        / np.maximum(marginal, _EPS)
    )
    return post / post.sum(axis=1, keepdims=True)


def pred_census_name(
    df: pd.DataFrame,
    lname_col: str,
    fname_col: str,
    year: int = 2020,
    prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict race/ethnicity from first and last name using census tables.

    Combines the census surname table p(race|last) and first-name table
    p(race|first) via naive Bayes (conditional independence of the two names
    given race). Six categories: white, black, api, aian, 2prace, hispanic.

    Out-of-dictionary surnames fall back to the census LSTM (four categories;
    aian/2prace are NaN on those rows). The ``basis`` column records what each
    row's estimate used: ``dict_both``, ``dict_last``, ``lstm+dict_first``,
    or ``lstm``.

    Args:
        df: Input DataFrame.
        lname_col: Last-name column.
        fname_col: First-name column.
        year: Census year for the surname table (2000, 2010, or 2020;
            first-name data is 2020 regardless).
        prior: Optional target class distribution (see statistical principles
            docs); reweights posteriors from the census population marginal.

    Returns:
        DataFrame with probability columns per category, ``race`` (argmax),
        and ``basis``.
    """
    from .pred_census_ln import pred_census_ln

    if lname_col not in df.columns or fname_col not in df.columns:
        raise ValueError("lname_col and fname_col must exist in the DataFrame")

    rdf = df.copy()
    last_keys = _norm_names(rdf[lname_col])
    first_keys = _norm_names(rdf[fname_col])

    last_table = _Tables.census_last(year)
    first_table = _Tables.census_first()
    marginal6 = _Tables.census_marginal()

    p_last6 = last_table[CENSUS_PCT_COLS].reindex(last_keys).to_numpy() / 100
    p_first6 = first_table[CENSUS_PCT_COLS].reindex(first_keys).to_numpy() / 100
    # Renormalize over non-suppressed cells
    with np.errstate(invalid="ignore"):
        p_last6 = np.nan_to_num(p_last6) / np.maximum(
            np.nan_to_num(p_last6).sum(axis=1, keepdims=True), _EPS
        )
        p_first6 = np.nan_to_num(p_first6) / np.maximum(
            np.nan_to_num(p_first6).sum(axis=1, keepdims=True), _EPS
        )

    has_last = ~np.isnan(last_table[CENSUS_PCT_COLS[0]].reindex(last_keys).to_numpy())
    has_first = ~np.isnan(
        first_table[CENSUS_PCT_COLS[0]].reindex(first_keys).to_numpy()
    )

    n = len(rdf)
    probs6 = np.full((n, len(CENSUS_CATS)), np.nan)
    basis = np.empty(n, dtype=object)

    # Dictionary paths (six categories)
    both = has_last & has_first
    probs6[both] = _bayes_combine(p_last6[both], p_first6[both], marginal6)
    basis[both] = "dict_both"
    last_only = has_last & ~has_first
    probs6[last_only] = p_last6[last_only]
    basis[last_only] = "dict_last"

    # LSTM fallback for out-of-dictionary surnames (four categories)
    ood = ~has_last
    if ood.any():
        lstm_in = rdf.loc[ood, [lname_col]].copy()
        lstm_out = pred_census_ln(lstm_in, lname_col, year=year)
        p_lstm4 = lstm_out[LSTM_CATS].to_numpy()

        cat_idx = [CENSUS_CATS.index(c) for c in LSTM_CATS]
        marginal4 = marginal6[cat_idx] / marginal6[cat_idx].sum()

        p_first4 = p_first6[ood][:, cat_idx]
        first4_valid = has_first[ood] & (p_first4.sum(axis=1) > _EPS)
        p_first4 = p_first4 / np.maximum(p_first4.sum(axis=1, keepdims=True), _EPS)

        combined4 = np.where(
            first4_valid[:, None],
            _bayes_combine(p_lstm4, p_first4, marginal4),
            p_lstm4,
        )
        block = np.full((int(ood.sum()), len(CENSUS_CATS)), np.nan)
        block[:, cat_idx] = combined4
        probs6[ood] = block
        basis[ood] = np.where(first4_valid, "lstm+dict_first", "lstm")

    if prior is not None:
        full = ~np.isnan(probs6).any(axis=1)
        if full.any():
            probs6[full] = apply_prior(
                probs6[full],
                CENSUS_CATS,
                prior,
                dict(zip(CENSUS_CATS, marginal6, strict=True)),
            )
        four = ~full
        if four.any():
            cat_idx = [CENSUS_CATS.index(c) for c in LSTM_CATS]
            marginal4 = marginal6[cat_idx] / marginal6[cat_idx].sum()
            prior4 = {c: prior[c] for c in LSTM_CATS if c in prior}
            if len(prior4) != len(LSTM_CATS):
                raise ValueError(f"prior must include all of {LSTM_CATS}")
            probs6[np.ix_(four, cat_idx)] = apply_prior(
                probs6[np.ix_(four, cat_idx)],
                LSTM_CATS,
                prior4,
                dict(zip(LSTM_CATS, marginal4, strict=True)),
            )

    for i, cat in enumerate(CENSUS_CATS):
        rdf[cat] = probs6[:, i]
    with np.errstate(invalid="ignore", all="ignore"):
        argmax = np.nanargmax(probs6, axis=1)
    rdf["race"] = [CENSUS_CATS[i] for i in argmax]
    rdf["basis"] = basis
    return rdf


def pred_voter_name(
    df: pd.DataFrame,
    lname_col: str,
    fname_col: str,
    prior: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict race/ethnicity from the Rosenman-Olivella-Imai dictionaries.

    Five categories (white, black, hispanic, asian, other) with self-reported
    race from six Southern-state voter files. First+last combined via naive
    Bayes; the ``basis`` column records ``dict_both``, ``dict_last``,
    ``dict_first``, or ``none`` (probabilities NaN when neither name is in
    the dictionaries).

    Args:
        df: Input DataFrame.
        lname_col: Last-name column.
        fname_col: First-name column.
        prior: Optional target class distribution; reweights posteriors from
            the six-state voter-population marginal.

    Returns:
        DataFrame with the five probability columns, ``race``, and ``basis``.
    """
    if lname_col not in df.columns or fname_col not in df.columns:
        raise ValueError("lname_col and fname_col must exist in the DataFrame")

    rdf = df.copy()
    last_keys = _norm_names(rdf[lname_col])
    first_keys = _norm_names(rdf[fname_col])

    last_table = _Tables.rosenman("last")
    first_table = _Tables.rosenman("first")
    marginal = _Tables.rosenman_marginal()

    p_last = last_table.reindex(last_keys).to_numpy()
    p_first = first_table.reindex(first_keys).to_numpy()
    has_last = ~np.isnan(p_last[:, 0])
    has_first = ~np.isnan(p_first[:, 0])

    n = len(rdf)
    probs = np.full((n, len(VOTER_CATS)), np.nan)
    basis = np.empty(n, dtype=object)

    both = has_last & has_first
    probs[both] = _bayes_combine(p_last[both], p_first[both], marginal)
    basis[both] = "dict_both"
    last_only = has_last & ~has_first
    probs[last_only] = p_last[last_only]
    basis[last_only] = "dict_last"
    first_only = has_first & ~has_last
    probs[first_only] = p_first[first_only]
    basis[first_only] = "dict_first"
    basis[~has_last & ~has_first] = "none"

    if prior is not None:
        found = has_last | has_first
        if found.any():
            probs[found] = apply_prior(
                probs[found],
                VOTER_CATS,
                prior,
                dict(zip(VOTER_CATS, marginal, strict=True)),
            )

    for i, cat in enumerate(VOTER_CATS):
        rdf[cat] = probs[:, i]
    race = np.full(n, np.nan, dtype=object)
    found = has_last | has_first
    if found.any():
        race[found] = [VOTER_CATS[i] for i in np.nanargmax(probs[found], axis=1)]
    rdf["race"] = race
    rdf["basis"] = basis

    logger.info(f"basis counts: {pd.Series(basis).value_counts().to_dict()}")
    return rdf


def census_fn_main(argv: list[str] | None = None) -> int:
    """CLI for census first-name lookup."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Append Census 2020 demographic data by first name"
    )
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("-f", "--first", required=True, help="First name column")
    parser.add_argument("-o", "--output", default="census-fn-output.csv")
    parser.add_argument(
        "-c", "--conf", type=float, default=None, help="Wilson interval level"
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rdf = census_fn(df, args.first, conf_int=args.conf)
    rdf.to_csv(args.output, index=False)
    logger.info(f"Output written: {args.output}")
    return 0


def pred_census_name_main(argv: list[str] | None = None) -> int:
    """CLI for census first+last prediction."""
    args = arg_parser(
        argv if argv is not None else sys.argv[1:],
        title="Predict race/ethnicity from first and last name (census tables)",
        default_out="census-name-output.csv",
        default_year=2020,
        year_choices=[2000, 2010, 2020],
        first=True,
    )
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rdf = pred_census_name(df, args.last, args.first, year=args.year)
    rdf.to_csv(args.output, index=False)
    logger.info(f"Output written: {args.output}")
    return 0


def pred_voter_name_main(argv: list[str] | None = None) -> int:
    """CLI for voter-dictionary prediction."""
    args = arg_parser(
        argv if argv is not None else sys.argv[1:],
        title="Predict race/ethnicity from voter-file name dictionaries",
        default_out="voter-name-output.csv",
        default_year=2022,
        year_choices=[2022],
        first=True,
    )
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rdf = pred_voter_name(df, args.last, args.first)
    rdf.to_csv(args.output, index=False)
    logger.info(f"Output written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(census_fn_main())
