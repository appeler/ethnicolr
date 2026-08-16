#!/usr/bin/env python
"""Name -> country-of-origin prediction.

Predicts the likely country of origin of a name over ~130 country classes,
using an LSTM trained on Wikidata people (citizenship as the origin label;
melting-pot citizenships and ambiguous historical unions excluded — see
scripts/data-acquisition/wiki/mappings/country_to_origin.csv).

The reference population is people notable enough for Wikipedia/Wikidata, and
citizenship is a proxy for name origin — for migrants the two can differ.
Country-level classification from a name alone is genuinely ambiguous; prefer
the full probability distribution, `coverage=` conformal sets, or aggregate
country columns into regions by summing.
"""

import logging
import sys

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WikiOriginModel(EthnicolrModelClass):
    MODELFN = "models/wiki/lstm/wiki_origin_lstm_pt.pt"
    VOCABFN = "models/wiki/lstm/wiki_origin_vocab_pt.csv"
    RACEFN = "models/wiki/lstm/wiki_origin_race_pt.csv"

    NGRAMS = 2
    FEATURE_LEN = 25

    @classmethod
    def pred_wiki_origin(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        fname_col: str,
        conf_int: float = 1.0,
        num_iter: int = 100,
        prior: dict[str, float] | None = None,
        coverage: float | None = None,
    ) -> pd.DataFrame:
        """Predict the likely country of origin from first and last name.

        Args:
            df: Input DataFrame.
            lname_col: Last-name column.
            fname_col: First-name column.
            conf_int: Confidence level for Monte Carlo dropout intervals
                (1.0 = point predictions).
            num_iter: Monte Carlo iterations when conf_int < 1.
            prior: Optional target distribution over origin countries.
            coverage: Optional conformal coverage level (0.80/0.90/0.95);
                adds an ``origin_set`` column.

        Returns:
            DataFrame with one probability column per country, ``origin``
            (the argmax), and ``origin_set`` when coverage is requested.
        """
        if lname_col not in df.columns or fname_col not in df.columns:
            raise ValueError("lname_col and fname_col must exist in the DataFrame")

        working = df.copy()
        working["__name"] = (
            working[lname_col].fillna("").astype(str).str.strip()
            + " "
            + working[fname_col].fillna("").astype(str).str.strip()
        ).str.strip()

        # The base declares VOCABFN/RACEFN as `str | None` because non-LSTM
        # models have neither. This is an LSTM model and sets both; narrowing
        # them in the subclass does not work, since a mutable ClassVar is
        # invariant and `str` is not assignable to `str | None`.
        assert cls.VOCABFN is not None and cls.RACEFN is not None

        rdf = cls.transform_and_pred(
            df=working,
            newnamecol="__name",
            vocab_fn=cls.VOCABFN,
            race_fn=cls.RACEFN,
            model_fn=cls.MODELFN,
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int,
            prior=prior,
            coverage=coverage,
        )

        rdf = rdf.drop(columns=["__name"], errors="ignore")
        renames = {"race": "origin"}
        if "race_set" in rdf.columns:
            renames["race_set"] = "origin_set"
        return rdf.rename(columns=renames)


pred_wiki_origin = WikiOriginModel.pred_wiki_origin


def main(argv: list[str] | None = None) -> int:
    """CLI for name -> country-of-origin prediction."""
    args = arg_parser(
        argv if argv is not None else sys.argv[1:],
        title="Predict country of origin from first and last name",
        default_out="wiki-origin-output.csv",
        default_year=2026,
        year_choices=[2026],
        first=True,
    )
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    rdf = pred_wiki_origin(df, args.last, args.first, conf_int=args.conf)
    rdf.to_csv(args.output, index=False)
    logger.info(f"Output written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
