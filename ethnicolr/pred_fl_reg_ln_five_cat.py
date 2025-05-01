#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Florida Last Name 5-Category Race/Ethnicity Prediction Module.

Uses LSTM models trained on Florida voter registration data to predict
race/ethnicity from last names, collapsed to 5 categories.
"""

import sys
import os
import logging
import pandas as pd
from typing import Optional, List
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloridaRegLnFiveCatModel(EthnicolrModelClass):
    MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_5_cat{0:s}.h5"
    VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab_5_cat{0:s}.csv"
    RACEFN = "models/fl_voter_reg/lstm/fl_ln_five_cat_race{0:s}.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def pred_fl_reg_ln(cls,
                       df: pd.DataFrame,
                       lname_col: str,
                       num_iter: int = 100,
                       conf_int: float = 1.0,
                       year: int = 2022) -> pd.DataFrame:
        """
        Predict race/ethnicity using Florida last name model (5-category version).
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' does not exist in the DataFrame.")

        suffix = "_2022" if year == 2022 else ""
        logger.info(f"Using FL 5-cat model for year {year}")

        rdf = cls.transform_and_pred(
            df=df.copy(),
            newnamecol=lname_col,
            vocab_fn=cls.VOCABFN.format(suffix),
            race_fn=cls.RACEFN.format(suffix),
            model_fn=cls.MODELFN.format(suffix),
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int
        )

        return rdf


# CLI alias
pred_fl_reg_ln_five_cat = FloridaRegLnFiveCatModel.pred_fl_reg_ln


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using the Florida registration 5-cat model",
            default_out="fl-pred-ln-five-cat-output.csv",
            default_year=2022,
            year_choices=[2017, 2022]
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_fl_reg_ln_five_cat(
            df=df,
            lname_col=args.last,
            num_iter=args.iter,
            conf_int=args.conf,
            year=args.year
        )

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"📦 Output written: {args.output} ({len(rdf)} rows)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing model files: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
