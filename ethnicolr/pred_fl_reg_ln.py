#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Florida Last Name Race/Ethnicity Prediction Module.

Uses an LSTM model trained on Florida voter registration data
to predict race/ethnicity from last names.
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


class FloridaRegLnModel(EthnicolrModelClass):
    """
    Florida last-name model for race/ethnicity prediction.
    """
    MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm.h5"
    VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab.csv"
    RACEFN = "models/fl_voter_reg/lstm/fl_ln_race.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def pred_fl_reg_ln(cls,
                       df: pd.DataFrame,
                       lname_col: str,
                       num_iter: int = 100,
                       conf_int: float = 1.0) -> pd.DataFrame:
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' does not exist in the DataFrame.")

        logger.info(f"Predicting race/ethnicity for {len(df)} rows using Florida LSTM model")

        rdf = cls.transform_and_pred(
            df=df.copy(),
            newnamecol=lname_col,
            vocab_fn=cls.VOCABFN,
            race_fn=cls.RACEFN,
            model_fn=cls.MODELFN,
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int
        )

        logger.info(f"Prediction complete. Added columns: {', '.join(set(rdf.columns) - set(df.columns))}")
        return rdf


# CLI alias
pred_fl_reg_ln = FloridaRegLnModel.pred_fl_reg_ln


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using Florida registration model",
            default_out="fl-pred-ln-output.csv",
            default_year=2017,
            year_choices=[2017]
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_fl_reg_ln(
            df=df,
            lname_col=args.last,
            num_iter=args.iter,
            conf_int=args.conf
        )

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"📦 Output written: {args.output} ({len(rdf)} rows)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing model file: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
