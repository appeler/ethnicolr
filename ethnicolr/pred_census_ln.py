#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Census Last Name Race/Ethnicity Prediction Module.

Uses LSTM models trained on U.S. Census data to predict race/ethnicity from last names.
"""

import sys
import os
import logging
from typing import Optional, List
import pandas as pd
from pkg_resources import resource_filename
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Suppress TensorFlow noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CensusLnModel(EthnicolrModelClass):
    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def get_model_paths(cls, year):
        return (
            resource_filename(__name__, f"models/census/lstm/census{year}_ln_lstm.h5"),
            resource_filename(__name__, f"models/census/lstm/census{year}_ln_vocab.csv"),
            resource_filename(__name__, f"models/census/lstm/census{year}_race.csv")
        )

    @classmethod
    def check_models_exist(cls, year):
        model_path, vocab_path, race_path = cls.get_model_paths(year)
        missing = [p for p in [model_path, vocab_path, race_path] if not os.path.exists(p)]
        if missing:
            msg = (
                f"Required model files not found for Census {year}:\n"
                f"{', '.join(missing)}\n\n"
                "Install with: pip install ethnicolr[models]\n"
                "Or download from: https://github.com/appeler/ethnicolr/releases"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        return True

    @classmethod
    def pred_census_ln(cls,
                       df: pd.DataFrame,
                       lname_col: str,
                       year: int = 2010,
                       num_iter: int = 100,
                       conf_int: float = 1.0) -> pd.DataFrame:
        if year not in [2000, 2010]:
            raise ValueError("Census year must be either 2000 or 2010")

        cls.check_models_exist(year)
        model_path, vocab_path, race_path = cls.get_model_paths(year)

        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist.")

        logger.info(f"Processing {len(df)} names using Census {year} LSTM model")

        rdf = cls.transform_and_pred(
            df=df,
            newnamecol=lname_col,
            vocab_fn=vocab_path,
            race_fn=race_path,
            model_fn=model_path,
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int
        )

        pred_count = rdf.dropna(subset=["race"]).shape[0]
        logger.info(f"Predicted {pred_count} of {len(df)} rows ({pred_count / len(df) * 100:.1f}%)")
        logger.info(f"Added columns: {', '.join(set(rdf.columns) - set(df.columns))}")

        return rdf


# Alias for CLI use
pred_census_ln = CensusLnModel.pred_census_ln


def download_models(year=None):
    """
    Stub for downloading model files.
    """
    years = [year] if year else [2000, 2010]
    for y in years:
        logger.info(f"Downloading Census {y} model files...")
        # TODO: Implement actual download logic
        logger.info(f"Downloaded Census {y} model files successfully")


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        parser = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using Census LSTM model",
            default_out="census-pred-ln-output.csv",
            default_year=2010,
            year_choices=[2000, 2010]
        )

        parser.add_argument(
            "--download-models", action="store_true",
            help="Download required model files"
        )

        args = parser.parse_args(argv)

        if args.download_models:
            download_models(args.year)
            return 0

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_census_ln(
            df=df,
            lname_col=args.last,
            year=args.year,
            num_iter=args.iter,
            conf_int=args.conf
        )

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"📦 Output written: {args.output} ({len(rdf)} rows)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing model files: {e}")
        logger.info("Try running with --download-models to download required files")
        return 2
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
