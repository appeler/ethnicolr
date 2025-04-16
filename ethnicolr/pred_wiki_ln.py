#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wikipedia Last Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity from last names using an LSTM model trained on Wikipedia data.
"""

import sys
import os
import logging
from typing import List, Optional
import pandas as pd
from pkg_resources import resource_filename
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikiLnModel(EthnicolrModelClass):
    """
    Wikipedia Last Name prediction model.
    """
    MODELFN = "models/wiki/lstm/wiki_ln_lstm.h5"
    VOCABFN = "models/wiki/lstm/wiki_ln_vocab.csv"
    RACEFN = "models/wiki/lstm/wiki_race.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def get_model_paths(cls):
        return (
            resource_filename(__name__, cls.MODELFN),
            resource_filename(__name__, cls.VOCABFN),
            resource_filename(__name__, cls.RACEFN)
        )

    @classmethod
    def check_models_exist(cls):
        model_path, vocab_path, race_path = cls.get_model_paths()
        missing_files = [
            path for path in [model_path, vocab_path, race_path] if not os.path.exists(path)
        ]

        if missing_files:
            error_msg = (
                f"Required model files not found:\n{', '.join(missing_files)}\n\n"
                "Install models using: pip install ethnicolr[models]\n"
                "Or download from: https://github.com/appeler/ethnicolr/releases"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return True

    @classmethod
    def pred_wiki_ln(cls,
                     df: pd.DataFrame,
                     lname_col: str,
                     num_iter: int = 100,
                     conf_int: float = 1.0) -> pd.DataFrame:
        """
        Predict race/ethnicity using only the last name column.
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist in the DataFrame.")

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()

        # Drop rows with empty or missing last names
        before = len(working_df)
        working_df = working_df[
            working_df[lname_col].fillna("").astype(str).str.strip().str.len() > 0
        ]
        after = len(working_df)

        if before > after:
            logger.warning(f"Removed {before - after} rows with empty or missing last names.")

        if after == 0:
            raise ValueError("No valid last names to process. Please check your input data.")

        try:
            logger.info(f"Processing {after} last names using Wikipedia LSTM model")

            rdf = cls.transform_and_pred(
                df=working_df,
                newnamecol=lname_col,
                vocab_fn=vocab_path,
                race_fn=race_path,
                model_fn=model_path,
                ngrams=cls.NGRAMS,
                maxlen=cls.FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int
            )

            predicted = rdf.dropna(subset=["race"]).shape[0]
            logger.info(f"Predicted {predicted} of {after} rows ({predicted / after * 100:.1f}%)")
            logger.info(f"Added columns: {', '.join(set(rdf.columns) - set(df.columns))}")

            return rdf

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# For backward compatibility
pred_wiki_ln = WikiLnModel.pred_wiki_ln


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using Wikipedia model",
            default_out="wiki-pred-ln-output.csv",
            default_year=2017,
            year_choices=[2017]
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_wiki_ln(
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
        logger.error(f"Missing model files: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
