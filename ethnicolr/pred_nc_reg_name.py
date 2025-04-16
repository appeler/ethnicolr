#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
North Carolina Voter Registration Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity using full names based on an LSTM model trained on NC voter data (12-category).
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


class NCRegNameModel(EthnicolrModelClass):
    """
    North Carolina 12-category full name prediction model.
    """
    MODELFN = "models/nc_voter_reg/lstm/nc_voter_name_lstm_oversample.h5"
    VOCABFN = "models/nc_voter_reg/lstm/nc_voter_name_vocab_oversample.csv"
    RACEFN = "models/nc_voter_reg/lstm/nc_name_race.csv"

    NGRAMS = (2, 3)
    FEATURE_LEN = 25

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
    def pred_nc_reg_name(cls,
                         df: pd.DataFrame,
                         lname_col: str,
                         fname_col: str,
                         num_iter: int = 100,
                         conf_int: float = 1.0) -> pd.DataFrame:
        """
        Predict race/ethnicity using full names and the NC voter registration model.
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist.")
        if fname_col not in df.columns:
            raise ValueError(f"The first name column '{fname_col}' doesn't exist.")

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()

        # Safe temporary column name
        temp_col = "__ethnicolr_temp_name"
        while temp_col in working_df.columns:
            temp_col += "_"

        logger.info(f"Processing {len(working_df)} names")

        working_df[temp_col] = (
            working_df[lname_col].fillna("").astype(str).str.strip() + " " +
            working_df[fname_col].fillna("").astype(str).str.strip()
        ).str.title()

        before = len(working_df)
        working_df = working_df[working_df[temp_col].str.strip().str.len() > 0]
        after = len(working_df)

        if before > after:
            logger.warning(f"Removed {before - after} rows with empty or invalid names.")

        if after == 0:
            raise ValueError("No valid names to process. Please check your input data.")

        try:
            logger.info(f"Applying NC voter name model (confidence interval: {conf_int})")

            rdf = cls.transform_and_pred(
                df=working_df,
                newnamecol=temp_col,
                vocab_fn=vocab_path,
                race_fn=race_path,
                model_fn=model_path,
                ngrams=cls.NGRAMS,
                maxlen=cls.FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int
            )

            if temp_col in rdf.columns:
                rdf.drop(columns=[temp_col], inplace=True)

            predicted = rdf.dropna(subset=["race"]).shape[0]
            logger.info(f"Predicted {predicted} of {after} names ({predicted / after * 100:.1f}%)")
            logger.info(f"Added columns: {', '.join(set(rdf.columns) - set(df.columns))}")

            return rdf

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Alias for backwards compatibility
pred_nc_reg_name = NCRegNameModel.pred_nc_reg_name


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by name using NC 12-category voter registration model",
            default_out="nc-pred-name-output.csv",
            default_year=2017,
            year_choices=[2017],
            first=True
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_nc_reg_name(
            df=df,
            lname_col=args.last,
            fname_col=args.first,
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
