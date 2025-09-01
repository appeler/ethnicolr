#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Florida Voter Registration Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity using full names based on an LSTM model trained on Florida voter data.
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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FloridaRegNameModel(EthnicolrModelClass):
    """
    Florida full-name LSTM prediction model.
    """

    MODELFN = "models/fl_voter_reg/lstm/fl_all_name_lstm.h5"
    VOCABFN = "models/fl_voter_reg/lstm/fl_all_name_vocab.csv"
    RACEFN = "models/fl_voter_reg/lstm/fl_name_race.csv"

    NGRAMS = 2
    FEATURE_LEN = 25

    @classmethod
    def get_model_paths(cls):
        return (
            resource_filename(__name__, cls.MODELFN),
            resource_filename(__name__, cls.VOCABFN),
            resource_filename(__name__, cls.RACEFN),
        )

    @classmethod
    def check_models_exist(cls):
        model_path, vocab_path, race_path = cls.get_model_paths()
        missing_files = [
            path
            for path in [model_path, vocab_path, race_path]
            if not os.path.exists(path)
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
    def pred_fl_reg_name(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        fname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict race/ethnicity using the Florida voter LSTM model.
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist.")
        if fname_col not in df.columns:
            raise ValueError(f"The first name column '{fname_col}' doesn't exist.")

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()
        original_length = len(working_df)

        # Generate a unique temp name column
        temp_col = "__ethnicolr_temp_name"
        while temp_col in working_df.columns:
            temp_col += "_"

        logger.info(f"Processing {original_length} full names")

        # Create original full name column for reference
        working_df["__name"] = (
            working_df[lname_col].fillna("").astype(str).str.strip()
            + " "
            + working_df[fname_col].fillna("").astype(str).str.strip()
        ).str.strip()

        # Build full name and sanitize for processing
        working_df[temp_col] = working_df["__name"].str.title()

        # Add normalization info columns
        working_df["name_normalized"] = working_df["__name"]
        working_df["name_normalized_clean"] = working_df[temp_col]

        # Track which names will be skipped and why
        empty_original = working_df["__name"].str.strip().str.len() == 0
        empty_after_clean = working_df[temp_col].str.strip().str.len() == 0

        # Create processing status column
        working_df["processing_status"] = "processed"
        working_df.loc[empty_original, "processing_status"] = "skipped_empty_original"
        working_df.loc[empty_after_clean & ~empty_original, "processing_status"] = (
            "skipped_empty_after_normalization"
        )

        # Count what we're about to skip
        to_skip = working_df["processing_status"] != "processed"
        skipped_count = to_skip.sum()

        if skipped_count > 0:
            skip_reasons = working_df[to_skip]["processing_status"].value_counts()
            logger.warning(f"Will skip {skipped_count} names:")
            for reason, count in skip_reasons.items():
                logger.warning(f"  - {count} names: {reason}")

        # Separate processable and skipped names
        processable_df = working_df[~to_skip].copy()
        skipped_df = working_df[to_skip].copy()

        if len(processable_df) == 0:
            logger.warning(
                "No valid names to process after cleaning. Returning original data with status info."
            )
            result_df = working_df.copy()
            result_df["race"] = None
            return result_df

        try:
            logger.info(
                f"Applying Florida voter name model to {len(processable_df)} processable names (confidence interval: {conf_int})"
            )

            # Run prediction only on processable names
            pred_df = cls.transform_and_pred(
                df=processable_df,
                newnamecol=temp_col,
                vocab_fn=vocab_path,
                race_fn=race_path,
                model_fn=model_path,
                ngrams=cls.NGRAMS,
                maxlen=cls.FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int,
            )

            # For skipped names, add empty prediction columns
            if len(skipped_df) > 0:
                # Get all prediction columns from successful predictions
                pred_columns = set(pred_df.columns) - set(working_df.columns)
                for col in pred_columns:
                    if col not in skipped_df.columns:
                        if col == "race":
                            skipped_df[col] = None
                        else:
                            skipped_df[col] = float("nan")

            # Combine results
            result_df = pd.concat([pred_df, skipped_df], ignore_index=True)

            # Sort by original order if possible
            if "__rowindex" in result_df.columns:
                result_df = result_df.sort_values("__rowindex").reset_index(drop=True)

            # Clean up temporary columns
            columns_to_drop = [temp_col, "__rowindex"]
            result_df.drop(
                columns=[col for col in columns_to_drop if col in result_df.columns],
                inplace=True,
                errors="ignore",
            )

            predicted = result_df["race"].notna().sum()
            logger.info(
                f"Successfully predicted {predicted} of {original_length} names ({predicted / original_length * 100:.1f}%)"
            )

            if skipped_count > 0:
                logger.info(
                    f"Skipped {skipped_count} names - see 'processing_status' column for details"
                )

            logger.info(
                f"Added columns: {', '.join(set(result_df.columns) - set(df.columns))}"
            )

            return result_df

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# For backward compatibility
pred_fl_reg_name = FloridaRegNameModel.pred_fl_reg_name


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by name using Florida registration model",
            default_out="fl-pred-name-output.csv",
            default_year=2017,
            year_choices=[2017],
            first=True,
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_fl_reg_name(
            df=df,
            lname_col=args.last,
            fname_col=args.first,
            num_iter=args.iter,
            conf_int=args.conf,
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
