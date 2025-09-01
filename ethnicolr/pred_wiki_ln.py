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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    def pred_wiki_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict race/ethnicity using only the last name column.
        """
        if lname_col not in df.columns:
            raise ValueError(
                f"The last name column '{lname_col}' doesn't exist in the DataFrame."
            )

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()
        original_length = len(working_df)

        logger.info(f"Processing {original_length} last names")

        # Create normalized name column for tracking
        working_df["name_normalized"] = (
            working_df[lname_col].fillna("").astype(str).str.strip()
        )

        # Track which names will be skipped and why
        empty_original = working_df["name_normalized"].str.len() == 0

        # Create processing status column
        working_df["processing_status"] = "processed"
        working_df.loc[empty_original, "processing_status"] = "skipped_empty_original"

        # Count what we're about to skip
        to_skip = working_df["processing_status"] != "processed"
        skipped_count = to_skip.sum()

        if skipped_count > 0:
            logger.warning(
                f"Will skip {skipped_count} names with empty/missing last names"
            )

        # Separate processable and skipped names
        processable_df = working_df[~to_skip].copy()
        skipped_df = working_df[to_skip].copy()

        if len(processable_df) == 0:
            logger.warning(
                "No valid last names to process. Returning original data with status info."
            )
            result_df = working_df.copy()
            result_df["race"] = None
            return result_df

        try:
            logger.info(
                f"Applying Wikipedia last name model to {len(processable_df)} processable names (confidence interval: {conf_int})"
            )

            # Run prediction only on processable names
            pred_df = cls.transform_and_pred(
                df=processable_df,
                newnamecol=lname_col,
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
            columns_to_drop = ["__rowindex"]
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
            year_choices=[2017],
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_wiki_ln(
            df=df, lname_col=args.last, num_iter=args.iter, conf_int=args.conf
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
