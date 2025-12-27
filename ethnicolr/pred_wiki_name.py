#!/usr/bin/env python
"""
Wikipedia Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity using full names based on LSTM models trained on Wikipedia data.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import unicodedata
from importlib import resources

import numpy as np
import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Normalize a name by removing accents and punctuation.

    The Wikipedia models expect names containing only ASCII letters.
    This helper strips diacritics, drops any non-letter characters and
    collapses multiple spaces. The cleaned string is title cased to match
    the model's training format.

    Args:
        name: Input name string to normalize.

    Returns:
        Cleaned name with only ASCII letters, title cased.

    Example:
        >>> _normalize_name('José García-Smith')
        'Jose Garcia Smith'
    """
    # Remove accents
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("utf-8")
    # Drop everything except letters and spaces
    name = re.sub(r"[^A-Za-z ]+", " ", name)
    # Condense whitespace and strip
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


class WikiNameModel(EthnicolrModelClass):
    """Wikipedia-based full name prediction model.

    LSTM model trained on Wikipedia data for predicting race/ethnicity
    from full names (first + last). Provides detailed ethnic categories
    beyond the basic Census classifications.
    """

    MODELFN = "models/wiki/lstm/wiki_name_lstm.h5"
    VOCABFN = "models/wiki/lstm/wiki_name_vocab.csv"
    RACEFN = "models/wiki/lstm/wiki_name_race.csv"

    NGRAMS = 2
    FEATURE_LEN = 25

    @classmethod
    def get_model_paths(cls):
        """Get file paths for Wikipedia name model components.

        Returns:
            Tuple of (model_path, vocab_path, race_path) as strings.
        """
        base_path = resources.files(__name__.split(".")[0])
        vocab_path = str(base_path / cls.VOCABFN) if cls.VOCABFN else None
        race_path = str(base_path / cls.RACEFN) if cls.RACEFN else None
        return (
            str(base_path / cls.MODELFN),
            vocab_path,
            race_path,
        )

    @classmethod
    def check_models_exist(cls):
        """Verify that all required model files exist.

        Raises:
            FileNotFoundError: If any model files are missing.

        Returns:
            True if all files exist.
        """
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
    def pred_wiki_name(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        fname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """Predict race/ethnicity from full names using Wikipedia model.

        Uses LSTM trained on Wikipedia biographical data to predict detailed
        ethnic categories. Handles name normalization and provides comprehensive
        processing status tracking.

        Args:
            df: Input DataFrame containing first and last names.
            lname_col: Column name containing last names.
            fname_col: Column name containing first names.
            num_iter: Monte Carlo iterations for confidence intervals.
            conf_int: Confidence level (1.0 for point estimates).

        Returns:
            DataFrame with original data plus prediction columns:
            - 'race': Predicted ethnicity category
            - Probability columns for each ethnicity
            - 'processing_status': Name processing outcome
            - 'name_normalized': Original combined name
            - 'name_normalized_clean': Cleaned name used for prediction
            - Confidence bounds if conf_int < 1.0

        Raises:
            ValueError: If required columns are missing.
            FileNotFoundError: If model files are not found.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'first': ['John', 'Maria', 'Chen'],
            ...     'last': ['Smith', 'Garcia', 'Wang']
            ... })
            >>> result = WikiNameModel.pred_wiki_name(df, 'last', 'first')
            >>> print(result[['first', 'last', 'race']].head())
              first    last        race
            0  John   Smith  GreaterEuropean
            1 Maria  Garcia      Hispanic
            2  Chen    Wang  EastAsian
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist.")
        if fname_col not in df.columns:
            raise ValueError(f"The first name column '{fname_col}' doesn't exist.")

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()
        working_df["__rowindex"] = np.arange(len(working_df))
        original_length = len(working_df)

        # Create a safe temporary name column
        temp_col = "__ethnicolr_temp_name"
        while temp_col in working_df.columns:
            temp_col += "_"

        logger.info(f"Processing {original_length} names")

        # Create original full name column for reference (before normalization)
        working_df["__name"] = (
            working_df[lname_col].fillna("").astype(str).str.strip()
            + " "
            + working_df[fname_col].fillna("").astype(str).str.strip()
        ).str.strip()

        # Create normalized full name for prediction
        working_df[temp_col] = working_df["__name"].apply(_normalize_name)

        # Add normalization info column to track what happened to each name
        working_df["name_normalized"] = working_df["__name"]
        working_df["name_normalized_clean"] = working_df[temp_col]

        # Track which names will be skipped and why
        empty_after_norm = working_df[temp_col].str.strip().str.len() == 0
        empty_original = working_df["__name"].str.strip().str.len() == 0

        # Create processing status column
        working_df["processing_status"] = "processed"
        working_df.loc[empty_original, "processing_status"] = "skipped_empty_original"
        working_df.loc[empty_after_norm & ~empty_original, "processing_status"] = (
            "skipped_empty_after_normalization"
        )

        # Count what we're about to skip
        to_skip = working_df["processing_status"] != "processed"
        skipped_count = to_skip.sum()

        if skipped_count > 0:
            status_series: pd.Series = working_df[to_skip]["processing_status"]  # type: ignore
            skip_reasons = status_series.value_counts()
            logger.warning(f"Will skip {skipped_count} names:")
            for reason, count in skip_reasons.items():
                logger.warning(f"  - {count} names: {reason}")

        # Separate processable and skipped names
        processable_df = working_df[~to_skip].copy()
        skipped_df = working_df[to_skip].copy()

        # Ensure we have DataFrames, not Series
        assert isinstance(processable_df, pd.DataFrame)
        assert isinstance(skipped_df, pd.DataFrame)

        if len(processable_df) == 0:
            logger.warning(
                "No valid names to process after cleaning. Returning original data with status info."
            )
            result_df = working_df.copy()
            result_df["race"] = None
            return result_df

        try:
            logger.info(
                f"Applying Wikipedia name model to {len(processable_df)} processable names (confidence interval: {conf_int})"
            )

            # Run prediction only on processable names
            if vocab_path is None or race_path is None:
                raise ValueError(
                    "Vocabulary and race files must be provided for LSTM models"
                )

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

            # Combine results - handle empty DataFrames explicitly to avoid deprecation warnings
            if pred_df.empty and skipped_df.empty:
                result_df = pd.DataFrame()
            elif pred_df.empty:
                result_df = skipped_df.reset_index(drop=True)
            elif skipped_df.empty:
                result_df = pred_df.reset_index(drop=True)
            else:
                result_df = pd.concat([pred_df, skipped_df], ignore_index=True)

            # Sort by original order if possible
            if "__rowindex" in result_df.columns:
                result_df = result_df.sort_values(by="__rowindex").reset_index(
                    drop=True
                )

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
pred_wiki_name = WikiNameModel.pred_wiki_name


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for Wikipedia name predictions.

    Provides CLI access to Wikipedia-based race/ethnicity prediction
    using full names. Processes CSV files and outputs predictions.

    Args:
        argv: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code: 0 success, 1 general error, 2 missing files, 3 invalid data.
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by name using Wikipedia model",
            default_out="wiki-pred-name-output.csv",
            default_year=2017,
            year_choices=[2017],
            first=True,
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_wiki_name(
            df=df,
            lname_col=args.last,
            fname_col=args.first,
            num_iter=args.iter,
            conf_int=args.conf,
        )

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        rdf.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"Output written: {args.output} ({len(rdf)} rows)")

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
