#!/usr/bin/env python
"""
North Carolina Voter Registration Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity using full names based on an LSTM model trained on NC voter data (12-category).
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    def pred_nc_reg_name(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        fname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict race/ethnicity from full names using North Carolina voter registration LSTM model.

        This function uses an LSTM neural network model trained on North Carolina voter registration
        data to predict detailed race/ethnicity categories from combined first and last names.
        The NC model provides 12-category predictions with Hispanic/Latino and race intersections.

        Performance Context:
            The NC model provides more granular racial/ethnic categorization than other models:
            - Full name predictions: ~75-85% accuracy for major categories
            - Hispanic/Latino intersections: Detailed HL+{race} and NL+{race} categories
            - Regional variations: Optimized for North Carolina demographic patterns
            - 12-category system captures nuanced racial/ethnic identities

        Args:
            df: Input DataFrame containing names to predict.
            lname_col: Name of the column containing last names. Must exist in df.
            fname_col: Name of the column containing first names. Must exist in df.
            num_iter: Number of Monte Carlo iterations for confidence interval estimation.
                     Only used when conf_int < 1.0. Default: 100.
            conf_int: Confidence interval level (0.0-1.0). When < 1.0, enables Monte Carlo
                     sampling to generate confidence intervals. Default: 1.0 (disabled).

        Returns:
            DataFrame with original data plus 12-category race prediction columns:
            - race: Predicted race category (most likely from 12 categories below)

            Hispanic/Latino + Race categories:
            - HL+A: Hispanic/Latino + Asian [0.0-1.0]
            - HL+B: Hispanic/Latino + Black [0.0-1.0]
            - HL+I: Hispanic/Latino + American Indian [0.0-1.0]
            - HL+M: Hispanic/Latino + Multiracial [0.0-1.0]
            - HL+O: Hispanic/Latino + Other [0.0-1.0]
            - HL+W: Hispanic/Latino + White [0.0-1.0]

            Non-Hispanic/Latino + Race categories:
            - NL+A: Non-Hispanic/Latino + Asian [0.0-1.0]
            - NL+B: Non-Hispanic/Latino + Black [0.0-1.0]
            - NL+I: Non-Hispanic/Latino + American Indian [0.0-1.0]
            - NL+M: Non-Hispanic/Latino + Multiracial [0.0-1.0]
            - NL+O: Non-Hispanic/Latino + Other [0.0-1.0]
            - NL+W: Non-Hispanic/Latino + White [0.0-1.0]

            Processing transparency columns (2024 enhancement):
            - processing_status: 'processed', 'skipped_empty_original', 'skipped_empty_after_normalization'
            - __name: Full name used for processing (last + " " + first)
            - name_normalized: Original combined name before cleaning
            - name_normalized_clean: Name after title case normalization

            When conf_int < 1.0, additional confidence interval columns:
            - {category}_mean: Monte Carlo mean probability for each category
            - {category}_std: Monte Carlo standard deviation
            - {category}_lb: Lower confidence bound
            - {category}_ub: Upper confidence bound

        Raises:
            ValueError: If lname_col or fname_col does not exist in df.
            FileNotFoundError: If required model files are missing.

        Example:
            >>> import pandas as pd
            >>> from ethnicolr.pred_nc_reg_name import pred_nc_reg_name
            >>>
            >>> df = pd.DataFrame({
            ...     'last': ['Garcia', 'Smith', 'Kim', 'Washington'],
            ...     'first': ['Maria', 'John', 'Jin', 'Keisha'],
            ...     'id': [1, 2, 3, 4]
            ... })
            >>> result = pred_nc_reg_name(df, 'last', 'first')
            >>> print(result[['last', 'first', 'race', 'HL+W', 'NL+W', 'NL+A']])
                  last   first   race     HL+W     NL+W     NL+A
            0   Garcia   Maria   HL+W    0.734    0.124    0.032
            1    Smith    John   NL+W    0.089    0.821    0.054
            2      Kim     Jin   NL+A    0.012    0.089    0.734
            3 Washington Keisha  NL+B    0.045    0.123    0.021

            >>> # Check processing status
            >>> print(result['processing_status'].value_counts())
            processed    4

        Note:
            - Provides 12-category intersectional race/ethnicity predictions
            - HL+ prefix indicates Hispanic/Latino ethnicity + racial category
            - NL+ prefix indicates Non-Hispanic/Latino + racial category
            - Model trained on North Carolina voter data with regional context
            - Enhanced name processing preserves diverse name formats
            - All probability columns sum to 1.0 across the 12 categories
        """
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist.")
        if fname_col not in df.columns:
            raise ValueError(f"The first name column '{fname_col}' doesn't exist.")

        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()

        working_df = df.copy()
        original_length = len(working_df)

        # Safe temporary column name
        temp_col = "__ethnicolr_temp_name"
        while temp_col in working_df.columns:
            temp_col += "_"

        logger.info(f"Processing {original_length} names")

        # Create original full name column for reference
        working_df["__name"] = (
            working_df[lname_col].fillna("").astype(str).str.strip()
            + " "
            + working_df[fname_col].fillna("").astype(str).str.strip()
        ).str.strip()

        # Build full name for processing
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

            # Add all expected NC prediction columns with NaN values
            nc_categories = [
                "HL+A",
                "HL+B",
                "HL+I",
                "HL+M",
                "HL+O",
                "HL+W",
                "NL+A",
                "NL+B",
                "NL+I",
                "NL+M",
                "NL+O",
                "NL+W",
            ]
            for category in nc_categories:
                result_df[category] = float("nan")

            return result_df

        try:
            logger.info(
                f"Applying NC voter name model to {len(processable_df)} processable names (confidence interval: {conf_int})"
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


# Alias for backwards compatibility
pred_nc_reg_name = NCRegNameModel.pred_nc_reg_name


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by name using NC 12-category voter registration model",
            default_out="nc-pred-name-output.csv",
            default_year=2017,
            year_choices=[2017],
            first=True,
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_nc_reg_name(
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
