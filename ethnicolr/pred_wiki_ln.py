#!/usr/bin/env python
"""
Wikipedia Last Name-based Race/Ethnicity Prediction Module.

Predicts race/ethnicity from last names using an LSTM model trained on Wikipedia data.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .model_base import ModelType, register_model
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@register_model(ModelType.WIKI_LSTM)
class WikiLnModel(EthnicolrModelClass):
    """Wikipedia-based last name prediction model.

    LSTM model trained on Wikipedia data for predicting race/ethnicity
    from last names only. Provides detailed ethnic categories.
    """

    # Required abstract class attributes
    SUPPORTED_CATEGORIES = [
        "Asian,GreaterEastAsian,EastAsian",
        "Asian,GreaterEastAsian,Japanese",
        "Asian,IndianSubContinent",
        "GreaterAfrican,Africans",
        "GreaterAfrican,Muslim",
        "GreaterEuropean,British",
        "GreaterEuropean,EastEuropean",
        "GreaterEuropean,Jewish",
        "GreaterEuropean,WestEuropean,French",
        "GreaterEuropean,WestEuropean,Germanic",
        "GreaterEuropean,WestEuropean,Hispanic",
        "GreaterEuropean,WestEuropean,Italian",
        "GreaterEuropean,WestEuropean,Nordic",
    ]

    # Model file paths
    MODELFN = "models/wiki/lstm/wiki_ln_lstm.h5"
    VOCABFN = "models/wiki/lstm/wiki_ln_vocab.csv"
    RACEFN = "models/wiki/lstm/wiki_race.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def get_model_paths(cls):
        """Get file paths for Wikipedia last name model components.

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
    def pred_wiki_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """Predict race/ethnicity from last names using Wikipedia model.

        Uses LSTM trained on Wikipedia data to predict detailed ethnic
        categories from last names. Handles missing/empty names gracefully.

        Args:
            df: Input DataFrame containing last names.
            lname_col: Column name containing last names.
            num_iter: Monte Carlo iterations for confidence intervals.
            conf_int: Confidence level (1.0 for point estimates).

        Returns:
            DataFrame with original data plus prediction columns:
            - 'race': Predicted ethnicity category
            - Probability columns for each ethnicity
            - 'processing_status': Name processing outcome
            - 'name_normalized': Normalized last name
            - Confidence bounds if conf_int < 1.0

        Raises:
            ValueError: If last name column is missing.
            FileNotFoundError: If model files are not found.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'surname': ['Smith', 'Garcia', 'Wang']})
            >>> result = WikiLnModel.pred_wiki_ln(df, 'surname')
            >>> print(result[['surname', 'race']].head())
              surname        race
            0   Smith  GreaterEuropean
            1  Garcia      Hispanic
            2    Wang      EastAsian
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

        # Ensure we have DataFrames, not Series
        assert isinstance(processable_df, pd.DataFrame)
        assert isinstance(skipped_df, pd.DataFrame)

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
            if vocab_path is None or race_path is None:
                raise ValueError(
                    "Vocabulary and race files must be provided for LSTM models"
                )

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

    # Abstract method implementations
    @classmethod
    def predict(cls, df: pd.DataFrame, name_col: str, **kwargs) -> pd.DataFrame:
        """
        Generate race/ethnicity predictions for Wikipedia model.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing last names to predict.
            **kwargs: Additional parameters (num_iter, conf_int).

        Returns:
            DataFrame with race/ethnicity predictions.
        """
        cls.validate_input(df, name_col)
        return cls.pred_wiki_ln(df, name_col, **kwargs)

    @classmethod
    def predict_with_confidence(
        cls,
        df: pd.DataFrame,
        name_col: str,
        conf_int: float = 0.95,
        num_iter: int = 100,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing last names to predict.
            conf_int: Confidence interval level (0.0-1.0).
            num_iter: Number of Monte Carlo iterations.

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        cls.validate_input(df, name_col)
        return cls.pred_wiki_ln(
            df, name_col, num_iter=num_iter, conf_int=conf_int, **kwargs
        )


# For backward compatibility
pred_wiki_ln = WikiLnModel.pred_wiki_ln  # type: ignore


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for Wikipedia last name predictions.

    Provides CLI access to Wikipedia-based race/ethnicity prediction
    using last names only. Processes CSV files and outputs predictions.

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
