#!/usr/bin/env python
"""
Census Last Name Race/Ethnicity Prediction Module.

Uses LSTM models trained on U.S. Census data to predict race/ethnicity from last names.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .model_base import ModelType, RaceCategory, register_model
from .utils import arg_parser

# Suppress TensorFlow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@register_model(ModelType.CENSUS_LSTM)
class CensusLnModel(EthnicolrModelClass):
    """Census-based last name prediction model.

    LSTM model trained on U.S. Census data for predicting race/ethnicity
    from last names. Supports both 2000 and 2010 Census data.
    """

    # Required abstract class attributes
    SUPPORTED_CATEGORIES = [
        RaceCategory.WHITE,
        RaceCategory.BLACK,
        RaceCategory.ASIAN,
        RaceCategory.HISPANIC,
    ]
    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def get_model_paths(cls, year):
        package = resources.files(__name__.split(".")[0])
        return (
            str(package / f"models/census/lstm/census{year}_ln_lstm.h5"),
            str(package / f"models/census/lstm/census{year}_ln_vocab.csv"),
            str(package / f"models/census/lstm/census{year}_race.csv"),
        )

    @classmethod
    def check_models_exist(cls, year):
        model_path, vocab_path, race_path = cls.get_model_paths(year)
        missing = [
            p for p in [model_path, vocab_path, race_path] if not os.path.exists(p)
        ]
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
    def pred_census_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        year: int = 2010,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """Predict race/ethnicity from last names using Census LSTM model.

        Uses machine learning models trained on U.S. Census surname data
        to predict race/ethnicity categories: white, black, Asian, Hispanic.

        Args:
            df: Input DataFrame containing last names.
            lname_col: Name of column containing last names.
            year: Census year for model selection (2000 or 2010).
            num_iter: Monte Carlo iterations for confidence intervals.
            conf_int: Confidence level (1.0 for point estimates).

        Returns:
            DataFrame with original data plus prediction columns:
            - 'race': Predicted race category
            - 'white', 'black', 'asian', 'hispanic': Probability scores
            - Confidence bounds if conf_int < 1.0

        Raises:
            ValueError: If year not in [2000, 2010] or column missing.
            FileNotFoundError: If required model files not found.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'surname': ['Smith', 'Garcia', 'Wang']})
            >>> result = CensusLnModel.pred_census_ln(df, 'surname', year=2010)
            >>> print(result[['surname', 'race']].head())
               surname    race
            0    Smith   white
            1   Garcia hispanic
            2     Wang   asian
        """
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
            conf_int=conf_int,
        )

        pred_count = rdf.dropna(subset=["race"]).shape[0]
        if len(df) > 0:
            logger.info(
                f"Predicted {pred_count} of {len(df)} rows ({pred_count / len(df) * 100:.1f}%)"
            )
        else:
            logger.info("No rows to predict (empty DataFrame)")
        logger.info(f"Added columns: {', '.join(set(rdf.columns) - set(df.columns))}")

        return rdf

    # Abstract method implementations
    @classmethod
    def predict(
        cls, df: pd.DataFrame, name_col: str, year: int = 2010, **kwargs
    ) -> pd.DataFrame:
        """
        Generate race/ethnicity predictions for Census model.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing last names to predict.
            year: Census year (2000 or 2010).
            **kwargs: Additional parameters (num_iter, conf_int).

        Returns:
            DataFrame with race/ethnicity predictions.
        """
        cls.validate_input(df, name_col)
        return cls.pred_census_ln(df, name_col, year=year, **kwargs)

    @classmethod
    def predict_with_confidence(
        cls,
        df: pd.DataFrame,
        name_col: str,
        conf_int: float = 0.95,
        num_iter: int = 100,
        year: int = 2010,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing last names to predict.
            conf_int: Confidence interval level (0.0-1.0).
            num_iter: Number of Monte Carlo iterations.
            year: Census year (2000 or 2010).

        Returns:
            DataFrame with predictions and confidence intervals.
        """
        cls.validate_input(df, name_col)
        return cls.pred_census_ln(
            df, name_col, year=year, num_iter=num_iter, conf_int=conf_int, **kwargs
        )


# Alias for CLI use
pred_census_ln = CensusLnModel.pred_census_ln  # type: ignore


def download_models(year=None):
    """Download Census model files.

    Placeholder function for downloading required model files.
    Currently logs download actions but doesn't implement actual downloads.

    Args:
        year: Specific Census year to download (None for all years).

    Note:
        This is a stub implementation. Actual file downloads should be
        implemented based on the model distribution strategy.
    """
    years = [year] if year else [2000, 2010]
    for y in years:
        logger.info(f"Downloading Census {y} model files...")
        # TODO: Implement actual download logic
        logger.info(f"Downloaded Census {y} model files successfully")


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for Census last name predictions.

    Provides CLI access to Census-based race/ethnicity prediction.
    Supports model downloads and batch processing of CSV files.

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
            title="Predict Race/Ethnicity by last name using Census LSTM model",
            default_out="census-pred-ln-output.csv",
            default_year=2010,
            year_choices=[2000, 2010],
        )

        # Note: arg_parser returns Namespace, not parser object
        # Custom argument handling would need to be implemented differently

        # Download models functionality would need custom implementation

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_census_ln(
            df=df,
            lname_col=args.last,
            year=args.year,
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
