#!/usr/bin/env python
"""
Florida Last Name Race/Ethnicity Prediction Module.

Uses an LSTM model trained on Florida voter registration data
to predict race/ethnicity from last names.
"""

import logging
import os
import sys
from importlib import resources

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .model_base import ModelType, RaceCategory, register_model
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@register_model(ModelType.FLORIDA_LSTM)
class FloridaRegLnModel(EthnicolrModelClass):
    """Florida voter registration LSTM model for last name prediction.

    LSTM model trained on Florida voter registration data to predict
    race/ethnicity from last names. Uses 5-category classification.
    """

    # Required abstract class attributes
    SUPPORTED_CATEGORIES = [
        RaceCategory.ASIAN,
        RaceCategory.HISPANIC,
        RaceCategory.NH_BLACK,
        RaceCategory.NH_WHITE,
    ]
    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def get_model_paths(cls):
        package = resources.files(__name__.split(".")[0])
        return (
            str(package / "models/fl_voter_reg/lstm/fl_all_ln_lstm.h5"),
            str(package / "models/fl_voter_reg/lstm/fl_all_ln_vocab.csv"),
            str(package / "models/fl_voter_reg/lstm/fl_ln_race.csv"),
        )

    @classmethod
    def check_models_exist(cls):
        model_path, vocab_path, race_path = cls.get_model_paths()
        missing = [
            p for p in [model_path, vocab_path, race_path] if not os.path.exists(p)
        ]
        if missing:
            msg = (
                f"Required model files not found for Florida model:\n"
                f"{', '.join(missing)}\n\n"
                "Install with: pip install ethnicolr[models]\n"
                "Or download from: https://github.com/appeler/ethnicolr/releases"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        return True

    @classmethod
    def pred_fl_reg_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict race/ethnicity from last names using Florida voter registration LSTM model.

        This function uses an LSTM neural network model trained on Florida voter registration
        data to predict race/ethnicity probabilities from last names. The model outputs
        4-category race predictions with optional confidence intervals.

        Performance Context:
            The Florida voter registration dataset provides strong regional context for
            predictions. Accuracy varies by name frequency and racial demographics:
            - Common names: ~75-85% accuracy
            - Regional Hispanic names: ~80-90% accuracy
            - Less common surnames: ~60-75% accuracy

        Args:
            df: Input DataFrame containing names to predict.
            lname_col: Name of the column containing last names. Column must exist in df.
            num_iter: Number of Monte Carlo iterations for confidence interval estimation.
                     Only used when conf_int < 1.0. Default: 100.
            conf_int: Confidence interval level (0.0-1.0). When < 1.0, enables Monte Carlo
                     sampling to generate confidence intervals. Default: 1.0 (disabled).

        Returns:
            DataFrame with original data plus race prediction columns:
            - race: Predicted race category (asian, hispanic, nh_black, nh_white)
            - asian: Probability of Asian ethnicity [0.0-1.0]
            - hispanic: Probability of Hispanic ethnicity [0.0-1.0]
            - nh_black: Probability of Non-Hispanic Black [0.0-1.0]
            - nh_white: Probability of Non-Hispanic White [0.0-1.0]

            When conf_int < 1.0, additional confidence interval columns:
            - {race}_mean: Monte Carlo mean probability
            - {race}_std: Monte Carlo standard deviation
            - {race}_lb: Lower confidence bound
            - {race}_ub: Upper confidence bound

        Raises:
            ValueError: If lname_col does not exist in df.
            FileNotFoundError: If required model files are missing.

        Example:
            >>> import pandas as pd
            >>> from ethnicolr.pred_fl_reg_ln import pred_fl_reg_ln
            >>>
            >>> df = pd.DataFrame({
            ...     'last': ['Garcia', 'Smith', 'Rodriguez', 'Chen'],
            ...     'id': [1, 2, 3, 4]
            ... })
            >>> result = pred_fl_reg_ln(df, 'last')
            >>> print(result[['last', 'race', 'hispanic', 'nh_white']])
                  last     race  hispanic   nh_white
            0   Garcia  hispanic     0.823     0.142
            1    Smith  nh_white     0.091     0.821
            2 Rodriguez  hispanic     0.891     0.078
            3     Chen     asian     0.034     0.156

            >>> # With confidence intervals
            >>> result_conf = pred_fl_reg_ln(df, 'last', conf_int=0.9, num_iter=50)
            >>> print(result_conf[['last', 'race', 'hispanic_mean', 'hispanic_lb']])

        Note:
            - Predictions work best with clean, alphabetic last names
            - Model was trained on Florida voter data and may perform better on
              names common in Florida demographics
            - For full name predictions, see pred_fl_reg_name()
            - All probability columns sum to 1.0 across race categories
        """
        if lname_col not in df.columns:
            raise ValueError(
                f"The last name column '{lname_col}' does not exist in the DataFrame."
            )

        logger.info(
            f"Predicting race/ethnicity for {len(df)} rows using Florida LSTM model"
        )

        model_path, vocab_path, race_path = cls.get_model_paths()

        rdf = cls.transform_and_pred(
            df=df.copy(),
            newnamecol=lname_col,
            vocab_fn=vocab_path,
            race_fn=race_path,
            model_fn=model_path,
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int,
        )

        logger.info(
            f"Prediction complete. Added columns: {', '.join(set(rdf.columns) - set(df.columns))}"
        )
        return rdf

    # Abstract method implementations
    @classmethod
    def predict(cls, df: pd.DataFrame, name_col: str, **kwargs) -> pd.DataFrame:
        """
        Generate race/ethnicity predictions for Florida model.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing last names to predict.
            **kwargs: Additional parameters (num_iter, conf_int).

        Returns:
            DataFrame with race/ethnicity predictions.
        """
        cls.validate_input(df, name_col)
        cls.check_models_exist()
        return cls.pred_fl_reg_ln(df, name_col, **kwargs)

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
        cls.check_models_exist()
        return cls.pred_fl_reg_ln(
            df, name_col, num_iter=num_iter, conf_int=conf_int, **kwargs
        )


# CLI alias
pred_fl_reg_ln = FloridaRegLnModel.pred_fl_reg_ln  # type: ignore


def main(argv: list[str] | None = None) -> int:
    """
    Command-line interface for Florida voter registration last name prediction.

    This function provides CLI access to the Florida LSTM model for predicting race/ethnicity
    from last names. It handles argument parsing, file I/O, error handling, and logging.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].
              Expected format: [input_file, -l, last_col, -o, output_file, ...]

    Returns:
        Exit code:
        - 0: Success
        - 1: Unhandled error
        - 2: Missing model files
        - 3: Invalid input data

    Example Usage:
        Command line:
        $ pred_fl_reg_ln input.csv -l surname -o predictions.csv
        $ pred_fl_reg_ln data.csv -l last_name -c 0.9 -i 50 -o results_with_ci.csv

        Python:
        >>> from ethnicolr.pred_fl_reg_ln import main
        >>> exit_code = main(['data.csv', '-l', 'last', '-o', 'output.csv'])
        >>> print(f"Process completed with exit code: {exit_code}")

    CLI Arguments:
        input_file: Path to CSV file with names
        -l, --last: Column name containing last names
        -o, --output: Output file path (default: fl-pred-ln-output.csv)
        -c, --conf: Confidence interval level 0.0-1.0 (default: 1.0, disabled)
        -i, --iter: Monte Carlo iterations for CI (default: 100)
        -y, --year: Model year, currently only 2017 supported

    Note:
        Input CSV should have headers and contain the specified last name column.
        Output includes all original columns plus race prediction probabilities.
    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using Florida registration model",
            default_out="fl-pred-ln-output.csv",
            default_year=2017,
            year_choices=[2017],
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_fl_reg_ln(
            df=df, lname_col=args.last, num_iter=args.iter, conf_int=args.conf
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
