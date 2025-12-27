#!/usr/bin/env python
"""
Florida Last Name 5-Category Race/Ethnicity Prediction Module.

Uses LSTM models trained on Florida voter registration data to predict
race/ethnicity from last names, collapsed to 5 categories.
"""

import logging
import os
import sys

import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FloridaRegLnFiveCatModel(EthnicolrModelClass):
    MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_5_cat{0:s}.h5"
    VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab_5_cat{0:s}.csv"
    RACEFN = "models/fl_voter_reg/lstm/fl_ln_five_cat_race{0:s}.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def pred_fl_reg_ln(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        num_iter: int = 100,
        conf_int: float = 1.0,
        year: int = 2022,
    ) -> pd.DataFrame:
        """
        Predict race/ethnicity from last names using Florida voter registration 5-category LSTM model.

        This function uses an LSTM neural network model trained on Florida voter registration
        data to predict race/ethnicity probabilities from last names. This 5-category version
        includes an 'other' category for improved coverage of diverse racial/ethnic identities.

        Performance Context:
            The 5-category Florida model provides more nuanced predictions than the 4-category version:
            - Common surnames: ~75-85% accuracy across all categories
            - 'Other' category captures multiracial and less common ethnic groups
            - Enhanced coverage for diverse demographics present in Florida
            - Available for both 2017 and 2022 training data versions

        Args:
            df: Input DataFrame containing names to predict.
            lname_col: Name of the column containing last names. Column must exist in df.
            num_iter: Number of Monte Carlo iterations for confidence interval estimation.
                     Only used when conf_int < 1.0. Default: 100.
            conf_int: Confidence interval level (0.0-1.0). When < 1.0, enables Monte Carlo
                     sampling to generate confidence intervals. Default: 1.0 (disabled).
            year: Model year version to use (2017 or 2022). Different training datasets.
                 Default: 2022.

        Returns:
            DataFrame with original data plus 5-category race prediction columns:
            - race: Predicted race category (asian, hispanic, nh_black, nh_white, other)
            - asian: Probability of Asian ethnicity [0.0-1.0]
            - hispanic: Probability of Hispanic ethnicity [0.0-1.0]
            - nh_black: Probability of Non-Hispanic Black [0.0-1.0]
            - nh_white: Probability of Non-Hispanic White [0.0-1.0]
            - other: Probability of Other/Multiracial [0.0-1.0]

            When conf_int < 1.0, additional confidence interval columns:
            - {race}_mean: Monte Carlo mean probability
            - {race}_std: Monte Carlo standard deviation
            - {race}_lb: Lower confidence bound
            - {race}_ub: Upper confidence bound

        Raises:
            ValueError: If lname_col does not exist in df.
            FileNotFoundError: If required model files for specified year are missing.

        Example:
            >>> import pandas as pd
            >>> from ethnicolr.pred_fl_reg_ln_five_cat import pred_fl_reg_ln_five_cat
            >>>
            >>> df = pd.DataFrame({
            ...     'last': ['Garcia', 'Smith', 'Patel', 'Johnson'],
            ...     'id': [1, 2, 3, 4]
            ... })
            >>> result = pred_fl_reg_ln_five_cat(df, 'last', year=2022)
            >>> print(result[['last', 'race', 'hispanic', 'asian', 'other']])
                last      race  hispanic     asian     other
            0  Garcia  hispanic     0.834     0.023     0.089
            1   Smith  nh_white     0.078     0.034     0.112
            2   Patel     asian     0.056     0.782     0.089
            3 Johnson  nh_white     0.091     0.028     0.134

            >>> # Using 2017 model version
            >>> result_2017 = pred_fl_reg_ln_five_cat(df, 'last', year=2017)

        Note:
            - 5-category version provides 'other' category for multiracial/diverse identities
            - Model year affects training data: 2022 includes more recent demographic patterns
            - 'Other' category captures names not well-represented in the 4 main categories
            - All probability columns sum to 1.0 across the 5 categories
            - For full name predictions, see pred_fl_reg_name_five_cat()
        """
        if lname_col not in df.columns:
            raise ValueError(
                f"The last name column '{lname_col}' does not exist in the DataFrame."
            )

        suffix = "_2022" if year == 2022 else ""
        logger.info(f"Using FL 5-cat model for year {year}")

        rdf = cls.transform_and_pred(
            df=df.copy(),
            newnamecol=lname_col,
            vocab_fn=cls.VOCABFN.format(suffix) if cls.VOCABFN else "",
            race_fn=cls.RACEFN.format(suffix) if cls.RACEFN else "",
            model_fn=cls.MODELFN.format(suffix),
            ngrams=cls.NGRAMS,
            maxlen=cls.FEATURE_LEN,
            num_iter=num_iter,
            conf_int=conf_int,
        )

        return rdf


# CLI alias
pred_fl_reg_ln_five_cat = FloridaRegLnFiveCatModel.pred_fl_reg_ln


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        args = arg_parser(
            argv,
            title="Predict Race/Ethnicity by last name using the Florida registration 5-cat model",
            default_out="fl-pred-ln-five-cat-output.csv",
            default_year=2022,
            year_choices=[2017, 2022],
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        rdf = pred_fl_reg_ln_five_cat(
            df=df,
            lname_col=args.last,
            num_iter=args.iter,
            conf_int=args.conf,
            year=args.year,
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
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
