#!/usr/bin/env python

import logging
import warnings
from importlib.resources import as_file, files
from itertools import chain

import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM as _KerasLSTM  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import sequence  # type: ignore

from .model_base import AbstractLSTMModel, InvalidInputError

logger = logging.getLogger(__name__)


class _CompatLSTM:
    """Backward-compatible LSTM layer.

    Older versions of Keras exposed a ``time_major`` argument on ``LSTM``
    layers. The pre-trained models distributed with *ethnicolr* were saved
    with this argument set to ``False``. Keras 3 removed the keyword which
    causes deserialization to fail with ``ValueError``. This shim accepts the
    deprecated argument but simply ignores it, allowing legacy models to load
    on newer Keras versions."""

    def __new__(cls, *args, time_major=False, **kwargs):
        # ``time_major`` is unused but preserved for compatibility.
        # Return actual LSTM instance, ignoring time_major parameter
        return _KerasLSTM(*args, **kwargs)


class EthnicolrModelClass(AbstractLSTMModel):
    """
    Base class for ethnicolr machine learning models.

    This class provides the foundation for all LSTM-based race/ethnicity prediction models
    in the ethnicolr package. It handles common functionality including model loading,
    text preprocessing, n-gram generation, prediction with confidence intervals, and
    Monte Carlo sampling for uncertainty estimation.

    Architecture:
        All ethnicolr models use LSTM neural networks trained on character-level n-grams
        extracted from names. The models predict probability distributions over
        race/ethnicity categories using softmax output layers.

    Model Types Supported:
        - Census surname lookup models (census_ln.py)
        - Census LSTM prediction models (pred_census_ln.py)
        - Wikipedia name models (pred_wiki_*.py)
        - Florida voter registration models (pred_fl_reg_*.py)
        - North Carolina voter registration models (pred_nc_reg_*.py)

    Common Workflow:
        1. Load vocabulary, race labels, and pre-trained LSTM model
        2. Preprocess and normalize input names
        3. Extract character n-grams from names
        4. Convert n-grams to model input sequences
        5. Generate predictions using LSTM model
        6. Apply confidence interval estimation if requested
        7. Return DataFrame with probabilities and predicted race

    Attributes:
        vocab (list): Character n-gram vocabulary loaded from model files.
        race (list): Race/ethnicity category labels loaded from model files.
        model (tf.keras.Model): Pre-trained LSTM model for predictions.
        model_year (int): Year version of the loaded model (affects training data).

    Class Constants (defined in subclasses):
        MODELFN (str): Path to .h5 model file within package
        VOCABFN (str): Path to vocabulary CSV file within package
        RACEFN (str): Path to race labels CSV file within package
        NGRAMS (int | tuple): N-gram size(s) for feature extraction
        FEATURE_LEN (int): Maximum sequence length for model input

    Example:
        >>> # Subclass implementation pattern
        >>> class MyModel(EthnicolrModelClass):
        ...     MODELFN = "models/my_model.h5"
        ...     VOCABFN = "models/my_vocab.csv"
        ...     RACEFN = "models/my_races.csv"
        ...     NGRAMS = 2
        ...     FEATURE_LEN = 20
        ...
        ...     @classmethod
        ...     def my_prediction_func(cls, df, name_col):
        ...         return cls.transform_and_pred(df, name_col, ...)

    Note:
        - This class is not intended for direct instantiation
        - Subclasses must define model file paths and parameters
        - All models use TensorFlow/Keras backend
        - Model files are distributed separately from the package
    """

    # Model caches per class to prevent conflicts between different model types
    _model_cache = {}  # Dict[str, Dict] - keyed by class name

    @classmethod
    def _get_cache_key(cls) -> str:
        """Get cache key for this specific model class."""
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def _ensure_cache_initialized(cls):
        """Initialize cache for this model class if not exists."""
        cache_key = cls._get_cache_key()
        if cache_key not in cls._model_cache:
            cls._model_cache[cache_key] = {
                "vocab": None,
                "race": None,
                "model": None,
                "model_year": None,
                "vocab_dict": None,
            }

    @classmethod
    def _get_cache(cls) -> dict:
        """Get cache dictionary for this model class."""
        cls._ensure_cache_initialized()
        return cls._model_cache[cls._get_cache_key()]

    @classmethod
    def get_vocab(cls):
        """Get vocabulary list for this model."""
        return cls._get_cache()["vocab"]

    @classmethod
    def set_vocab(cls, value):
        """Set vocabulary list for this model."""
        cache = cls._get_cache()
        cache["vocab"] = value
        cache["vocab_dict"] = None  # Reset dict cache when vocab changes

    @classmethod
    def get_race(cls):
        """Get race categories list for this model."""
        return cls._get_cache()["race"]

    @classmethod
    def set_race(cls, value):
        """Set race categories list for this model."""
        cls._get_cache()["race"] = value

    @classmethod
    def get_model(cls):
        """Get trained model for this model."""
        return cls._get_cache()["model"]

    @classmethod
    def set_model(cls, value):
        """Set trained model for this model."""
        cls._get_cache()["model"] = value

    @classmethod
    def get_model_year(cls):
        """Get model year for this model."""
        return cls._get_cache()["model_year"]

    @classmethod
    def set_model_year(cls, value):
        """Set model year for this model."""
        cls._get_cache()["model_year"] = value

    @staticmethod
    def test_and_norm_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Validate presence of ``col`` and drop rows with NaNs while preserving duplicates.

        Earlier implementations removed duplicate values which could silently drop
        rows. This caused a mismatch between the number of input rows and the
        predictions returned. The function now keeps duplicates but logs how many
        were found so callers remain informed.
        """
        if col not in df.columns:
            raise ValueError(f"The column '{col}' does not exist in the DataFrame.")

        original_length = len(df)

        # Track NaN removals
        nan_mask = df[col].isna()
        nan_count = nan_mask.sum()
        if nan_count > 0:
            logger.info(f"Removing {nan_count} rows with NaN values in column '{col}'")
            df = df.dropna(subset=[col])

        if df.empty:
            logger.warning("The name column has no non-NaN values.")

        # Log duplicates but keep them so prediction results align with inputs
        dup_count = df.duplicated(subset=[col]).sum()
        if dup_count > 0:
            logger.info(
                f"Preserving {dup_count} duplicate rows based on column '{col}'"
            )

        final_length = len(df)
        if original_length > 0:
            percentage = final_length / original_length * 100
            logger.info(
                f"Data filtering summary: {original_length} -> {final_length} rows (kept {percentage:.1f}%)"
            )
        else:
            logger.info("Data filtering summary: Empty input DataFrame")

        return df

    @staticmethod
    def n_grams(seq, n: int = 1):
        """Generate n-grams from a sequence.

        Creates overlapping n-grams from an input sequence by shifting tokens.
        Used for feature extraction in LSTM models.

        Args:
            seq: Input sequence (string or list) to generate n-grams from.
            n: Size of n-grams to generate (default 1 for unigrams).

        Returns:
            Iterator yielding tuples of n consecutive elements.

        Example:
            >>> list(EthnicolrModelClass.n_grams('hello', 2))
            [('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')]
        """

        def shift_token(i):
            return (el for j, el in enumerate(seq) if j >= i)

        shiftedTokens = (shift_token(i) for i in range(n))
        return zip(*shiftedTokens, strict=False)

    @staticmethod
    def range_ngrams(seq, ngramRange=(1, 2)):
        """Generate n-grams for a range of n values.

        Creates n-grams of multiple sizes from a single sequence, useful for
        models that use multiple n-gram features simultaneously.

        Args:
            seq: Input sequence to generate n-grams from.
            ngramRange: Tuple (start, stop) defining the range of n-gram sizes.
                       Default (1, 2) generates unigrams only.

        Returns:
            Iterator yielding n-grams of all specified sizes.

        Example:
            >>> list(EthnicolrModelClass.range_ngrams('abc', (1, 3)))
            [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c')]
        """
        return chain(*(EthnicolrModelClass.n_grams(seq, i) for i in range(*ngramRange)))

    @classmethod
    def find_ngrams(cls, vocab, text: str, n) -> list:
        """Convert text n-grams to vocabulary indices.

        Generates n-grams from input text and maps them to indices in a
        pre-defined vocabulary. Unknown n-grams are mapped to index 0.

        PERFORMANCE: Uses O(1) dictionary lookup instead of O(n) list.index().

        Args:
            vocab: List of vocabulary items for index lookup.
            text: Input string to process.
            n: N-gram size, or tuple (start, stop) for range of sizes.

        Returns:
            List of vocabulary indices corresponding to text n-grams.
            Unknown n-grams map to index 0.

        Example:
            >>> vocab = ['<UNK>', 'th', 'he', 'sm']
            >>> EthnicolrModelClass.find_ngrams(vocab, 'smith', 2)
            [3, 0, 0, 0]  # 'sm' found at index 3, others unknown
        """
        # Build vocabulary dictionary for O(1) lookups if not cached
        cache = cls._get_cache()
        if cache["vocab_dict"] is None or len(cache["vocab_dict"]) != len(vocab):
            cache["vocab_dict"] = {gram: idx for idx, gram in enumerate(vocab)}

        if isinstance(n, tuple):
            ngram_iter = EthnicolrModelClass.range_ngrams(text, n)
        else:
            ngram_iter = zip(*[text[i:] for i in range(n)], strict=False)

        return [cache["vocab_dict"].get("".join(gram), 0) for gram in ngram_iter]

    @classmethod
    def transform_and_pred(
        cls,
        df: pd.DataFrame,
        newnamecol: str,
        vocab_fn: str,
        race_fn: str,
        model_fn: str,
        ngrams,
        maxlen: int,
        num_iter: int,
        conf_int: float,
    ) -> pd.DataFrame:
        """Transform names to features and generate predictions.

        Core prediction method that loads models, converts names to n-gram features,
        and generates race/ethnicity predictions with optional confidence intervals.
        Handles both point predictions and Monte Carlo sampling for uncertainty.

        Args:
            df: Input DataFrame containing names to predict.
            newnamecol: Column name containing the names to predict on.
            vocab_fn: Path to vocabulary file for n-gram mapping.
            race_fn: Path to race/ethnicity labels file.
            model_fn: Path to trained LSTM model file.
            ngrams: N-gram size (int) or range (tuple) for feature extraction.
            maxlen: Maximum sequence length for padding/truncation.
            num_iter: Number of Monte Carlo iterations for confidence intervals.
            conf_int: Confidence interval level (1.0 for point prediction).

        Returns:
            DataFrame with original data plus prediction columns:
            - 'race': Predicted race/ethnicity category
            - Probability columns for each race/ethnicity
            - Confidence interval bounds (if conf_int < 1.0)

        Raises:
            FileNotFoundError: If model files don't exist.
            ValueError: If required columns are missing.

        Example:
            >>> df = pd.DataFrame({'name': ['Smith', 'Garcia']})
            >>> result = cls.transform_and_pred(
            ...     df, 'name', 'vocab.csv', 'race.csv', 'model.h5',
            ...     ngrams=2, maxlen=20, num_iter=100, conf_int=0.95
            ... )
            >>> print(result.columns)
            ['name', 'race', 'white', 'black', 'asian', 'hispanic', ...]
        """
        # Load model resources and prepare data
        vocab_path = files("ethnicolr") / vocab_fn
        model_path = files("ethnicolr") / model_fn
        race_path = files("ethnicolr") / race_fn

        df = df.copy()
        original_index = df.index.copy()  # Preserve original index
        df = cls.test_and_norm_df(df, newnamecol)

        # Handle empty DataFrames by returning empty result with correct structure
        if df.empty:
            # Load race categories to know what columns to create
            if cls.get_race() is None:
                with as_file(race_path) as race_file:
                    cls.set_race(pd.read_csv(race_file).race.tolist())

            result_df = df.copy()
            result_df["race"] = None
            # Add race columns (these will be empty but maintain structure)
            for race in cls.get_race():
                result_df[race] = float("nan")
            return result_df

        df[newnamecol] = df[newnamecol].astype(str).str.strip().str.title()
        rowindex_added = "__rowindex" not in df.columns
        if rowindex_added:
            df["__rowindex"] = np.arange(len(df))

        # Load model, vocab, and race label set once
        if cls.get_model() is None:
            with as_file(vocab_path) as vocab_file:
                cls.set_vocab(pd.read_csv(vocab_file).vocab.tolist())
            with as_file(race_path) as race_file:
                cls.set_race(pd.read_csv(race_file).race.tolist())
            # ``time_major`` argument was removed in Keras 3. Models bundled with
            # ethnicolr were trained with ``time_major=False`` which causes
            # deserialization errors on newer Keras versions. We register a
            # compatibility LSTM that simply ignores the argument.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Argument `input_length` is deprecated",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Do not pass an `input_shape`/`input_dim` argument",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Argument `decay` is no longer supported",
                    category=UserWarning,
                )
                logger.debug(f"Loading model from: {model_path}")
                with as_file(model_path) as model_file:
                    model = load_model(
                        model_file,
                        custom_objects={"LSTM": _CompatLSTM},
                        compile=False,
                    )
                logger.debug(f"Model loaded successfully: {type(model).__name__}")
                cls.set_model(model)

        # Vectorize input
        logger.debug(f"Vectorizing {len(df)} names using {ngrams}-grams")
        X = [cls.find_ngrams(cls.get_vocab(), name, ngrams) for name in df[newnamecol]]
        X = sequence.pad_sequences(X, maxlen=maxlen)
        logger.debug(f"Padded sequences to shape: {X.shape}")

        if conf_int == 1:
            proba = cls.get_model()(X, training=False).numpy()
            proba_df = pd.DataFrame(proba, columns=cls.get_race())
            proba_df["race"] = proba_df.idxmax(axis=1)
            # Use original index for alignment
            proba_df.index = df.index
            final_df = pd.concat([df, proba_df], axis=1)

        else:
            lower_perc = (0.5 - conf_int / 2) * 100
            upper_perc = (0.5 + conf_int / 2) * 100

            logger.info(
                f"Generating {num_iter} samples for CI [{lower_perc:.1f}%, {upper_perc:.1f}%]"
            )
            logger.debug(f"Model input shape: {X.shape}, Data types: {X.dtype}")

            all_preds = [
                cls.get_model()(X, training=True).numpy() for _ in range(num_iter)
            ]
            logger.debug(
                f"Generated {len(all_preds)} prediction arrays, shape: {all_preds[0].shape}"
            )
            stacked = np.vstack(all_preds)
            pdf = pd.DataFrame(stacked, columns=cls.get_race())
            pdf["__rowindex"] = np.tile(df["__rowindex"].to_numpy(), num_iter)

            agg = {
                col: [
                    "mean",
                    "std",
                    lambda x: np.percentile(x, q=lower_perc),
                    lambda x: np.percentile(x, q=upper_perc),
                ]
                for col in cls.get_race()
            }

            summary = pdf.groupby("__rowindex").agg(agg).reset_index()

            # Flatten column names
            summary.columns = [
                "_".join(filter(None, map(str, col))) for col in summary.columns
            ]
            summary.columns = [
                col.replace("<lambda_0>", "lb").replace("<lambda_1>", "ub")
                for col in summary.columns
            ]

            # Choose race with highest mean
            means = [col for col in summary.columns if col.endswith("_mean")]
            race_cols: pd.Series = summary[means].idxmax(axis=1)  # type: ignore
            summary["race"] = race_cols.str.replace("_mean", "")

            # Add basic probability columns (same as mean values for compatibility)
            for col in cls.get_race():
                summary[col] = summary[f"{col}_mean"]

            # Convert CI columns to float
            for suffix in ["_lb", "_ub"]:
                target = [col for col in summary.columns if col.endswith(suffix)]
                summary[target] = summary[target].astype(float)

            # Align rowindex column name for join
            summary.rename(columns={"__rowindex_": "__rowindex"}, inplace=True)

            final_df = df.merge(summary, on="__rowindex", how="left")

        # Clean up
        if rowindex_added:
            final_df.drop(columns=["__rowindex"], inplace=True, errors="ignore")

        # Restore original index only if lengths match
        # If rows were filtered out, we can't restore the original index directly
        if len(final_df) == len(original_index):
            final_df.index = original_index
        else:
            # Use reindex to handle missing indices properly, filling with appropriate defaults
            final_df = final_df.reindex(original_index)

        return final_df

    # Abstract method implementations
    @classmethod
    def predict(cls, df: pd.DataFrame, name_col: str, **kwargs) -> pd.DataFrame:
        """
        Implementation of abstract predict method.
        Delegates to transform_and_pred with model-specific parameters.
        """
        # This base implementation requires subclasses to override with their specific parameters
        raise NotImplementedError("Subclasses must implement predict() method")

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
        Implementation of abstract predict_with_confidence method.
        """
        # This base implementation requires subclasses to override with their specific parameters
        raise NotImplementedError(
            "Subclasses must implement predict_with_confidence() method"
        )

    @classmethod
    def get_supported_categories(cls) -> list[str]:
        """Get list of race/ethnicity categories this model predicts."""
        if hasattr(cls, "SUPPORTED_CATEGORIES"):
            result: list[str] = []
            for cat in cls.SUPPORTED_CATEGORIES:
                if hasattr(cat, "value"):
                    result.append(cat.value)  # type: ignore
                else:
                    result.append(cat)  # type: ignore
            return result
        else:
            # Fallback to runtime race categories if loaded
            race_list = cls.get_race()
            return race_list if race_list else []

    @classmethod
    def validate_input(cls, df: pd.DataFrame, name_col: str) -> None:
        """
        Validate input DataFrame and column.

        Raises:
            InvalidInputError: If validation fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise InvalidInputError("Input must be a pandas DataFrame")

        if df.empty:
            raise InvalidInputError("Input DataFrame is empty")

        if name_col not in df.columns:
            raise InvalidInputError(f"Column '{name_col}' not found in DataFrame")

        if bool(df[name_col].isna().all()):
            raise InvalidInputError(f"All values in column '{name_col}' are NaN")
