#!/usr/bin/env python

import logging
from importlib.resources import files
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model_base import AbstractLSTMModel, InvalidInputError
from .torch_utils import (
    NameLSTM,
    apply_prior,
    conformal_sets,
    get_device,
    load_model_stats,
    pad_sequences,
)

logger = logging.getLogger(__name__)


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

    Common Workflow:
        1. Load vocabulary, race labels, and pre-trained LSTM model
        2. Preprocess and normalize input names
        3. Extract character n-grams from names
        4. Convert n-grams to model input sequences
        5. Generate predictions using LSTM model
        6. Apply confidence interval estimation if requested
        7. Return DataFrame with probabilities and predicted race

    Class Constants (defined in subclasses):
        MODELFN (str): Path to .pt model state dict within package
        VOCABFN (str): Path to vocabulary CSV file within package
        RACEFN (str): Path to race labels CSV file within package
        NGRAMS (int | tuple): N-gram size(s) for feature extraction
        FEATURE_LEN (int): Maximum sequence length for model input

    Note:
        - This class is not intended for direct instantiation
        - Subclasses must define model file paths and parameters
        - All models use the PyTorch backend
    """

    # Loaded model resources keyed by resolved model path. Keying by path (not
    # class) lets one class serve multiple model variants (e.g. the Florida
    # five-category 2017 and 2022 models) without stale-cache mixups.
    _model_cache: dict[str, dict] = {}

    @staticmethod
    def _resolve_path(fn: str) -> Path:
        """Resolve a model resource path (package-relative or absolute)."""
        p = Path(fn)
        return p if p.is_absolute() else Path(str(files("ethnicolr"))) / fn

    @classmethod
    def _load_resources(cls, vocab_fn: str, race_fn: str, model_fn: str) -> dict:
        """Load and cache vocab, race labels, and model for a model file."""
        key = str(cls._resolve_path(model_fn))
        entry = cls._model_cache.get(key)
        if entry is None:
            vocab = pd.read_csv(cls._resolve_path(vocab_fn)).vocab.tolist()
            race = pd.read_csv(cls._resolve_path(race_fn)).race.tolist()
            device = get_device()
            logger.info(f"Loading model {model_fn} on {device}")
            model = NameLSTM(vocab_size=len(vocab), num_classes=len(race))
            state = torch.load(
                cls._resolve_path(model_fn), map_location=device, weights_only=True
            )
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            stats = load_model_stats(cls._resolve_path(model_fn))
            entry = {
                "vocab_dict": {gram: idx for idx, gram in enumerate(vocab)},
                "race": race,
                "model": model,
                "device": device,
                "stats": stats,
                "temperature": stats["temperature"] if stats else 1.0,
            }
            cls._model_cache[key] = entry
        return entry

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
    def find_ngrams(cls, vocab_dict: dict, text: str, n) -> list:
        """Convert text n-grams to vocabulary indices.

        Generates n-grams from input text and maps them to indices in a
        pre-defined vocabulary. Unknown n-grams are mapped to index 0.

        Args:
            vocab_dict: Mapping of n-gram string to vocabulary index.
            text: Input string to process.
            n: N-gram size, or tuple (start, stop) for range of sizes.

        Returns:
            List of vocabulary indices corresponding to text n-grams.
            Unknown n-grams map to index 0.

        Example:
            >>> vocab_dict = {'<UNK>': 0, 'th': 1, 'he': 2, 'sm': 3}
            >>> EthnicolrModelClass.find_ngrams(vocab_dict, 'smith', 2)
            [3, 0, 0, 0]  # 'sm' found at index 3, others unknown
        """
        if isinstance(n, tuple):
            ngram_iter = EthnicolrModelClass.range_ngrams(text, n)
        else:
            ngram_iter = zip(*[text[i:] for i in range(n)], strict=False)

        return [vocab_dict.get("".join(gram), 0) for gram in ngram_iter]

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
        prior: dict[str, float] | None = None,
        coverage: float | None = None,
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
            model_fn: Path to trained LSTM model state dict.
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
        """
        df = df.copy()
        original_index = df.index.copy()  # Preserve original index
        df = cls.test_and_norm_df(df, newnamecol)

        # Handle empty DataFrames by returning empty result with correct structure
        if df.empty:
            race = pd.read_csv(cls._resolve_path(race_fn)).race.tolist()
            result_df = df.copy()
            result_df["race"] = None
            # Add race columns (these will be empty but maintain structure)
            for r in race:
                result_df[r] = float("nan")
            return result_df

        df[newnamecol] = df[newnamecol].astype(str).str.strip().str.title()
        rowindex_added = "__rowindex" not in df.columns
        if rowindex_added:
            df["__rowindex"] = np.arange(len(df))

        entry = cls._load_resources(vocab_fn, race_fn, model_fn)
        model = entry["model"]
        race = entry["race"]
        stats = entry["stats"]
        temperature = entry["temperature"]

        if (prior is not None or coverage is not None) and conf_int != 1:
            raise ValueError(
                "prior and coverage require point predictions (conf_int=1)"
            )
        if prior is not None and coverage is not None:
            raise ValueError(
                "prior and coverage cannot be combined: the conformal "
                "guarantee is calibrated for unreweighted probabilities"
            )
        if (prior is not None or coverage is not None) and stats is None:
            raise ValueError(
                "this model has no calibration stats file; run "
                "scripts/model-training/calibrate_model.py"
            )
        if coverage is not None:
            available = stats["conformal_quantiles"] if stats else {}
            key = f"{coverage:.2f}"
            if key not in available:
                raise ValueError(
                    f"coverage must be one of {sorted(available)}, got {coverage}"
                )

        # Vectorize input
        logger.debug(f"Vectorizing {len(df)} names using {ngrams}-grams")
        X = [
            cls.find_ngrams(entry["vocab_dict"], name, ngrams)
            for name in df[newnamecol]
        ]
        X = pad_sequences(X, maxlen)
        X_tensor = torch.from_numpy(X).to(entry["device"])

        if conf_int == 1:
            with torch.no_grad():
                proba = (
                    torch.softmax(model(X_tensor) / temperature, dim=1).cpu().numpy()
                )
            if prior is not None:
                proba = apply_prior(
                    proba, race, prior, stats["train_class_distribution"]
                )
            proba_df = pd.DataFrame(proba, columns=race)
            proba_df["race"] = proba_df.idxmax(axis=1)
            if coverage is not None:
                qhat = stats["conformal_quantiles"][f"{coverage:.2f}"]
                proba_df["race_set"] = conformal_sets(proba, race, qhat)
            proba_df.index = df.index
            final_df = pd.concat([df, proba_df], axis=1)

        else:
            lower_perc = (0.5 - conf_int / 2) * 100
            upper_perc = (0.5 + conf_int / 2) * 100

            logger.info(
                f"Generating {num_iter} samples for CI [{lower_perc:.1f}%, {upper_perc:.1f}%]"
            )

            model.train()  # Enable dropout for Monte Carlo sampling
            with torch.no_grad():
                preds = np.stack(
                    [
                        torch.softmax(model(X_tensor) / temperature, dim=1)
                        .cpu()
                        .numpy()
                        for _ in range(max(1, num_iter))
                    ],
                    axis=0,
                )
            model.eval()

            mean = preds.mean(axis=0)
            std = preds.std(axis=0, ddof=1)
            lb = np.percentile(preds, lower_perc, axis=0)
            ub = np.percentile(preds, upper_perc, axis=0)

            summary = pd.DataFrame(index=df.index)
            for i, r in enumerate(race):
                summary[f"{r}_mean"] = mean[:, i]
                summary[f"{r}_std"] = std[:, i]
                summary[f"{r}_lb"] = lb[:, i]
                summary[f"{r}_ub"] = ub[:, i]
                summary[r] = mean[:, i]
            summary["race"] = pd.DataFrame(mean, columns=race, index=df.index).idxmax(
                axis=1
            )
            final_df = pd.concat([df, summary], axis=1)

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
        return []

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
