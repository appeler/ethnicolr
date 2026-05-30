#!/usr/bin/env python
"""
Census Last Name Race/Ethnicity Prediction Module.

Uses PyTorch LSTM models trained on U.S. Census data to predict race/ethnicity from last names.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .utils import arg_parser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RACES = ["api", "black", "hispanic", "white"]
NGRAMS = 2
FEATURE_LEN = 20


class CensusLSTM(nn.Module):
    """Character-bigram LSTM for race/ethnicity prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)


class CensusLnModel:
    """Census-based last name prediction using PyTorch."""

    _models: dict[int, nn.Module] = {}
    _vocabs: dict[int, dict[str, int]] = {}
    _device: torch.device | None = None

    @classmethod
    def get_device(cls) -> torch.device:
        if cls._device is None:
            if torch.cuda.is_available():
                cls._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                cls._device = torch.device("mps")
            else:
                cls._device = torch.device("cpu")
        return cls._device

    @classmethod
    def get_model_paths(cls, year: int) -> tuple[Path, Path, Path]:
        package = resources.files(__name__.split(".")[0])
        base = Path(str(package)) / "models" / "census" / "lstm"
        return (
            base / f"census{year}_ln_lstm_pytorch.pt",
            base / f"census{year}_ln_vocab_pytorch.csv",
            base / f"census{year}_race_pytorch.csv",
        )

    @classmethod
    def load_model(cls, year: int) -> nn.Module:
        if year in cls._models:
            return cls._models[year]

        model_path, vocab_path, _ = cls.get_model_paths(year)

        if not model_path.exists():
            raise FileNotFoundError(
                f"PyTorch model not found: {model_path}\n"
                f"Train with: python scripts/model-training/census/train_census_lstm_pytorch.py --year {year}"
            )

        device = cls.get_device()
        logger.info(f"Loading Census {year} PyTorch model on {device}...")

        # Load vocabulary first to get vocab size
        vocab_df = pd.read_csv(vocab_path)
        vocab_list = vocab_df["vocab"].tolist()
        cls._vocabs[year] = {word: idx for idx, word in enumerate(vocab_list)}

        # Create model with correct vocab size and load state dict
        model = CensusLSTM(vocab_size=len(vocab_list))
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        cls._models[year] = model

        return model

    @classmethod
    def get_vocab(cls, year: int) -> dict[str, int]:
        if year not in cls._vocabs:
            cls.load_model(year)
        return cls._vocabs[year]

    @classmethod
    def predict_with_confidence(
        cls,
        df: pd.DataFrame,
        lname_col: str,
        year: int = 2010,
        conf_int: float = 1.0,
        num_iter: int = 100,
    ) -> pd.DataFrame:
        """Class method interface for CLI compatibility."""
        return pred_census_ln(
            df, lname_col, year=year, num_iter=num_iter, conf_int=conf_int
        )

    @classmethod
    def names_to_sequences(cls, names: list[str], vocab: dict[str, int]) -> np.ndarray:
        sequences = []
        for name in names:
            name = str(name).strip().title()
            seq = []
            for i in range(len(name) - NGRAMS + 1):
                bigram = name[i : i + NGRAMS]
                seq.append(vocab.get(bigram, 0))
            sequences.append(seq)

        # Pad sequences (pre-padding)
        padded = np.zeros((len(sequences), FEATURE_LEN), dtype=np.int64)
        for i, seq in enumerate(sequences):
            if len(seq) > FEATURE_LEN:
                padded[i] = seq[:FEATURE_LEN]
            elif len(seq) > 0:
                padded[i, -len(seq) :] = seq
        return padded


def pred_census_ln(
    df: pd.DataFrame,
    lname_col: str,
    year: int = 2010,
    num_iter: int = 100,
    conf_int: float = 1.0,
) -> pd.DataFrame:
    """Predict race/ethnicity from last names using Census PyTorch model.

    Uses machine learning models trained on U.S. Census surname data
    to predict race/ethnicity categories: white, black, Asian, Hispanic.

    Args:
        df: Input DataFrame containing last names.
        lname_col: Name of column containing last names.
        year: Census year (2000, 2010, or 2020).
        num_iter: Monte Carlo iterations for confidence intervals.
        conf_int: Confidence level (1.0 for point estimates only).

    Returns:
        DataFrame with original data plus prediction columns:
        - 'race': Predicted race category
        - 'white', 'black', 'api', 'hispanic': Probability scores
        - Confidence bounds if conf_int < 1.0

    Raises:
        ValueError: If year not in [2000, 2010, 2020] or column missing.
        FileNotFoundError: If required model files not found.

    Example:
        >>> import pandas as pd
        >>> from ethnicolr import pred_census_ln
        >>> df = pd.DataFrame({'surname': ['Smith', 'Garcia', 'Wang']})
        >>> result = pred_census_ln(df, 'surname', year=2020)
        >>> print(result[['surname', 'race']].head())
           surname    race
        0    Smith   white
        1   Garcia hispanic
        2     Wang     api
    """
    if year not in [2000, 2010, 2020]:
        raise ValueError("Census year must be 2000, 2010, or 2020")

    if lname_col not in df.columns:
        raise ValueError(f"Column '{lname_col}' not found in DataFrame")

    # Check for duplicate column names
    if df.columns.duplicated().any() and (df.columns == lname_col).sum() > 1:
        raise ValueError(f"Duplicate column name '{lname_col}' found in DataFrame")

    # Handle empty DataFrame
    if len(df) == 0:
        result = df.copy()
        for race in RACES:
            result[race] = pd.Series(dtype=float)
        result["race"] = pd.Series(dtype=str)
        return result

    model = CensusLnModel.load_model(year)
    vocab = CensusLnModel.get_vocab(year)
    device = CensusLnModel.get_device()

    logger.info(f"Predicting {len(df)} names using Census {year} PyTorch model")

    # Convert names to sequences (handle categorical and null values)
    col_data = df[lname_col]
    if hasattr(col_data, "cat"):  # Categorical column
        col_data = col_data.astype(str)
    names = col_data.fillna("").astype(str).tolist()
    X = CensusLnModel.names_to_sequences(names, vocab)
    X_tensor = torch.from_numpy(X).to(device)

    # Get predictions
    with torch.no_grad():
        if conf_int < 1.0:
            # Monte Carlo dropout for confidence intervals
            model.train()  # Enable dropout
            predictions = []
            for _ in range(max(1, num_iter)):
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
            predictions_arr = np.stack(predictions, axis=0)

            # Calculate statistics
            mean_probs = predictions_arr.mean(axis=0)
            std_probs = predictions_arr.std(axis=0)

            alpha = 1 - conf_int
            lower = np.percentile(predictions_arr, alpha / 2 * 100, axis=0)
            upper = np.percentile(predictions_arr, (1 - alpha / 2) * 100, axis=0)

            model.eval()
        else:
            # Point estimates only
            logits = model(X_tensor)
            mean_probs = torch.softmax(logits, dim=1).cpu().numpy()
            std_probs = None
            lower = None
            upper = None

    # Build result DataFrame
    result = df.copy()

    # Normalize names to title case (as done during processing)
    # Convert to str first to handle categorical columns
    result[lname_col] = result[lname_col].astype(str).fillna("").str.strip().str.title()

    # Always add base probability columns
    for i, race in enumerate(RACES):
        result[race] = mean_probs[:, i]

    # Add confidence interval columns when requested
    if (
        conf_int < 1.0
        and std_probs is not None
        and lower is not None
        and upper is not None
    ):
        for i, race in enumerate(RACES):
            result[f"{race}_mean"] = mean_probs[:, i]
            result[f"{race}_std"] = std_probs[:, i]
            result[f"{race}_lb"] = lower[:, i]
            result[f"{race}_ub"] = upper[:, i]

    # Add predicted race
    pred_indices = mean_probs.argmax(axis=1)
    result["race"] = [RACES[i] for i in pred_indices]

    pred_count = result["race"].notna().sum()
    logger.info(f"Predicted {pred_count} of {len(df)} rows")

    return result


def main(argv: list[str] | None = None) -> int:
    """Command-line interface for Census last name predictions.

    Provides CLI access to Census-based race/ethnicity prediction.

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
            default_year=2020,
            year_choices=[2000, 2010, 2020],
        )

        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        logger.info(f"Loaded {len(df)} records")

        result = pred_census_ln(
            df=df,
            lname_col=args.last,
            year=args.year,
            num_iter=args.iter,
            conf_int=args.conf,
        )

        if os.path.exists(args.output):
            logger.warning(f"Overwriting existing file: {args.output}")

        result.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(f"Output written: {args.output} ({len(result)} rows)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing model files: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
