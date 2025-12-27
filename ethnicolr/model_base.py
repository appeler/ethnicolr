#!/usr/bin/env python
"""
Abstract base classes for ethnicolr prediction models.

Provides formal interfaces and contracts for different types of prediction models,
ensuring consistency and making it easier to extend the library with new models.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import pandas as pd

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""

    CENSUS_LOOKUP = "census_lookup"  # Direct census statistics lookup
    CENSUS_LSTM = "census_lstm"  # LSTM trained on census data
    WIKI_LSTM = "wiki_lstm"  # LSTM trained on Wikipedia data
    FLORIDA_LSTM = "florida_lstm"  # LSTM trained on Florida voter data
    NC_LSTM = "nc_lstm"  # LSTM trained on NC voter data


class RaceCategory(Enum):
    """Enumeration of common race/ethnicity categories."""

    # Standard 4-category system (Census LSTM)
    WHITE = "white"
    BLACK = "black"
    ASIAN = "api"  # Asian/Pacific Islander
    HISPANIC = "hispanic"

    # Extended Florida 5-category system
    NH_WHITE = "nh_white"  # Non-Hispanic White
    NH_BLACK = "nh_black"  # Non-Hispanic Black
    OTHER = "other"  # Other/Multiracial

    # Extended North Carolina system
    HL_ASIAN = "HL+A"  # Hispanic Latino + Asian
    HL_BLACK = "HL+B"  # Hispanic Latino + Black
    HL_INDIAN = "HL+I"  # Hispanic Latino + American Indian
    HL_MULTIRACIAL = "HL+M"  # Hispanic Latino + Multiracial
    HL_OTHER = "HL+O"  # Hispanic Latino + Other
    HL_WHITE = "HL+W"  # Hispanic Latino + White
    NL_ASIAN = "NL+A"  # Non-Latino + Asian
    NL_BLACK = "NL+B"  # Non-Latino + Black
    NL_INDIAN = "NL+I"  # Non-Latino + American Indian
    NL_MULTIRACIAL = "NL+M"  # Non-Latino + Multiracial
    NL_OTHER = "NL+O"  # Non-Latino + Other
    NL_WHITE = "NL+W"  # Non-Latino + White


class PredictionError(Exception):
    """Base exception for prediction-related errors."""

    pass


class ModelNotFoundError(PredictionError):
    """Raised when required model files are not found."""

    pass


class InvalidInputError(PredictionError):
    """Raised when input data is invalid or malformed."""

    pass


class AbstractEthnicolrModel(ABC):
    """
    Abstract base class for all ethnicolr prediction models.

    Defines the common interface that all prediction models must implement,
    including model loading, caching, and prediction methods.
    """

    # Subclasses must define these class attributes
    MODEL_TYPE: ClassVar[ModelType]
    SUPPORTED_CATEGORIES: ClassVar[list[RaceCategory | str]]
    MODELFN: ClassVar[str]  # Path to model file
    VOCABFN: ClassVar[str | None] = None  # Path to vocabulary (LSTM only)
    RACEFN: ClassVar[str | None] = None  # Path to race labels (LSTM only)

    @classmethod
    @abstractmethod
    def predict(cls, df: pd.DataFrame, name_col: str, **kwargs: Any) -> pd.DataFrame:
        """
        Generate race/ethnicity predictions for input names.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing names to predict.
            **kwargs: Model-specific parameters.

        Returns:
            DataFrame with original data plus prediction columns.

        Raises:
            InvalidInputError: If input data is invalid.
            ModelNotFoundError: If required model files are missing.
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_categories(cls) -> list[str]:
        """Get list of race/ethnicity categories this model predicts."""
        pass

    @classmethod
    @abstractmethod
    def validate_input(cls, df: pd.DataFrame, name_col: str) -> None:
        """
        Validate input DataFrame and column.

        Args:
            df: Input DataFrame to validate.
            name_col: Name of column to validate.

        Raises:
            InvalidInputError: If validation fails.
        """
        pass


class AbstractLSTMModel(AbstractEthnicolrModel):
    """
    Abstract base class for LSTM-based prediction models.

    Extends the base model interface with LSTM-specific functionality
    including n-gram processing, confidence intervals, and model caching.
    """

    # LSTM-specific class attributes
    NGRAMS: ClassVar[int | tuple[int, int]]  # N-gram configuration
    FEATURE_LEN: ClassVar[int]  # Maximum sequence length

    @classmethod
    @abstractmethod
    def predict_with_confidence(
        cls,
        df: pd.DataFrame,
        name_col: str,
        conf_int: float = 0.95,
        num_iter: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing names to predict.
            conf_int: Confidence interval level (0.0-1.0).
            num_iter: Number of Monte Carlo iterations.
            **kwargs: Additional model-specific parameters.

        Returns:
            DataFrame with predictions and confidence interval columns.
        """
        pass


class AbstractCensusModel(AbstractEthnicolrModel):
    """
    Abstract base class for Census-based models.

    Handles Census-specific functionality including year selection
    and demographic percentage lookup.
    """

    SUPPORTED_YEARS: ClassVar[list[int]]  # Supported census years

    @classmethod
    @abstractmethod
    def predict_by_year(
        cls, df: pd.DataFrame, name_col: str, year: int, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Generate predictions using specific census year.

        Args:
            df: Input DataFrame containing names.
            name_col: Column containing names to predict.
            year: Census year to use for predictions.
            **kwargs: Additional parameters.

        Returns:
            DataFrame with year-specific predictions.
        """
        pass


class ModelRegistry:
    """
    Registry for managing available prediction models.

    Provides discovery and instantiation of models by type,
    making it easy to add new models without modifying existing code.
    """

    _models: dict[ModelType, type[AbstractEthnicolrModel]] = {}

    @classmethod
    def register(cls, model_class: type[AbstractEthnicolrModel]) -> None:
        """Register a model class with the registry."""
        if not hasattr(model_class, "MODEL_TYPE"):
            raise ValueError(
                f"Model class {model_class.__name__} must define MODEL_TYPE"
            )

        cls._models[model_class.MODEL_TYPE] = model_class
        logger.debug(
            f"Registered model: {model_class.__name__} ({model_class.MODEL_TYPE.value})"
        )

    @classmethod
    def get_model(cls, model_type: ModelType) -> type[AbstractEthnicolrModel]:
        """Get model class by type."""
        if model_type not in cls._models:
            raise ValueError(f"Model type {model_type.value} not registered")
        return cls._models[model_type]

    @classmethod
    def get_available_models(cls) -> dict[ModelType, type[AbstractEthnicolrModel]]:
        """Get all registered models."""
        return cls._models.copy()

    @classmethod
    def list_model_types(cls) -> list[ModelType]:
        """Get list of available model types."""
        return list(cls._models.keys())


def register_model(model_type: ModelType):
    """Decorator for auto-registering model classes."""

    def decorator(
        model_class: type[AbstractEthnicolrModel],
    ) -> type[AbstractEthnicolrModel]:
        model_class.MODEL_TYPE = model_type
        ModelRegistry.register(model_class)
        return model_class

    return decorator
