"""
Shared helper functions for ethnicolr tests.
"""

import numpy as np
import pandas as pd


def validate_probabilities_sum_to_one(
    df: pd.DataFrame, race_columns: list[str], tolerance: float = 0.1
) -> bool:
    """
    Validate that probability columns sum to approximately 1.0 for each row.

    Args:
        df: DataFrame with prediction results
        race_columns: List of column names containing race probabilities
        tolerance: Allowable deviation from 1.0

    Returns:
        True if all rows sum to approximately 1.0
    """
    available_columns = [col for col in race_columns if col in df.columns]
    if not available_columns:
        return False

    sums = df[available_columns].sum(axis=1)
    return all(abs(sums - 1.0) <= tolerance)


def validate_prediction_columns(df: pd.DataFrame, expected_columns: list[str]) -> bool:
    """
    Validate that DataFrame contains expected prediction columns.

    Args:
        df: DataFrame to check
        expected_columns: List of expected column names

    Returns:
        True if all expected columns are present
    """
    return all(col in df.columns for col in expected_columns)


def validate_race_prediction_consistency(
    df: pd.DataFrame,
    true_race_col: str = "true_race",
    predicted_race_col: str = "race",
    min_accuracy: float = 0.6,
    model_type: str = "census",
) -> bool:
    """
    Validate that predicted race has reasonable accuracy for test data.

    Args:
        df: DataFrame with prediction results
        true_race_col: Column name with true race labels
        predicted_race_col: Column name with predicted race labels
        min_accuracy: Minimum accuracy threshold (default 0.6 for 60%)
        model_type: Type of model for realistic accuracy expectations

    Returns:
        True if prediction accuracy meets minimum threshold
    """
    # Model-specific realistic accuracy thresholds
    model_thresholds = {
        "census": 0.6,  # Census models perform well
        "florida": 0.5,  # Florida models moderate performance
        "nc": 0.2,  # NC models with complex categories and known limitations
        "wiki": 0.45,  # Wiki models variable performance
    }

    # Use model-specific threshold if available, otherwise use provided min_accuracy
    if model_type in model_thresholds:
        min_accuracy = model_thresholds[model_type]
    if true_race_col not in df.columns or predicted_race_col not in df.columns:
        return False

    if len(df) == 0:
        return True

    accuracy = (df[true_race_col] == df[predicted_race_col]).mean()
    return accuracy >= min_accuracy


def validate_no_nan_in_predictions(df: pd.DataFrame, race_columns: list[str]) -> bool:
    """
    Validate that prediction columns contain no NaN values.

    Args:
        df: DataFrame to check
        race_columns: List of race probability column names

    Returns:
        True if no NaN values found in race columns
    """
    available_columns = [col for col in race_columns if col in df.columns]
    if not available_columns:
        return True  # No columns to check

    return not df[available_columns].isnull().any().any()


def validate_probability_ranges(df: pd.DataFrame, race_columns: list[str]) -> bool:
    """
    Validate that all probability values are between 0 and 1.

    Args:
        df: DataFrame to check
        race_columns: List of race probability column names

    Returns:
        True if all values are in [0, 1] range
    """
    available_columns = [col for col in race_columns if col in df.columns]
    if not available_columns:
        return True  # No columns to check

    values = df[available_columns].values
    return np.all((values >= 0) & (values <= 1))


def validate_mc_dropout_columns(df: pd.DataFrame, mean_columns: list[str]) -> bool:
    """
    Validate that Monte Carlo dropout mean columns are present and valid.

    Args:
        df: DataFrame to check
        mean_columns: Expected mean columns (for example, ``white_mc_mean``).

    Returns:
        True if Monte Carlo dropout columns are present and valid.
    """
    if not validate_prediction_columns(df, mean_columns):
        return False

    return validate_probability_ranges(
        df, mean_columns
    ) and validate_no_nan_in_predictions(df, mean_columns)


def get_race_columns_for_model(model_type: str) -> list[str]:
    """
    Get the expected race columns for different model types.

    Args:
        model_type: Type of model ('census', 'wiki', 'florida_4cat',
            'florida_5cat', 'nc')

    Returns:
        List of expected race column names
    """
    if model_type == "census":
        return ["api", "black", "hispanic", "white"]
    if model_type == "wiki":
        return [
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
    if model_type == "florida_4cat":
        return ["asian", "hispanic", "nh_black", "nh_white"]
    if model_type == "florida_5cat":
        return ["asian", "hispanic", "nh_black", "nh_white", "other"]
    if model_type == "nc":
        return [
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
    raise ValueError(f"Unknown model type: {model_type}")


def get_mc_mean_columns_for_model(model_type: str) -> list[str]:
    """
    Get the expected Monte Carlo dropout mean columns for a model type.

    Args:
        model_type: Type of model

    Returns:
        List of expected mean column names
    """
    base_columns = get_race_columns_for_model(model_type)
    return [col + "_mc_mean" for col in base_columns]


def validate_basic_prediction_output(
    df: pd.DataFrame, model_type: str, with_uncertainty: bool = False
) -> dict:
    """
    Comprehensive validation of prediction output for any model type.

    Args:
        df: DataFrame with prediction results
        model_type: Type of model being tested
        with_uncertainty: Whether MC-dropout uncertainty summaries are expected

    Returns:
        Dictionary with validation results
    """
    race_columns = get_race_columns_for_model(model_type)
    results = {
        "has_expected_columns": validate_prediction_columns(df, race_columns),
        "probabilities_sum_to_one": validate_probabilities_sum_to_one(df, race_columns),
        "no_nan_values": validate_no_nan_in_predictions(df, race_columns),
        "valid_probability_ranges": validate_probability_ranges(df, race_columns),
        "has_race_column": "race" in df.columns,
    }

    if with_uncertainty:
        mean_columns = get_mc_mean_columns_for_model(model_type)
        results.update(
            {
                "has_mc_dropout_columns": validate_prediction_columns(df, mean_columns),
                "mc_probabilities_sum_to_one": validate_probabilities_sum_to_one(
                    df, mean_columns
                ),
                "mc_no_nan": validate_no_nan_in_predictions(df, mean_columns),
                "mc_valid_ranges": validate_probability_ranges(df, mean_columns),
            }
        )

    results["all_validations_passed"] = all(results.values())
    return results


def assert_prediction_quality(
    df: pd.DataFrame, model_type: str, with_uncertainty: bool = False
):
    """
    Assert that prediction output meets quality standards.

    Args:
        df: DataFrame with prediction results
        model_type: Type of model being tested
        with_uncertainty: Whether MC-dropout uncertainty summaries are expected

    Raises:
        AssertionError: If any validation fails
    """
    results = validate_basic_prediction_output(df, model_type, with_uncertainty)

    assert results["has_expected_columns"], (
        f"Missing expected columns for {model_type} model"
    )
    assert results["probabilities_sum_to_one"], "Probabilities do not sum to 1.0"
    assert results["no_nan_values"], "Found NaN values in prediction columns"
    assert results["valid_probability_ranges"], (
        "Found probability values outside [0,1] range"
    )
    assert results["has_race_column"], "Missing 'race' column with predicted race"

    if with_uncertainty:
        assert results["has_mc_dropout_columns"], "Missing MC-dropout columns"
        assert results["mc_probabilities_sum_to_one"], (
            "MC-dropout mean probabilities do not sum to 1.0"
        )
        assert results["mc_no_nan"], "Found NaN values in MC-dropout columns"
        assert results["mc_valid_ranges"], "Found MC-dropout values outside [0,1] range"


def create_test_summary(
    test_name: str,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    model_type: str,
    with_uncertainty: bool = False,
) -> str:
    """
    Create a summary of test results for debugging.

    Args:
        test_name: Name of the test
        input_df: Input DataFrame
        output_df: Output DataFrame with predictions
        model_type: Type of model tested
        with_uncertainty: Whether MC-dropout uncertainty summaries were used

    Returns:
        String summary of the test
    """
    validation_results = validate_basic_prediction_output(
        output_df, model_type, with_uncertainty
    )

    summary = f"""
Test: {test_name}
Model Type: {model_type}
Input rows: {len(input_df)}
Output rows: {len(output_df)}
With MC-dropout uncertainty summaries: {with_uncertainty}

Validation Results:
"""
    for check, passed in validation_results.items():
        status = "✓" if passed else "✗"
        summary += f"  {status} {check}\n"

    if not validation_results["all_validations_passed"]:
        summary += f"\nOutput columns: {list(output_df.columns)}\n"
        summary += f"Sample output:\n{output_df.head()}\n"

    return summary
