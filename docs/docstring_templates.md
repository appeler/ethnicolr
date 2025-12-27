# Docstring Templates for ethnicolr

This document provides standard templates for consistent documentation across the ethnicolr codebase.

## Core Prediction Function Template

```python
def pred_model_name(
    df: pd.DataFrame,
    name_col: str,
    additional_params...
) -> pd.DataFrame:
    """Predict race/ethnicity from names using [Model Name] LSTM model.

    Uses machine learning models trained on [data source] to predict race/ethnicity
    categories. This model works best with [specific characteristics] and typically
    achieves [accuracy range]% accuracy on diverse name datasets.

    **Model Performance:**
    - Best suited for: [specific use cases]
    - Expected accuracy: [range]% on typical datasets
    - Limitations: [known limitations]

    **Categories Predicted:**
    - [list of race/ethnicity categories]

    Args:
        df: Input DataFrame containing names to predict.
        name_col: Name of column containing [type] names.
        [additional parameters with clear descriptions]

    Returns:
        DataFrame with original data plus prediction columns:
        - 'race': Predicted race/ethnicity category
        - '[category1]', '[category2]', etc.: Probability scores (0-1)
        - Confidence interval columns if conf_int < 1.0

    Raises:
        ValueError: If required column missing or invalid parameters.
        FileNotFoundError: If model files not found (install with pip install ethnicolr[models]).

    Example:
        Basic prediction:

        >>> import pandas as pd
        >>> from ethnicolr import pred_model_name
        >>> df = pd.DataFrame({'names': ['Smith', 'Garcia', 'Zhang', 'Johnson']})
        >>> result = pred_model_name(df, 'names')
        >>> print(result[['names', 'race', 'confidence']].head())
           names      race  confidence
        0  Smith     white        0.85
        1 Garcia  hispanic        0.92
        2  Zhang     asian        0.88
        3 Johnson    black        0.79

        With confidence intervals:

        >>> result_conf = pred_model_name(df, 'names', conf_int=0.9)
        >>> print(result_conf[['names', 'race', 'white_mean']].head())

    See Also:
        - pred_other_model: Alternative model for [different use case]
        - [model]_cli: Command-line interface for batch processing

    Note:
        [Any important usage notes, performance tips, or warnings]
    """
```

## CLI Main Function Template

```python
def main(argv: list[str] | None = None) -> int:
    """Command-line interface for [model name] predictions.

    Provides batch processing of CSV files with [model] race/ethnicity predictions.
    Supports confidence intervals, custom output formats, and error handling.

    Args:
        argv: Command-line arguments (uses sys.argv[1:] if None).

    Returns:
        Exit code: 0 for success, non-zero for errors:
        - 1: General error (invalid arguments, processing failure)
        - 2: Missing input files or model files
        - 3: Invalid data format or empty results

    Example:
        Basic usage:

        $ pred_model_name input.csv -l lastname -o predictions.csv

        With confidence intervals:

        $ pred_model_name data.csv -l surname -c 0.9 -i 50 -o results.csv

        Custom column names:

        $ pred_model_name names.csv -l family_name -f given_name -o output.csv

    CLI Options:
        input: Path to input CSV file with name columns
        -l, --last: Column name for last names (required)
        -f, --first: Column name for first names (required for full-name models)
        -o, --output: Output file path (default: [model]_output.csv)
        -c, --conf: Confidence level 0.0-1.0 (default: 1.0 for point estimates)
        -i, --iter: Monte Carlo iterations for confidence intervals (default: 100)

    Note:
        Requires model files. Install with: pip install ethnicolr[models]
    """
```

## Utility Function Template

```python
def utility_function(param1: type, param2: type = default) -> return_type:
    """Brief description of utility function purpose.

    Longer description explaining what the function does, how it works,
    and when to use it. Include any important algorithmic details.

    Args:
        param1: Clear description of first parameter and its constraints.
        param2: Description of optional parameter with default behavior.

    Returns:
        Description of return value and its format/structure.

    Raises:
        ExceptionType: When and why this exception is raised.

    Example:
        >>> result = utility_function("example", param2="custom")
        >>> print(result)
        expected_output

    Note:
        Any important implementation details or usage warnings.
    """
```

## Class Documentation Template

```python
class ModelClass:
    """Brief description of class purpose and functionality.

    Longer description explaining the class's role in the system,
    its relationship to other classes, and key design decisions.

    This class provides [main functionality] and is designed for [use cases].
    It implements [key algorithms/patterns] and manages [key resources].

    Attributes:
        class_attr: Description of class-level attributes.
        instance_attr: Description of instance attributes.

    Example:
        >>> model = ModelClass()
        >>> result = model.main_method(data)
        >>> print(result.summary())

    Note:
        Any important usage notes, thread safety, or performance considerations.
    """

    def method_name(self, param: type) -> type:
        """Method description following function template above."""
```

## Module-Level Documentation Template

```python
"""Module Name: Brief Description

Longer description of module purpose, main functionality, and relationship
to other modules. Explain the primary use cases and key classes/functions.

This module provides [main capabilities] for [target users]. It implements
[key algorithms] and integrates with [other components].

Main Functions:
    - function1: Brief description
    - function2: Brief description

Main Classes:
    - Class1: Brief description
    - Class2: Brief description

Example:
    Basic usage:

    >>> from ethnicolr.module import main_function
    >>> result = main_function(data, parameters)
    >>> print(result.head())

See Also:
    - Related module: Description of relationship
    - External docs: URL if applicable

Note:
    Any module-level warnings, dependencies, or configuration requirements.
"""
```

## Documentation Standards

### Style Guidelines
- **Format**: Use Google-style docstrings consistently
- **Length**: One-line summary, then detailed description
- **Sections**: Use standard sections (Args, Returns, Raises, Example, Note, See Also)
- **Examples**: Include realistic, executable examples with expected outputs
- **Types**: Use proper type hints in function signatures

### Content Requirements
- **Performance context**: Include accuracy expectations and limitations
- **Error handling**: Document all exception conditions and solutions
- **Cross-references**: Link to related functions and CLI commands
- **Usage guidance**: Explain when and why to use each function

### Quality Checklist
- [ ] One-line summary is clear and specific
- [ ] Detailed description provides context and use cases
- [ ] All parameters are documented with types and constraints
- [ ] Return value format is clearly specified
- [ ] All exceptions are documented with conditions
- [ ] Examples are realistic and show expected outputs
- [ ] Performance expectations are stated
- [ ] Cross-references are provided where relevant
- [ ] Any limitations or warnings are noted
