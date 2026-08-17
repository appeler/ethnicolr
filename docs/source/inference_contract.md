# Inference result contract

Ethnicolr returns name-pattern estimates under inference contract version 1.0.
The contract separates whether a model produced a probability distribution from
whether Ethnicolr chose to return a label. This distinction matters because a
model can score a name while abstaining from a label when the evidence is
ambiguous.

The contract is intended to be shared by Pranaam, Instate, Outkast, Naampy, and
other name-inference packages. Each package retains its target-specific
probability columns; the common columns make results interpretable and testable
across packages.

If an input column uses a name reserved for estimator output, Ethnicolr preserves
the input under `input_<name>`. It adds a numeric suffix when that name already
exists.

## Common columns

| Column | Meaning |
| --- | --- |
| `inference_contract_version` | Version of this result contract. |
| `estimate_type` | Always `name-pattern estimate`. |
| `target` | Quantity estimated, such as `race-ethnicity` or `country-origin`. |
| `input_scope` | Name components used: `first-name`, `last-name`, or `full-name`. |
| `predicted_label` | Highest-probability returned label, or missing after abstention. |
| `predicted_probability` | Probability of `predicted_label` on a 0 to 1 scale. |
| `scored` | Whether the model produced a usable probability distribution. |
| `script_supported` | Whether the input script is supported. |
| `abstained` | Whether the package declined to return a label. |
| `abstention_reason` | Machine-readable reason for abstention. |
| `model_id` | Stable identifier for the model or dictionary. |
| `model_version` | Version of the package producing the estimate. |
| `model_revision` | Immutable revision of every artifact used for inference. |
| `reference_population` | Population represented by the training or lookup data. |
| `calibration_status` | Whether and how probability calibration was validated. |
| `calibration_reference` | Data population used to assess or fit calibration. |
| `uncertainty_method` | Method used for uncertainty output, when requested. |
| `uncertainty_level` | Requested interval, range, or coverage level. |

## Required invariants

- An unscored row must abstain.
- A row that does not abstain must be scored and have a predicted label and
  probability.
- An unsupported-script row cannot be scored.
- Target probabilities on a scored row must be finite, between zero and one,
  and sum to one, apart from documented rounding.
- `predicted_probability` must equal the target probability associated with
  `predicted_label`.
- `model_revision` must identify an immutable, complete artifact bundle. A
  mutable branch name such as `main` or `latest` is not a revision.
- A scored row may still abstain when a package applies an uncertainty rule. In
  that case the probability distribution remains available but
  `predicted_label` is missing.

## Abstention reasons

The shared vocabulary is `missing-name`, `no-letters`, `unsupported-script`,
`out-of-vocabulary`, `out-of-dictionary`, `uncertain-score`, and
`insufficient-evidence`. Packages may add a reason only when none of these is
accurate.

## Uncertainty columns

Statistical interval columns use `<probability_column>_lower` and
`<probability_column>_upper`. Monte Carlo dropout summaries use
`<probability_column>_mc_mean`, `_mc_std`, `_mc_lower`, and `_mc_upper` because
their empirical quantiles are not automatically confidence intervals.

## Intended use

These outputs describe patterns associated with names in a stated reference
population. They do not establish a person's identity, ancestry, citizenship,
religion, caste, race, or ethnicity. Do not use them for individual profiling
or consequential decisions. Prefer aggregate analysis, disclose abstention and
coverage, and report sensitivity to the reference population.
