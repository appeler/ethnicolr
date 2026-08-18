# ethnicolr

[![CI](https://github.com/appeler/ethnicolr/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/ethnicolr/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ethnicolr.svg)](https://pypi.org/project/ethnicolr/)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://appeler.github.io/ethnicolr/)
[![PePy downloads](https://static.pepy.tech/badge/ethnicolr)](https://www.pepy.tech/projects/ethnicolr)

Ethnicolr estimates race, ethnicity, or country-of-origin patterns associated
with names. It provides exact lookups from published name tables, dictionary
estimators that combine first and last names, and PyTorch models trained on
U.S. Census, voter-registration, and Wikipedia/Wikidata data.

Every result is a name-pattern estimate tied to a stated reference population.
It is not evidence of a person's identity, ancestry, citizenship, race, or
ethnicity. Do not use Ethnicolr for individual profiling or decisions about
employment, education, credit, housing, policing, health care, eligibility, or
access to services.

## Install

Ethnicolr supports Python 3.11 through 3.14.

```bash
pip install ethnicolr
```

Neural model weights live in
[`gojiberries/ethnicolr`](https://huggingface.co/gojiberries/ethnicolr), not in
the Python wheel. The first use of a neural estimator downloads only its weight
file from a full, package-pinned Hugging Face commit. Later calls use the local
Hugging Face cache. Set `ETHNICOLR_MODEL_CACHE` to choose the cache directory,
or `ETHNICOLR_MODEL_DIR` to use a local mirror of the model repository. Exact
Census and voter-file lookup tables remain in the wheel.

## Quick start

```python
import pandas as pd

from ethnicolr import estimate_census_full_name

names = pd.DataFrame(
    {
        "surname": ["Smith", "Garcia", "Zhang"],
        "first_name": ["Tyrone", "Maria", "Wei"],
    }
)

estimates = estimate_census_full_name(
    names,
    surname_column="surname",
    first_name_column="first_name",
    year=2020,
)

print(
    estimates[
        [
            "surname",
            "first_name",
            "predicted_label",
            "predicted_probability",
            "abstained",
            "model_revision",
        ]
    ]
)
```

The returned data retains the input columns, adds the full target-specific
probability distribution, and appends the shared inference fields described
below.

## Public API

Use `lookup_*` when a function returns published table values. Use
`estimate_*` when a function combines evidence or runs a statistical model.

| Function | Input | Source and target |
| --- | --- | --- |
| `lookup_census_surname` | surname | Census race/ethnicity proportions |
| `lookup_census_first_name` | first name | Census 2020 race/ethnicity proportions |
| `estimate_census_surname` | surname | Census-trained race/ethnicity model |
| `estimate_census_full_name` | full name | Census tables with model fallback |
| `estimate_voter_file_full_name` | full name | Six-state voter-file race/ethnicity frequencies |
| `estimate_florida_voter_surname` | surname | Florida voter-file race/ethnicity model |
| `estimate_florida_voter_full_name` | full name | Florida voter-file race/ethnicity model |
| `estimate_north_carolina_voter_full_name` | full name | North Carolina voter-file race/ethnicity model |
| `estimate_wikipedia_surname` | surname | Wikipedia/Wikidata race/ethnicity model |
| `estimate_wikipedia_full_name` | full name | Wikipedia/Wikidata race/ethnicity model |
| `estimate_wikipedia_origin` | full name | Wikipedia/Wikidata country-of-origin model |

Required data and column arguments may be positional or named. Optional
arguments are keyword-only. The main optional arguments are:

- `uncertainty_level`: a value strictly between 0 and 1. Census lookups add
  Wilson bounds. Neural models add Monte Carlo dropout summaries.
- `mc_iterations`: number of Monte Carlo dropout draws when
  `uncertainty_level` is set.
- `target_prior`: target-population class proportions used to reweight model
  probabilities.
- `conformal_coverage`: requested marginal coverage for a conformal prediction
  set. Supported levels depend on the model artifact.

`target_prior` and `conformal_coverage` cannot be used together. Neither can be
combined with `uncertainty_level`. Ethnicolr reports the exact conflicting
arguments and the remedy.

## Result contract

All public functions implement inference contract version 1.0. Results include:

| Column | Meaning |
| --- | --- |
| `estimate_type` | `name-pattern estimate` |
| `target` | Quantity estimated, such as `race-ethnicity` or `country-origin` |
| `input_scope` | `first-name`, `last-name`, or `full-name` |
| `predicted_label` | Highest-probability label, or missing after abstention |
| `predicted_probability` | Probability of `predicted_label` on a 0 to 1 scale |
| `scored` | Whether the function produced a usable probability distribution |
| `script_supported` | Whether the estimator supports the input script |
| `abstained` | Whether Ethnicolr declined to return a label |
| `abstention_reason` | Machine-readable explanation for abstention |
| `model_id` | Stable model or table identifier |
| `model_version` | Ethnicolr package version |
| `model_revision` | SHA-256 revision of the complete runtime artifact bundle |
| `reference_population` | Population represented by the source data |
| `calibration_status` | Status of probability calibration validation |
| `uncertainty_method` | Requested uncertainty method, when present |
| `uncertainty_level` | Requested uncertainty level, when present |

Blank names, unsupported scripts, dictionary misses without a fallback, and
inputs with no learned features abstain. Their target probabilities and labels
remain missing. Ethnicolr does not silently map unsupported inputs to a default
class distribution.

See the [inference contract](docs/source/inference_contract.md) for invariants
and the shared abstention vocabulary.

## Uncertainty and calibration

Ethnicolr distinguishes three different operations:

1. Wilson bounds quantify sampling uncertainty in published Census lookup
   proportions.
2. Monte Carlo dropout summaries describe variation under repeated stochastic
   model passes. They are not confidence intervals.
3. Conformal prediction sets target marginal coverage for names exchangeable
   with a model's calibration data.

The [statistical principles](docs/source/statistical_principles.md) explain
calibration, target-prior adjustment, conformal sets, evaluation weighting,
and the first-name/last-name independence assumption. The
[model cards](docs/source/model_cards.md) report model-specific evidence.

## Current model evidence

The current neural models have the following held-out results. The model cards
define each evaluation population and report calibration metrics.

| Model | Classes | Accuracy | Top-3 accuracy |
| --- | ---: | ---: | ---: |
| Census 2000 surname | 4 | 0.833 | 0.993 |
| Census 2010 surname | 4 | 0.808 | 0.984 |
| Census 2020 surname | 4 | 0.807 | 0.988 |
| Florida voter surname | 5 | 0.588 | 0.947 |
| Florida voter full name | 5 | 0.677 | 0.948 |
| North Carolina voter full name | 12 | 0.425 | 0.896 |
| Wikipedia/Wikidata surname | 13 | 0.775 | 0.907 |
| Wikipedia/Wikidata full name | 13 | 0.863 | 0.954 |
| Wikipedia/Wikidata origin | 90 | 0.626 | 0.809 |

Florida and North Carolina use source-disjoint splits. Their earlier metrics
were invalid because balancing before splitting allowed source rows to cross
partitions. The 2.0 artifacts replace those models and their calibration data.

## Command line

The `ethnicolr` command exposes the supported surname estimators.

```bash
# List command-line models and their sources
ethnicolr models list --detailed

# Estimate from Census 2020 surnames
ethnicolr estimate census-surname names.csv \
  --last-column surname \
  --output census-estimates.csv

# Request Monte Carlo dropout summaries
ethnicolr estimate wikipedia-surname names.csv \
  --last-column surname \
  --uncertainty-level 0.90 \
  --mc-iterations 200 \
  --output wikipedia-estimates.csv

# Use the default quick estimator
ethnicolr quick-estimate names.csv --last-column surname
```

Run `ethnicolr --help` or `ethnicolr estimate --help` for the current command
surface. Removed per-model scripts are not compatibility aliases in 2.0.

## Name handling

Pass names as recorded unless you have a documented normalization rule for your
application. Ethnicolr trims surrounding whitespace and performs the model's
required normalization. It preserves the original rows and reports whether the
script is supported. Do not delete diacritics or coerce unsupported scripts into
Latin text merely to force a score.

## Runtime device

Inference uses CUDA when available and otherwise uses CPU. Set
`ETHNICOLR_DEVICE` to `cpu`, `cuda`, or `mps` to override device selection. MPS
is opt-in because some virtualized Apple Silicon environments advertise MPS but
return incorrect LSTM output.

## Data

Ethnicolr uses:

- U.S. Census surname tables for 2000, 2010, and 2020
- the U.S. Census 2020 first-name table
- Florida voter-registration records from 2022
- North Carolina voter-registration records
- six-state voter-file name frequencies from Alabama, Florida, Georgia,
  Louisiana, North Carolina, and South Carolina
- Wikipedia/Wikidata person records produced by the repository's reproducible
  acquisition pipeline

Each estimator reports its reference population. Sources, licenses, acquisition
steps, and known limitations are documented in the data directories and model
cards.

## Development

```bash
uv sync --all-groups
make lint
make test
make docs
make build
```

`make ci` runs the local release checks. The project also uses the reusable
`py-canon` CI workflow and `preen` for package auditing.

## Documentation

The full documentation includes:

- [inference result contract](docs/source/inference_contract.md)
- [statistical principles](docs/source/statistical_principles.md)
- [model cards and reliability diagrams](docs/source/model_cards.md)

The Sphinx site includes this README directly so installation and API examples
have one maintained source.

## Authors and conduct

Ethnicolr was created by Suriyan Laohaprapanon and Gaurav Sood. Contributions
are welcome. Participants must follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/3/0/).

## License

Ethnicolr is released under the [MIT License](https://opensource.org/license/mit).

## Related packages

- [ethnicolr2](https://github.com/appeler/ethnicolr2)
- [naampy](https://github.com/appeler/naampy)
- [nc_race_ethnicity](https://github.com/appeler/nc_race_ethnicity)
- [parsernaam](https://github.com/appeler/parsernaam)
