---
license: mit
library_name: pytorch
tags:
  - name-analysis
  - demographic-research
  - ethnicolr
---

# Ethnicolr model artifacts

This repository stores the neural model artifacts used by the `ethnicolr`
Python package. Install the package for its public API, input validation,
calibration, abstention behavior, and pinned artifact resolution:

```bash
pip install ethnicolr
```

Ethnicolr returns name-pattern estimates tied to a reference population. The
outputs are not evidence of a person's identity, ancestry, citizenship, race,
or ethnicity. Do not use them for individual profiling or decisions about
employment, education, credit, housing, policing, health care, eligibility, or
access to services.

## Included models

| Model | Input | Categories | Top-1 | Top-3 |
| --- | --- | ---: | ---: | ---: |
| Census 2000 | surname | 4 | 0.833 | 0.993 |
| Census 2010 | surname | 4 | 0.808 | 0.984 |
| Census 2020 | surname | 4 | 0.807 | 0.988 |
| Florida voter | surname | 5 | 0.588 | 0.947 |
| Florida voter | full name | 5 | 0.677 | 0.948 |
| North Carolina voter | full name | 12 | 0.425 | 0.896 |
| Wikipedia/Wikidata | surname | 13 | 0.775 | 0.907 |
| Wikipedia/Wikidata | full name | 13 | 0.863 | 0.954 |
| Wikipedia/Wikidata origin | full name | 90 | 0.626 | 0.809 |

Metrics come from held-out evaluation populations documented in each model's
statistics JSON. Florida and North Carolina use source-disjoint splits. North
Carolina results are weighted back to pre-deduplication voter frequency.
Metrics are not directly comparable across data sources.

## Artifact layout

Each model bundle contains:

- a PyTorch state dictionary (`*.pt`);
- an ordered, schema-versioned character n-gram vocabulary (`*_vocab_*.json`);
- ordered class labels (`*_labels_*.json`);
- calibration statistics and conformal quantiles (`*_stats_*.json`);
- a training manifest for newly retrained models (`*_training_*.json`).

The Python package pins this repository by a full commit SHA. It downloads only
the requested weight file through the Hugging Face cache. Set
`ETHNICOLR_MODEL_CACHE` to choose a package-specific cache directory.

## Method

The neural models encode overlapping character bigrams and use a PyTorch LSTM.
Surname sequences use a maximum length of 20; full-name sequences use 25.
Vocabulary is learned from training rows only. Florida and North Carolina split
source rows before balancing, preventing the source overlap found in older
artifacts.

Temperature scaling and adaptive conformal prediction sets use separate halves
of the held-out data. The statistics files report the calibration weighting,
temperature, ECE, multiclass Brier score, observed coverage, and mean prediction
set size.

## Sources and limitations

The models draw on U.S. Census surname tables, Florida and North Carolina voter
files, and Wikipedia/Wikidata biographies. These sources represent different
populations and encode their measurement choices and historical biases.
Wikipedia citizenship is only a proxy for name origin. Voter-file categories
and coverage do not represent the full population.

Names are ambiguous, transliteration changes information, and performance can
shift across time, geography, language, and subgroups. Prefer the full
probability distribution or a calibrated prediction set over a single label.
The package abstains on unsupported scripts and inputs without known features.

See the [Ethnicolr repository](https://github.com/appeler/ethnicolr) for source
code, acquisition scripts, model cards, evaluation details, and licenses.
