# Ethnicolr development guide

Ethnicolr provides name-pattern estimates from published lookup tables and
PyTorch models. It does not infer a person's identity, ancestry, citizenship,
race, or ethnicity. Do not describe or build individual profiling or
consequential-decision uses.

## Development commands

```bash
uv sync --all-groups
make lint
make test
make docs
make build
make ci
```

`make ci` is the local release gate. It runs Ruff, Pyright, all tests with
coverage, the warnings-as-errors Sphinx build, and the distribution build.

## Public API

Public functions use two verbs:

- `lookup_*` returns values from a published name table.
- `estimate_*` combines evidence or runs a statistical model.

Required DataFrame and column arguments may be positional. Optional arguments
are keyword-only. Use `data`, `surname_column`, and `first_name_column`; do not
introduce abbreviated variants. There are no backward-compatibility aliases.

Every public result follows inference contract 1.0 in
`docs/source/inference_contract.md`. Probabilities use the 0 to 1 scale.
Unsupported inputs abstain instead of receiving a default distribution.

## Package architecture

- `ethnicolr/api.py` exposes dictionary and hybrid estimators.
- `ethnicolr/neural_name_model.py` owns shared neural inference.
- `ethnicolr/inference.py` validates options and adds result metadata.
- `ethnicolr/model_artifacts.py` resolves immutable model bundles.
- `ethnicolr/model_metadata.py` validates ordered JSON vocabularies and labels.
- `ethnicolr/runtime_tables.py` validates typed Parquet lookup tables.
- Source-specific modules define the public Census, Florida, North Carolina,
  and Wikipedia estimators.

Use descriptive snake-case names for files, functions, parameters, and local
variables. Use singular nouns for one object and plural nouns for collections.
Match established terms within a module. Do not encode type, position, or an
implementation accident in a name.

## Model artifacts

Neural weights are stored in `gojiberries/ethnicolr` on Hugging Face. The
package pins a full commit SHA and downloads only the requested weight through
the Hugging Face cache. `ETHNICOLR_MODEL_CACHE` selects a cache directory;
`ETHNICOLR_MODEL_DIR` selects a local mirror.

The wheel includes schema-validated JSON vocabularies, labels, calibration
statistics, training manifests, and typed Parquet runtime tables. It does not
include model weights, CSV metadata, or raw training data.

## Training and evaluation

`scripts/model-training/train_name_lstm.py` trains the Florida, North Carolina,
and Wikipedia models. Census has its source-specific trainer under
`scripts/model-training/census/`. `scripts/model-training/calibrate_model.py`
fits temperature scaling and conformal prediction sets.

Split source rows before balancing or augmentation. Learn vocabulary only from
training rows. Keep calibration fitting and conformal evaluation disjoint.
Report the evaluation unit and weighting with every metric.
