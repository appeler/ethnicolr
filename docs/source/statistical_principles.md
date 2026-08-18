# Statistical principles

Ethnicolr reports name-pattern probabilities with their reference population,
calibration status, and assumptions. This page defines those statistical
claims and their limits.

## What the probabilities mean

Every scored prediction returns a full probability distribution over classes
(one column per class). After validated post-hoc calibration,
`hispanic = 0.83` means: *among
held-out evaluation names to which the model assigns ≈0.83, the true class is
hispanic ≈83% of the time*, under the evaluation weighting documented in each
model's stats file (see "Whose probabilities?" below).

Two caveats apply to any name-based model:

1. **Names are genuinely ambiguous.** Even a perfect model cannot assign
   "Smith" to one class with certainty; the honest output is a spread-out
   distribution. Low confidence is information, not failure.
2. **The probability is about a population, not a person.** p(race | name) is
   a statement about people who share a name, not a determination about an
   individual.

## Calibration (temperature scaling)

Each model ships with a stats file (`*_stats_pt.json`) produced by
`scripts/model-training/calibrate_model.py`:

- The model's deterministic held-out split is divided into a calibration half
  and an evaluation half.
- A single temperature `T` is fitted on the calibration half by minimizing
  negative log-likelihood (Guo et al. 2017); shipped probabilities are
  `softmax(logits / T)`. Temperature scaling never changes the ranking of
  classes, so accuracy is untouched.
- Expected calibration error (ECE, 15 bins) and multiclass Brier score are
  reported before and after on the evaluation half, together with full
  reliability-diagram bins. Reliability plots are rendered in the model cards.

Calibration quality varies by source. Each artifact records both pre-scaling
and post-scaling results so the package does not assume temperature scaling
helped. Conformal coverage is checked separately on the evaluation half.

## Whose probabilities? (weighting)

Several training sets deduplicate names or balance classes, so a naive
held-out evaluation would answer "for a random unique *name*" rather than
"for a random *person*". Each stats file records its `calibration_weighting`:

- census: person-weighted by construction (names sampled by population count,
  labels drawn from census race shares);
- Florida: a source-disjoint, person-level sample of registered voters;
- NC: names were deduplicated for training, so held-out rows are re-inflated
  by their pre-deduplication frequency in the voter file (person-weighted);
- wiki: one row per unique notable person (Wikipedia/Wikidata); there is no
  natural person-frequency to inflate to, and the reference population is
  "people notable enough for Wikipedia", not any national population.

## Base rates and the `target_prior=` argument

Models trained on class-balanced data (Florida and North Carolina) output
probabilities that answer: *which class, if all classes were equally common?*
Applied to a real population this overstates rare classes by construction.
The fix is a Bayes adjustment, exposed on each estimator that returns model
probabilities:

```python
estimate_florida_voter_surname(
    df,
    "last",
    target_prior={
        "asian": 0.03,
        "hispanic": 0.27,
        "nh_black": 0.15,
        "nh_white": 0.50,
        "other": 0.05,
    },
)
```

computes `p_adj(y|x) ∝ p_cal(y|x) · π_target(y) / π_train(y)`, where
`π_train` is read from the stats file. Pass the demographic margins of *your*
population (a state, a county, a census tract). With tract-level margins this
is the name-likelihood step of BISG-style methods (Elliott et al. 2009;
Imai & Khanna 2016); for full BISG pipelines see
[surgeo](https://pypi.org/project/surgeo/) (Python) or
[wru](https://github.com/kosukeimai/wru) (R). Ethnicolr's calibrated
likelihoods are designed to compose with them, covering names absent from
census dictionaries.

## Conformal prediction sets (`conformal_coverage=`)

```python
estimate_wikipedia_full_name(df, "last", "first", conformal_coverage=0.90)
```

For a model with a valid calibration artifact, this adds a `race_set` column:
the smallest set of classes whose calibrated
probability mass reaches the conformal quantile fitted on held-out data
(split-conformal with adaptive prediction sets). The guarantee: *among names
exchangeable with the calibration data, the true class falls inside the set
at least 90% of the time*. Each stats file reports an empirical check
(`conformal_empirical_coverage`). Valid shipped models are within 2 to 3
percentage points of nominal. Set sizes adapt per name: unambiguous names get singletons,
ambiguous names get honestly larger sets. Supported levels: 0.80, 0.90, 0.95.

Fine print, stated plainly:

- Coverage is **marginal**. It is averaged over names and not guaranteed per
  subgroup. Group-conditional coverage is reported in the model cards where
  it deviates materially.
- The guarantee assumes your names are **exchangeable with the calibration
  distribution** (census sample, voter files, Wikipedia). Under distribution
  shift, treat the level as approximate.
- `conformal_coverage=` cannot be combined with `target_prior=` because
  reweighting invalidates the stored quantiles. It also cannot be combined
  with `uncertainty_level`.

## Monte Carlo dropout variation (`uncertainty_level=`)

Monte Carlo dropout summaries are not confidence intervals. The `*_mc_mean`,
`*_mc_std`, `*_mc_lower`, and `*_mc_upper` columns answer a different question:
*how stable is the probability estimate itself under model uncertainty?* A
probability can be precisely estimated yet wrong, or noisy yet calibrated,
so intervals and conformal sets complement rather than replace each other.
The percentile range carries no frequentist coverage guarantee.

## Responsible-use boundary and abstention

All outputs are name-pattern estimates tied to a reference population, not
determinations about a person. They must not be used for individual or
consequential decisions. Inference APIs report `script_supported`, `abstained`,
and `abstention_reason`; blank inputs, unsupported scripts, dictionary misses
without a fallback, and inputs with no known model features receive no class
probabilities. Every result also reports the package version, model identifier,
reference population, calibration status/reference, uncertainty method/level,
and a SHA-256 revision of the complete runtime artifact bundle.

## Dictionary estimators and the independence assumption

`lookup_census_first_name`, `estimate_census_full_name`, and
`estimate_voter_file_full_name` are *dictionary* estimators: exact
conditional frequencies from public tables (Census 2020 first-name and
surname files; six-state voter-file name tables), no neural network involved
for in-dictionary names.

Combining first and last names uses naive Bayes:

```
p(race | first, last) ∝ p(race | last) · p(race | first) / π(race)
```

which assumes first and last names are conditionally independent given race.
This is an approximation. Culturally correlated first/last pairs (e.g. a
distinctively Irish first name with an Irish surname) make the combined
posterior somewhat overconfident, because the two names partially repeat the
same evidence. The `evidence_basis` column records exactly what evidence each row's
estimate used, including surname-model fallback for out-of-dictionary surnames.

Reference populations differ and are part of each estimator's meaning:
census tables describe the 2020 US enumerated population; the voter
dictionaries describe registered voters in AL/FL/GA/LA/NC/SC (their implied
race marginal ships in `rosenman_stats.json` and anchors the `target_prior=`
adjustment).

## Exact intervals for census lookups

`lookup_census_surname(..., uncertainty_level=0.95)` and `lookup_census_first_name(..., uncertainty_level=0.95)` add
Wilson score bounds computed from the published name counts. These capture
sampling uncertainty in the published proportions. The uncertainty is near zero for common
names, honest for rare ones. The 2020 counts additionally carry the Census
Bureau's disclosure-avoidance noise (±3 per cell at 95% probability), which
the intervals do not model; for counts above a few hundred it is negligible.

## The origin model's reference population

`estimate_wikipedia_origin` predicts the likely *country of origin of a name* over
90 country classes, trained on Wikidata people using citizenship as the
label. Two caveats are part of its meaning: the reference population is
people notable enough for Wikipedia/Wikidata, and citizenship is a proxy for
name origin. For migrants the two differ. Melting-pot citizenships and
ambiguous historical unions are excluded from training for this reason; see
the auditable mapping in
`scripts/data-acquisition/wiki/mappings/country_to_origin.csv`. With 90
classes, single-country answers are often uncertain. Prefer the
full distribution, `conformal_coverage=` sets, or aggregate columns into regions, e.g.
`df[["Sweden", "Norway", "Denmark", "Finland", "Iceland"]].sum(axis=1)`.

## References

- Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural
  Networks.* ICML.
- Elliott et al. (2009). *Using the Census Bureau's surname list to improve
  estimates of race/ethnicity.* Health Services and Outcomes Research
  Methodology (BISG).
- Imai & Khanna (2016). *Improving Ecological Inference by Predicting
  Individual Ethnicity from Voter Registration Records.* Political Analysis.
- Imai, Olivella & Rosenman (2022). *Addressing census data problems in race
  imputation via fully Bayesian Improved Surname Geocoding.* Science Advances.
- Rosenman, Olivella & Imai (2023). *Race and ethnicity data for first,
  middle, and surnames.* Scientific Data.
- Angelopoulos & Bates (2023). *Conformal Prediction: A Gentle Introduction.*
  Foundations and Trends in Machine Learning.
- Gelman, Hill & Vehtari (2020). *Regression and Other Stories.* (On
  calibrated probabilistic prediction and base rates.)
