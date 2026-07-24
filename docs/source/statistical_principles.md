# Statistical principles

ethnicolr aims to be a *calibrated name-likelihood engine*: every number it
outputs should mean what it says, every guarantee should be checkable, and
every assumption should be written down. This page is the contract.

## What the probabilities mean

Every prediction returns a full probability distribution over classes (one
column per class). After post-hoc calibration, `hispanic = 0.83` means: *among
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

Empirically, the 1.x-generation models are close to calibrated out of the box
(ECE ≈ 0.01, T ≈ 1.0–1.05) — the point of this machinery is that this is
*measured*, not assumed, and will be re-verified for every future model.

## Whose probabilities? (weighting)

Several training sets deduplicate names or balance classes, so a naive
held-out evaluation would answer "for a random unique *name*" rather than
"for a random *person*". Each stats file records its `calibration_weighting`:

- census: person-weighted by construction (names sampled by population count,
  labels drawn from census race shares);
- FL 4-category: a person-level sample of registered voters;
- FL 5-category: person-level within classes, but *class-balanced* — see
  base rates below;
- NC: names were deduplicated for training, so held-out rows are re-inflated
  by their pre-deduplication frequency in the voter file (person-weighted);
- wiki: one row per unique notable person (Wikipedia/Wikidata); there is no
  natural person-frequency to inflate to, and the reference population is
  "people notable enough for Wikipedia", not any national population.

## Base rates and the `prior=` argument

Models trained on class-balanced data (FL five-category, NC) output
probabilities that answer: *which class, if all classes were equally common?*
Applied to a real population this overstates rare classes by construction.
The fix is a one-line Bayes adjustment, exposed on every prediction function:

```python
pred_fl_reg_ln_five_cat(df, "last", prior={"asian": .03, "hispanic": .27,
                                           "nh_black": .15, "nh_white": .50,
                                           "other": .05})
```

computes `p_adj(y|x) ∝ p_cal(y|x) · π_target(y) / π_train(y)`, where
`π_train` is read from the stats file. Pass the demographic margins of *your*
population (a state, a county, a census tract). With tract-level margins this
is the name-likelihood step of BISG-style methods (Elliott et al. 2009;
Imai & Khanna 2016); for full BISG pipelines see
[surgeo](https://pypi.org/project/surgeo/) (Python) or
[wru](https://github.com/kosukeimai/wru) (R) — ethnicolr's calibrated
likelihoods are designed to compose with them, covering names absent from
census dictionaries.

## Conformal prediction sets (`coverage=`)

```python
pred_wiki_name(df, "last", "first", coverage=0.90)
```

adds a `race_set` column: the smallest set of classes whose calibrated
probability mass reaches the conformal quantile fitted on held-out data
(split-conformal with adaptive prediction sets). The guarantee: *among names
exchangeable with the calibration data, the true class falls inside the set
at least 90% of the time* — verified empirically in each stats file
(`conformal_empirical_coverage`, within ±2–3pp of nominal for all shipped
models). Set sizes adapt per name: unambiguous names get singletons,
ambiguous names get honestly larger sets. Supported levels: 0.80, 0.90, 0.95.

Fine print, stated plainly:

- Coverage is **marginal** — averaged over names, not guaranteed per
  subgroup. Group-conditional coverage is reported in the model cards where
  it deviates materially.
- The guarantee assumes your names are **exchangeable with the calibration
  distribution** (census sample, voter files, Wikipedia). Under distribution
  shift, treat the level as approximate.
- `coverage=` cannot be combined with `prior=`: reweighting invalidates the
  stored quantiles. Nor with `conf_int<1`.

## Monte Carlo confidence intervals (`conf_int=`)

MC-dropout intervals (`*_lb`, `*_ub` columns) answer a different question:
*how stable is the probability estimate itself under model uncertainty?* A
probability can be precisely estimated yet wrong, or noisy yet calibrated —
so intervals and conformal sets complement rather than replace each other.
These intervals carry no frequentist coverage guarantee.

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
