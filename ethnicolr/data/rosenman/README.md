# Rosenman-Olivella-Imai name-race dictionaries

P(race | name) dictionaries built from six Southern-state voter files
(AL, FL, GA, LA, NC, SC), from:

> Rosenman, E., Olivella, S. & Imai, K. Race and ethnicity data for first,
> middle, and surnames. *Scientific Data* 10, 299 (2023).
> https://doi.org/10.7910/DVN/YL2OXB (CC0 1.0)

Files here are converted verbatim from the Dataverse release by
`scripts/data-acquisition/census/fetch_rosenman_dictionaries.py`:

- `first_name_race.parquet`: 135,777 first names × 5 race probabilities
- `last_name_race.parquet`: 338,169 surnames × 5 race probabilities
- `rosenman_stats.json`: provenance and the implied voter-population race
  marginal (recovered via Bayes' rule from the paired probability matrices),
  used as π_train by the `target_prior=` adjustment.

Used by `ethnicolr.estimate_voter_file_full_name`.
