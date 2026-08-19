## Data from Wikipedia

The data were originally collected by a team lead by Steven Skiena as part of the project to build a classifier for race and ethnicity based on names. The team scraped Wikipedia to produce a novel database of over 140k name/race associations. For details of the how the data was collected, see [Name-ethnicity classification from open sources](http://dl.acm.org/citation.cfm?id=1557032) (for reference, see below).

The team has two papers (reference for one of the papers can be found below; the other paper is forthcoming) on novel ways of building a classifier. The team also provided public APIs for the classifiers they built, at `textmap.com/ethnicity` and `data-prism.com`; both services are now offline.

If you use this data, please cite:

@inproceedings{ambekar2009name,
  title={Name-ethnicity classification from open sources},
  author={Ambekar, Anurag and Ward, Charles and Mohammed, Jahangir and Male, Swapna and Skiena, Steven},
  booktitle={Proceedings of the 15th ACM SIGKDD international conference on Knowledge Discovery and Data Mining},
  pages={49--58},
  year={2009},
  organization={ACM}
}

## Refreshing with Wikidata (2026 pipeline)

The wiki models are now trained on this 2009-era file (kept at
`scripts/data-acquisition/source-tables/wiki/wiki_name_race.csv`) **merged with fresh
Wikidata-derived names**. The pipeline is fully scripted and needs no
credentials (public QLever SPARQL endpoint):

```bash
cd scripts/data-acquisition/wiki
python fetch_wikidata_people.py            # ~100 country files + P172 file (~1 GB)
python prepare_wiki_data.py                # -> ../raw/wiki_name_race_2026.csv.gz
cd ../../model-training
python train_name_lstm.py wikipedia_surname
python train_name_lstm.py wikipedia_full_name
```

Labeling policy: an explicit Wikidata ethnic-group statement (P172), mapped via
`scripts/data-acquisition/wiki/mappings/ethnic_group_to_category.csv`, takes
precedence; otherwise citizenship (P27) is mapped via
`mappings/country_to_category.csv`. Citizenship in melting-pot countries
(US, Canada, Australia, NZ, Brazil, ...) is treated as no signal. Such people
are only included when they carry a mapped P172 statement. People whose
signals map to conflicting categories are dropped. The mapping tables are the
auditable core of the pipeline; edit them and re-run to change policy.
