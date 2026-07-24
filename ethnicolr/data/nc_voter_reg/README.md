# NC Voter Registration Data

Training data comes from the restricted [NC Voter Registration Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NEFUBN)
dataset on Harvard Dataverse (`ncvoter_Statewide.zip`). Access requires a
Dataverse account with permission; set `DATAVERSE_API_TOKEN` and use
`scripts/data-acquisition/download_dataverse.py` to download, then
`scripts/data-acquisition/nc_voter_reg/prepare_nc_data.py` to build the
training CSV consumed by `scripts/model-training/train_name_lstm.py`.
