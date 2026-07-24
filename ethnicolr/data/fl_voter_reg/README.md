# Florida Voter Registration Data

Training data comes from the restricted [Florida Voter Registration Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBIG3F)
dataset on Harvard Dataverse (2017: `20170207_VoterDetail.7z`, 2022:
`20220621_VoterDetail_2.7z`). Access requires a Dataverse account with
permission; set `DATAVERSE_API_TOKEN` and use
`scripts/data-acquisition/download_dataverse.py` to download, then
`scripts/data-acquisition/fl_voter_reg/prepare_fl_data.py` to build the
training CSVs consumed by `scripts/model-training/train_name_lstm.py`.
