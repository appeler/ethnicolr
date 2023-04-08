#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser


class NCRegNameModel(EthnicolrModelClass):
    MODELFN = "models/nc_voter_reg/lstm/nc_voter_name_lstm_oversample.h5"
    VOCABFN = "models/nc_voter_reg/lstm/nc_voter_name_vocab_oversample.csv"
    RACEFN = "models/nc_voter_reg/lstm/nc_name_race.csv"

    NGRAMS = (2, 3)
    FEATURE_LEN = 25

    @classmethod
    def pred_nc_reg_name(cls,
                         df: pd.DataFrame,
                         lname_col: str, 
                         fname_col: str, 
                         num_iter: int=100,
                         conf_int: float=1.0) -> pd.DataFrame:
        """Predict the race+ethnicity by the full name using the
        North Carolina 12 category voter model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the first and last name
                columns.
            lname_col (str): Column name for the last name.
            fname_col (str or int): Column name for the first name.
            num_iter (int): Number of iterations do calculate the confidence interval. Default is 100.
            conf_int (float): What confidence interval? Default is 1, which means just the point estimate.            

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        if lname_col not in df.columns:
            raise Exception(f"The {lname_col} column doesn't exist in the dataframe.")
        if fname_col not in df.columns:
            raise Exception(f"The {fname_col} column doesn't exist in the dataframe.")

        df['__name'] = (df[lname_col].str.strip()
                        + ' ' + df[fname_col].str.strip()).str.title()

        rdf = cls.transform_and_pred(df=df,
                                     newnamecol='__name',
                                     vocab_fn=cls.VOCABFN,
                                     race_fn=cls.RACEFN,
                                     model_fn=cls.MODELFN,
                                     ngrams=cls.NGRAMS,
                                     maxlen=cls.FEATURE_LEN,
                                     num_iter=num_iter,
                                     conf_int=conf_int)

        del rdf['__name']
        return rdf


pred_nc_reg_name = NCRegNameModel.pred_nc_reg_name


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by name using NC 12 category voter registration model", 
                default_out = "fl-pred-name-output.csv", 
                default_year = 2017, 
                year_choices = [2017],
                first = True)

    df = pd.read_csv(args.input)
   
    rdf = pred_nc_reg_name(df = df, 
                           lname_col = args.last, 
                           fname_col = args.first, 
                           num_iter = args.iter, 
                           conf_int = args.conf)


    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
