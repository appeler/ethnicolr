#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser


class FloridaRegLnFiveCatModel(EthnicolrModelClass):
    MODELFN = "models/fl_voter_reg/lstm/fl_all_ln_lstm_5_cat{0:s}.h5"
    VOCABFN = "models/fl_voter_reg/lstm/fl_all_ln_vocab_5_cat{0:s}.csv"
    RACEFN = "models/fl_voter_reg/lstm/fl_ln_five_cat_race{0:s}.csv"

    NGRAMS = 2
    FEATURE_LEN = 20

    @classmethod
    def pred_fl_reg_ln(cls, 
                       df: pd.DataFrame, 
                       lname_col: str, 
                       num_iter: int=100, 
                       conf_int: float=1.0, 
                       year: int=2022) -> pd.DataFrame:

        """Predict the race/ethnicity of the last name using the Florida voter
        registration data model.

        Args:
            df (:obj:`DataFrame`): Pandas DataFrame containing the last name
                column.
            lname_col (str): Column name for the last name.
            num_iter (int): Number of iterations do calculate the confidence interval. Default is 100.
            conf_int (float): What confidence interval? Default is 1, which means just the point estimate.
            year (int): the year of the model. Default = 2022. 

        Returns:
            DataFrame: Pandas DataFrame with additional columns:
                - `race` the predict result
                - Additional columns for probability of each classes.

        """

        year = '_2022' if year == 2022 else ''

        rdf = cls.transform_and_pred(df=df,
                                     newnamecol=lname_col,
                                     vocab_fn=cls.VOCABFN.format(year),
                                     race_fn=cls.RACEFN.format(year),
                                     model_fn=cls.MODELFN.format(year),
                                     ngrams=cls.NGRAMS,
                                     maxlen=cls.FEATURE_LEN,
                                     num_iter=num_iter,
                                     conf_int=conf_int)

        return rdf


pred_fl_reg_ln_five_cat = FloridaRegLnFiveCatModel.pred_fl_reg_ln


def main(argv=sys.argv[1:]) -> None:
    args = arg_parser(argv, 
                title = "Predict Race/Ethnicity by last name using the Florida registration 5 cat. model", 
                default_out = "fl-pred-ln-five-cat-output.csv", 
                default_year = 2022, 
                year_choices = [2017, 2022])

    df = pd.read_csv(args.input)

    rdf = pred_fl_reg_ln_five_cat(df, args.last, args.iter, args.conf,
                                  args.year)

    print(f"Saving output to file: `{args.output}`")
    rdf.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
