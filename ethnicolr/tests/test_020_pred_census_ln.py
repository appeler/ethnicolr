#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for pred_census_ln.py

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr.pred_census_ln import pred_census_ln
from . import capture

race = ["api_mean", "black_mean", "hispanic_mean", "white_mean"]


class TestCensusLn(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "true_race": "white"},
            {"last": "zhang", "true_race": "api"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_census_ln_2000(self):
        odf = pred_census_ln(self.df, "last", 2000)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_census_ln_2010(self):
        odf = pred_census_ln(self.df, "last", 2010)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))


if __name__ == "__main__":
    unittest.main()
