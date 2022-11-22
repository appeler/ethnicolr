#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for FL voter registration models

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr.pred_fl_reg_ln import pred_fl_reg_ln
from ethnicolr.pred_fl_reg_name import pred_fl_reg_name
from ethnicolr.pred_fl_reg_ln_five_cat import pred_fl_reg_ln_five_cat
from ethnicolr.pred_fl_reg_name_five_cat import pred_fl_reg_name_five_cat
from pkg_resources import resource_filename

from . import capture

race = ["asian", "hispanic", "nh_black", "nh_white"]
race5 = ["asian", "hispanic", "nh_black", "nh_white", "other"]

race_mean = ["asian_mean", "hispanic_mean", "nh_black_mean", "nh_white_mean"]
race5_mean = ["asian_mean", "hispanic_mean", "nh_black_mean", "nh_white_mean",
              "other_mean"]


class TestPredFL(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "zhang", "first": "simon", "true_race": "asian"},
            {"last": "torres", "first": "raul", "true_race": "hispanic"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_fl_reg_ln(self):
        odf = pred_fl_reg_ln(self.df, "last")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_name(self):
        odf = pred_fl_reg_name(self.df, "last", "first")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_ln_five_cat(self):
        odf = pred_fl_reg_ln_five_cat(self.df, "last")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race5]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_name_five_cat(self):
        odf = pred_fl_reg_name_five_cat(self.df, "last", "first")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race5]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_ln_mean(self):
        odf = pred_fl_reg_ln(self.df, "last", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_name_mean(self):
        odf = pred_fl_reg_name(self.df, "last", "first", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_ln_five_cat_mean(self):
        odf = pred_fl_reg_ln_five_cat(self.df, "last", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race5_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_fl_reg_name_five_cat_mean(self):
        odf = pred_fl_reg_name_five_cat(self.df, "last", "first", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race5_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

if __name__ == "__main__":
    unittest.main()
