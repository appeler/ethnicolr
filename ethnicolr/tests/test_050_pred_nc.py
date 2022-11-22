#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for NC voter registration models

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr.pred_nc_reg_name import pred_nc_reg_name

from . import capture

race = [
    "HL+A",
    "HL+B",
    "HL+I",
    "HL+M",
    "HL+O",
    "HL+W",
    "NL+A",
    "NL+B",
    "NL+I",
    "NL+M",
    "NL+O",
    "NL+W",
]

race_mean = [
    "HL+A_mean",
    "HL+B_mean",
    "HL+I_mean",
    "HL+M_mean",
    "HL+O_mean",
    "HL+W_mean",
    "NL+A_mean",
    "NL+B_mean",
    "NL+I_mean",
    "NL+M_mean",
    "NL+O_mean",
    "NL+W_mean",
]


class TestPredNC(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "hernandez", "first": "hector", "true_race": "HL+O"},
            {"last": "zhang", "first": "simon", "true_race": "NL+A"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_nc_reg_name(self):
        odf = pred_nc_reg_name(self.df, "last", "first")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_nc_reg_name(self):
        odf = pred_nc_reg_name(self.df, "last", "first", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

if __name__ == "__main__":
    unittest.main()
