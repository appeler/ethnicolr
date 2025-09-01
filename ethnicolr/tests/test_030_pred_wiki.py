#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Wiki models

"""

import os
import shutil
import unittest
import pandas as pd
from ethnicolr.pred_wiki_ln import pred_wiki_ln
from ethnicolr.pred_wiki_name import pred_wiki_name
from . import capture

race = [
    "Asian,GreaterEastAsian,EastAsian",
    "Asian,GreaterEastAsian,Japanese",
    "Asian,IndianSubContinent",
    "GreaterAfrican,Africans",
    "GreaterAfrican,Muslim",
    "GreaterEuropean,British",
    "GreaterEuropean,EastEuropean",
    "GreaterEuropean,Jewish",
    "GreaterEuropean,WestEuropean,French",
    "GreaterEuropean,WestEuropean,Germanic",
    "GreaterEuropean,WestEuropean,Hispanic",
    "GreaterEuropean,WestEuropean,Italian",
    "GreaterEuropean,WestEuropean,Nordic",
]

race_mean = [
    "Asian,GreaterEastAsian,EastAsian_mean",
    "Asian,GreaterEastAsian,Japanese_mean",
    "Asian,IndianSubContinent_mean",
    "GreaterAfrican,Africans_mean",
    "GreaterAfrican,Muslim_mean",
    "GreaterEuropean,British_mean",
    "GreaterEuropean,EastEuropean_mean",
    "GreaterEuropean,Jewish_mean",
    "GreaterEuropean,WestEuropean,French_mean",
    "GreaterEuropean,WestEuropean,Germanic_mean",
    "GreaterEuropean,WestEuropean,Hispanic_mean",
    "GreaterEuropean,WestEuropean,Italian_mean",
    "GreaterEuropean,WestEuropean,Nordic_mean",
]


class TestPredWiki(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "first": "john",
             "true_race": "GreaterEuropean,British"},
            {
                "last": "zhang",
                "first": "simon",
                "true_race": "Asian,GreaterEastAsian,EastAsian",
            },
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_wiki_ln(self):
        odf = pred_wiki_ln(self.df, "last")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_wiki_name(self):
        odf = pred_wiki_name(self.df, "last", "first")
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_wiki_ln_mean(self):
        odf = pred_wiki_ln(self.df, "last", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_wiki_name_mean(self):
        odf = pred_wiki_name(self.df, "last", "first", conf_int=0.9)
        self.assertTrue(
            all(
                odf[[col for col in odf.columns
                     if col in race_mean]].sum(axis=1).round(1)
                == 1.0
            )
        )
        self.assertTrue(all(odf.true_race == odf.race))

    def test_pred_wiki_name_preserves_duplicates(self):
        dupe_df = pd.DataFrame(
            [
                {"last": "O'Neil", "first": "John"},
                {"last": "ONeil", "first": "John"},
            ]
        )
        out = pred_wiki_name(dupe_df, "last", "first")
        self.assertEqual(len(out), len(dupe_df))
        self.assertListEqual(out["last"].tolist(), dupe_df["last"].tolist())
        self.assertListEqual(out["first"].tolist(), dupe_df["first"].tolist())
        self.assertTrue(out["processing_status"].eq("processed").all())
        self.assertEqual(out["race"].nunique(), 1)

    def test_pred_wiki_name_handles_accents_and_initials(self):
        tricky_df = pd.DataFrame(
            [
                {"last": "Szathmáry", "first": "Emöke"},
                {"last": "McMillan", "first": "A."},
                {"last": "FitzGerald", "first": "John"},
                {"last": "Edwards", "first": "N"},
                {"last": "Sauder", "first": "E"},
                {"last": "Aguilar", "first": "J."},
            ]
        )
        out = pred_wiki_name(tricky_df, "last", "first")
        self.assertEqual(len(out), len(tricky_df))
        self.assertTrue(out["processing_status"].eq("processed").all())

if __name__ == "__main__":
    unittest.main()
