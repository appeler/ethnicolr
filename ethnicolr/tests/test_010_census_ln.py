#!/usr/bin/env python

"""
Tests for census_ln.py

"""

import unittest

import pandas as pd

from ethnicolr.census_ln import census_ln


class TestCensusLn(unittest.TestCase):
    def setUp(self):
        names = [
            {"last": "smith", "true_race": "white"},
            {"last": "zhang", "true_race": "api"},
        ]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_census_ln_2000(self):
        odf = census_ln(self.df, "last", 2000)
        self.assertIn("pctwhite", odf.columns)

    def test_census_ln_2010(self):
        odf = census_ln(self.df, "last", 2010)
        self.assertIn("pcthispanic", odf.columns)


if __name__ == "__main__":
    unittest.main()
