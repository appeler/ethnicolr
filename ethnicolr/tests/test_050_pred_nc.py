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


class TestPredNC(unittest.TestCase):

    def setUp(self):
        names = [{'last': 'smith', 'first': 'john', 'true_race': 'NL+M'},
                 {'last': 'zhang', 'first': 'simon', 'true_race': 'NL+A'}]
        self.df = pd.DataFrame(names)

    def tearDown(self):
        pass

    def test_pred_nc_reg_name(self):
        odf = pred_nc_reg_name(self.df, 'last', 'first')
        self.assertTrue(all(odf.sum(axis=1).round(1) == 1.0))
        self.assertTrue(all(odf.true_race == odf.race))


if __name__ == '__main__':
    unittest.main()
