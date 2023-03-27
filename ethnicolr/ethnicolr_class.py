#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from pkg_resources import resource_filename

from .utils import column_exists, fixup_columns, transform_and_pred, arg_parser

MODELFN = "models/census/lstm/census{0:d}_ln_lstm.h5"
VOCABFN = "models/census/lstm/census{0:d}_ln_vocab.csv"
RACEFN = "models/census/lstm/census{0:d}_race.csv"

MODEL = resource_filename(__name__, MODELFN)
VOCAB = resource_filename(__name__, VOCABFN)
RACE = resource_filename(__name__, RACEFN)

NGRAMS = 2
FEATURE_LEN = 20

class EthnicolrModelClass:
    vocab = None
    race = None
    model = None
    model_year = None

# Constructor
  def __init__(self, vocab, race, model, model_year):
    self.vocab = vocab
    self.race = race
    self.model = model
    self.model_year = model_year
 
 