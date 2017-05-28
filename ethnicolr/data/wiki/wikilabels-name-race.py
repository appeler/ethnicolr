#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from nameparser import HumanName


def parse_name(c):
    name = HumanName(c)
    return pd.Series({'first': name.first, 'last': name.last,
                      'middle': name.middle, 'suffix': name.suffix})


# Load original wiki label data
df = pd.read_csv('WikiLabels.tar.gz', sep='\t', compression='gzip', skiprows=7,
                 names=['a', 'b', 'c', 'd'], usecols=['b', 'd'])
df.dropna(subset=['b', 'd'], inplace=True)

# Parse name
df[['first', 'last', 'middle', 'suffix']] = df.b.apply(lambda c: parse_name(c))

# Write the output to CSV file
df.columns = ['full_name', 'race', 'name_first', 'name_last', 'name_middle',
              'name_suffix']
df.to_csv('wiki_name_race.csv',
          columns=['name_last', 'name_suffix', 'name_first', 'name_middle',
                   'race'], index=False, encoding='utf-8')
