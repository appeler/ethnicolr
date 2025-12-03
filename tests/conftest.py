"""
Shared test fixtures and configuration for ethnicolr tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_census_names():
    """Basic test data for census-based predictions."""
    return pd.DataFrame([
        {"last": "smith", "true_race": "white"},
        {"last": "zhang", "true_race": "api"},
        {"last": "garcia", "true_race": "hispanic"},
        {"last": "johnson", "true_race": "black"},
        {"last": "patel", "true_race": "api"},
    ])


@pytest.fixture
def sample_wiki_names():
    """Test data for Wikipedia-based predictions."""
    return pd.DataFrame([
        {"last": "smith", "first": "john", "true_race": "GreaterEuropean,British"},
        {"last": "zhang", "first": "simon", "true_race": "Asian,GreaterEastAsian,EastAsian"},
        {"last": "garcia", "first": "maria", "true_race": "GreaterEuropean,WestEuropean,Hispanic"},
        {"last": "nakamura", "first": "akira", "true_race": "Asian,GreaterEastAsian,Japanese"},
        {"last": "johnson", "first": "michael", "true_race": "GreaterAfrican,Africans"},
    ])


@pytest.fixture
def sample_florida_names():
    """Test data for Florida voter registration predictions."""
    return pd.DataFrame([
        {"last": "zhang", "first": "simon", "true_race": "asian"},
        {"last": "torres", "first": "raul", "true_race": "hispanic"},
        {"last": "williams", "first": "james", "true_race": "nh_black"},
        {"last": "smith", "first": "sarah", "true_race": "nh_white"},
        {"last": "kumar", "first": "raj", "true_race": "other"},
    ])


@pytest.fixture
def sample_nc_names():
    """Test data for North Carolina voter registration predictions."""
    return pd.DataFrame([
        {"last": "hernandez", "first": "hector", "true_race": "HL+O"},
        {"last": "zhang", "first": "simon", "true_race": "NL+A"},
        {"last": "smith", "first": "john", "true_race": "NL+W"},
        {"last": "williams", "first": "james", "true_race": "NL+B"},
        {"last": "garcia", "first": "maria", "true_race": "HL+W"},
    ])


@pytest.fixture
def extensive_names():
    """Larger, more realistic test dataset."""
    names_data = [
        # European names
        {"last": "smith", "first": "john", "expected_major": "white"},
        {"last": "johnson", "first": "emily", "expected_major": "white"},
        {"last": "williams", "first": "david", "expected_major": "white"},
        {"last": "brown", "first": "sarah", "expected_major": "white"},
        {"last": "davis", "first": "michael", "expected_major": "white"},
        {"last": "miller", "first": "jennifer", "expected_major": "white"},
        {"last": "wilson", "first": "christopher", "expected_major": "white"},
        {"last": "moore", "first": "lisa", "expected_major": "white"},
        {"last": "taylor", "first": "daniel", "expected_major": "white"},
        {"last": "anderson", "first": "mary", "expected_major": "white"},

        # East Asian names
        {"last": "zhang", "first": "wei", "expected_major": "asian"},
        {"last": "wang", "first": "li", "expected_major": "asian"},
        {"last": "li", "first": "ming", "expected_major": "asian"},
        {"last": "chen", "first": "xiao", "expected_major": "asian"},
        {"last": "liu", "first": "ling", "expected_major": "asian"},
        {"last": "kim", "first": "sung", "expected_major": "asian"},
        {"last": "park", "first": "ji", "expected_major": "asian"},
        {"last": "nakamura", "first": "akira", "expected_major": "asian"},
        {"last": "tanaka", "first": "yuki", "expected_major": "asian"},
        {"last": "sato", "first": "takeshi", "expected_major": "asian"},

        # Hispanic names
        {"last": "garcia", "first": "jose", "expected_major": "hispanic"},
        {"last": "rodriguez", "first": "maria", "expected_major": "hispanic"},
        {"last": "martinez", "first": "carlos", "expected_major": "hispanic"},
        {"last": "hernandez", "first": "ana", "expected_major": "hispanic"},
        {"last": "lopez", "first": "luis", "expected_major": "hispanic"},
        {"last": "gonzalez", "first": "sofia", "expected_major": "hispanic"},
        {"last": "perez", "first": "miguel", "expected_major": "hispanic"},
        {"last": "sanchez", "first": "carmen", "expected_major": "hispanic"},
        {"last": "ramirez", "first": "diego", "expected_major": "hispanic"},
        {"last": "torres", "first": "elena", "expected_major": "hispanic"},

        # South Asian names
        {"last": "patel", "first": "raj", "expected_major": "asian"},
        {"last": "sharma", "first": "priya", "expected_major": "asian"},
        {"last": "gupta", "first": "amit", "expected_major": "asian"},
        {"last": "singh", "first": "simran", "expected_major": "asian"},
        {"last": "kumar", "first": "vikram", "expected_major": "asian"},
        {"last": "agarwal", "first": "neha", "expected_major": "asian"},
        {"last": "jain", "first": "rohan", "expected_major": "asian"},
        {"last": "shah", "first": "kavita", "expected_major": "asian"},

        # African names
        {"last": "jackson", "first": "jamal", "expected_major": "black"},
        {"last": "robinson", "first": "keisha", "expected_major": "black"},
        {"last": "walker", "first": "tyrone", "expected_major": "black"},
        {"last": "wright", "first": "latoya", "expected_major": "black"},
        {"last": "hall", "first": "deshawn", "expected_major": "black"},
        {"last": "allen", "first": "shaniqua", "expected_major": "black"},
        {"last": "young", "first": "malik", "expected_major": "black"},
        {"last": "king", "first": "ebony", "expected_major": "black"},
    ]

    return pd.DataFrame(names_data)


@pytest.fixture
def edge_case_names():
    """Names with special characters, empty fields, and edge cases."""
    return pd.DataFrame([
        {"last": "O'Brien", "first": "Patrick", "note": "apostrophe"},
        {"last": "Van Der Berg", "first": "Anna", "note": "spaces_in_last"},
        {"last": "Müller", "first": "Björn", "note": "unicode_characters"},
        {"last": "D'Angelo", "first": "Maria", "note": "apostrophe_italian"},
        {"last": "José-Luis", "first": "Santiago", "note": "hyphen_in_first"},
        {"last": "Smith", "first": "A.", "note": "initial_only"},
        {"last": "Williams", "first": "J", "note": "single_letter"},
        {"last": "UPPERCASE", "first": "lowercase", "note": "case_variations"},
        {"last": "MacLeod", "first": "Ian", "note": "scottish_prefix"},
        {"last": "de Silva", "first": "Carlos", "note": "portuguese_prefix"},
    ])


@pytest.fixture
def problematic_data():
    """DataFrames with various data quality issues."""
    return {
        "empty_df": pd.DataFrame(),
        "missing_columns": pd.DataFrame([{"name": "smith"}]),
        "null_values": pd.DataFrame([
            {"last": None, "first": "john"},
            {"last": "smith", "first": None},
        ]),
        "empty_strings": pd.DataFrame([
            {"last": "", "first": "john"},
            {"last": "smith", "first": ""},
        ]),
        "whitespace_only": pd.DataFrame([
            {"last": "   ", "first": "john"},
            {"last": "smith", "first": "   "},
        ]),
    }
