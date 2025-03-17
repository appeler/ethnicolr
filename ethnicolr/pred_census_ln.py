#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Census Last Name Race/Ethnicity Prediction Module.

This module uses machine learning models based on Census data to predict
race/ethnicity from last names.
"""
import sys
import logging
from typing import List, Optional, Union
import pandas as pd
import os
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CensusLnModel(EthnicolrModelClass):
    """
    Census Last Name prediction model.
    
    This class implements race/ethnicity prediction based on last names
    using LSTM models trained on Census data.
    """
    # Model file paths
    MODELFN = "models/census/lstm/census{0:d}_ln_lstm.h5"
    VOCABFN = "models/census/lstm/census{0:d}_ln_vocab.csv"
    RACEFN = "models/census/lstm/census{0:d}_race.csv"
    
    # Model parameters
    NGRAMS = 2
    FEATURE_LEN = 20
    
    @classmethod
    def pred_census_ln(cls, 
                       df: pd.DataFrame,
                       lname_col: str,
                       year: int = 2010,
                       num_iter: int = 100,
                       conf_int: float = 1.0) -> pd.DataFrame:
        """
        Predict race/ethnicity based on last names using Census-trained models.
        
        This method applies a deep learning model trained on Census surnames
        data to predict the likely racial/ethnic composition for each last name.
        
        Args:
            df (pd.DataFrame): Pandas DataFrame containing the last name column
            lname_col (str): Column name for the last name
            year (int): Census model year to use (2000 or 2010, default: 2010)
            num_iter (int): Number of iterations for confidence interval calculation (default: 100)
            conf_int (float): Confidence interval level (default: 1.0 for point estimate only)
            
        Returns:
            pd.DataFrame: Original DataFrame with additional columns:
                - race: Predicted most likely race/ethnicity
                - black: Probability of being Black/African American
                - api: Probability of being Asian/Pacific Islander
                - white: Probability of being White
                - hispanic: Probability of being Hispanic/Latino
                
        Raises:
            ValueError: If year is not 2000 or 2010
            FileNotFoundError: If model files are not found
        """
        if year not in [2000, 2010]:
            raise ValueError("Census year must be either 2000 or 2010")
        
        # Check if model files exist
        model_path = cls.MODELFN.format(year)
        vocab_path = cls.VOCABFN.format(year)
        race_path = cls.RACEFN.format(year)
        
        # These paths are relative to the package, convert to absolute paths
        for path in [model_path, vocab_path, race_path]:
            if not os.path.exists(path):
                logger.error(f"Required model file not found: {path}")
                raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Predicting race/ethnicity for {len(df)} records using Census {year} model")
        logger.info(f"Using confidence interval: {conf_int}, iterations: {num_iter}")
        
        # Track original column count for verification
        original_cols = set(df.columns)
        
        try:
            # Apply the model
            rdf = cls.transform_and_pred(
                df=df,
                newnamecol=lname_col,
                vocab_fn=vocab_path,
                race_fn=race_path,
                model_fn=model_path,
                ngrams=cls.NGRAMS,
                maxlen=cls.FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int
            )
            
            # Calculate statistics on results
            new_cols = set(rdf.columns) - original_cols
            pred_count = rdf.dropna(subset=['race']).shape[0]
            
            logger.info(f"Successfully predicted {pred_count} of {len(df)} records ({pred_count/len(df)*100:.1f}%)")
            logger.info(f"Added columns: {', '.join(new_cols)}")
            
            return rdf
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Function alias for backward compatibility
pred_census_ln = CensusLnModel.pred_census_ln

def main(argv: Optional[List[str]] = None) -> int:
    """
    Command-line interface for pred_census_ln function.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    if argv is None:
        argv = sys.argv[1:]
    
    try:
        args = arg_parser(
            argv, 
            title="Predict Race/Ethnicity by last name using Census last name model", 
            default_out="census-pred-ln-output.csv", 
            default_year=2010, 
            year_choices=[2000, 2010]
        )
        
        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Input file contains {len(df)} records")
        
        rdf = pred_census_ln(
            df=df, 
            lname_col=args.last, 
            year=args.year, 
            num_iter=args.iter, 
            conf_int=args.conf
        )
        
        logger.info(f"Saving output to file: {args.output}")
        rdf.to_csv(args.output, index=False)
        logger.info("Processing complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
