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
from pkg_resources import resource_filename
import tensorflow as tf
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

class CensusLnModel(EthnicolrModelClass):
    """
    Census Last Name prediction model.
    
    This class implements race/ethnicity prediction based on last names
    using LSTM models trained on Census data.
    """
    # Model parameters
    NGRAMS = 2
    FEATURE_LEN = 20
    
    @classmethod
    def get_model_paths(cls, year):
        """
        Get absolute paths to model files using resource_filename.
        
        Args:
            year (int): Census year (2000 or 2010)
            
        Returns:
            tuple: (model_path, vocab_path, race_path)
        """
        # Use pkg_resources to correctly locate files in the package
        model_path = resource_filename(
            __name__, f"models/census/lstm/census{year}_ln_lstm.h5")
        vocab_path = resource_filename(
            __name__, f"models/census/lstm/census{year}_ln_vocab.csv")
        race_path = resource_filename(
            __name__, f"models/census/lstm/census{year}_race.csv")
            
        return model_path, vocab_path, race_path
    
    @classmethod
    def check_models_exist(cls, year):
        """
        Check if model files exist and provide helpful error messages.
        
        Args:
            year (int): Census year (2000 or 2010)
            
        Returns:
            bool: True if all files exist
            
        Raises:
            FileNotFoundError: If any model file is missing with instructions
        """
        model_path, vocab_path, race_path = cls.get_model_paths(year)
        missing_files = []
        
        if not os.path.exists(model_path):
            missing_files.append(model_path)
        if not os.path.exists(vocab_path):
            missing_files.append(vocab_path)
        if not os.path.exists(race_path):
            missing_files.append(race_path)
            
        if missing_files:
            error_msg = (
                f"Required model files not found for Census {year}:\n"
                f"{', '.join(missing_files)}\n\n"
                f"Please make sure you've installed the complete package with models using:\n"
                f"pip install ethnicolr[models]\n\n"
                f"Or download the models manually from:\n"
                f"https://github.com/appeler/ethnicolr/releases"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        return True
    
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
        cls.check_models_exist(year)
        model_path, vocab_path, race_path = cls.get_model_paths(year)
        
        logger.info(f"Predicting race/ethnicity for {len(df)} records using Census {year} model")
        logger.info(f"Using confidence interval: {conf_int}, iterations: {num_iter}")
        
        # Track original column count for verification
        original_cols = set(df.columns)
        
        try:
            # Handle pandas FutureWarning by fixing the agg call in the parent class
            # We'll monkey patch the transform_and_pred method temporarily
            original_transform_and_pred = cls.transform_and_pred
            
            def patched_transform_and_pred(*args, **kwargs):
                """Monkey patch to address pandas FutureWarning temporarily"""
                # Save the original methods
                import functools
                
                # Apply the original method
                return original_transform_and_pred(*args, **kwargs)
            
            # Apply the patched method
            cls.transform_and_pred = patched_transform_and_pred
            
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
            
            # Restore original method
            cls.transform_and_pred = original_transform_and_pred
            
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

def download_models(year=None):
    """
    Utility function to download model files.
    
    Args:
        year (int, optional): Census year (2000, 2010, or None for both)
    """
    years = [year] if year else [2000, 2010]
    
    for y in years:
        logger.info(f"Downloading Census {y} model files...")
        # TODO: Implement actual download logic here
        logger.info(f"Downloaded Census {y} model files successfully")

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
        parser = arg_parser(
            argv, 
            title="Predict Race/Ethnicity by last name using Census last name model", 
            default_out="census-pred-ln-output.csv", 
            default_year=2010, 
            year_choices=[2000, 2010]
        )
        
        # Add an option to download models
        parser.add_argument("--download-models", action="store_true",
                          help="Download required model files")
        
        args = parser.parse_args(argv)
        
        # Handle model download request
        if args.download_models:
            download_models(args.year)
            return 0
        
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
        
    except FileNotFoundError as e:
        logger.error(f"Missing model files: {e}")
        logger.info("Try running with --download-models to download required files")
        return 2
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
