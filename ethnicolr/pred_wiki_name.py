#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wikipedia Name-based Race/Ethnicity Prediction Module.

This module provides functionality to predict race/ethnicity using full names
based on models trained on Wikipedia data.
"""
import sys
import logging
from typing import List, Optional, Union
import pandas as pd
import os
from pkg_resources import resource_filename
from .ethnicolr_class import EthnicolrModelClass
from .utils import arg_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WikiNameModel(EthnicolrModelClass):
    """
    Wikipedia Full Name prediction model.
    
    This class implements race/ethnicity prediction based on full names
    using LSTM models trained on Wikipedia data.
    """
    # Model file paths
    MODELFN = "models/wiki/lstm/wiki_name_lstm.h5"
    VOCABFN = "models/wiki/lstm/wiki_name_vocab.csv"
    RACEFN = "models/wiki/lstm/wiki_name_race.csv"
    
    # Model parameters
    NGRAMS = 2
    FEATURE_LEN = 25
    
    @classmethod
    def get_model_paths(cls):
        """
        Get absolute paths to model files using resource_filename.
        
        Returns:
            tuple: (model_path, vocab_path, race_path)
        """
        model_path = resource_filename(__name__, cls.MODELFN)
        vocab_path = resource_filename(__name__, cls.VOCABFN)
        race_path = resource_filename(__name__, cls.RACEFN)
            
        return model_path, vocab_path, race_path
    
    @classmethod
    def check_models_exist(cls):
        """
        Check if model files exist and provide helpful error messages.
        
        Returns:
            bool: True if all files exist
            
        Raises:
            FileNotFoundError: If any model file is missing with instructions
        """
        model_path, vocab_path, race_path = cls.get_model_paths()
        missing_files = []
        
        if not os.path.exists(model_path):
            missing_files.append(model_path)
        if not os.path.exists(vocab_path):
            missing_files.append(vocab_path)
        if not os.path.exists(race_path):
            missing_files.append(race_path)
            
        if missing_files:
            error_msg = (
                f"Required model files not found:\n"
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
    def pred_wiki_name(cls, 
                       df: pd.DataFrame,
                       lname_col: str, 
                       fname_col: str, 
                       num_iter: int = 100,
                       conf_int: float = 1.0) -> pd.DataFrame:
        """
        Predict race/ethnicity by full name using the Wikipedia model.
        
        This method combines first and last names, then applies a deep learning model
        to predict the likely racial/ethnic category.
        
        Args:
            df (pd.DataFrame): Pandas DataFrame containing first and last name columns
            lname_col (str): Column name for the last name
            fname_col (str): Column name for the first name
            num_iter (int): Number of iterations for confidence interval calculation (default: 100)
            conf_int (float): Confidence interval level (default: 1.0 for point estimate only)
            
        Returns:
            pd.DataFrame: Original DataFrame with additional columns:
                - race: Predicted most likely race/ethnicity
                - Additional probability columns for each racial/ethnic category
                
        Raises:
            ValueError: If name columns don't exist in the DataFrame
            FileNotFoundError: If model files are not found
        """
        # Validate required columns
        if lname_col not in df.columns:
            raise ValueError(f"The last name column '{lname_col}' doesn't exist in the DataFrame")
        if fname_col not in df.columns:
            raise ValueError(f"The first name column '{fname_col}' doesn't exist in the DataFrame")
        
        # Check if model files exist
        cls.check_models_exist()
        model_path, vocab_path, race_path = cls.get_model_paths()
        
        # Create a clean copy of the DataFrame to avoid modifying the original
        working_df = df.copy()
        
        # Combine names and normalize
        logger.info(f"Processing {len(working_df)} names")
        working_df['__name'] = (
            working_df[lname_col].str.strip() + ' ' + 
            working_df[fname_col].str.strip()
        ).str.title()
        
        # Remove rows with empty combined names
        name_count_before = len(working_df)
        working_df = working_df[working_df['__name'].str.strip().str.len() > 0]
        name_count_after = len(working_df)
        
        if name_count_before > name_count_after:
            logger.warning(f"Removed {name_count_before - name_count_after} rows with empty names")
        
        if name_count_after == 0:
            logger.error("No valid names to process after cleaning")
            raise ValueError("No valid names to process. Please check input data.")
        
        # Track original columns for verification
        original_cols = set(working_df.columns)
        
        try:
            logger.info(f"Applying Wikipedia name model (confidence interval: {conf_int})")
            
            # Apply the model
            rdf = cls.transform_and_pred(
                df=working_df,
                newnamecol='__name',
                vocab_fn=vocab_path,
                race_fn=race_path,
                model_fn=model_path,
                ngrams=cls.NGRAMS,
                maxlen=cls.FEATURE_LEN,
                num_iter=num_iter,
                conf_int=conf_int
            )
            
            # Clean up temporary column
            del rdf['__name']
            
            # Calculate statistics on results
            new_cols = set(rdf.columns) - original_cols
            pred_count = rdf.dropna(subset=['race']).shape[0]
            
            logger.info(f"Successfully predicted {pred_count} of {len(working_df)} names ({pred_count/len(working_df)*100:.1f}%)")
            logger.info(f"Added columns: {', '.join(sorted(new_cols))}")
            
            return rdf
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Function alias for backward compatibility
pred_wiki_name = WikiNameModel.pred_wiki_name

def main(argv: Optional[List[str]] = None) -> int:
    """
    Command-line interface for pred_wiki_name function.
    
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
            title="Predict Race/Ethnicity by name using Wikipedia model", 
            default_out="wiki-pred-name-output.csv", 
            default_year=2017, 
            year_choices=[2017],
            first=True
        )
        
        logger.info(f"Reading input file: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Input file contains {len(df)} records")
        
        rdf = pred_wiki_name(
            df=df, 
            lname_col=args.last, 
            fname_col=args.first, 
            num_iter=args.iter, 
            conf_int=args.conf
        )
        
        logger.info(f"Saving output to file: {args.output}")
        rdf.to_csv(args.output, index=False)
        logger.info("Processing complete")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Missing model files: {e}")
        return 2
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 3
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
