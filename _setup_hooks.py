"""
Setup hooks for ethnicolr.
This module contains custom commands that run during the setup process.
"""

import os
import platform
import sys
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand

# Define model files that might need to be downloaded
MODEL_FILES = [
    "models/census/lstm/census2000_ln_lstm.h5",
    "models/census/lstm/census2000_ln_vocab.csv",
    "models/census/lstm/census2000_race.csv",
    "models/census/lstm/census2010_ln_lstm.h5",
    "models/census/lstm/census2010_ln_vocab.csv",
    "models/census/lstm/census2010_race.csv",
]

def download_models():
    """
    Download model files if they don't exist.
    """
    import urllib.request
    import os
    
    base_url = "https://github.com/appeler/ethnicolr/releases/latest/download"
    base_dir = os.path.join("ethnicolr")
    
    for model_file in MODEL_FILES:
        target_dir = os.path.join(base_dir, os.path.dirname(model_file))
        target_file = os.path.join(base_dir, model_file)
        
        # Create directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        
        # Download file if it doesn't exist
        if not os.path.exists(target_file):
            try:
                print(f"Downloading {model_file}...")
                urllib.request.urlretrieve(
                    f"{base_url}/{model_file}", 
                    target_file
                )
                print(f"Downloaded {model_file} successfully.")
            except Exception as e:
                print(f"Error downloading {model_file}: {e}")
                print(f"You can manually download it from {base_url}/{model_file}")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        # Download models after develop installation
        download_models()
        print("Development installation completed with model downloads.")


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        # Download models after regular installation
        download_models()
        print("Installation completed with model downloads.")


class Tox(TestCommand):
    """Run tests with tox."""
    
    user_options = [("tox-args=", "a", "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        import shlex

        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        tox.cmdline(args=args)
