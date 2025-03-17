"""
Modern setup hooks for ethnicolr.

This module implements hook points for setuptools using entry points,
replacing the need for custom command classes in setup.py.
"""

import os
import sys
import urllib.request
from pathlib import Path
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand
import atexit

# Define model files that need to be downloaded
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
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<3.9
        import importlib_resources as pkg_resources
    
    try:
        # Get the package installation path
        import ethnicolr
        base_dir = Path(ethnicolr.__file__).parent
        base_url = "https://github.com/appeler/ethnicolr/releases/latest/download"
        
        for model_file in MODEL_FILES:
            target_file = base_dir / model_file
            target_dir = target_file.parent
            
            # Create directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file if it doesn't exist
            if not target_file.exists():
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
    except Exception as e:
        print(f"Error during model download: {e}")


def post_develop(command, *args, **kwargs):
    """Hook that runs after the develop command."""
    print("Running post-develop hook")
    download_models()
    print("Development installation completed with model downloads.")
    return command


def post_install(command, *args, **kwargs):
    """Hook that runs after the install command."""
    print("Running post-install hook")
    download_models()
    print("Installation completed with model downloads.")
    return command


class ToxTest(TestCommand):
    """Custom test command that runs tox."""
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


def register_commands(dist):
    """
    Register custom commands with setuptools.
    
    This function is called by setuptools via the entry point
    'setuptools.finalize_distribution_options'.
    """
    # Register the tox test command
    if 'test' in dist.commands:
        dist.cmdclass['test'] = ToxTest
    
    # Register hooks for develop and install commands
    if 'develop' in dist.commands:
        original_run = dist.cmdclass.get('develop', develop).run
        
        def custom_develop_run(self):
            original_run(self)
            post_develop(self)
        
        dist.cmdclass.setdefault('develop', develop).run = custom_develop_run
    
    if 'install' in dist.commands:
        original_run = dist.cmdclass.get('install', install).run
        
        def custom_install_run(self):
            original_run(self)
            post_install(self)
        
        dist.cmdclass.setdefault('install', install).run = custom_install_run


# If we need to do something at the end of installation
def finalize():
    """Run at the end of installation."""
    # We could potentially validate the model files here
    pass


# Register the finalize function to run at exit
atexit.register(finalize)


# Add a console script for model downloads
def download_cli():
    """Command-line interface for downloading models."""
    print("Downloading model files...")
    download_models()
    print("Download complete!")
    return 0
