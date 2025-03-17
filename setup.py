"""
A setuptools based setup module for ethnicolr.

Predicts Race/Ethnicity Based on Sequence of Characters in Names.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os
import sys
import platform
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand

# To use a consistent encoding
from codecs import open
from os import path

# Setup script directory
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

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


# Determine TensorFlow dependency based on platform
def get_tensorflow_requirement():
    """Get the appropriate TensorFlow requirement based on platform."""
    if platform.machine() == 'aarch64':
        return "tensorflow-aarch64>=2.7.2,<2.16"
    else:
        return "tensorflow>=2.7.2,<2.16"

# Core requirements for the package
INSTALL_REQUIRES = [
    get_tensorflow_requirement(),
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    "h5py>=3.1.0",
    "tqdm>=4.62.0"  # For download progress
]

# Development and testing requirements
DEV_REQUIRES = [
    "check-manifest",
    "flake8",
    "black",
    "isort",
]

TEST_REQUIRES = [
    "coverage",
    "pytest",
    "pytest-cov",
    "tox",
]

DOCS_REQUIRES = [
    "sphinx",
    "sphinx_rtd_theme",
]

# Package data to be included
PACKAGE_DATA = {
    "ethnicolr": [
        # Census data
        "data/census/census_2000.csv",
        "data/census/census_2010.csv",
        "data/census/readme.md",
        "data/census/*.pdf",
        "data/census/*.R",
        
        # Wiki data
        "data/wiki/*.*",
        
        # Model files and notebooks
        "models/*.ipynb",
        "models/*.md",
        "models/census/lstm/*.h5",
        "models/census/lstm/*.csv",
        "models/wiki/lstm/*.h5",
        "models/wiki/lstm/*.csv",
        "models/fl_voter_reg/lstm/*.h5",
        "models/fl_voter_reg/lstm/*.csv",
        "models/nc_voter_reg/lstm/*.h5",
        "models/nc_voter_reg/lstm/*.csv",
        
        # Examples and input data
        "data/input*.csv",
        "examples/*.ipynb",
    ],
}

setup(
    name="ethnicolr",
    version="0.10.0",  # Bump version for model download improvements
    description="Predict Race/Ethnicity Based on Sequence of Characters in Names",
    long_description=long_description,
    long_description_content_type="text/x-rst",  # Specify content type
    
    # Project URLs
    url="https://github.com/appeler/ethnicolr",
    project_urls={
        "Bug Reports": "https://github.com/appeler/ethnicolr/issues",
        "Documentation": "https://github.com/appeler/ethnicolr#readme",
        "Source Code": "https://github.com/appeler/ethnicolr",
    },
    
    # Author details
    author="Suriyan Laohaprapanon, Gaurav Sood",
    author_email="suriyant@gmail.com, gsood07@gmail.com",
    
    # License
    license="MIT",
    
    # Classifiers
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",
        
        # Audience and topic
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    # Keywords
    keywords="race ethnicity names demographics machine-learning nlp",
    
    # Packages
    packages=find_packages(exclude=["data", "docs", "tests", "scripts"]),
    
    # Runtime dependencies
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    
    # Extra features
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
        "docs": DOCS_REQUIRES,
        "all": DEV_REQUIRES + TEST_REQUIRES + DOCS_REQUIRES,
        "models": [],  # Just a marker for full installation with models
    },
    
    # Package data
    package_data=PACKAGE_DATA,
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "census_ln=ethnicolr.census_ln:main",
            "pred_census_ln=ethnicolr.pred_census_ln:main",
            "pred_wiki_name=ethnicolr.pred_wiki_name:main",
            "pred_wiki_ln=ethnicolr.pred_wiki_ln:main",
            "pred_fl_reg_name=ethnicolr.pred_fl_reg_name:main",
            "pred_fl_reg_ln=ethnicolr.pred_fl_reg_ln:main",
            "pred_fl_reg_ln_five_cat=ethnicolr.pred_fl_reg_ln_five_cat:main",
            "pred_fl_reg_name_five_cat=ethnicolr.pred_fl_reg_name_five_cat:main",
            "pred_nc_reg_name=ethnicolr.pred_nc_reg_name:main",
            # Add a model download utility
            "ethnicolr_download_models=ethnicolr.utils.download:main",
        ],
    },
    
    # Custom commands
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
        "test": Tox,
    },
    tests_require=TEST_REQUIRES,
)
