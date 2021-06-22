"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand

# To use a consistent encoding
from codecs import open
from os import path, system

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        print("TODO: PostDevelopCommand")
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        print("TODO: PostInstallCommand")
        install.run(self)


class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import tox
        import shlex
        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        tox.cmdline(args=args)

setup(
    name='ethnicolr',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.6.0',

    description='Predict Race/Ethnicity Based on Sequence of Characters in the Name',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/appeler/ethnicolr',

    # Author details
    author='Suriyan Laohaprapanon, Gaurav Sood',
    author_email='suriyant@gmail.com, gsood07@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],

    # What does your project relate to?
    keywords='race ethnicity names',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['data', 'docs', 'tests', 'scripts']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'pandas',
        'h5py',
        'Keras==2.4.3',
        'numpy==1.19.5',
        'tensorflow==2.5.0'
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'ethnicolr': ['data/census/census_2000.csv',
                      'data/census/census_2010.csv',
                      'data/census/readme.md',
                      'data/census/*.pdf',
                      'data/census/*.R',
                      'data/wiki/*.*',
                      'models/*.ipynb',
                      'models/*.md',
                      'models/census/lstm/*.h5',
                      'models/census/lstm/*.csv',
                      'models/wiki/lstm/*.h5',
                      'models/wiki/lstm/*.csv',
                      'models/fl_voter_reg/lstm/*.h5',
                      'models/fl_voter_reg/lstm/*.csv',
                      'models/nc_voter_reg/lstm/*.h5',
                      'models/nc_voter_reg/lstm/*.csv',
                      'data/input*.csv',
                      'examples/*.ipynb'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'ethnicolr' will be installed into '<sys.prefix>/ethnicolr'

    #data_files=[('ethnicolr', ['ethnicolr/data/test.txt'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'census_ln=ethnicolr.census_ln:main',
            'pred_census_ln=ethnicolr.pred_census_ln:main',
            'pred_wiki_name=ethnicolr.pred_wiki_name:main',
            'pred_wiki_ln=ethnicolr.pred_wiki_ln:main',
            'pred_fl_reg_name=ethnicolr.pred_fl_reg_name:main',
            'pred_fl_reg_ln=ethnicolr.pred_fl_reg_ln:main',
            'pred_fl_reg_ln_five_cat=ethnicolr.pred_fl_reg_ln_five_cat:main',
            'pred_fl_reg_name_five_cat=ethnicolr.pred_fl_reg_name_five_cat:main',
            'pred_nc_reg_name=ethnicolr.pred_nc_reg_name:main',
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
        'test': Tox,
    },
    tests_require=['tox'],
)
