# Adapted from
# https://github.com/kennethreitz/setup.py

from setuptools import find_packages, setup
from Cython.Build import cythonize

# Package meta-data.
NAME = 'bpe'
DESCRIPTION = 'sklearn and keras compatible byte pair encoding'
VERSION = '0.0.0'
URL = 'https://github.com/dantegates/byte-pair-encoding'
REQUIRES_PYTHON = '>=3.6.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # package: version
    'Cython>=0.28.5,<0.29.0'
]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=REQUIRED,
    ext_modules=cythonize("bpe/utils.pyx")
)
