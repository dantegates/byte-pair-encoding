from distutils.core import setup
from Cython.Build import cythonize

setup(name='Byte pair encoder',
      ext_modules=cythonize("bpe/bpe_utils.pyx"))