from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension("dither", ["dither.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='dither',
    ext_modules=cythonize(extensions),
    script_args = ["build_ext", "--inplace"]
)