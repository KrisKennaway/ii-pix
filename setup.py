from setuptools import setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(
        ["dither_dhr.pyx", "dither_shr.pyx"],
        annotate=True,
        compiler_directives={'language_level': "3"}
    )
)
