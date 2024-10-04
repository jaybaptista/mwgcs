
from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

package = Extension('cy_evolve', ['cy_evolve.pyx'], include_dirs=[np.get_include()])
setup(ext_modules=cythonize([package]))
