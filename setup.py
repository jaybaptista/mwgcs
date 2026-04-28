#!/usr/bin/env python

# from distutils.core import setup
from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

setup(
    name="mwgcs",
    version="1.0",
    description="Globular cluster formation and evolution in Symphony",
    author="Jay Baptista",
    author_email="jaymarie@stanford.edu",
    packages=["mwgcs"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "asdf",
        "h5py",
        "symlib",
        "astropy"
    ],
    include_dirs=[np.get_include()]
)