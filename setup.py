#!/usr/bin/env python

# from distutils.core import setup
from setuptools import setup
import numpy as np

setup(
    name="gchords",
    version="1.0",
    description="Globular cluster formation and evolution in Symphony",
    author="Jay Baptista",
    author_email="jaymarie@stanford.edu",
    packages=["gchords"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "asdf",
        "h5py",
        "symlib",
        "astropy",
    ],
    include_dirs=[np.get_include()]
)