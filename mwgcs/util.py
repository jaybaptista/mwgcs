import asdf
import astropy.constants as c
import astropy.units as u
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
import symlib
from tqdm import tqdm

import os



# def tidal_radius(R, m, M, s):
#     # rt,2 from https://academic.oup.com/mnras/article/474/3/3043/4638541#106350025
#     # R is the galactocentric radius
#     # m is the mass of the orbiting body
#     # M = M(R) is the enclosed mass at radius R
#     # s = (dln M / dln R)|_R=R

#     mass_ratio = (m / M)**(1/3)
#     slope_term = (3 - s)**(1/3)
#     return R * mass_ratio / slope_term

