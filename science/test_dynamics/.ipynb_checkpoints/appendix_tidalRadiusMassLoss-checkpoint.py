import numpy as np
import matplotlib.pyplot as plt
import asdf
from matplotlib import rc
from scipy.optimize import curve_fit
import symlib
import sys
sys.path.append('/sdf/home/j/jaymarie/software/gravitree/python')
import gravitree
from scipy.integrate import quad
import astropy.constants as cons
import astropy.units as u
from tqdm import tqdm
from glob import glob

from plot import setFonts
setFonts()

from gala.potential import PlummerPotential

from gala.units import galactic

sim_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/SymphonyMilkyWay/Halo023"

from mwgcs import Simulation, GCSystem, MassProfile, NFW, Einasto, sampleDwarfGCMF

sim = Simulation(sim_dir)

