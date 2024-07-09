import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt
import random
import symlib

from .fit import MassProfile

from gala.potential import PlummerPotential
from gala.units import galactic

def getLambda(q_gc, m_gc, isHost=False):
    
    # baryonic potential
    r_exp = 3 * u.kpc
    r_half = np.log(2) * r_exp
    a = (r_half / (1.3)).value
    m0 = 4e10
    
    pot_b = PlummerPotential(m0, a, units=galactic) 
    H_ij_b = pot.hessian(q_gc)
    tensor_b = (-(1/3)*np.trace(H_ij_b) * np.identity(3) + H_ij_b)
    
    # dark matter potential
    
    
    
    # lam_b = (np.max(np.abs(np.linalg.eig(tensor)[0]))).to(1/u.Gyr**2)
    # lam_dm is wrong
    # lam_dm = ((c.G * (_m * u.Msun) / (rtidal * u.kpc)**3).to(u.s**(-2))).to(u.Gyr**(-2))
    return lam_dm + lam_b

def getTidalFrequency(gc_dist, _m, rtidal):
    alpha = getLambda(gc_dist, _m, rtidal).to(u.s**(-2))
    return (alpha.value/3)**(1/2)

def getTidalTimescale(gc_dist, gc_mass, rtidal):
    tidal_frequency = getTidalFrequency(gc_mass, rtidal) / u.s
    t_tid = 10 * u.Gyr * (gc_mass / 2e5)**(2/3) / (tidal_frequency / (100 * u.Gyr**(-1)))
    t_tid = (t_tid.to(u.Gyr)).value
    return t_tid

def getMassLossRate(gc_dist, gc_mass, rtidal):
    # Msun / Gyr
    return -gc_mass / getTidalTimescale(gc_dist, gc_mass, rtidal)