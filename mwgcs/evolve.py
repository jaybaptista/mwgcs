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


def tidalRadius(gc_dist, gc_mass, profile : MassProfile):
    # the mass profile should have already been fit at this point
    # TODO: 
    slope = None
    if gc_dist < profile.r_conv:
        profile.fit()
        slope = profile._ein.analyticSlope(np.array([gc_dist]))
    else:
        # interpolate around gc_dist
        _r, _rho = profile.density_rs()
        _slope = _rho * _r**3 * 4 * np.pi / profile.mass(gc_dist)
        f = interp1d(_r, _slope, kind="cubic")
        slope = f(gc_dist)
    
    enclosed_mass = profile.mass(gc_dist)
    r_tidal = gc_dist * ((gc_mass / enclosed_mass) / (3 - slope))**(1/3)
    # print("subhalo mass: ", enclosed_mass, "mass slope", slope)
    return r_tidal

def getLambda(_m, rtidal):
    return ((c.G * (_m * u.Msun) / (rtidal * u.kpc)**3).to(u.s**(-2))).to(u.Gyr**(-2))

def getTidalFrequency(_m, rtidal):
    alpha = (c.G * (_m * u.Msun) / (rtidal * u.kpc)**3).to(u.s**(-2))
    # print("frequency in Gyr:", (alpha.to(u.Gyr**(-2))/3)**(1/2))
    return (alpha.value/3)**(1/2)

def getTidalTimescale(gc_mass, rtidal):
    tidal_frequency = getTidalFrequency(gc_mass, rtidal) / u.s
    t_tid = 10 * u.Gyr * (gc_mass / 2e5)**(2/3) / (tidal_frequency / (100 * u.Gyr**(-1)))
    t_tid = (t_tid.to(u.Gyr)).value
    return t_tid

def getMassLossRate(gc_mass, mvir, rtidal):
    # Msun / Gyr
    return -gc_mass / getTidalTimescale(gc_mass, mvir, rtidal)

# ############################################

# def assignParticles(mp, mgc, bindEnergy=None):
#     prob = mp / np.sum(mp)
    
#     if bindEnergy is not None:
#         bound_mask = bindEnergy < 0
#         prob[~bound_mask] = 0 # set unbound stars to zero prob
    
#     tag_idx  = np.random.choice(np.arange(len(prob)), size=len(mgc), replace=False, p=prob)
#     return tag_idx

# def distancesToParticle(pos, tag_id):
#     pos_tag = pos[tag_id]
#     transformed_pos = pos - pos_tag
#     dist = (transformed_pos[:, 0]**2 + transformed_pos[:, 1]**2 + transformed_pos[:, 2]**2)**(1/2)
#     return dist

# def getPotentials(pos, ok, params):
#     rmax, vmax, PE, order = symlib.profile_info(params, pos[ok])
#     pot = PE * (vmax**2)
#     return pot