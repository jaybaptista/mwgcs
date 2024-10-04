import numpy as np
import matplotlib.pyplot as plt
import asdf
import symlib
import sys
sys.path.append('/sdf/home/j/jaymarie/software/gravitree/python')
import gravitree
import astropy.units as u
from tqdm import tqdm

import time # benchmarking

from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics.nbody import DirectNBody
from gala.units import galactic, UnitSystem

sys.path.append('/sdf/home/j/jaymarie/mwgcs/science')
from plot import setFonts
setFonts()

sim_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/SymphonyMilkyWay/Halo023"

from mwgcs import Simulation, GCSystem, MassProfile, NFW, Einasto, sampleDwarfGCMF, getMassLossRate, mdot_gg23, getTidalTimescale, CMassLoss

dm     = gp.NFWPotential(m=1e12, r_s=25., units=galactic)
baryon = gp.PlummerPotential(m=4e10, b=1.6, units=galactic)
pot    = dm + baryon
# pot = dm
# palomar 5 orbital parameters
r_apo = 15.5
r_peri = 6.5
ecc = 0.41

v_frac = np.linspace(.01, .95, 10)
q0 = [r_apo, 0., 0.] * u.kpc
v_init = pot.circular_velocity(q0).value[0]
ang_mom = (r_apo * u.kpc * (v_init * v_frac * u.km/u.s)).to(u.kpc**2 / u.Myr)
v_inits = np.vstack([np.zeros(len(v_frac)), v_init * v_frac, np.zeros(len(v_frac))]).T * u.km / u.s
arr_orbits = []

dt=1e-2*u.Myr

for v_i in tqdm(v_inits):
    w0 = gd.PhaseSpacePosition(pos=q0, vel=v_i)
    particle_pot = [None]
    nbody = DirectNBody(w0, particle_pot, external_potential=pot)
    T = 2*np.pi*r_apo * u.kpc / pot.circular_velocity(q0) 
    orbits = nbody.integrate_orbit(dt=dt, t1=0, t2=T)
    arr_orbits.append(orbits)

def getOrbitalPeriod(orbit):

    # all units in kpc2 / Myr2
    E = orbit.energy()[0][0].value
    
    ang_mom = np.sqrt(np.sum(orbit[:, 0].angular_momentum().T**2, axis=1))[0].value
    
    def getPotential(r):
        q = [0., r, 0.] * u.kpc
        return pot(q).value

    _apo, _peri = orbit.apocenter()[0].value, orbit.pericenter()[0].value
    
    def integrand(r):
        return (2*(E - getPotential(r)) - (ang_mom**2 / r**2))**(-1/2)

    res = quad(integrand, _peri, _apo)[0]
    return 2 * res * u.Myr

def estimateOrbitalPeriod(orbit):
    # find index it flips sign
    #arg rel extrema
    v = np.sqrt(orbit[:, 0].v_x**2 + orbit[:, 0].v_y**2 + orbit[:, 0].v_z**2).to(u.km/u.s).value
    extr_idx = argrelextrema(v, np.less)
    estimated_period = orbit.t[extr_idx[0][0]]
    return estimated_period

def getTidalStrength(q):
    hess = pot.hessian(q)
    tidal_tensor = hess - np.multiply((1/3)*np.trace(hess, axis1=0, axis2=1), np.tile(np.identity(3), (hess.shape[2], 1, 1)).T)
    eigs = np.linalg.eigvals(tidal_tensor.T)
    lam = np.max(np.abs(eigs), axis=1)
    return lam

test_radii     = np.logspace(-2, 2, 200)
test_qs        = np.vstack([np.zeros(len(test_radii)), test_radii, np.zeros(len(test_radii))]).T * u.kpc
test_strengths = np.log10([getTidalStrength(q).to(u.Gyr**(-2)).value[0] for q in test_qs])
interp_log_strength = interp1d(np.log10(test_radii), test_strengths, kind='linear', fill_value='extrapolate')


def get_dM_m(m0, orbit):
    print('wtf')
    start_time = time.time()
    
    r = np.sqrt(np.sum(orbit[:, 0].xyz**2, axis=0))
    masses = np.zeros(len(r)) + m0
    
    strengths = np.zeros(len(r))
    print("1 --- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    end_idx = np.argmin(abs(orbit.t - estimateOrbitalPeriod(orbit))) # time index that roughly matches the orbital period
    print("2 --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    ang_mom = np.sqrt(np.sum(orbit[:, 0].angular_momentum().T**2, axis=1))[0].value
    _apo, _peri = orbit.apocenter()[0].value, orbit.pericenter()[0].value
    print("3 --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    l = 10**(interp_log_strength(np.log10(r.value))) / u.Gyr**2
    print("4 --- %s seconds ---" % (time.time() - start_time))
    print(end_idx)
    masses = CMassLoss(m0, l, end_idx, dt)
    
    strengths = np.log10(l.value)
    dM_orb = m0 - np.min(masses)

    tree = {
        'dM_m': dM_orb / m0,
        'peri': _peri,
        'apo': _apo,
        'L': ang_mom,
        'mass': masses[:end_idx+1],
        'logLambda': strengths[:end_idx+1],
        'radii': r[:end_idx+1],
        'time': orbit.t[:end_idx+1]
    }
    
    return tree

trees = []
m_init = 3.5e4

for orbit in tqdm(arr_orbits):
    trees.append(get_dM_m(m_init, orbit))

for k, tree in enumerate(trees):
    af = asdf.AsdfFile(tree=tree)
    af.write_to('data/pal5_orbit_v_{:.2f}_apo_{:.2f}.asdf'.format(v_frac[k], r_apo))