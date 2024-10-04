import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import asdf
import symlib
import sys
sys.path.append('/sdf/home/j/jaymarie/software/gravitree/python')
import gravitree
import astropy.units as u
from tqdm import tqdm
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

from mwgcs import CMassLoss

######### Bulge-Disk Potential ###########

dm     = gp.NFWPotential(m=1e12, r_s=25., units=galactic)
baryon = gp.PlummerPotential(m=4e10, b=1.6, units=galactic)
pot    = dm + baryon

##########################################

##### palomar 5 orbital parameters #######
r_apo  = 15.5
r_peri = 6.5
ecc    = 0.41

##########################################

# simulation parameters

dt     = 1e-2*u.Myr
m_init = 3.5e4

# set up variable apocenters
N_apo      = 5
N_v        = 10
apocenters = np.linspace(10., 200., N_apo)
# fraction of circular velocity to initialize test particle
v_frac     = np.linspace(.005, .95, N_v)

########## helper functions ##############


def getOrbitalPeriod(orbit):
    # calculates the orbital period numerically
    
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
    # approximates the orbital period
    # find index it flips sign
    
    v = np.sqrt(orbit[:, 0].v_x**2 + orbit[:, 0].v_y**2 + orbit[:, 0].v_z**2).to(u.km/u.s).value
    extr_idx = argrelextrema(v, np.less)
    estimated_period = orbit.t[extr_idx[0][0]]
    return estimated_period

def getTidalStrength(q):
    # calculates the tidal strength using the hessian -> tidal tensor
    hess = pot.hessian(q)
    tidal_tensor = hess - np.multiply((1/3)*np.trace(hess, axis1=0, axis2=1), np.tile(np.identity(3), (hess.shape[2], 1, 1)).T)
    eigs = np.linalg.eigvals(tidal_tensor.T)
    lam = np.max(np.abs(eigs), axis=1)
    return lam


def get_dM_m(m0, orbit):
    
    # returns the orbital properties + mass evolution of a particle
    # given an initial mass and the orbit
    
    r           = np.sqrt(np.sum(orbit[:, 0].xyz**2, axis=0))
    masses      = np.zeros(len(r)) + m0
    strengths   = np.zeros(len(r))
    end_idx     = np.argmin(abs(orbit.t - estimateOrbitalPeriod(orbit))) # time index that roughly matches the orbital period
    ang_mom     = np.sqrt(np.sum(orbit[:, 0].angular_momentum().T**2, axis=1))[0].value
    _apo, _peri = orbit.apocenter()[0].value, orbit.pericenter()[0].value
    l           = 10**(interp_log_strength(np.log10(r.value))) / u.Gyr**2
    masses      = CMassLoss(m0, l, end_idx, dt) 
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
        'time': orbit.t[:end_idx+1].to(u.Gyr),
        'end_idx': end_idx
    }
    
    return tree

# FOR SPEED: interpolate the tidal strength at different radii
test_radii     = np.logspace(-2, 2, 200)
test_qs        = np.vstack([np.zeros(len(test_radii)), test_radii, np.zeros(len(test_radii))]).T * u.kpc
test_strengths = np.log10([getTidalStrength(q).to(u.Gyr**(-2)).value[0] for q in test_qs])
interp_log_strength = interp1d(np.log10(test_radii), test_strengths, kind='linear', fill_value='extrapolate')

#####################################

fig, ax = plt.subplots(dpi=200, figsize=(3,3))
ax.set_xlabel(r'$r_\mathrm{peri}$ [kpc]')
ax.set_ylabel(r'$\frac{\Delta M}{M}$ over 1 orbital period')

cmap = mpl.colormaps['plasma']
colors = cmap(np.linspace(0, 1, N_apo))

# loop through different apocenters
for k, r_apo in enumerate(tqdm(apocenters)):
    
    q0        = [r_apo, 0., 0.] * u.kpc
    v_circ    = pot.circular_velocity(q0).value[0]
    v_inits   = np.vstack([np.zeros(len(v_frac)), v_circ * v_frac, np.zeros(len(v_frac))]).T * u.km / u.s
    # ang_mom = (r_apo * u.kpc * (v_circ * v_frac * u.km/u.s)).to(u.kpc**2 / u.Myr)
    orbits    = []
    
    for v_i in v_inits:
        
        w0           = gd.PhaseSpacePosition(pos=q0, vel=v_i) # initialize the phase space position of particle
        particle_pot = [None] # no particle contributions to potential at the moment...
        nbody        = DirectNBody(w0, particle_pot, external_potential=pot) # setup direct nbody
        T            = 2*np.pi*r_apo * u.kpc / pot.circular_velocity(q0) # run simulation for two orbital periods
        
        # integrate orbits
        _o = nbody.integrate_orbit(dt=dt, t1=0, t2=T)
        orbits.append(_o)

    trees = []

    for orbit in orbits:
        trees.append(get_dM_m(m_init, orbit))
    
    r_peri = []
    dM_m = []
    end_indices = []
    period = []
    apo = []
    
    for tree in trees:
        r_peri.append(tree['peri'])
        dM_m.append(tree['dM_m'])
        end_indices.append(tree['end_idx'])
        period.append(tree['time'][-1])
        apo.append(tree['apo'])

        af = asdf.AsdfFile(tree)
        af.write_to('orbit_apo_{:.2f}_peri_{:.2f}.asdf'.format(tree['apo'], tree['peri']))
        

    _f = interp1d(r_peri, dM_m, kind='linear')
    _r = np.linspace(_f.x.min(), _f.x.max(), 100)
    
    ax.scatter(r_peri, dM_m, c=colors[k], marker='o', label=r'$r_\text{{apo}} = {:.2f} \text{{ kpc}}$'.format(r_apo))
    ax.plot(_r, _f(_r), c=colors[k])

ax.legend()
plt.savefig('singleOrbitMassLoss.pdf', bbox_inches='tight')