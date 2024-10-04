import numpy as np
import matplotlib.pyplot as plt
import asdf
import symlib
import sys
sys.path.append('/sdf/home/j/jaymarie/software/gravitree/python')
import gravitree
import astropy.units as u
from tqdm import tqdm

from scipy.stats import norm

import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics.nbody import DirectNBody
from gala.units import galactic, UnitSystem
sys.path.append('/sdf/home/j/jaymarie/mwgcs/science/')
from plot import setFonts
setFonts()

from gala.dynamics import PhaseSpacePosition
from astropy.coordinates import CylindricalRepresentation, CylindricalDifferential
from astropy.units import Quantity

from mwgcs import getMassLossRate, mdot_gg23

def tidalRadius(r, msat, pot):
    
    q_ref = [r.to(u.kpc).value , 0., 0.] * u.kpc
    mass = pot.mass_enclosed(q_ref).to(u.Msun)
    density = pot.density(q_ref).to(u.Msun / u.kpc**3)
    logMassSlope = ((r / mass) * 4 * np.pi * r**2 * density).decompose()
    
    f = 3 - logMassSlope
    
    return (msat/(f*mass))**(1/3) * r


# note this is only for the toy model, working w
# sim data needs to use the Profile class to interface
# properly w all the Einasto potentials
def ga(r, pot):
    # get hessian at [r, 0, 0]
    q_ref         = [r.value, 0., 0.,] * u.kpc
    d2phi_dr2     = pot.hessian(q_ref)[0, 0].to(u.Myr**(-2))
    v_circ        = pot.circular_velocity(q_ref)
    orbital_freq2 = ((v_circ / r)**2).to(u.Myr**(-2))
    g_a           = orbital_freq2 - d2phi_dr2
    return g_a

def f_t():
    # tidal factor, may be variable in future
    return .8

def Racc(r_apo, r_peri, pot):
    return (ga(r_peri, pot) / ga(r_apo, pot)).decompose().value

# @np.vectorize
def sample_k(r_apo, r_peri, pot):
    
    # optimized Kupper+12 initial conditions 
    k_r_mu = 2.
    k_vt_mu = .3
    # dispersions
    Y = .15 * (f_t()**2) * (Racc(r_apo, r_peri, pot))**(2/3)

    @np.vectorize
    def get_disp(x):
        return np.min([.4, x])

    k_r_sigma = k_vt_sigma = get_disp(Y)
    
    k_z_mu = k_vz_mu = 0. # symmetry
    k_z_sigma = k_vz_sigma = .5

    # these are set to zero in literature
    k_phi = 0.
    k_vr  = 0.
    k_r = norm.rvs(loc=k_r_mu, scale=k_r_sigma)
    k_vt = norm.rvs(loc=k_vt_mu, scale=k_vt_sigma)
    k_z = norm.rvs(loc=k_vz_mu, scale=k_vz_sigma)
    k_vz = norm.rvs(loc=k_vz_mu, scale=k_vz_sigma)

    return k_phi, k_vr, k_r, k_vt, k_z, k_vz

def w_ejecta(w_sat, m_sat, pot, r_apo, r_peri):
    cyl = w_sat.represent_as('cylindrical')
    
    r_sat, phi_sat, z_sat = cyl.pos.rho, cyl.pos.phi, cyl.pos.z
    vr_sat, vt_sat, vz_sat = cyl.vel.d_rho, cyl.vel.d_phi, cyl.vel.d_z

    vt_sat = vt_sat.to(u.rad / u.s)
    
    k_phi, k_vr, k_r, k_vt, k_z, k_vz = sample_k(r_apo, r_peri, pot)

    r_t = tidalRadius(r_sat, m_sat, pot)

    # positions
    r_ej = Quantity([r_sat + (k_r * r_t), r_sat - (k_r * r_t)])
    phi_ej = Quantity([phi_sat + (k_phi * (r_t / r_sat)) * u.rad, phi_sat - (k_phi * (r_t / r_sat)) * u.rad] )
    z_ej = Quantity([z_sat + (k_z * (r_t / r_sat)) * u.kpc, z_sat - (k_z * (r_t / r_sat)) * u.kpc]) # this is in the orbital plane
    # velocities

    vr_ej = Quantity([vr_sat + k_vr * vr_sat, vr_sat - k_vr * vr_sat])
    
    _tmp = (pot.circular_velocity(w_sat.pos.to_cartesian().get_xyz()) * (r_t / r_sat) * u.rad / u.kpc).to(u.rad / u.s)
    
    vt_ej = Quantity([
        vt_sat + k_vt * _tmp, vt_sat - k_vt * _tmp])
        
        # not sure if I'm doing this right?
    vz_ej = Quantity([k_vz * (r_t / r_sat) * (u.km / u.s), -k_vz * (r_t / r_sat) * (u.km / u.s)])

    _p_cyl = CylindricalRepresentation(
        r_ej.to(u.kpc),
        phi_ej.to(u.rad),
        z_ej.to(u.kpc))

    _v_cyl = CylindricalDifferential(
        vr_ej.reshape(2, 1).to(u.km / u.s),
        vt_ej.to(u.rad / u.s),
        vz_ej.to(u.km / u.s))

    cyl_ej = PhaseSpacePosition(_p_cyl, _v_cyl)
    w_ej = cyl_ej.represent_as('cartesian')
    return w_ej

def ejection_rate(theta_r, r_apo, r_peri, pot):
    _Racc     = Racc(r_apo, r_peri, pot)
    zeta      = np.exp(1.4 * (np.log(_Racc))**(3./4.))
    alpha     = _Racc**(.55)
    x = (f_t() * _Racc)
    theta_mid = -.1 + .7 * (x / (7 + x))
    
    


    
