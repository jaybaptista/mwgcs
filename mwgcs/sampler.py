import abc
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm 

#################################################################
# Samplers for the GC System mass, should return a single value for mass

def EadieSampler(stellar_mass, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    system_mass = 1/(1 + np.exp(-(b0 + (b1 * np.log10(stellar_mass))))) * (g0 + (g1*np.log10(stellar_mass)))
    return system_mass        

#################################################################
# Samplers for the individual GC tags, should return a number of GC masses
def DwarfGCMF(stellar_mass,
              mass_light_ratio = 10.,
              system_mass_sampler = EadieSampler):
    
    # NOTE—make sure that you have a keyword "system_mass_sampler"
    # which is a function that takes in stellar mass in Msun
    # and outputs the bulk GC system mass.
    
    gc_peak_mass = 10**(system_mass_sampler(stellar_mass))

    min_gband = -5.5
    max_gband = -9.5
    
    min_mass = mass_light_ratio * 10**(0.4*(5.03 - max_gband))
    max_mass = mass_light_ratio * 10**(0.4*(5.03 - min_gband))
    
    gc_mass_range = np.logspace(np.log10(min_mass), np.log10(max_mass))

    def _gclf(mass, M_mean=-7., M_sigma=0.7):
        # get magnitude from mass
        mag = 5.03 - 2.5 * np.log10(mass)
        gclf_value = norm.pdf(mag, loc=M_mean, scale=M_sigma)
        return gclf_value
    
    def r(M):
        _a = quad(_gclf, min_mass, M)[0]
        _b = quad(_gclf, min_mass, max_mass)[0]
        return _a / _b
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interp1d(cdf, gc_mass_range, kind="linear")

    accumulated_mass = [inv_cdf(np.random.uniform(0, 1))]
    last_sampled_mass = accumulated_mass[0]

    while np.sum(accumulated_mass) < gc_peak_mass:
        # sample a mass
        sampled_mass = inv_cdf(np.random.uniform(0, 1))

        # add it to the running total
        accumulated_mass.append(sampled_mass)

        # keep track of the last sampled mass
        last_sampled_mass = sampled_mass

    ratio = (np.sum(accumulated_mass) - gc_peak_mass) / last_sampled_mass 
    
    if (np.random.uniform(0, 1) > ratio) & (len(accumulated_mass) > 1):
        accumulated_mass = accumulated_mass[:-1]

    return accumulated_mass

#################################################################
# Samplers for the stream phase space distribution

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