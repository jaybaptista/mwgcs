import asdf
import astropy.constants as c
import astropy.units as u
from colossus.cosmology import cosmology

sys.path.append('/sdf/home/j/jaymarie/software/gravitree/python')
import gravitree

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

import random

from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d

import symlib
import sys

class Simulation():
    """
    This is just a helper class to get what I want from the simulation
    """
    def __init__(self, sim_dir, seed=96761):
        random.seed(seed) 
        self.sim_dir = sim_dir
        self.params = symlib.simulation_parameters(sim_dir)
        self.scale_factors = np.array(symlib.scale_factors(sim_dir))
        self.col_params = symlib.colossus_parameters(self.params)
        self.cosmo = cosmology.setCosmology("cosmo", params=self.col_params)
        self.z = (1/self.scale_factors) - 1
        self.partClass = symlib.Particles(sim_dir)
        
        self.rs, self.hist = symlib.read_rockstar(self.sim_dir)
        self.sf, self.shist = symlib.read_symfind(self.sim_dir)
        
        self.gal_halo = gal_halo = symlib.GalaxyHaloModel(
            symlib.StellarMassModel(
                symlib.UniverseMachineMStar(),
                symlib.DarkMatterSFH() # swapped this one out
        ),
            symlib.ProfileModel(
                symlib.Jiang2019RHalf(),
                symlib.PlummerProfile()
            ),
            symlib.MetalModel(
                symlib.Kirby2013Metallicity(),
                symlib.Kirby2013MDF(model_type="gaussian"),
                symlib.FlatFeHProfile(),
                symlib.GaussianCoupalaCorrelation()
            )
        )
        
        self.buffer_snap = 235
        self.particles = None
    
    def getRedshift(self, snapshot):
        return self.z[snapshot]
    
    def getScaleFactor(self, snapshot):
        return self.scale_factors[snapshot]
        
    def getParticles(self, snap):
        
        if (self.buffer_snap != snap) or (self.particles is None):
            self.buffer_snap = snap
            self.particles = self.partClass.read(snap, mode="all")
            
        return self.particles
        
        self.particles = self.partClass.read(snap, mode="all")
        return self.particles
    
    def getConvergenceRadius(self, snapshot):
        # Calculate convergence radius
        a = self.getScaleFactor(snapshot)
        
        H0 = (100 * self.params['h100'] * u.km / u.s / u.Mpc).decompose().value
        
        rho_crit = (3 / (8 * np.pi * c.G.to(u.kpc**3 / u.Msun / u.s**2).value)) * (H0)**2
        
        # rho_m0 = self.params['Om0'] * rho_crit / a**3 # divide by a^3
        
        rho_m = self.params['Om0'] * rho_crit
        
        l_0 = (self.params['mp'] / self.params['h100'] / rho_m)**(1/3)
        
        z = self.getRedshift(snapshot)
        
        l = a * l_0 
        return (5.5e-2 * l, 3 * self.params['eps']/self.params['h100'] * a)

    

class Einasto():
    """
    This is a Profile class that holds all the helper functions for the fit
    """
    def __init__(self, Rs=None, logrho2=None, alpha=None):
        """
        Sets initial fitting parameters
        """
        if alpha is None:
            self.alpha = .18 
        else:
            self.alpha = alpha
        
        if Rs is None:
            self.default_Rs = 1
        else: 
            self.default_Rs = Rs
        
        if logrho2 is None:
            self.default_logrho2 = 0
        else:
            self.default_logrho2 = logrho2
    
    def A(self):
        """
        Assuming an Einasto profile, the radius at which the density curve is maximal is A times the scale radius Rs. 
        """
        return 1.715 * (self.alpha**(-.00183)) * (self.alpha + 0.0817)**(-.179488)
        
    def density(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Analytic form of the density profile
        """
        if Rs is None:
            Rs = self.default_Rs
        if logrho2 is None:
            logrho2 = self.default_logrho2
        if alpha is None:
            alpha = self.alpha
        
        # Rmax = 2.164 * Rs
        Rmax = self.A() * Rs
        return (10**logrho2) * np.exp(-(2 / self.alpha) * (((self.A() * r) / Rmax)**(self.alpha) - 1))
    
    def logdensity(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Log of the density profile, used primarily for fitting
        """
        return np.log10(self.density(r, Rs, logrho2, alpha))

    def density_integrand(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Integrand used to compute enclosed mass of the profile
        """
        return self.density(r, Rs, logrho2, alpha) * 4 * np.pi * r**2
    
    def mass(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Enclosed mass of the profile
        """
        m = quad(self.density_integrand, 0, r, (Rs, logrho2, alpha))[0]
        return m
    
    def v_circ(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Returns circular velocity
        """
        mass = self.mass(r, Rs, logrho2, alpha) * u.Msun
        vel = ((c.G * mass / (r * u.kpc))**(1/2)).to(u.km / u.s).value
        return vel
    
    def massSlope(self, radii, Rs=None, logrho2=None, alpha=None):
        mass = [self.mass(r_i, Rs, logrho2, alpha) for r_i in radii]
        dlnr = np.gradient(radii, edge_order=2) / radii
        dlnm = np.gradient(mass, edge_order=2) / mass
        mass_profile = interp1d(radii, dlnm/dlnr)
        return mass_profile
    
    def getVmax(self):
        radii = 10**np.linspace(-.5, 2.5)
        velocities = [self.v_circ(r) for r in radii]
        return np.max(velocities)
        
class MassProfile():
    """
    This class interacts with a Simulation object and references a subhalo and snapshot 
    to compute the density profile according to Rockstar-style binning
    """
    
    def __init__(self,
                 sim,
                 snap,
                 sh_id,
                 boundOnly=False,
                 subsample_frac=0.
                ):
        
        p = sim.getParticles(snap)
        params = sim.params
        
        self.pos = p[sh_id]['x']
        self.vel = p[sh_id]['v']
        
        self.sh_pos = sim.rs[sh_id, snap]['x']
        self.sh_vel = sim.rs[sh_id, snap]['v']
        self.rvir = sim.rs[sh_id, snap]['rvir']
        
        if not sim.sf[sh_id, snap]['ok_rs']:
            self.sh_pos = sim.rs[sh_id, snap]['x']
            self.sh_vel = sim.rs[sh_id, snap]['v']
        
        self.rvir = sim.rs[sh_id, snap]['rvir']
        self.eps = sim.params['eps']
        self.h100 = sim.params['h100']
        self.mp = sim.params['mp']
        
        self.bins_ok = False
        
        part_limit_fit = 25
        resample_counter = 0
        
        # this is bugged right now:
        while (subsample_frac > 0) and not self.bins_ok:
            n = len(self.pos) 
            rand_index = random.sample(range(n), int(subsample_frac * n))
            
            self.pos = p[sh_id]['x'][rand_index]
            self.vel = p[sh_id]['v'][rand_index]
            self.mp = self.mp / subsample_frac
            _dist = np.sqrt(np.sum((self.pos-self.sh_pos)**2, axis=1))
            
            mask = _dist < self.rvir
            
            resample_counter +=1
            # loop through until you find good enough choice
            # of bins
            if np.sum(mask) > part_limit_fit:
                self.bins_ok = False
                break
            elif resample_counter > 10:
                raise Exception("Resampled more than 10 times, ending...")

                
        self.dist = np.sqrt(np.sum((self.pos-self.sh_pos)**2, axis=1))
        self.r_conv = np.max(sim.getConvergenceRadius(snap))
        self._ein = None
        
        self.z = sim.getRedshift(snap)
        
        self.boundOnly = boundOnly
        
        if boundOnly:
            print("Calculating binding energies...")
            self.bE = gravitree.binding_energy(
                self.pos - self.sh_pos, # change particle frame to frame of the subhalo
                self.vel - self.sh_vel, # same here
                self.mp / self.h100,
                self.eps / self.h100,
                n_iter=3)

            self.bound = self.bE < 0
        
        

        
    def particleCount(self, r):
        
        if self.boundOnly:
            return np.sum((self.dist < r) & self.bound)
        
        return np.sum(self.dist < r)
    
    def mass(self, r):
        return self.mp * self.particleCount(r)
    
    def density(self, r, dr=.001):
        vol = (4*np.pi / 3) * (r**3 - (r-dr)**3)
        counts = 0
        if self.boundOnly:
            counts = np.sum((self.dist < r) & (self.dist > (r-dr)) & self.bound)
        else:
            counts = np.sum((self.dist < r) & (self.dist > (r-dr)))
        
        mass = counts * self.mp
        
        return mass / vol
    
    def v_circ(self, r):
        return (c.G * self.mass(r) * u.Msun / (r * u.kpc))**(1/2)
    
    def density_rs(self, bins=50):
        mask = self.dist < self.rvir # cut on distance
        
        # edge case is when there are less particles than bins desired
        low_res = np.sum(mask) < bins
        if low_res:
            
            bins = np.sum(mask)
            print("low res", bins)
        
        _dist = self.dist[mask]
        indices = np.argsort(_dist) # sort by radius
        idx_bins = np.array_split(indices, bins) # split index bins roughly evenly
        
        self._rs_idx_bins = idx_bins
        
        bin_edges = np.concatenate(
            [
                [0],
                [_dist[i[-1]] for i in idx_bins] # rightmost bin edge
            ]
        )
        
        self._rs_bin_edges = bin_edges
        
        bin_volume = (4*np.pi / 3) * ((bin_edges[1:])**3 - (bin_edges[:-1])**3)
        
        self._rs_bin_volume = bin_volume
        
        self._rs_bin_count = np.array([
            np.sum(
                (_dist >= bin_edges[i]) & (_dist < bin_edges[i+1])) for i in range(len(bin_edges)-1)])
        
        if low_res:
            self._rs_bin_count = np.ones(bins)
        
        bin_density = np.array(self.mp * self._rs_bin_count / bin_volume)
        
        rad_bins = np.array_split(_dist[indices], bins) # split particles up by their sorted positions
        avg_rad = np.array([np.mean(i) for i in rad_bins]) # get the mean of each bin, this is the radius used in the fit
        
        return avg_rad, bin_density

    def fit(self, ein=None):
        if ein is None:
            ein = Einasto()
        self._ein = ein # remove this after debugging
        radii, density_profile = self.density_rs() # fit the desnity profile based on rockstar binning
        
        radial_mask = radii > self.r_conv 
        self.bins = np.array(radii)[radial_mask]
        self.logdata = np.log10(np.array(density_profile)[radial_mask])
        if (self.bins.shape != self.logdata.shape):
            print("FUCK")
        popt, pcov = curve_fit(ein.logdensity,
                       self.bins,
                       self.logdata,
                       p0=[20, .6, .18], # some random values
                       maxfev = 5000
                      )
        
        _ein = Einasto(*popt)
        
        self._popt = popt
        self._ein = _ein
        
        return _ein.density, _ein.mass, _ein.v_circ
    
    def massSlope(self, radii):
        mass = [self.mass(r_i) for r_i in radii]
        dlnr = np.gradient(radii, edge_order=2) / radii
        dlnm = np.gradient(mass, edge_order=2) / mass
        mass_profile = interp1d(radii, dlnm/dlnr)
        return mass_profile
    

def lognorm_hurdle(ms, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    mgc = 1/(1 + np.exp(-(b0 + (b1 * np.log10(ms))))) * (g0 + (g1*np.log10(ms)))
    return mgc

def mass_func(m, logmc=5.4, logDelta=5.9):
        dM = 10**logDelta
        mc = 10**logmc
        return (m+dM)**(-2) * np.exp(-(m+dM)/mc)

def sampleGCMFHurdle(ms, Mmin = 1e3, Mmax = 1e7):
    ms_peak_idx  = np.argmax(ms)
    ms_peak = ms[ms_peak_idx]
    gc_peak_mass = 10**(lognorm_hurdle(ms_peak))
    gc_mass_range = np.logspace(np.log10(Mmin), np.log10(Mmax))
    
    def r(M):
        num = quad(mass_func, Mmin, M)[0]
        den = quad(mass_func, Mmin, Mmax)[0]
        return num / den
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interpolate.interp1d(cdf, gc_mass_range, kind="linear")
    
    accumulated_mass  = [inv_cdf(np.random.uniform(0, 1))]
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

    return accumulated_mass, gc_mass_range


def getTidalRadius(gc_dist, gc_mass, sh_x, sh_rvir, bE=None, offset=0):
    """
    gc_dist: radial distance from subhalo center to gc
    gc_mass: mass of the gc
    sh_x: position of particles in the subhalo
    sh_rvir: virial radius of the subhalo
    bE: binding energies of the particles
    offset: position of the subhalo relative to central or simbox origin
    """
    
    # get slope
    g = massProfile(sh_x, sh_rvir, bE=bE, offset=offset)
    
    sh_m = mass(gc_dist, sh_x, mp=281981.0)
    return gc_dist * ((gc_mass / sh_m) / (2 - g(gc_dist)))**(1/3)



mass_light_ratio = 2.

def lognorm_hurdle(ms, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    mgc = 1/(1 + np.exp(-(b0 + (b1 * np.log10(ms))))) * (g0 + (g1*np.log10(ms)))
    return mgc

#########################################################################################################################################################

def MilkyWayGCMF(m, logmc=5.4, logDelta=5.9):
        dM = 10**logDelta
        mc = 10**logmc
        return (m+dM)**(-2) * np.exp(-(m+dM)/mc)

def sampleMilkyWayGCMF(ms, Mmin = 1e3, Mmax = 1e7):
    ms_peak_idx  = np.argmax(ms)
    ms_peak = ms[ms_peak_idx]
    gc_peak_mass = 10**(lognorm_hurdle(ms_peak))
    gc_mass_range = np.logspace(np.log10(Mmin), np.log10(Mmax))
    
    def r(M):
        num = integrate.quad(MilkyWayGCMF, Mmin, M)[0]
        den = integrate.quad(MilkyWayGCMF, Mmin, Mmax)[0]
        return num / den
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interpolate.interp1d(cdf, gc_mass_range, kind="linear")
    
    accumulated_mass  = [inv_cdf(np.random.uniform(0, 1))]
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

    return accumulated_mass, gc_mass_range


#########################################################################################################################################################


def DwarfGCMF(mass, M_mean=-7., M_sigma=0.7):
    
    # get magnitude from mass
    
    mag = 5.03 - 2.5 * np.log10(mass)
    gclf_value = norm.pdf(mag, loc=M_mean, scale=M_sigma)
    
    return gclf_value
    
def sampleDwarfGCMF(ms):
    # This will only work for dwarfs between 5.5 and 8.5 log star mass
    
    gc_peak_mass = 10**(lognorm_hurdle(ms))
    
    min_gband = -5.5
    max_gband = -9.5
    min_mass = mass_light_ratio * 10**(0.4*(5.03 - max_gband))
    max_mass = mass_light_ratio * 10**(0.4*(5.03 - min_gband))
    
    gc_mass_range = np.logspace(np.log10(min_mass), np.log10(max_mass))
    
    def r(M):
        num = integrate.quad(DwarfGCMF, min_mass, M)[0]
        den = integrate.quad(DwarfGCMF, min_mass, max_mass)[0]
        return num / den
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interpolate.interp1d(cdf, gc_mass_range, kind="linear")
    
    accumulated_mass  = [inv_cdf(np.random.uniform(0, 1))]
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

    return accumulated_mass, gc_mass_range

def assignParticles(mp, mgc, bindEnergy=None):
    prob = mp / np.sum(mp)
    
    if bindEnergy is not None:
        bound_mask = bindEnergy < 0
        prob[~bound_mask] = 0 # set unbound stars to zero prob
    
    tag_idx  = np.random.choice(np.arange(len(prob)), size=len(mgc), replace=False, p=prob)
    return tag_idx

def distancesToParticle(pos, tag_id):
    pos_tag = pos[tag_id]
    transformed_pos = pos - pos_tag
    dist = (transformed_pos[:, 0]**2 + transformed_pos[:, 1]**2 + transformed_pos[:, 2]**2)**(1/2)
    return dist

def getTidalFrequency(mvir, rtidal):
    
    alpha = (c.G * (mvir * u.Msun) / (rtidal * u.kpc)**3).to(u.s**(-2))
    print(alpha)
    return (alpha.value/3)**(1/2)

def getTidalTimescale(gc_mass, mvir, rtidal):
    tidal_frequency = getTidalFrequency(mvir, rtidal) / u.s
    
    t_tid = 10 * u.Gyr * (gc_mass / 2e5)**(2/3) / (tidal_frequency / (100 * u.Gyr**(-1)))
    
    t_tid = (t_tid.to(u.Gyr)).value
    
    return t_tid

def getMassLossRate(gc_mass, mvir, rtidal):
    # Msun / Gyr
    return -gc_mass / getTidalTimescale(gc_mass, mvir, rtidal)


# def getPotentials(pos, ok, params):
#     rmax, vmax, PE, order = symlib.profile_info(params, pos[ok])
#     pot = PE * (vmax**2)
#     return pot

