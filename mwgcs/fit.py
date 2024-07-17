import astropy.constants as c
import astropy.units as u
import numpy as np
import random
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma, gammaincc
import gravitree
import jax
import jax.numpy as jnp
import jax.scipy as jsc

from gala.potential import PlummerPotential
from gala.units import galactic

class NFW():
    """
    This is a Profile class for an NFW profile
    """
    
    def __init__(self, mvir, rvir, c):
        
        self.type = "nfw"
        self.mvir = mvir
        self.rvir = rvir
        self.c = c        
        self.Rs = self.rvir / self.c
        
        self.scaleDensity = (
            self.mvir / \
            (4 * np.pi * (self.Rs)**3 * \
             (np.log(1+self.c) - (self.c/(1+self.c))))).value
        
    def mass(self, r):
        return 4 * np.pi * self.scaleDensity * self.Rs**3 * (np.log((self.Rs + r) / self.Rs) - (r / (self.Rs + r)))
    
    def density(self, r):
        return self.scaleDensity / ((r/self.Rs) * (1+ (r/self.Rs))**2)
    
    def analyticSlope(self, r):
        _m = np.vectorize(self.mass)
        m = _m(r)
        return 4 * np.pi * (r**3) * self.density(r) / m
    
    def analyticPotential(self, r):
        return -(4 * np.pi * c.G * self.scaleDensity * self.Rs**3 / r) * np.log(1 + (r / self.Rs))

class Einasto():
    """
    This is a Profile class that holds all the helper functions for the fit
    """
    def __init__(self, alpha, scaleRadius, logScaleDensity):
        """
        Sets initial fitting parameters.

        Parameters
        ----------
        Rs : float (optional, default=1)
            Scale radius of the Einasto profile in kpc
        
        logScaleDensity : float (optional, default=0)
            Log of the density at Rs in Msun/kpc^3
        
        alpha : float (optional, default=0.18)
            Shape parameter of the Einasto profile
        """

        self.alpha = alpha
        self.Rs = scaleRadius
        self.logScaleDensity = logScaleDensity

    def A(self):
        """
        Assuming an Einasto profile, the radius at which the density curve is maximal is A times the scale radius Rs.

        Returns
        -------
        float
            A parameter for the Einasto profile
        """
        return 1.715 * (self.alpha**(-.00183)) * (self.alpha + 0.0817)**(-.179488)
        
    def density(self, r, log=False):
        """
        Analytic form of the density profile

        Parameters
        ----------
        r : float
            Radius at which to evaluate the density profile
        
        Rs : float (optional, default=None)
            Scale radius of the Einasto profile
        
        logScaleDensity : float (optional, default=None)
            Log of the density at Rs
        
        alpha : float (optional, default=None)
            Shape parameter of the Einasto profile

        Returns
        -------
        float
            Density at radius r
        """
        
        Rmax = self.A() * self.Rs
        scaleDensity = 10**self.logScaleDensity
        rho = scaleDensity * np.exp(-(2 / self.alpha) * (((self.A() * r) / Rmax)**(self.alpha) - 1))

        if log:
            return np.log10(rho)
        
        return rho
    
    def mass(self, r):
        """
        Enclosed mass of the profile at radius r
        """
        density_integrand = lambda r: self.density(r) * 4 * np.pi * r**2
        m = quad(density_integrand, 0, r)[0]
        return m
    
    def circularVelocity(self, r):
        """
        Returns circular velocity at radius r in kpc
        """
        _g = c.G.to(u.km**3 / u.Msun / u.s**2).value
        kpc_to_km = 3.086e16

        mass = self.mass(r)
        vel = np.sqrt(_g * mass / (r * kpc_to_km))
        return vel
    
    def massSlope(self, r):
        """
        Returns the mass slope given an array of radii(dlnm/dlnr)

        Parameters
        ----------
        radii : array
            Array of radii at which to evaluate the mass slope
        
        Rs : float (optional, default=None)
            Scale radius of the Einasto profile
        
        logrho2 : float (optional, default=None)
            Log of the density at Rs
        
        alpha : float (optional, default=None)
            Shape parameter of the Einasto profile

        Returns
        -------
        interp1d
            Interpolated function of the mass slope
        """
        # vectorize mass function
        _m = np.vectorize(self.mass)

        m = _m(r)
        dlnr = np.gradient(r, edge_order=2) / r
        dlnm = np.gradient(m, edge_order=2) / m

        mass_profile = interp1d(r, dlnm/dlnr)

        return mass_profile
    
    def analyticSlope(self, r):
        """
        Returns the mass slope given
        ----------        
        radii : array
            Array of radii at which to evaluate the mass slope
        """

        _m = np.vectorize(self.mass)
        _rho = np.vectorize(self.density)

        m = _m(r)
        rho = _rho(r)        

        return 4 * np.pi * (r**2) * rho * (r / m)
    
    def potential(self, q, MW=True):
        """
        Returns the potential at q = [x, y, z]
        """

        r = jnp.sqrt(jnp.sum(q[0]**2 + q[1]**2 + q[2]**2))
        _a = self.alpha
        _g = c.G.to(u.kpc**3 / u.Gyr**2 / u.Msun).value
        _tilde = (_a * self.Rs**_a / 2)**_a
        scaleDensity = 10**self.logScaleDensity

        def lowerIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammainc(a, x) * jsc.special.gamma(a)

            if tilde:
                return base * _tilde
            else:
                return base
        
        def upperIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammaincc(a, x) * jsc.special.gamma(a)

            if tilde:
                return base * _tilde
            else:
                return base

        _sr = 2 * r**_a / (_a * self.Rs**_a)

        tmp1 = 4*np.pi*_g*scaleDensity*jnp.exp(2/_a) / _a
        tmp2 = (1/r) * lowerIncompleteGamma(3/_a, _sr, tilde=True)
        tmp3 = upperIncompleteGamma(2/_a, _sr, tilde=True)


        if MW:
            _pot = PlummerPotential(m=4e10*u.Msun, b=1.6*u.kpc, units=galactic)
            return _pot.value(q) + tmp1 * (tmp2 + tmp3)
        else:
            return tmp1 * (tmp2 + tmp3)
    
    def tidalTensor(self, q):
        """
        Returns the tidal tensor at position q = [x, y, z]
        """
        hessian = jax.hessian(self.potential, argnums=0)(q)
        tt = -(1/3) * jnp.trace(hessian) * jnp.identity(3) + hessian
        return tt
    
    def tidalStrength(self, q):
        """
        Returns the tidal strength at position q = [x, y, z]
        """
        lam = np.max(np.abs(jnp.linalg.eigvals(self.tidalTensor(q))))
        return lam

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
        
        self.type = "ein"
        self.pos = p[sh_id]['x']
        self.vel = p[sh_id]['v']
        self.sh_pos = sim.rs[sh_id, snap]['x']
        self.sh_vel = sim.rs[sh_id, snap]['v']
        self.rvir = sim.rs[sh_id, snap]['rvir']
        
        if not sim.sf[sh_id, snap]['ok_rs']:
            self.sh_pos = sim.rs[sh_id, snap]['x']
            self.sh_vel = sim.rs[sh_id, snap]['v']
        
        self.rvir = sim.rs[sh_id, snap]['rvir']
        self.eps = sim.params['eps'] / sim.params['h100']
        self.h100 = sim.params['h100']
        self.mp = sim.params['mp'] / sim.params['h100']
        
        self.bins_ok = False
        
        # This is bugged right now, need to fix ################

        part_limit_fit = 25
        resample_counter = 0
       
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

        ########################################################
                
        self.dist = np.sqrt(np.sum((self.pos-self.sh_pos)**2, axis=1))
        self.r_conv = np.max(sim.getConvergenceRadius(snap))
        self.profile = Einasto(.18, 10, 6)
        
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
    
    def enclosedDensity(self, r, dr=.001):
        vol = (4*np.pi / 3) * (r**3 - (r-dr)**3)
        counts = 0
        
        if self.boundOnly:
            counts = np.sum((self.dist < r) & (self.dist > (r-dr)) & self.bound)
        else:
            counts = np.sum((self.dist < r) & (self.dist > (r-dr)))
        
        mass = counts * self.mp
        
        rho = mass / vol
        
        return rho
    
    def circularVelocity(self, r):
        _g = c.G.to(u.km**3 / u.Msun / u.s**2).value
        kpc_to_km = 3.086e16
        mass = self.mass(r)
        vel = np.sqrt(_g * mass / (r * kpc_to_km))
        return vel
    
    def density(self, bins=50):
        
        # Select particles within the virial radius
        mask = self.dist < self.rvir
        
        # Edge case is when there are less particles than bins desired
        low_res = np.sum(mask) < bins
        
        if low_res: 
            bins = np.sum(mask)
            print("Subhalo has insufficient particle count, rebinning with ", bins, " bins")
        
        _d = self.dist[mask] # only consider particles within rvir

        # bin particles by radius
        sorted_indices = np.argsort(_d) # sort by radius
        bin_idx = np.array_split(sorted_indices, bins) # split index bins roughly evenly
        bin_edges = np.concatenate(
            [
                [0],
                [_d[i[-1]] for i in bin_idx] # rightmost bin edge
            ]
        )
        bin_volume = (4*np.pi / 3) * ((bin_edges[1:])**3 - (bin_edges[:-1])**3)
        
        self.bin_idx = bin_idx
        self.bin_edges = bin_edges
        self.bin_volume = bin_volume
        self.bin_count = np.array([
            np.sum(
                (_d >= bin_edges[i]) & (_d < bin_edges[i+1])) for i in range(len(bin_edges)-1)])
        
        if low_res:
            self.bin_count = np.ones(bins)
        
        # calculate the density in each bin
        bin_density = np.array(self.mp * self.bin_count / self.bin_volume)

        # get average radius in each bin
        _br = np.array_split(_d[sorted_indices], bins) # split particles up by their sorted positions
        bin_radius = np.array([np.mean(i) for i in _br]) # get the mean of each bin, this is the radius used in the fit
        
        return bin_radius, bin_density

    def fit(self):
        r, rho = self.density() # fit the desnity profile based on rockstar binning
        
        # mask out particles within the convergence radius
        mask = r > self.r_conv

        # fit the density profile 
        
        self.bins = np.array(r)[mask]
        
        self.logdata = np.log10(np.array(rho)[mask])

        popt, pcov = curve_fit(
                    self.profile.density,
                    self.bins,
                    self.logdata,
                    p0=[.18, 20, .6], # some random values
                    maxfev = 10000
                    )
        
        self.profile = Einasto(*popt)
        
        self.profile_params = popt
    
    def massSlope(self, r):
        _m = np.vectorize(self.mass)
        m = _m(r)

        dlnr = np.gradient(r, edge_order=2) / r
        dlnm = np.gradient(m, edge_order=2) / m

        slope = interp1d(r, dlnm/dlnr)

        return slope