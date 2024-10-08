import abc
import astropy.constants as c
import astropy.units as u
import numpy as np
import random
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma, gammaincc
from scipy.stats import norm
import gravitree
import jax
import jax.numpy as jnp
import jax.scipy as jsc

from gala.potential import PlummerPotential
from gala.units import galactic


########################################################################################################

class SphericalHaloProfile(abc.ABC):

    def __init__(self, q, q_sh, mp, rvir, a=1.0, **kwargs):
        # particle positions with shape [3, N_particles] (Cartesian)
        self.q = q 
        
        # position of the subhalo
        self.q_sh = q_sh

        # particle mass
        self.mp = mp

        # virial radius of halo
        self.rvir = rvir

        # distances to center
        self.r = np.sqrt(np.sum((self.q - self.q_sh)**2, axis=1))

        self.a = a

    def particle_count(self, r):
        return np.sum(self.r <= r)
        
    def menc(self, r):
        return self.mp * self.particle_count(r)
        
    def getDensityProfile(self, bins=50):
        # Select particles within the virial radius
        mask = self.r < self.rvir
        _r = self.r[mask] # only consider particles within rvir
       
        # Edge case is when there are less particles than bins desired
        low_res = np.sum(mask) < bins
        
        if low_res: 
            bins = np.sum(mask)
            print("Subhalo has insufficient particle count, rebinning with ", bins, " bins")

        # bin particles by radius
        sorted_indices = np.argsort(_r) # sort by radius
        
        bin_idx = np.array_split(sorted_indices, bins) # split index bins roughly evenly
        
        bin_edges = np.concatenate(
            [
                [0],
                [_r[i[-1]] for i in bin_idx] # rightmost bin edge
            ]
        )
        
        bin_volume = (4*np.pi / 3) * ((bin_edges[1:])**3 - (bin_edges[:-1])**3)

        
        bin_count = np.array([
            np.sum((_r >= bin_edges[i]) & (_r < bin_edges[i+1])) \
            for i in range(len(bin_edges)-1)])
        
        if low_res:
            bin_count = np.ones(bins)
        
        # calculate the density in each bin
        bin_density = np.array(self.mp * bin_count / bin_volume)

        # get average radius in each bin
        # split particles up by their sorted positions
        _br = np.array_split(_r[sorted_indices], bins) 

        # get the mean of each bin, this is the radius used in the fit
        bin_radius = np.array([np.mean(i) for i in _br]) 
        
        return bin_radius, bin_density

    def getRadialAccelerationProfile(self, rvir, eps, bins=100):
        
        _menc = np.vectorize(self.menc)
        sampling_radii = np.logspace(-2, np.log10(5 * rvir), bins)
        
        mass = _menc(sampling_radii)
        _G = 1.3938323614347172e-22 # units of km * kpc2 / (Msun s2)

        # apply force softening to the bins that are located
        # within the softening radius
        softening = eps * self.a # multiply by a 
        
        # accelerations = _G * mass / (sampling_radii**2 + softening**2)
        accelerations = _G * mass / (sampling_radii**2)
        
        return sampling_radii, accelerations
        
    @abc.abstractmethod
    def fit(self, **kwargs):
        # This is where one would implement their fitting scheme
        # to obtain the parameters they want.
        self.params = {}
        
        return self.params

class SymphonyHaloProfile(SphericalHaloProfile):

    def einasto_log_density(self, r, alpha, Rs, logScaleDensity):
        A = 1.715 * (alpha**(-.00183)) * (alpha + 0.0817)**(-.179488)
        Rmax = A * Rs
        scaleDensity = 10**logScaleDensity
        rho = scaleDensity * np.exp(-(2 / alpha) * (((A * r) / Rmax)**(alpha) - 1))
        return np.log10(rho)

    def fit(self, r_conv, bins=50):
        
        radii, rho = self.getDensityProfile(bins)
    
        mask = radii > r_conv
        
        radii = self.bins = np.array(radii)[mask]
        logrho = np.log10(np.array(rho)[mask])

        popt, pcov = curve_fit(
                    self.einasto_log_density,
                    radii,
                    logrho,
                    p0=[.18, 10, 5], # some random values
                    bounds=((.01, .01, -2.), (.3, 100., 10.)), 
                    maxfev = 100000,
                    nan_policy='omit'
                    )
        alpha, Rs, logScaleDensity = popt
        
        self.params = {
            "alpha": alpha,
            "Rs": Rs,
            "logScaleDensity": logScaleDensity
        }

        return self.params
    
########################################################################################################


class Profile(abc.ABC):

    def __init__():
        pass

    # @abc.abstractmethod
    # def set_params(self, **kwargs):
    #     pass

    @abc.abstractmethod
    def mass(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def density(self, **kwargs):
        pass

    @abc.abstractmethod
    def potential(self, **kwargs):
        pass

    @abc.abstractmethod
    def hessian(self, **kwargs):
        pass

    def acceleration(self, r, softening = 0.):
        _menc = self.mass(r)
        _g = 1.3938323614347172e-22 # in units of kpc^2/Msun * (km/s^2)
        return _g * _menc / (r**2 + softening**2)

    @abc.abstractmethod
    def tidalStrength(self, **kwargs):
        pass

class Einasto(Profile):
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
        else:
            return rho

    def mass(self, r):
        """
        Enclosed mass of the profile at radius r
        """
        integ = lambda r: self.density(r) * 4 * np.pi * r**2
        menc = quad(integ, 0, r)[0]
        return menc

    def potential(self, q):
        """
        Returns the potential at q = [x, y, z]
        """

        r = (q[0]**2 + q[1]**2 + q[2]**2)**(1/2)

        if r == 0.:
            return 0.

        m = _m(r)
        rho = _rho(r)        

        return 4 * np.pi * (r**2) * rho * (r / m)
    
    def potential(self, q):
        """
        Returns the potential at q = [x, y, z]
        """


        _a = self.alpha
        # _g = 4.498502151469554e-06 # units of kpc^3/Gyr^2/Msun
        _g = 4.30091727e-06 # in units of kpc/Msun * (km/s)^2
        
        scaleDensity = 10**self.logScaleDensity

        def lowerIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammainc(a, x) * jsc.special.gamma(a)
            _tilde = (_a * self.Rs**_a / 2)**a
            
            if tilde:
                return base * _tilde
            else:
                return base
        
        def upperIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammaincc(a, x) * jsc.special.gamma(a)
            _tilde = (_a * self.Rs**_a / 2)**a
            if tilde:
                return base * _tilde
            else:
                return base

        _sr = 2 * r**_a / (_a * self.Rs**_a)

        tmp1 = 4*np.pi*_g*scaleDensity*jnp.exp(2/_a) / _a
        tmp2 = (1/r) * lowerIncompleteGamma(3/_a, _sr, tilde=True)
        tmp3 = upperIncompleteGamma(2/_a, _sr, tilde=True)

        return - tmp1 * (tmp2 + tmp3) 
        
    def hessian(self, r):
        # spherical symmetry
        q = np.array([r, 0., 0.])
        return jax.hessian(self.potential, argnums=0)(q)

    def tidalStrength(self, r):
        
        hess = self.hessian(r)
        
        tidal_tensor = - hess
        
        eigenvalues, eigenvectors = np.linalg.eig(tidal_tensor)

        l1 = eigenvalues[0] 
        l2 = eigenvalues[1]
        l3 = eigenvalues[2] # most negative eigenvalue? 
        omega = - (1/3) * (l1 + l2 + l3)

        return eigenvalues[0] + omega

class NFW(Profile):

    def __init__(self, mvir, rvir, cvir):
        self.mvir = mvir
        self.rvir = rvir
        self.cvir = cvir

        self.Rs = rvir / cvir
        self.rho0 = (mvir / (4 * np.pi * self.Rs**3)) / (jnp.log(1+cvir) - (cvir / (1+cvir)))

    def mass(self, r):
        """
        Enclosed mass of the profile at radius r
        """
        integ = lambda r: self.density(r) * 4 * np.pi * r**2
        menc = quad(integ, 0, r)[0]
        return menc
        
    def density(self, r):
        return (self.rho0) / ((r/self.Rs) * (1 + (r/self.Rs))**2)

    def potential(self, q):
        
        r = (q[0]**2 + q[1]**2 + q[2]**2)**(1/2)
        
        _G = 4.30091727e-06 # in units of kpc/Msun * (km/s)^2
        
        return - ((4 * np.pi * _G * self.rho0 * self.Rs**3) / r) * jnp.log(1 + (r/self.Rs))

    def hessian(self, r):
        # spherical symmetry
        
        q = np.array([r, 0., 0.])
        
        return jax.hessian(self.potential, argnums=0)(q)

    def tidalStrength(self, r):
        
        hess = self.hessian(r)
        
        tidal_tensor = - hess
        
        eigenvalues, eigenvectors = np.linalg.eig(tidal_tensor)

        l1 = eigenvalues[0] 
        l2 = eigenvalues[1]
        l3 = eigenvalues[2] # most negative eigenvalue? 
        omega = - (1/3) * (l1 + l2 + l3)

        return eigenvalues[0] + omega