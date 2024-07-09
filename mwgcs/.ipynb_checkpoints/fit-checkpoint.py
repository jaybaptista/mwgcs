import astropy.constants as c
import astropy.units as u
import numpy as np
import random
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.special import gamma, gammaincc
import gravitree

class NFW():
    """
    This is a Profile class for an NFW profile
    """
    
    def __init__(self, mvir, rvir, conc):
        
        self.type = "nfw"
        
        # set up parameters
        self.mvir = mvir
        self.rvir = rvir
        self.conc = conc
        
        # calculate aux params
        
        self.Rs = self.rvir / self.conc
        
        # units of density is Msun/kpc3
        self.rho0 = (
            self.mvir * u.Msun / \
            (4 * np.pi * (self.Rs * u.kpc)**3 * \
             (np.log(1+self.conc) - (self.conc/(1+self.conc))))).value
        
    def mass(self, r):
        return 4 * np.pi * self.rho0 * self.Rs**3 * (np.log((self.Rs + r) / self.Rs) - (r / (self.Rs + r)))
    
    def density(self, r):
        return self.rho0 / ((r/self.Rs) * (1+ (r/self.Rs))**2)
    
    def analyticSlope(self, r):
        _mass = self.mass(r)
        return 4 * np.pi * (r**3) * self.density(r) / _mass
    
    def analyticPotential(self, r):
        return - (4 * np.pi * c.G * self.rho0 * (u.Msun / u.kpc**3) * (self.Rs * u.kpc)**3 / r) * np.log(1 + (r / (self.Rs * u.kpc)).decompose().value)
    
    def getTidalTensor(self, r):
        # REPLACE THIS WITH GALA'S HESSIAN (mathematica one here is wrong)
        # !!!
        #
        #
        
        # a = - c.G * self.mvir / (np.log(1+self.conc) - self.conc/(1+self.conc))
        # H_rr =  (a/r**3) * (((r * (3*r + 2 * self.Rs))/(r+self.Rs)**2) - 2*np.log(((r+self.Rs) / self.Rs).value))
        
        H_rr = 4 * np.pi * c.G * self.rho0 * (self.Rs * u.kpc)**3 * (u.Msun / u.kpc**3) * (r*(3*r + 2 * (self.Rs * u.kpc)) - 2. * (r + (self.Rs * u.kpc))**2 * np.log(((r + (self.Rs * u.kpc))/(self.Rs * u.kpc)).value)) / (r**3 * (r + (self.Rs * u.kpc))**2)
        
        H_rr = H_rr.to(u.Gyr**(-2)).value
        H_ij = np.diag(np.array([H_rr, 0, 0]))
        tensor = (-(1/3)*np.trace(H_ij) * np.identity(3) + H_ij) * (u.Gyr**(-2))
        return tensor

class Einasto():
    """
    This is a Profile class that holds all the helper functions for the fit
    """
    def __init__(self, Rs=None, logrho2=None, alpha=None):
        """
        Sets initial fitting parameters

        Parameters
        ----------
        Rs : float (optional, default=1)
            Scale radius of the Einasto profile
        
        logrho2 : float (optional, default=0)
            Log of the density at Rs
        
        alpha : float (optional, default=0.18)
            Shape parameter of the Einasto profile
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

        Returns
        -------
        float
            A parameter for the Einasto profile
        """
        return 1.715 * (self.alpha**(-.00183)) * (self.alpha + 0.0817)**(-.179488)
        
    def density(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Analytic form of the density profile

        Parameters
        ----------
        r : float
            Radius at which to evaluate the density profile
        
        Rs : float (optional, default=None)
            Scale radius of the Einasto profile
        
        logrho2 : float (optional, default=None)
            Log of the density at Rs
        
        alpha : float (optional, default=None)
            Shape parameter of the Einasto profile

        Returns
        -------
        float
            Density at radius r
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

        Parameters
        ----------
        r : float
            Radius at which to evaluate the density profile
        
        Rs : float (optional, default=None)
            Scale radius of the Einasto profile
        
        logrho2 : float (optional, default=None)
            Log of the density at Rs
        
        alpha : float (optional, default=None)
            Shape parameter of the Einasto profile
        
        Returns
        -------
        float
            Log of the density at radius r
        """
        return np.log10(self.density(r, Rs, logrho2, alpha))

    def density_integrand(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Integrand used to compute enclosed mass of the profile
        """
        return self.density(r, Rs, logrho2, alpha) * 4 * np.pi * r**2
    
    def mass(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Enclosed mass of the profile at radius r
        """
        m = quad(self.density_integrand, 0, r, (Rs, logrho2, alpha))[0]
        return m
    
    def v_circ(self, r, Rs=None, logrho2=None, alpha=None):
        """
        Returns circular velocity at radius r
        """
        mass = self.mass(r, Rs, logrho2, alpha) * u.Msun
        vel = ((c.G * mass / (r * u.kpc))**(1/2)).to(u.km / u.s).value
        return vel
    
    def massSlope(self, radii, Rs=None, logrho2=None, alpha=None):
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
        mass = [self.mass(r_i, Rs, logrho2, alpha) for r_i in radii]
        dlnr = np.gradient(radii, edge_order=2) / radii
        dlnm = np.gradient(mass, edge_order=2) / mass
        mass_profile = interp1d(radii, dlnm/dlnr)
        return mass_profile
    
    def analyticSlope(self, radii, Rs=None, logrho2=None, alpha=None):
        mass = [self.mass(r_i, Rs, logrho2, alpha) for r_i in radii]
        
        return [self.density_integrand(r_i, Rs, logrho2, alpha) for r_i in radii] * radii / mass
    
    def getVmax(self):
        """
        Returns the maximum circular velocity of the Einasto profile
        """

        radii = 10**np.linspace(-.5, 2.5)
        velocities = [self.v_circ(r) for r in radii]
        return np.max(velocities)
    
    def analyticPotential(self, r, m=1e12 * u.Msun):
        
        dn = 2 / self.alpha
        n = 1 / self.alpha
        s = (dn)**n * (r / self.default_Rs)
        h = self.default_Rs * u.kpc / (dn)**n
        
        t1 = gammaincc(3*n, s**(1/n))
        t2 = gammaincc(2*n, s**(1/n)) * s * (gamma(2*n) / gamma(3*n))
        
        return (c.G * m / (h * s)) * (1 - t1 + t2)
    
    def getHessianRR(self, r):
        r = r * u.kpc
        dn = 2 / self.alpha
        n = 1 / self.alpha
        s = ((dn)**n * (r / (self.default_Rs * u.kpc))).decompose().value
        rs = self.default_Rs * u.kpc
        
        halo_mass = self.mass(1e3) * u.Msun # sorry phil this is just a messy way of getting the halo mass
        
        _a = (c.G * halo_mass / r**3) 
        _b = np.exp(-s**(1/n)) * (s**2) * (dn**n * r * (-n + s**(1/n)) - (s**(1/n)**(1+n)) * rs)
        _c = 2 * n**2 * rs * gammaincc(3*n, s**(1/n))
        _d = n**2 * rs * gamma(3 * n)
        
        H_rr = _a * (- 2 + (_b + _c)/_d)
        H_rr = H_rr.to(u.Gyr**(-2)).value
        
        return H_rr
    
    # create M(r) -> Phi(r)
    # 
    
    
    def analyticTidalTensor(self, r):

        H_rr = self.getHessianRR(r.value)
        
        H_ij = np.diag(np.array([H_rr, 0, 0]))
        tensor = (-(1/3)*np.trace(H_ij) * np.identity(3) + H_ij) * (u.Gyr**(-2))
        return tensor
        
    
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

        ########################################################
                
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
        
        _dist = self.dist[mask] # only consider particles within rvir

        indices = np.argsort(_dist) # sort by radius
        idx_bins = np.array_split(indices, bins) # split index bins roughly evenly
        bin_edges = np.concatenate(
            [
                [0],
                [_dist[i[-1]] for i in idx_bins] # rightmost bin edge
            ]
        )
        bin_volume = (4*np.pi / 3) * ((bin_edges[1:])**3 - (bin_edges[:-1])**3)
        
        self._rs_idx_bins = idx_bins
        self._rs_bin_edges = bin_edges
        self._rs_bin_volume = bin_volume
        self._rs_bin_count = np.array([
            np.sum(
                (_dist >= bin_edges[i]) & (_dist < bin_edges[i+1])) for i in range(len(bin_edges)-1)])
        
        if low_res:
            self._rs_bin_count = np.ones(bins)
        
        # calculate the density in each bin
        bin_density = np.array(self.mp * self._rs_bin_count / bin_volume)

        # get average radius in each bin
        rad_bins = np.array_split(_dist[indices], bins) # split particles up by their sorted positions
        avg_rad = np.array([np.mean(i) for i in rad_bins]) # get the mean of each bin, this is the radius used in the fit
        
        return avg_rad, bin_density

    def fit(self, ein=None):
        if ein is None:
            ein = Einasto()
        self._ein = ein # remove this after debugging
        radii, density_profile = self.density_rs() # fit the desnity profile based on rockstar binning
        
        # mask out the convergence radius
        radial_mask = radii > self.r_conv

        # fit the density profile 
        self.bins = np.array(radii)[radial_mask]
        self.logdata = np.log10(np.array(density_profile)[radial_mask])

        popt, pcov = curve_fit(ein.logdensity,
                       self.bins,
                       self.logdata,
                       p0=[20, .6, .18], # some random values
                       maxfev = 10000
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
    
    
