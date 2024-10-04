import asdf
import astropy.constants as c
import astropy.units as u
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
import symlib
import os

class Simulation():
    """
    This is just a helper class to get what I want from the simulation
    """
    def __init__(self, sim_dir, seed=96761):
        """
        Initialize the simulation class

        Parameters

        sim_dir : str
            The directory of the simulation
        seed : int
            The seed for the random number generator
        """
        # Set the seed
        random.seed(seed) 

        # Set the simulation directory
        self.sim_dir = sim_dir

        # Get the simulation parameters
        self.params = symlib.simulation_parameters(sim_dir)
        self.scale_factors = np.array(symlib.scale_factors(sim_dir))
        self.col_params = symlib.colossus_parameters(self.params)
        self.cosmo = cosmology.setCosmology("cosmo", params=self.col_params)
        self.z = (1/self.scale_factors) - 1

        # Get the particle class
        self.partClass = symlib.Particles(sim_dir)
        
        # Get the rockstar and symfind data
        self.rs, self.hist = symlib.read_rockstar(self.sim_dir)
        self.sf, self.shist = symlib.read_symfind(self.sim_dir)
        self.um = symlib.read_um(self.sim_dir)
        
        # Get the galaxy halo model for star tagging
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
        
        # Set up particle cache for speedup
        self.buffer_snap = 235
        self.particles = None

    def getSimulationName(self):
        return os.path.split(self.sim_dir)[-1]

    
    def getRedshift(self, snapshot):
        """
        Get the redshift of a snapshot

        Parameters

        snapshot : int
            The snapshot number
        
        Returns

        z : float
            The redshift of the snapshot
        """
        return self.z[snapshot]
    
    def getScaleFactor(self, snapshot):
        """
        Get the scale factor of a snapshot

        Parameters

        snapshot : int
            The snapshot number
        
        Returns

        a : float
            The scale factor of the snapshot
        """
        return self.scale_factors[snapshot]
        
    def getParticles(self, snap, mode="all"):
        """
        Get the particles at a given snapshot

        Parameters

        snap : int
            The snapshot number

        mode : str
            The mode to read the particles.
        
        Returns

        particles : object
            The particles at the snapshot
        """
        
        # If the snapshot is different from what is stored in cache,
        # or if the cache is empty, read the particles
        if (self.buffer_snap != snap) or (self.particles is None):
            self.buffer_snap = snap
            self.particles = self.partClass.read(snap, mode=mode)
        
        return self.particles
    
    def getConvergenceRadius(self, snapshot):
        """
        Get the convergence radius of a snapshot

        Parameters

        snapshot : int
            The snapshot number

        Returns

        r_conv : float
            The convergence radius of the snapshot
        """
        
        # Calculate convergence radius
        a = self.getScaleFactor(snapshot)
        
        # Get the Hubble constant
        H0 = (100 * self.params['h100'] * u.km / u.s / u.Mpc).decompose().value
        
        # Get the present day critical density 
        rho_crit = (3 / (8 * np.pi * c.G.to(u.kpc**3 / u.Msun / u.s**2).value)) * (H0)**2 
                
        # Get the present day matter density
        rho_m = self.params['Om0'] * rho_crit
        
        # Get the mean interparticle spacing
        l_0 = (self.params['mp'] / self.params['h100'] / rho_m)**(1/3)
        
        z = self.getRedshift(snapshot)      

        # Convert to physical units
        l = a * l_0 

        # Return the convergence radius
        return (5.5e-2 * l, 3 * self.params['eps']/self.params['h100'] * a)
