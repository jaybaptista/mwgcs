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
from tqdm import tqdm

import os

from .sym import Simulation
from .form import lognorm_hurdle, sampleMilkyWayGCMF, sampleDwarfGCMF
from .evolve import getMassLossRate, getTidalTimescale, getTidalFrequency
from .fit import MassProfile

# lookup tables

class EinastoLookupTable():
    def __init__(self, sim : Simulation, ):
        self.sim = sim
        self.df = asdf.AsdfFile()
        self.scale = 1.5
        self.n_sh = self.sim.rs.shape[0]
    
    def createLookupTable(self, write_dir):
        
        self.alpha = np.zeros(self.sim.rs.shape) - 1.
        self.rs = np.zeros(self.sim.rs.shape) - 1.
        self.logrho = np.zeros(self.sim.rs.shape) - 99.
        
        for k in tqdm(range(self.n_sh)):
            ok_snaps = np.where(self.sim.rs[k, :]['ok'])[0]
            for sn in tqdm(ok_snaps):
                prof = MassProfile(self.sim, sn, k)
                
                alpha, rs, logrho = np.nan, np.nan, np.nan
                
                try:
                    prof.fit()
                    alpha, rs, logrho = prof.profile_params
                except ValueError:
                    print("Error: possible infinite residual, ignoring fit.")
                
                self.alpha[k, sn] = alpha
                self.rs[k, sn] = rs
                self.logrho[k, sn] = logrho

        self.df['alpha'] = self.alpha
        self.df['rs'] = self.rs
        self.df['logrho'] = self.logrho

        self.df.write_to(os.path.join(write_dir , f'einasto_params_{self.sim.getSimulationName()}.asdf'))