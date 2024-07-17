import asdf
import astropy.constants as c
import astropy.units as u
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
import symlib

from .sym import Simulation
from .form import lognorm_hurdle, sampleMilkyWayGCMF, sampleDwarfGCMF
from .evolve import getMassLossRate, getTidalTimescale, getTidalFrequency
from .fit import MassProfile

def spawnSubhaloGC(ms):
    mgc = lognorm_hurdle(ms)
    gcm, _ = sampleMilkyWayGCMF(ms)
    
class GCSystem():
    def __init__(self, sim: Simulation, rsid: int):
        self.sim = sim
        self.rsid = rsid

        self.infall_snap = self.sim.hist[rsid]["first_infall_snap"]

        if self.infall_snap == -1:
            raise ValueError("Subhalo has not infallen, no tracking possible (at this point)...")
        
        # get stellar mass of the system
        ok_rs = self.sim.rs['ok'][self.rsid]
        
        # add max_snap variable
        snaps = np.arange(236)[ok_rs]
        times = sim.cosmo.hubbleTime(sim.z)
        times = times[snaps]
        self.dt = times[1:] - times[:-1]
        self.snaps = snaps[snaps >= self.infall_snap]

        self.ms_infall = self.sim.um["m_star"][self.infall_snap, ok_rs][self.rsid]
        self.mgc = lognorm_hurdle(self.ms_infall)# system mass
        self.gcm, self.gc_mass_range = sampleDwarfGCMF(self.mgc)
        
        self.gcm = np.array(self.gcm)
        self.gc_mass_range = np.array(self.gc_mass_range)

        stars, gals, ranks = symlib.tag_stars(sim.sim_dir, sim.gal_halo, target_subs=[self.rsid])

        # this is the z=0 distribution of stars... 
        # we can choose a PDF based on empirical models in a later update
        prob = stars[self.rsid]['mp'] / np.sum(stars[self.rsid]['mp'])

        tag_idx  = np.random.choice(
            np.arange(len(prob)),
            size = len(self.gcm),
            replace = False,
            p = prob)
        
        # tag_idx is the particle index
        self.tag_idx = tag_idx
        self.profile = MassProfile(self.sim, self.infall_snap, self.rsid)
        _ = self.profile.fit()

    def evolve(self, index, snap=None):
        # index here is the index of the tag_idx array
        if snap is None:
            snap = self.sim.buffer_snap
        
        self.t = np.zeros(len(self.snaps)+1)
        self.r = np.zeros(len(self.snaps)+1)
        self.lam = np.zeros(len(self.snaps)+1)
        
        self.evolved_mass = np.zeros(len(self.snaps)+1) + self.gcm[index]
        
        for i, sn in tqdm(enumerate(self.snaps[1:])):
            _particles = self.sim.getParticles(sn)[self.rsid]
            pos = _particles["x"][self.tag_idx[index]]
            r = np.sqrt(np.sum((pos - self.sim.rs[self.rsid, sn]['x'])**2))
            m = self.evolved_mass[i-1]
            self.profile = MassProfile(self.sim, sn, self.rsid)
            self.profile.fit()
            
            lam = self.profile.profile.tidalStrength([r, 0, 0])

            mass_loss = getMassLossRate(m, lam) * self.dt[i]
            freq = getTidalFrequency(lam)
            timescale = getTidalTimescale(m, lam)
            
            self.evolved_mass[i] = self.gcm[index] + mass_loss
            self.t[i] += self.dt[i]
            self.r[i] = r
            self.lam[i] = lam

        print("done")