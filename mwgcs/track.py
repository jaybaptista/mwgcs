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
from .evolve import tidalRadius, getMassLossRate, getTidalTimescale, getTidalFrequency
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

        prob = stars[self.rsid]['mp'] / np.sum(stars[self.rsid]['mp'])

        tag_idx  = np.random.choice(
            np.arange(len(prob)),
            size = len(self.gcm),
            replace = False,
            p = prob)
        
        # tag_idx is the particle index
        self.tag_idx = tag_idx
        self.profile = MassProfile(self.sim, self.infall_snap, self.rsid)

    def evolve(self, index, snap=None):
        # index here is the index of the tag_idx array
        if snap is None:
            snap = self.sim.buffer_snap
        
        self.evolved_mass = np.zeros(len(self.snaps)+1) + self.gcm[index]
        self.tidal_radii = np.zeros(len(self.snaps)+1)
        
        for i, sn in tqdm(enumerate(self.snaps[1:])):
            _particles = self.sim.getParticles(sn)[self.rsid]
            pos = _particles["x"][self.tag_idx[index]]
            _dist = np.sqrt(np.sum((pos - self.sim.rs[self.rsid, sn]['x'])**2))
            _mass = self.evolved_mass[i-1]
            self.profile = MassProfile(self.sim, sn, self.rsid)
            
            r_tidal = tidalRadius(_dist, _mass, self.profile)
            _mvir = self.sim.rs[self.rsid, snap]["m"]
            
            mass_loss = getMassLossRate(_mass, _mvir, r_tidal) * self.dt[i]
            freq = getTidalFrequency(_mvir, r_tidal)
            timescale = getTidalTimescale(_mass, _mvir, r_tidal)
            
            # self.tidal_radii[i] = tidalRadius(_dist, _mass, self.profile)
            # print("dist [kpc]:", _dist)
            # print("mass lost [Msun]:", mass_loss)
            # print("time elapsed [Gyr]: ", self.dt[i])
            # print("tidal freq [Hz]: ", freq)
            # print("tidal timescale [Gyr]:", timescale)
            # print("tidal radius [kpc]:", r_tidal)
            # self.evolved_mass[i] = self.gcm[index] + mass_loss

        return masses