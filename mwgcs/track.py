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

from .sym import Simulation
from .form import lognorm_hurdle, sampleMilkyWayGCMF, sampleDwarfGCMF
from .evolve import tidalRadius, getMassLossRate
from .fit import MassProfile

def spawnSubhaloGC(ms):
    mgc = lognorm_hurdle(ms)
    gcm, _ = sampleMilkyWayGCMF(ms)
    
class GCSystem():
    def __init__(self, sim: Simulation, rsid: int):
        self.sim = sim
        self.rsid = rsid

        self.infall_snap = sim.rs[sim.rs["id"] == rsid]["first_infall_snap"]

        if self.infall_snap == -1:
            raise ValueError("Subhalo has not infallen, no tracking possible (at this point)...")
        
        # get stellar mass of the system
        ok_rs = self.rs['ok'][self.rs_id]

        self.ms_infall = self.sim.um["m_star"][self.rsid, ok_rs][self.infall_snap]
        self.mgc = lognorm_hurdle(self.ms_infall)# system mass
        self.gcm, self.gc_mass_range = sampleDwarfGCMF(self.mgc)
        
        self.gcm = np.array(self.gcm)
        self.gc_mass_range = np.array(self.gc_mass_range)

        stars, gals, ranks = symlib.tag_stars(sim.sim_dir, sim.gal_halo, target_subs=[self.rsid])

        prob = stars[self.rsid] / np.sum(stars[self.rsid])

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
        _particles = self.sim.getParticles(snap)

        pos = _particles["x"][self.tag_idx[index]]
        _dist = np.sqrt(np.sum((pos - self.sim.rs[self.rsid, snap])**2, axis=1))
        _mass = self.gcm[index]
        r_tidal = tidalRadius(_dist, _mass, self.profile)
        _mvir = self.sim.rs[self.rsid, snap]["mvir"]
        mass_loss = getMassLossRate(_mass, _mvir, r_tidal)
        self.gcm[index] += mass_loss
        return self.gcm[index]