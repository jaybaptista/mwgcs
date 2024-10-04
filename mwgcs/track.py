import abc
import asdf
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
import astropy.constants as c
import symlib

from .sym import Simulation
from .form import lognorm_hurdle, sampleMilkyWayGCMF, sampleDwarfGCMF
from .evolve import getMassLossRate, getTidalTimescale, getTidalFrequency
# from .fit import MassProfile

########################################################################################################

class TreeTracker(abc.ABC):

    def __init__(self, catalog_path):
        # Note that the catalog has have been tagged!
        self.cluster_catalog = asdf.open(catalog_path)

    @abc.abstractmethod
    def write_tracking_catalog(self, **kwargs):
        pass


class SymphonyTracker(TreeTracker):

    def __init__(self, catalog_path):
        print("Reading in tree for tracking")
        super().__init__(self, catalog_path)

    
    

########################################################################################################
# def spawnSubhaloGC(ms):
#     mgc = lognorm_hurdle(ms)
#     gcm, _ = sampleMilkyWayGCMF(ms)
    
# class GCSystem():
#     def __init__(self, sim: Simulation, rsid: int):
#         self.sim = sim
#         self.rsid = rsid

#         self.infall_snap = self.sim.hist[rsid]["first_infall_snap"]

#         if self.infall_snap == -1:
#             raise ValueError("Subhalo has not infallen, no tracking possible (at this point)...")
        
#         # get stellar mass of the system
#         ok_rs = self.sim.rs['ok'][self.rsid]
        
#         # add max_snap variable
#         # snaps = np.arange(236)[ok_rs]

#         self.disrupt_snap = np.arange(236)[ok_rs][-1]

#         self.snaps = np.arange(self.infall_snap, 236, step=1, dtype=int)
        
#         times = sim.cosmo.hubbleTime(sim.z)
#         times = times[self.snaps]
#         self.dt = times[1:] - times[:-1]
        
#         # self.snaps = snaps[snaps >= self.infall_snap]

#         # talk to phil about this later
#         self.ms_infall = self.sim.um["m_star"][self.rsid, self.infall_snap]
        
        # add max_snap variable
        # snaps = np.arange(236)[ok_rs]

        # self.disrupt_snap = np.arange(236)[ok_rs][-1]

        # self.snaps = np.arange(self.infall_snap, 236, step=1, dtype=int)
        
        # times = sim.cosmo.hubbleTime(sim.z)
        # times = times[self.snaps]
        # self.dt = times[1:] - times[:-1]
        
        # # self.snaps = snaps[snaps >= self.infall_snap]

        # # talk to phil about this later
        # self.ms_infall = self.sim.um["m_star"][self.rsid, self.infall_snap]
        
        # self.mgc = lognorm_hurdle(self.ms_infall)# system mass
        # self.gcm, self.gc_mass_range = sampleDwarfGCMF(self.mgc)
        
#         self.gcm = np.array(self.gcm)
#         self.gc_mass_range = np.array(self.gc_mass_range)

#         stars, gals, ranks = symlib.tag_stars(sim.sim_dir, sim.gal_halo, target_subs=[self.rsid])

#         # this is the z=0 distribution of stars... 
#         # we can choose a PDF based on empirical models in a later update
#         prob = stars[self.rsid]['mp'] / np.sum(stars[self.rsid]['mp'])

#         tag_idx  = np.random.choice(
#             np.arange(len(prob)),
#             size = len(self.gcm),
#             replace = False,
#             p = prob)
        

#         # tag_idx is the particle index
#         self.tag_idx = tag_idx
#         self.profile = MassProfile(self.sim, self.infall_snap, self.rsid)
#         _ = self.profile.fit()

#     def getPositions(self, index, snap=None):

#         # if snap is None:
#         #     snap = self.sim.buffer_snap # starts out being the infall_snap

#         self.t = np.zeros(len(self.snaps)) # self.snap runs from infall_snap to 235
#         self.q = np.zeros((3, len(self.snaps)))

#         for i, sn in enumerate(tqdm(self.snaps[1:])):
#                 _particles = self.sim.getParticles(sn)[self.rsid] # call to symlib.Particles(sim_dir).read(snap, mode=mode)
#                 pos = _particles["x"][self.tag_idx[index]] # tag_idx are the indices of GC tags
#                 self.t[i+1] = self.t[i] + self.dt[i]
#                 self.q[:, i] = pos

#         return self.t, self.q
        

#     def evolve(self, index, snap=None):
#         # index here is the index of the tag_idx array
#         if snap is None:
#             snap = self.sim.buffer_snap
        
#         self.t = np.zeros(len(self.snaps))
#         self.r = np.zeros(len(self.snaps))
#         self.lam = np.zeros(len(self.snaps))
#         self.potential_flag = np.ones(len(self.snaps)+1, dtype=bool)
#         self.profile_params = []
        
#         self.evolved_mass = np.zeros(len(self.snaps)+1)
#         self.evolved_mass[0] = self.gcm[index]

#         # old way of doing this
#         for i, sn in tqdm(enumerate(self.snaps[1:])):
#             _particles = self.sim.getParticles(sn)[self.rsid]
#             pos = _particles["x"][self.tag_idx[index]]

#             subhalo_flag = sn >= self.disrupt_snap

#             potential_sh_id = self.rsid
            
#             if subhalo_flag:
#                 if sn == self.disrupt_snap:
#                     print('Subhalo disrupted... switching potential.')
#                 potential_sh_id = 0
            
#             q_subhalo = self.sim.rs[self.rsid, sn]['x']
#             q_ext     = self.sim.rs[0, sn]['x']
            
#             r_subhalo = np.sqrt(np.sum((pos - q_subhalo)**2))
#             r_ext     = np.sqrt(np.sum((pos - q_ext)**2))
            
#             m = self.evolved_mass[i]

#             # create a lookup table for this
#             # consider alternative constructor for MassProfile class
#             self.profile = MassProfile(self.sim, sn, potential_sh_id)
#             self.profile.fit()
            
#             # todo: add baryons
#             lam = self.profile.profile.tidalStrength(
#                 [r_subhalo, 0., 0.],
#                 [r_ext,     0., 0.],
#                 self.sim.rs[0, sn]['rvir'],
#                 self.sim.um["m_star"][0, sn],
#                 subhalo_flag)

#             self.lam[i] = lam

#         # masses = CMassLoss(
#         #     self.gcm[index],
#         #     self.lam,
#         #     end_idx,
#         #     dt)

#             # CMassLoss(m0, l, end_idx, dt)
#             mass_loss = getMassLossRate(m, lam) * self.dt[i]
#             freq = getTidalFrequency(lam)
#             timescale = getTidalTimescale(m, lam)

#             self.potential_flag[i] = subhalo_flag
#             self.evolved_mass[i+1] = m + mass_loss
#             self.t[i] += self.dt[i]
#             self.r[i] = r_ext if subhalo_flag else r_subhalo
            
#             self.profile_params.append(self.profile.profile_params)

#         print("done")

########################################################################################################
