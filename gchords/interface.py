import abc
import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import symlib
from colossus.cosmology import cosmology
from .um import UniverseMachineMStarFit

import agama
agama.setUnits(length=1, velocity=1, mass=1)
class Interface(abc.ABC):
    """
    interface is an abstract class that reads a simulations parameters
    and return the necessary information to run GChords
    """

    def __init__(self, **kwargs):
        self.cosmology_parameters = None
        self.scale_factors = None
        self.mp = None
        self.infall_properties = {
            "infall_snapshot": [],
            "halo_mass": [],
            "stellar_mass": [],
            "disrupt_snapshot": [],
            "preinfall_host_idx": [],
        }

        self.particles = None


class SymphonyInterface(Interface):
    def __init__(self, sim_dir, read_um=True,  **kwargs):
        super().__init__(**kwargs)
        self.sim_dir = sim_dir
        self.params = symlib.simulation_parameters(sim_dir)
        self.mp = self.params["mp"] / self.params["h100"]
        self.cosmology_parameters = symlib.colossus_parameters(self.params)
        self.cosmology = cosmology.setCosmology(
            "cosmo", params=self.cosmology_parameters
        )
        self.scale_factors = np.array(symlib.scale_factors(sim_dir))

        self.rs, hist = symlib.read_rockstar(sim_dir)
        self.infall_properties["infall_snapshot"] = hist["first_infall_snap"]
        self.infall_properties["halo_mass"] = self.rs["m"][
            np.arange(self.rs.shape[0]), hist["first_infall_snap"]
        ]

        # If UM outputs are available, read them. If not, compute them using the fit.
        if read_um:
            um = symlib.read_um(sim_dir)
            self.infall_properties["stellar_mass"] = um["m_star"][
                np.arange(um["m_star"].shape[0]), hist["first_infall_snap"]
            ]
        else:
            mpeaks = hist["mpeak"]
            infall_z = 1 / self.scale_factors[hist["first_infall_snap"]] - 1
            fit = UniverseMachineMStarFit()
            self.infall_properties["stellar_mass"] = np.array(
                [fit.m_star(mp_i, z_i) for mp_i, z_i in zip(mpeaks, infall_z)]
            )

        ok = self.rs["ok"]
        # TODO: make this less hacky.
        rev_idx = ok[:, ::-1].argmax(axis=1)
        has_true = ok.any(axis=1)
        last_true_idx = ok.shape[1] - 1 - rev_idx
        disrupt_snap = np.where(has_true, last_true_idx + 1, -1)
        disrupt_snap[disrupt_snap == self.rs.shape[1]] = -1
        self.infall_properties["disrupt_snapshot"] = disrupt_snap

        self.infall_properties["preinfall_host_idx"] = symlib.pre_infall_host(hist)

        self.particles = symlib.Particles(self.sim_dir)

    def get_gse_index(self):
        '''
        returns the GSE halo based on the Buch+2024 (https://arxiv.org/abs/2404.08043) criteria

        A GSE analog subhalo that merges with the MW host
        between 0.25 < adisrupt < 0.6 (i.e., 0.67 < zdisrupt < 3
        or between 6 and 11.5 Gyr ago) with Msub/Mhost > 0.2
        when the GSE analog achieves its peak mass (e.g.,
        Helmi 2020; Naidu et al. 2021).
        '''

        candidates = np.zeros(len(self.infall_properties["infall_snapshot"]), dtype=bool)
        candidates[1:] = True # exclude host

        infall_redshifts = 1 / self.scale_factors[self.infall_properties["infall_snapshot"]] - 1
        candidates &= (self.infall_properties["infall_snapshot"] >= 0) & (self.infall_properties["infall_snapshot"] < len(self.scale_factors))
        candidates &= (infall_redshifts > 0.67) & (infall_redshifts < 3.0)

        # mass ratio condition: Msub/Mhost > 0.2 at the snapshot of peak subhalo mass
        masked_masses = np.where(self.rs["ok"], self.rs["m"], 0.0)
        peak_snaps = np.argmax(masked_masses, axis=1)
        mpeaks = masked_masses[np.arange(len(peak_snaps)), peak_snaps]
        mhost_at_peak = self.rs["m"][0, peak_snaps]
        candidates &= (mpeaks / np.where(mhost_at_peak > 0, mhost_at_peak, np.inf)) > 0.2

        if np.any(candidates):
            gse_index = np.where(candidates)[0][np.argmax(self.infall_properties["halo_mass"][candidates])]
            return gse_index
        else:
            return -1
