import abc
import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import symlib
from colossus.cosmology import cosmology

from .sampler import FiducialGCHaloModel
from .um import UniverseMachineMStarFit
from .tag import expand, tag_energy_cut, energy, center

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

        rs, hist = symlib.read_rockstar(sim_dir)
        self.infall_properties["infall_snapshot"] = hist["first_infall_snap"]
        self.infall_properties["halo_mass"] = rs["m"][
            np.arange(rs.shape[0]), hist["first_infall_snap"]
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

        ok = rs["ok"]
        # TODO: make this less hacky.
        rev_idx = ok[:, ::-1].argmax(axis=1)
        has_true = ok.any(axis=1)
        last_true_idx = ok.shape[1] - 1 - rev_idx
        disrupt_snap = np.where(has_true, last_true_idx + 1, -1)
        disrupt_snap[disrupt_snap == rs.shape[1]] = -1
        self.infall_properties["disrupt_snapshot"] = disrupt_snap

        self.infall_properties["preinfall_host_idx"] = symlib.pre_infall_host(hist)


class GChords(object):
    def __init__(self, interfacer, gc_halo_model, **kwargs):
        self.interface = interfacer
        self.gc_halo_model = gc_halo_model
        self.particle_tags = None

    def generate_clusters(self, write_dir='particles.csv', seed=None, **kwargs):
        n_snapshots = len(self.interface.scale_factors)
        infall_snapshots = self.interface.infall_properties["infall_snapshot"]
        n_halos = len(infall_snapshots)

        if seed is not None:
            np.random.seed(seed)

        # particle tagging with Nimbus
        weights, _, _ = symlib.tag_stars(
            self.interface.sim_dir,
            self.gc_halo_model.nimbus_model,
        )

        df = pd.DataFrame(
                columns=[
                    "halo_index",
                    "infall_snap",
                    "disrupt_snap",
                    "gc_mass",
                    "preinfall_host_idx",
                    "nimbus_index",
                    "feh",
                    "a_form",
                    "infall_host_mstar",
                ]
            )
        
        rows = []

        for k in tqdm(np.arange(1, n_halos)):
            _, _, _mgcs = self.gc_halo_model.generate(
                halo_mass=self.interface.infall_properties["halo_mass"][k],
                stellar_mass=self.interface.infall_properties["stellar_mass"][k],
            )

            # skip halos with no GCs or infall onto non-central host
            if (_mgcs is None) or (self.interface.infall_properties["preinfall_host_idx"][k] != -1):
                continue

            mp = weights[k]['mp']
            
            # TODO: look carefulely at this.
            # e.g., if there aren't enough particles 
            if np.sum(mp) <= 0.0:
                # if I can't draw a particle tag, then I can't assign a GC, so skip this halo.
                # NOTE: this may set a resolution floor for GC formation
                continue
            
            p_draw = mp / np.sum(mp)
            draws = np.random.choice(len(mp), size=len(_mgcs), replace=False, p=p_draw)
            feh = weights[k]['feh'][draws]
            a_form = weights[k]['a_form'][draws]

            rows.append(
                pd.DataFrame(
                    {
                        "halo_index": np.repeat(k, len(_mgcs)),
                        "infall_snap": np.repeat(infall_snapshots[k], len(_mgcs)),
                        "disrupt_snap": np.repeat(
                            self.interface.infall_properties["disrupt_snapshot"][k],
                            len(_mgcs),
                        ),
                        "gc_mass": _mgcs,
                        "preinfall_host_idx": np.repeat(
                            self.interface.infall_properties["preinfall_host_idx"][k],
                            len(_mgcs),
                        ),
                        "infall_host_mstar": np.repeat(
                            self.interface.infall_properties["stellar_mass"][k],
                            len(_mgcs),
                        ),
                        "nimbus_index": draws,
                        "feh": feh,
                        "a_form": a_form,
                    }
                )
            )

        if not rows:
            df.to_csv(write_dir, index=False)

        self.particle_tags = pd.concat(rows, ignore_index=True)
        self.particle_tags.to_csv(write_dir, index=False)

    def track_clusters(self, comoving=False, write_dir='particles.npz'):
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        data = np.zeros((len(self.interface.scale_factors), len(self.particle_tags), 6)) * np.nan
        part = symlib.Particles(self.interface.sim_dir)
        for snapshot in tqdm(range(len(self.interface.scale_factors)), desc="Tracking particles across snapshots..."):
            # Load all the subhalos at a given snapshot and their corresponding particles
            particles = part.read(snapshot, mode="stars", comoving=comoving)
            p_flat = np.hstack(particles)

            sizes = np.array([len(p) for p in particles])
            edges = np.zeros(len(sizes) + 1, int)
            edges[1:] = np.cumsum(sizes)
            starts = edges[:-1]

            ok = self.particle_tags["infall_snap"] <= snapshot

            if ok.any():
                i_t = self.particle_tags["nimbus_index"][ok] + starts[self.particle_tags["halo_index"][ok]]
                data[snapshot, ok, :3] = p_flat[i_t]["x"]
                data[snapshot, ok, 3:] = p_flat[i_t]["v"]
        
        np.savez_compressed(write_dir, xv=data)

def is_bound(q, p, subhalo_pos, subhalo_vel, params):
    dq = q - subhalo_pos
    dp = p - subhalo_vel

    ke = np.sum(dp**2, axis=1) / 2
    ok = np.ones(len(ke), dtype=bool)

    if (dq.size == 0) or (len(ke) == 0):
        return np.array([], dtype=bool)

    for _ in range(3):
        if (np.sum(ok) == 0) or (len(dq) == 0):
            return ok
        _, vmax, pe, _ = symlib.profile_info(params, dq, ok=ok)
        E = ke + pe * vmax**2
        ok = E < 0

    return ok


def get_bounded_particles(q, p, subhalo_pos, subhalo_vel, params):
    ok = is_bound(q, p, subhalo_pos, subhalo_vel, params)
    return q[ok], p[ok]
