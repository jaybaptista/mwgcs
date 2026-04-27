import numpy as np
import pandas as pd
import symlib
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
class GChords(object):
    def __init__(self, interface, gc_halo_model, **kwargs):
        self.interface = interface
        self.gc_halo_model = gc_halo_model
        self.particle_tags = None
        # particle tagging with Nimbus
        self.weights, _, _ = symlib.tag_stars(
            self.interface.sim_dir,
            self.gc_halo_model.nimbus_model,
        )

        self.is_manually_tagged = False

    def set_particle_tags(self, particle_tags):
        self.particle_tags = particle_tags
        self.is_manually_tagged = True

    def set_particle_tracks(self, particle_tracks):
        self.particle_tracks = particle_tracks

    def generate_clusters(self, write_dir='particles.csv', seed=None, **kwargs):
        infall_snapshots = self.interface.infall_properties["infall_snapshot"]
        n_halos = len(infall_snapshots)

        if seed is not None:
            np.random.seed(seed)

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

        for k in np.arange(1, n_halos):
            _, _, _mgcs = self.gc_halo_model.generate(
                halo_mass=self.interface.infall_properties["halo_mass"][k],
                stellar_mass=self.interface.infall_properties["stellar_mass"][k],
            )

            # skip halos with no GCs or infall onto non-central host
            if (_mgcs is None) or (self.interface.infall_properties["preinfall_host_idx"][k] != -1):
                continue

            mp = self.weights[k]['mp']
            
            # TODO: look carefulely at this.
            # e.g., if there aren't enough particles 
            if np.sum(mp) <= 0.0:
                # if I can't draw a particle tag, then I can't assign a GC, so skip this halo.
                # NOTE: this may set a resolution floor for GC formation
                continue
            
            p_draw = mp / np.sum(mp)
            draws = np.random.choice(len(mp), size=len(_mgcs), replace=False, p=p_draw)
            feh = self.weights[k]['Fe_H'][draws]
            a_form = self.weights[k]['a_form'][draws]

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
        
        n_tracked_particles = len(self.particle_tags)
        indices = self.find_unique_particles()

        if self.is_manually_tagged:
            print('Manually tag particles--only track unique particles')    
            n_tracked_particles = len(indices)
        
        data = np.zeros((len(self.interface.scale_factors), n_tracked_particles, 6)) * np.nan
        
        for snapshot in tqdm(range(len(self.interface.scale_factors)), desc="Tracking particles across snapshots..."):
            # Load all the subhalos at a given snapshot and their corresponding particles
            particles = self.interface.particles.read(snapshot, mode="stars", comoving=comoving)
            p_flat = np.hstack(particles)
            
            # select unique particles if manually tagged, otherwise select all tagged particles
            ok = self.particle_tags["infall_snap"] <= snapshot
            if self.is_manually_tagged:
                ok &= np.isin(self.particle_tags["particle_index"], indices)

            if ok.any():
                i_t = self.particle_tags["particle_index"][ok]
                data[snapshot, ok, :3] = p_flat[i_t]["x"]
                data[snapshot, ok, 3:] = p_flat[i_t]["v"]
        
        self.particle_tracks = data
        self.particle_indices = indices
        np.savez_compressed(write_dir, xv=data, particle_index=indices)

    def find_unique_particles(self):
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        _p = self.interface.particles.read(len(self.interface.scale_factors), mode="stars", comoving=False)
        sizes = np.array([len(p) for p in _p])
        edges = np.zeros(len(sizes) + 1, int)
        edges[1:] = np.cumsum(sizes)
        starts = edges[:-1]
        
        # flattened index of particles tagged with GCs
        self.particle_tags["particle_index"] = self.particle_tags["nimbus_index"] + starts[self.particle_tags["halo_index"]]
        return self.particle_tags["particle_index"].unique()
    
    def compute_cluster_tidal_field(self, potential, write_dir='tidal_field.npz'):
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() first.")
        
        _t = []
        _st = []
        _int_st = []

        age_int = self.interface.cosmology.age(1/self.interface.scale_factors - 1) / 0.97779222

        for k in tqdm(range(self.particle_tracks.shape[1])):
            xv = self.particle_tracks[:, k, :]
            start_snapshot = np.where(~np.isnan(xv[:, 0]))[0][0]
            xv = xv[start_snapshot:]

            if start_snapshot == len(self.interface.scale_factors) - 1:
                _t.append([None])
                _st.append([None])
                _int_st.append([0.0])
        
            spl = [
                UnivariateSpline(
                    age_int[start_snapshot:], xv[:, i], s=0, k=min(3, len(age_int[start_snapshot:]) - 1)
                )
                for i in range(3)
            ]

            N_samples = 100 * len(age_int[start_snapshot:]) # sample 100 points per snapshot

            t_sample = np.linspace(age_int[start_snapshot], age_int[-1], N_samples)
            x_sample = np.array([spl_i(t_sample) for spl_i in spl]).T
            st_sample = potential.tidal_strength(x_sample, t=t_sample / 0.97779222) * 1.0459401725324529
            _t.append(t_sample)
            _st.append(st_sample)
            spl_st = PchipInterpolator(t_sample, np.sqrt(st_sample))
            _int_st.append(spl_st.integrate(t_sample[0], t_sample[-1]))

        np.savez_compressed(write_dir, tidal_field=_st, time=_t, integrated_tidal_field=_int_st)