import abc
import numpy as np
import os
import asdf
from tqdm import tqdm

from .fit import SymphonyHaloProfile

class TreeReader(abc.ABC):

    def __init__(self, snapshots, times, scale_factors, **kwargs):
        self.snapshots   = snapshots
        self.cosmic_time = times
        self.a           = scale_factors

    @abc.abstractmethod
    def set_subhalo_infall(self, halo_id, snapshots, halo_mass, end_snapshots, **kwargs):
        
        self.halo_id = halo_id
        
        # NOTE: Sentinel value—central galaxy should have infall_snap of -1
        # and should be the first entry. 
        
        self.infall_snap = snapshots
        self.infall_time = self.cosmic_time[snapshots]
        self.infall_a    = self.a[snapshots]
        self.infall_mass = halo_mass

        self.disrupt_snap = end_snapshots

    @abc.abstractmethod
    def set_subhalo_positions(self, positions, **kwargs):
        # NOTE: shape should be [len(subhalos), len(snaps)]
        # each entry is a position in [x, y, z] galactocentric
        self.sh_pos = positions
        
    @abc.abstractmethod
    def write_halo_catalog(self, write_dir, **kwargs):

        # if self.subhalo_positions is None:
        #     raise ValueError("No subhalo positions set!")
            
        tree = {
            "halo_id": self.halo_id,
            "infall_snap": self.infall_snap,
            "infall_mass": self.infall_mass,
            "disrupt_snap": self.disrupt_snap
            # "sh_pos": self.subhalo_positions 
        }
        
        _af = asdf.AsdfFile(tree)
        print("Saving halo catalog...")
        _af.write_to(write_dir) # reminder: compress this later

        self.halo_catalog = _af

    @abc.abstractmethod
    def write_cluster_catalog(self, gcs_mf, gcmf, write_dir=None, **kwargs):
        
        
        # loop over the infall masses
        
        tree = {
            "halo_id": [],
            "infall_snap": [],
            "disrupt_snap": [],
            "gc_mass": [],
            "particle_index": [],
            "gc_index": []
        }
        
        for halo_id, m_i in enumerate(tqdm(self.halo_catalog["infall_mass"])):
            # generate GC masses
            masses = np.array(gcmf(m_i, system_mass_sampler = gcs_mf))

            for mass in masses:
                tree["halo_id"].append(halo_id)
                tree["infall_snap"].append(self.halo_catalog["infall_snap"][halo_id])
                tree["disrupt_snap"].append(self.halo_catalog["disrupt_snap"][halo_id])
                tree["gc_mass"].append(mass)
                tree["particle_index"].append(None)
                tree["gc_index"].append(None)
        
        tree['gc_index'] = np.arange(len(tree["gc_index"]))
        
        _af = asdf.AsdfFile(tree)
        
        if write_dir is not None:
            print("Saving cluster catalog... NOTE—Cluster indices not yet assigned. Run `assign_cluster_tags` (assuming your particle reader has been coded in) to assign indices.")
            _af.write_to(write_dir)
            
        self.cluster_catalog = _af
    
    @abc.abstractmethod
    def assign_cluster_tags(self, **kwargs):
        
        # This should interface w/ self.cluster_catalog
        # as well as your particle reader.
        
        pass

    @abc.abstractmethod
    def write_tracking_catalog(self, write_dir, **kwargs):
        # This uses the tagged GC catalog to track corresponding GCs
        # across snapshots.
        
        # At each snapshot, the particle data is read in for tracking purposes.
        # Additionally, each subhalo at a given snapshot is fit to a 
        # generic Profile.
            
        self.tracking_catalog = {
            "snapshot": [],
            "halo_id": [],
            "pos": [],
            "vel": [],
            "particle_index": [],
            "gc_index": []
        }

    @abc.abstractmethod
    def write_potential_catalog(self, write_dir, **kwargs):
        self.potential_catalog = {
            # This uses the tagged GC catalog to fit relevant su
            # across snapshots.
                "snapshot": [],
                "halo_id": [],
                "pos": [],
                "vel": [],
                "fit_param": [],
                "type": [],
            }

    @abc.abstractmethod
    def write_acceleration_catalog(self, write_dir, **kwargs):
        self.acc_catalog = {
            # This uses the tagged GC catalog to fit relevant acc
            # across snapshots.
                "snapshot": [],
                "halo_id": [],
                "radii": [],
                "acc": []
            }

    @abc.abstractmethod
    def write_galaxy_parameters(self, write_dir, **kwargs):

        self.gal_params = {
                "snapshot": [],
                "halo_id": [],
                "params": [],
            }
        
    @abc.abstractmethod
    def write_tidal_strength_catalog(self, write_dir, **kwargs):
        self.lambda_catalog = {
            # This uses the tagged GC catalog to fit relevant tidal strengths
            # across snapshots.
                "snapshot": [],
                "halo_id": [],
                "radii": [],
                "lambda": []
            }
        

import symlib
from colossus.cosmology import cosmology
from .sampler import DwarfGCMF, EadieSampler

class SymphonyReader(TreeReader):

    def __init__(self, sim_dir, gcmf = DwarfGCMF, **kwargs):

        self.sim_dir = sim_dir
        
        snapshots = np.arange(0, 236, dtype=int)
        scale_factors = np.array(symlib.scale_factors(sim_dir))
        
        self.params = symlib.simulation_parameters(sim_dir)
        self.mp = self.params['mp'] / self.params['h100']
        self.eps = self.params['eps'] / self.params['h100']
        self.col_params = symlib.colossus_parameters(self.params)
        self.cosmo = cosmology.setCosmology("cosmo", params=self.col_params)
        self.z = (1/scale_factors) - 1

        times = self.cosmo.hubbleTime(self.z)
        
        super().__init__(snapshots, times, scale_factors)

        ############ Set subhalo infall characteristics #################
        # read in the rockstar catalogs
        
        self.rs, self.hist = symlib.read_rockstar(self.sim_dir)
        self.sf, _ = symlib.read_symfind(self.sim_dir)
        self.um = symlib.read_um(self.sim_dir)
        
        halo_id = np.arange(0, self.rs.shape[0], dtype=int)
        infall_snaps = self.hist["first_infall_snap"]
        infall_mass  = self.um["m_star"][halo_id, infall_snaps]

        # lowkey a hack :/
        ok_rs = np.array(self.rs['ok'], dtype=int)
        _index_mat = np.tile(np.arange(self.rs.shape[1]), (self.rs.shape[0], 1))
        ok_rs_idx = np.multiply(ok_rs, _index_mat)
        disrupt_snaps = np.max(ok_rs_idx, axis=1)
        
        self.set_subhalo_infall(halo_id, infall_snaps, infall_mass, disrupt_snaps)
        self.write_halo_catalog("./halo_catalog.asdf")
        self.write_cluster_catalog(gcs_mf = EadieSampler, gcmf = DwarfGCMF, write_dir = './cluster.asdf')

         # Get the galaxy halo model for star tagging
        self.gal_halo = symlib.GalaxyHaloModel(
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

        
        self.assign_cluster_tags(write_dir = './tagged_cluster.asdf')
        self.write_tracking_catalog(write_dir = './tracked_clusters.asdf')
        self.write_potential_catalog(write_dir = './tracked_potentials.asdf')
        self.write_acceleration_catalog(write_dir = './tracked_acc.asdf')
        #################################################################
        

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
        return self.a[snapshot]

    def getConvergenceRadius(self, snapshot):
        """
        Get the convergence radius of a snapshot

        Parameters

        snapshot : int
            The snapshot number

        Returns

        r_conv : float
            The convergence radius of the snapshot in kpc
        """
        
        # Calculate convergence radius
        a = self.getScaleFactor(snapshot)
        
        # Get the Hubble constant
        
        # factor = (100 * u.km / u.s / u.Mpc).decompose().value
        factor = 3.2407792894443648e-18 # units of 1/s
        H0 = self.params['h100'] * factor
        
        # Get the present day critical density
        # _G = c.G.to(u.kpc**3 / u.Msun / u.s**2).value
        _G = 4.517103049894965e-39 # in units of kpc3 / Msun / s2
        rho_crit = (3 / (8 * np.pi * _G)) * (H0)**2 
                
        # Get the present day matter density
        rho_m = self.params['Om0'] * rho_crit
        
        # Get the mean interparticle spacing
        l_0 = (self.params['mp'] / self.params['h100'] / rho_m)**(1/3)
        
        z = self.getRedshift(snapshot)      

        # Convert to physical units
        l = a * l_0 

        # Return the convergence radius
        return (5.5e-2 * l, 3 * self.params['eps']/self.params['h100'] * a)

    def set_subhalo_infall(self, halo_id, snapshots, halo_mass, end_snapshots, **kwargs):
        super().set_subhalo_infall(halo_id, snapshots, halo_mass, end_snapshots)

    def set_subhalo_positions(self, positions, **kwargs):
        positions = self.rs['x']
        super().set_subhalo_positions(positions)
    
    def write_halo_catalog(self, write_dir, **kwargs):
        if os.path.exists(write_dir):
            print("Found archived halo catalog...")
            self.halo_catalog = asdf.open(write_dir)
        else:
            super().write_halo_catalog(write_dir)

    def write_cluster_catalog(self, gcs_mf, gcmf, write_dir, **kwargs):
        if os.path.exists(write_dir):
            print("Found archived cluster catalog...")
            self.cluster_catalog = asdf.open(write_dir)
        else:
            super().write_cluster_catalog(gcs_mf, gcmf, write_dir)

    def assign_cluster_tags(self, write_dir):
        if os.path.exists(write_dir):
            print("Found archived tagged cluster catalog...")
            self.cluster_catalog = asdf.open(write_dir)
        else:
            particle_class = symlib.Particles(self.sim_dir)

            _infall_snaps = np.array(self.cluster_catalog['infall_snap'])
            
            loadable_snaps = np.unique(_infall_snaps[np.where(_infall_snaps != -1)[0]])                                

            for snap in tqdm(loadable_snaps):
                # load all particles for that snapshot
                particles = particle_class.read(snap, mode='stars')
                
                load_idx = np.where(np.array(self.cluster_catalog['infall_snap']) == snap)[0]
                
                
                # loop through all clusters identified with an
                # infall snap that matches the current particle load buffer
                if len(load_idx) > 0:
                    
                    load_halo_id = np.array(self.cluster_catalog['halo_id'])[load_idx]
                    stars, gals, ranks = symlib.tag_stars(
                                self.sim_dir,
                                self.gal_halo,
                                target_subs=load_halo_id)
                    
                    for halo_id in load_halo_id:
    
                        # calculate gc tag probabilities for the
                        # stars in the subhalo that was loaded
                        prob = stars[halo_id]['mp'] / np.sum(stars[halo_id]['mp'])
    
                        # draw tags
                        tag_idx  = np.random.choice(
                            np.arange(len(prob)),
                            size = len(load_idx),
                            replace = False,
                            p = prob)
        
                        for tag_id, i in zip(tag_idx, load_idx):
                            self.cluster_catalog['particle_index'][i] = tag_id
                else:
                    print("No subhalo infalls at this snap.")

            
            self.cluster_catalog['gc_index'] = np.arange(len(self.cluster_catalog['particle_index']))
            
            _af = self.cluster_catalog
            _af.write_to(write_dir)
    
    def write_tracking_catalog(self, write_dir, **kwargs):

        super().write_tracking_catalog(write_dir, **kwargs)
                
        if os.path.exists(write_dir):
            print("Found archived tracking catalog...")
            self.tracking_catalog = asdf.open(write_dir)
        else:
            
            particle_class = symlib.Particles(self.sim_dir)
            
            _infall_snaps = np.array(self.cluster_catalog['infall_snap'])

            # The starting snapshot is the first snapshot where the first subhalo with
            # a GC infalls.
            
            start_snap = np.min(np.unique(_infall_snaps[np.where(_infall_snaps != -1)[0]]))
            track_snaps = self.snapshots[start_snap:]

            def trackGCProperties(snapshot, halo_id, particle_index, gc_index, pos, vel):
                self.tracking_catalog["snapshot"].append(snapshot)
                self.tracking_catalog["halo_id"].append(halo_id)
                self.tracking_catalog["gc_index"].append(gc_index)
                self.tracking_catalog["particle_index"].append(particle_index)
                self.tracking_catalog["pos"].append(pos) # physical units pls
                self.tracking_catalog["vel"].append(vel) 
            
            for snap in tqdm(track_snaps):
                particles = particle_class.read(snap, mode='stars')
                
                # loop through every GC in the catalog
                
                for i, particle_index in enumerate(self.cluster_catalog['particle_index']):
                    
                    # should we track the GC at this snapshot
                    # i.e., has the GC fallen in yet and if it ever infalls?
                    
                    is_tracked = (particle_index is not None) & (self.cluster_catalog['infall_snap'][i] <= snap) & (self.cluster_catalog['infall_snap'][i] != -1)
                    
                    if is_tracked:
                        
                        pos = particles[self.cluster_catalog['halo_id'][i]]["x"][particle_index]
                        vel = particles[self.cluster_catalog['halo_id'][i]]["v"][particle_index]
                        
                        trackGCProperties(snap,
                                          self.cluster_catalog['halo_id'][i],
                                          self.cluster_catalog['particle_index'][i],
                                          self.cluster_catalog['gc_index'][i],
                                          pos,
                                          vel)
            _af = asdf.AsdfFile(self.tracking_catalog)
            _af.write_to(write_dir, all_array_compression="zlib")

                
            
    def write_potential_catalog(self, write_dir):
        
        super().write_potential_catalog(write_dir)
        
        if os.path.exists(write_dir):
            print("Found archived potential catalog...")
            self.potential_catalog = asdf.open(write_dir)
        else:
            print("Writing potential catalog...")
            particle_class = symlib.Particles(self.sim_dir)

            print("Particles loaded.")
            
            _infall_snaps = np.array(self.cluster_catalog['infall_snap'])
            start_snap = np.min(np.unique(_infall_snaps[np.where(_infall_snaps != -1)[0]]))
            track_snaps = self.snapshots[start_snap:]
            
            def trackPotentialProperties(snapshot, halo_id, pos, params, pot_type):
                self.potential_catalog["snapshot"].append(snapshot)
                self.potential_catalog["halo_id"].append(halo_id)
                self.potential_catalog["pos"].append(pos)
                self.potential_catalog["fit_param"].append(params)
                self.potential_catalog["type"].append(pot_type)

            for snap in tqdm(track_snaps):

                # read in particles at that snapshot
                particles = particle_class.read(snap, mode='all')

                # figure out which halos are okay to track
                n_halo = self.rs.shape[0]

                for i in range(n_halo):

                    # check if that halo is trackable
                    # i.e., is it flagged 'ok' by rockstar?
                    is_tracked = self.rs[i, snap]['ok']

                    if is_tracked:

                        # Bulk subhalo properties
                        pos = self.rs[i, snap]['x']
                        vel = self.rs[i, snap]['v']
                        mass = self.rs[i, snap]['m']
                        rvir = self.rs[i, snap]['rvir']

                        # Attempt to fit an Einasto profile to
                        # the density field.

                        q = particles[i]['x']
                        params = None

                        def fit_einasto():
                            profile = SymphonyHaloProfile(q,
                                                          pos,
                                                          self.mp,
                                                          rvir,
                                                          a=self.a[snap])
                        
                            params = profile.fit(np.max(self.getConvergenceRadius(snap)))
                            
                            trackPotentialProperties(snap, i, pos, params, 'einasto')

                        def fit_nfw():
                            params = {
                                "mvir": self.rs[i, snap]['m'],
                                "rvir": self.rs[i, snap]['rvir'],
                                "cvir": self.rs[i, snap]['cvir']
                            }
                            
                            # NFW profile
                            # mvir, c
                            trackPotentialProperties(snap, i, pos, params, 'nfw')
                        
                        try:
                            fit_einasto()
                        except ValueError:
                            print("Unable to fit, defaulting to NFW parameters.")
                            fit_nfw()
                        except OptimizeWarning:
                            # im not catching this right now, fix this.
                            print("Unable to fit, defaulting to NFW parameters.")
                            fit_nfw()
                        except RuntimeWarning:
                            # im not catching this right now, fix this.
                            print("Unable to fit, defaulting to NFW parameters.")
                            fit_nfw()
                        except:
                            fit_einasto()
                                        
                        
            _af = asdf.AsdfFile(self.potential_catalog)
            _af.write_to(write_dir)

    def write_acceleration_catalog(self, write_dir):

        super().write_acceleration_catalog(write_dir)
        
        if os.path.exists(write_dir):
            print("Found archived acceleration catalog...")
            self.acc_catalog = asdf.open(write_dir)
        else:
            print("Writing acceleration catalog...")
            particle_class = symlib.Particles(self.sim_dir)

            print("Particles loaded.")
            
            _infall_snaps = np.array(self.cluster_catalog['infall_snap'])
            start_snap = np.min(np.unique(_infall_snaps[np.where(_infall_snaps != -1)[0]]))
            track_snaps = self.snapshots[start_snap:]
            
            def trackAccelerations(snapshot, halo_id, radii, acc):
                self.acc_catalog["snapshot"].append(snapshot)
                self.acc_catalog["halo_id"].append(halo_id)
                self.acc_catalog["radii"].append(radii)
                self.acc_catalog["acc"].append(acc)

            for snap in tqdm(track_snaps):

            
                # read in particles at that snapshot
                particles = particle_class.read(snap, mode='all')
        
                # # figure out which halos are okay to track
                n_halo = self.rs.shape[0]
        
                for i in range(n_halo):
        
                    # check if that halo is trackable
                    # i.e., is it flagged 'ok' by rockstar?
                    is_tracked = self.rs[i, snap]['ok']
        
                    if is_tracked:

                        # very flexible, very slow...
            
                        # Bulk subhalo properties
                        pos = self.rs[i, snap]['x']
                        vel = self.rs[i, snap]['v']
                        mass = self.rs[i, snap]['m']
                        rvir = self.rs[i, snap]['rvir']

                        # select for bound particles only
                        q = particles[i]['x']
                        p = particles[i]['v']

                        dq = q - pos
                        dp = p - vel
                        
                        r = np.sqrt(np.sum(dq**2, axis=1))
                        order = np.argsort(r)
                    
                        ke = np.sum(dp**2, axis=1) / 2
                        ok = np.ones(len(ke), dtype=bool)
                    
                        for _ in range(3):
                            _, vmax, pe, _ = symlib.profile_info(self.params, dq, ok=ok)
                            E = ke + pe*vmax**2
                            ok = E < 0
        
                        print('halo loaded with ', len(q) ,'particles and r_vir =', rvir)
                        
                        profile = SymphonyHaloProfile(q[ok],
                                                      pos,
                                                      self.mp,
                                                      rvir,
                                                      a=self.a[snap])
        
                        

                        
                        radii, accelerations = profile.getRadialAccelerationProfile(rvir, self.eps)
                        trackAccelerations(snap, i, radii, accelerations)            
                                
            _af = asdf.AsdfFile(self.acc_catalog)
            _af.write_to(write_dir)

    def write_galaxy_parameters(self, write_dir):
        pass
    
    def write_tidal_strength_catalog(self, write_dir):

        super().write_tidal_strength_catalog(write_dir)
        
        # self.lambda_catalog = {
        #     # This uses the tagged GC catalog to fit relevant tidal strengths
        #     # across snapshots.
        #         "snapshot": [],
        #         "halo_id": [],
        #         "radii": [],
        #         "lambda": []
        #     }
        
        if os.path.exists(write_dir):
            print("Found archived tidal strength catalog...")
            self.lambda_catalog = asdf.open(write_dir)
        else:
            print("Writing tidal strength catalog...")
            
            
            def trackAccelerations(snapshot, halo_id, radii, acc):
                self.acc_catalog["snapshot"].append(snapshot)
                self.acc_catalog["halo_id"].append(halo_id)
                self.acc_catalog["radii"].append(radii)
                self.acc_catalog["acc"].append(acc)

            for snap in tqdm(track_snaps):
                        
                        radii, accelerations = profile.getRadialAccelerationProfile(rvir, self.eps)
                        trackAccelerations(snap, i, radii, accelerations)            
                                
            _af = asdf.AsdfFile(self.potential_catalog)
            _af.write_to(write_dir)


            
                
            