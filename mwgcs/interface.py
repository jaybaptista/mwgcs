import abc
import numpy as np
import os
import asdf
from tqdm import tqdm

import agama
agama.setUnits(length=1, velocity=1, mass=1)

from .fit import SymphonyHaloProfile

from scipy.stats import norm


class Interfacer(abc.ABC):
    def __init__(self, snapshots, times, scale_factors, **kwargs):
        self.snapshots = snapshots
        self.cosmic_time = times
        self.a = scale_factors

    @abc.abstractmethod
    def set_subhalo_infall(
        self, halo_id, snapshots, halo_mass, end_snapshots, **kwargs
    ):
        self.halo_id = halo_id

        # NOTE: Sentinel value—central galaxy should have infall_snap of -1
        # and should be the first entry.

        self.infall_snap = snapshots
        self.infall_time = self.cosmic_time[snapshots]
        self.infall_a = self.a[snapshots]
        self.infall_mass = halo_mass
        self.disrupt_snap = end_snapshots

    @abc.abstractmethod
    def set_subhalo_positions(self, positions, **kwargs):
        # NOTE: shape should be [len(subhalos), len(snaps)]
        # each entry is a position in [x, y, z] galactocentric
        self.sh_pos = positions


import symlib
from colossus.cosmology import cosmology
from .sampler import DwarfGCMF, EadieSampler, KGSampler


class SymphonyInterfacer(Interfacer):
    def __init__(self, sim_dir, gcmf=DwarfGCMF, **kwargs):
        self.sim_dir = sim_dir

        self.halo_label = os.path.split(sim_dir)[-1]

        if not os.path.exists(self.halo_label):
            print("Creating halo directory...")
            os.mkdir(self.halo_label)

        snapshots = np.arange(0, 236, dtype=int)
        scale_factors = np.array(symlib.scale_factors(sim_dir))

        self.params = symlib.simulation_parameters(sim_dir)
        self.mp = self.params["mp"] / self.params["h100"]
        self.eps = self.params["eps"] / self.params["h100"]
        self.col_params = symlib.colossus_parameters(self.params)
        self.cosmo = cosmology.setCosmology("cosmo", params=self.col_params)
        self.z = (1 / scale_factors) - 1

        times = self.cosmo.hubbleTime(self.z)

        super().__init__(snapshots, times, scale_factors)

        ############ Set subhalo infall characteristics #################
        # read in the rockstar catalogs

        self.rs, self.hist = symlib.read_rockstar(self.sim_dir)
        self.sf, _ = symlib.read_symfind(self.sim_dir)
        self.um = symlib.read_um(self.sim_dir)

        #
        self.particle_class = symlib.Particles(self.sim_dir)

        halo_id = np.arange(0, self.rs.shape[0], dtype=int)
        infall_snaps = self.hist["first_infall_snap"]
        infall_mass = self.um["m_star"][halo_id, infall_snaps]
        infall_halo_mass = self.rs["m"][halo_id, infall_snaps]

        # lowkey a hack :/ #####################
        # gets the disruption snapshot
        ok_rs = np.array(self.rs["ok"], dtype=int)
        _index_mat = np.tile(np.arange(self.rs.shape[1]), (self.rs.shape[0], 1))
        ok_rs_idx = np.multiply(ok_rs, _index_mat)
        disrupt_snaps = np.max(ok_rs_idx, axis=1)

        self.infall_snaps = infall_snaps
        self.infall_mass = infall_mass
        self.infall_halo_mass = infall_halo_mass
        self.disrupt_snaps = disrupt_snaps

        # Get the galaxy halo model for star tagging
        self.gal_halo = symlib.GalaxyHaloModel(
            symlib.StellarMassModel(
                symlib.UniverseMachineMStar(),
                symlib.DarkMatterSFH(),  # swapped this one out
            ),
            symlib.ProfileModel(symlib.Jiang2019RHalf(), symlib.PlummerProfile()),
            symlib.MetalModel(
                symlib.Kirby2013Metallicity(),
                symlib.Kirby2013MDF(model_type="gaussian"),
                symlib.FlatFeHProfile(),
                symlib.GaussianCoupalaCorrelation(),
            ),
        )

        ### interface with simulation outputs
        self.assign_particle_tags(
            KGSampler, DwarfGCMF, write_dir=os.path.join(self.halo_label, "./ParticleTags.npz")
        )
        self.track_particles(write_dir=os.path.join(self.halo_label, "./ParticleTracks.npz"))  # ps = phase space
        self.get_host_bfe("./BFECoefficients")
        self.get_subhalo_bfe("./BFESubhalo")
        # self.make_acceleration_cube(write_dir=os.path.join(self.halo_label, "./acc_cube.npz"))
        # self.make_mass_cube(write_dir=os.path.join(self.halo_label, "./mass_cube.npz"))
        # self.make_potential_cube(write_dir=os.path.join(self.halo_label, "./pot_cube.npz"))

        # simulation metadata
        self.getConvergenceRadii(write_dir=os.path.join(self.halo_label, "./rconv.npz"))

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
        factor = 3.2407792894443648e-18  # units of 1/s
        H0 = self.params["h100"] * factor

        # Get the present day critical density
        _G = 4.517103049894965e-39  # in units of kpc3 / Msun / s2
        rho_crit = (3 / (8 * np.pi * _G)) * (H0) ** 2

        # Get the present day matter density
        rho_m = self.params["Om0"] * rho_crit

        # Get the mean interparticle spacing
        l_0 = (self.params["mp"] / self.params["h100"] / rho_m) ** (1 / 3)

        z = self.getRedshift(snapshot)

        # Convert to physical units
        l = a * l_0

        # Return the convergence radius
        return np.max((5.5e-2 * l, 3 * self.params["eps"] / self.params["h100"] * a))

    def set_subhalo_infall(
        self, halo_id, snapshots, halo_mass, end_snapshots, **kwargs
    ):
        super().set_subhalo_infall(halo_id, snapshots, halo_mass, end_snapshots)

    def set_subhalo_positions(self, positions, **kwargs):
        positions = self.rs["x"]
        super().set_subhalo_positions(positions)

    def write_halo_catalog(self, write_dir, **kwargs):
        if os.path.exists(write_dir):
            print("Found archived halo catalog...")
            self.halo_catalog = asdf.open(write_dir)
        else:
            super().write_halo_catalog(write_dir)

    def getConvergenceRadii(self, write_dir):
        if os.path.exists(write_dir):
            print("Found archived convergence radii catalog...")
        else:
            data = np.zeros(len(self.snapshots))
            for k in self.snapshots:
                data[k] = self.getConvergenceRadius(k)
            
            np.savez_compressed(write_dir, data)

    def initialize_gc_array(self, system_mass_sampler, gc_mass_sampler):
        print("Initializing GC tag data structure...")

        # mask for subhalos that have infall snaps
        m = self.infall_snaps != -1

        # halo indices of subhalos that have infall snaps
        halo_indices = np.arange(len(self.infall_snaps))[m]

        # infall STELLAR masses of subhalos that have infall snaps
        infall_masses = self.infall_mass[m]
        infall_snaps = self.infall_snaps[m]
        disrupt_snaps = self.disrupt_snaps[m]
        infall_halo_mass = self.infall_halo_mass[m]

        _array_halo_indices = []
        _array_infall_snap = []
        _array_disrupt_snap = []
        _array_gc_masses = []

        for i, infall_mass in enumerate(tqdm(infall_masses)):
            # obtain individual GC masses for each GC system
            _gc_masses = gc_mass_sampler(infall_mass, system_mass_sampler=system_mass_sampler, halo_mass=infall_halo_mass[i])

            if _gc_masses is None:
                continue
            else:
                gc_masses = np.array(_gc_masses)

                _array_halo_indices.append(np.repeat(halo_indices[i], len(gc_masses)))
                _array_infall_snap.append(np.repeat(infall_snaps[i], len(gc_masses)))
                _array_disrupt_snap.append(np.repeat(disrupt_snaps[i], len(gc_masses)))
                print(len(gc_masses))
                _array_gc_masses.append(gc_masses)

        array_halo_indices = np.hstack(_array_halo_indices)  # int
        array_infall_snap = np.hstack(_array_infall_snap)  # int
        array_disrupt_snap = np.hstack(_array_disrupt_snap)  # int
        array_gc_masses = np.hstack(_array_gc_masses)  # float64

        # create structured array

        dtype = np.dtype(
            [
                ("halo_index", int),
                ("infall_snap", int),
                ("disrupt_snap", int),
                ("gc_mass", float),
            ]
        )

        _array = np.empty(len(array_halo_indices), dtype=dtype)

        _array["halo_index"] = array_halo_indices
        _array["infall_snap"] = array_infall_snap
        _array["disrupt_snap"] = array_disrupt_snap
        _array["gc_mass"] = array_gc_masses

        return _array

    def assign_particle_tags(
        self, system_mass_sampler, gc_mass_sampler, write_dir, tmp_dir="tmp.npz"
    ):

        tmp_save_dir = os.path.join(self.halo_label, tmp_dir)
        
        if os.path.exists(write_dir):
            print("Found particle tag `.npz`...")
            self.particle_tags = np.load(write_dir)["arr_0"]
        else:
            print("Assigning GC tags...")

            arr = None

            # initialize the GC array
            if os.path.exists(tmp_save_dir):
                arr = np.load(tmp_save_dir)["arr_0"]
            else:
                arr = self.initialize_gc_array(system_mass_sampler, gc_mass_sampler)
                np.savez_compressed(tmp_save_dir, arr)

            infall_snaps = self.infall_snaps[self.infall_snaps != -1]

            # create new structured array but with new column for particle_id (int)

            dtype = np.dtype(
                [
                    ("halo_index", int),
                    ("infall_snap", int),
                    ("disrupt_snap", int),
                    ("gc_mass", float),
                    ("nimbus_index", int),
                ]
            )

            particle_tag_arr = np.empty(len(arr), dtype=dtype)
            particle_tag_arr["halo_index"] = arr["halo_index"]
            particle_tag_arr["infall_snap"] = arr["infall_snap"]
            particle_tag_arr["disrupt_snap"] = arr["disrupt_snap"]
            particle_tag_arr["gc_mass"] = arr["gc_mass"]
            particle_tag_arr["nimbus_index"] = np.zeros(len(arr), dtype=int) - 1

            for snap in tqdm(infall_snaps):
                # particles = self.particle_class.read(snap, mode="stars")

                # indices to assign tags to
                indices = np.where(particle_tag_arr["infall_snap"] == snap)[0]
                halo_ids = particle_tag_arr["halo_index"][indices]

                for k, hid in zip(indices, halo_ids):
                    stars, gals, ranks = symlib.tag_stars(
                        self.sim_dir, self.gal_halo, target_subs=[hid]
                    )

                    prob = stars[hid]["mp"] / np.sum(stars[hid]["mp"])

                    # draw tag ids

                    # TODO: fix this, you might accidentally pick the same particle

                    particle_tag_index = np.random.choice(
                        np.arange(len(prob)), size=1, replace=False, p=prob
                    )

                    # NOTE: refactor particle_id to ``nimbus_index``
                    particle_tag_arr["nimbus_index"][k] = particle_tag_index
            
            np.savez_compressed(write_dir, particle_tag_arr)
            self.particle_tags = np.load(write_dir)["arr_0"]
            

    def track_particles(self, write_dir):
        if os.path.exists(write_dir):
            print("Found archived particle tracking cube...")
        else:
            # self.particle_tags

            particle_tag_indices = np.arange(len(self.particle_tags))

            # this structured array has shape (snapshot, particle index, position(3), velocity(3))
            tracking_cube = (
                np.zeros((self.rs.shape[1], len(particle_tag_indices), 6)) * np.nan
            )

            # loop over each snapshot
            for snapshot in tqdm(range(self.rs.shape[1])):
                # this should load all the subhalos at a given snapshot
                # and their corresponding particles
                particles = self.particle_class.read(snapshot, mode="stars")

                part_flat = np.hstack(particles)
                sizes = np.array([len(p) for p in particles])

                edges = np.zeros(len(sizes) + 1, int)
                edges[1:] = np.cumsum(sizes)
                starts = edges[:-1]
                # ends = edges[1:]

                ok = self.particle_tags["infall_snap"] <= snapshot

                if ok.any():

                    i_t = (
                        self.particle_tags["nimbus_index"][ok]
                        + starts[self.particle_tags["halo_index"][ok]]
                    )

                    tracking_cube[snapshot, ok, :3] = part_flat[i_t]["x"]
                    tracking_cube[snapshot, ok, 3:] = part_flat[i_t]["v"]

            np.savez_compressed(write_dir, tracking_cube)

    ###

    def get_host_bfe(self, write_dir):
        halo_index = 0 
        if not os.path.exists(write_dir):
            print("Creating halo directory...")
            os.mkdir(write_dir)
        for snapshot in tqdm(range(self.rs.shape[1])):

            coef_write_path = os.path.join(write_dir, f"coef_snap_{snapshot}.coef_mul")

            if os.path.exists(coef_write_path):
                print(f"Found coefficient file for snapshot {snapshot}.")
            else:
                particles = self.particle_class.read(snapshot, mode="all")
                q = particles[halo_index]["x"]
                ok = particles[halo_index]["ok"]
                
                masses = (np.ones(np.sum(ok)) * self.mp)

                pot = agama.Potential(
                    type="multipole",
                    particles=(q[ok], masses),
                    symmetry="none",
                    lmax=8,
                    rmin=0.01,
                    rmax=300
                )
                
                pot.export(coef_write_path)

    ### 
    
    def get_subhalo_bfe(self, write_dir):
        if not os.path.exists(write_dir):
            print("Creating halo directory...")
            os.mkdir(write_dir)
        for snapshot in tqdm(range(self.rs.shape[1])):
            snapshot_directory = os.path.join(write_dir, f"bfe_{snapshot}")
                
            if not os.path.exists(snapshot_directory):
                os.mkdir(snapshot_directory)

            particles = self.particle_class.read(snapshot, mode="smooth")
            
            for halo_index in range(self.rs.shape[0]):
                is_tracked = self.rs[halo_index, snapshot]["ok"]

                if is_tracked:
                
                    coef_write_path = os.path.join(snapshot_directory, f"coef_subhalo_{halo_index}.coef_mul")
        
                    if os.path.exists(coef_write_path):
                        print(f"Found halo {halo_index} coefficient file for snapshot {snapshot}.")
                    else:
                        
                        q = particles[halo_index]["x"]
                        ok = particles[halo_index]["ok"]

                        if np.sum(ok) < 100:
                            continue
    
                        subhalo_pos = self.rs[halo_index, snapshot]["x"]
                        rvir = self.rs[halo_index, snapshot]["rvir"]
                        
                        masses = (np.ones(np.sum(ok)) * self.mp)
        
                        pot = agama.Potential(
                            type="multipole",
                            particles=(q[ok] - subhalo_pos, masses), # offset expansion by the subhalo position
                            symmetry="none",
                            lmax=8,
                            rmin=0.001,
                            rmax=300.,
                            center=subhalo_pos,
                        )
                        
                        pot.export(coef_write_path)
    
    ### possibly deprecate these ###
    def make_potential_cube(self, write_dir):
        if os.path.exists(write_dir):
            print("Found archived potential cube...")
            self.pot_cube = np.load(write_dir)["arr_0"]
        else:
            cube = np.zeros((self.rs.shape[0], self.rs.shape[1], 5)) * np.nan

            # parameters 0-3 are the fit parameters
            # fourth parameter is 0 for einasto, 1 for nfw
            # fifth parameter is the logrh (baryonic)

            for snapshot in tqdm(range(self.rs.shape[1])):
                particles = self.particle_class.read(snapshot, mode="all")

                for halo_index in range(self.rs.shape[0]):
                    is_tracked = self.rs[halo_index, snapshot]["ok"]

                    if is_tracked:
                        subhalo_pos = self.rs[halo_index, snapshot]["x"]
                        subhalo_vel = self.rs[halo_index, snapshot]["v"]
                        rvir = self.rs[halo_index, snapshot]["rvir"]

                        logrh = np.log10(rh_rvir_relation(rvir, True))

                        # make a function in dynamics that superimposes
                        # the baryonic component, but we'll store it now
                        # for funsies :)

                        q = particles[halo_index]["x"]
                        p = particles[halo_index]["v"]

                        q, p = get_bounded_particles(
                            q, p, subhalo_pos, subhalo_vel, self.params
                        )

                        params = None

                        def fit_einasto():
                            profile = SymphonyHaloProfile(
                                q, subhalo_pos, self.mp, rvir, a=self.a[snapshot]
                            )

                            l_conv = self.getConvergenceRadius(snapshot)
                            fit_output = profile.fit(l_conv)

                            params = [
                                fit_output["alpha"],
                                fit_output["Rs"],
                                fit_output["logScaleDensity"],
                            ]

                            print("Einasto: ", params)
                            cube[halo_index, snapshot, :3] = params
                            cube[halo_index, snapshot, 3] = 0

                        def fit_nfw():
                            print("Unable to fit Einasto, switching to NFW.")
                            params = [
                                self.rs[halo_index, snapshot]["m"],
                                self.rs[halo_index, snapshot]["rvir"],
                                self.rs[halo_index, snapshot]["cvir"],
                            ]
                            print("NFW: ", params)
                            cube[halo_index, snapshot, :3] = params
                            cube[halo_index, snapshot, 3] = 1

                        try:
                            fit_einasto()
                        except:
                            fit_nfw()

                        cube[halo_index, snapshot, 4] = logrh

            np.savez_compressed(write_dir, cube)

    def make_acceleration_cube(self, write_dir):
        # check if the acceleration cube has already been made

        if os.path.exists(write_dir):
            print("Found archived acceleration cube...")
            self.acc_cube = np.load(write_dir)["arr_0"]
        else:
            radial_bin_count = 100
            cube = (
                np.zeros((self.rs.shape[0], self.rs.shape[1], radial_bin_count))
                * np.nan
            )

            # loop over each snapshot
            for snapshot in tqdm(range(self.rs.shape[1])):
                particles = self.particle_class.read(snapshot, mode="all")

                # loop over each halo
                for halo_index in range(self.rs.shape[0]):
                    # check if that halo is trackable
                    # i.e., is it flagged 'ok' by rockstar?
                    is_tracked = self.rs[halo_index, snapshot]["ok"]

                    if is_tracked:
                        # Bulk subhalo properties
                        pos = self.rs[halo_index, snapshot]["x"]
                        vel = self.rs[halo_index, snapshot]["v"]
                        mass = self.rs[halo_index, snapshot]["m"]

                        rvir = self.rs[halo_index, snapshot]["rvir"]

                        # select for bound particles only
                        q = particles[halo_index]["x"]
                        p = particles[halo_index]["v"]

                        dq = q - pos
                        dp = p - vel

                        r = np.sqrt(np.sum(dq**2, axis=1))
                        order = np.argsort(r)

                        ke = np.sum(dp**2, axis=1) / 2
                        ok = np.ones(len(ke), dtype=bool)

                        for _ in range(3):
                            _, vmax, pe, _ = symlib.profile_info(self.params, dq, ok=ok)
                            E = ke + pe * vmax**2
                            ok = E < 0

                        print(
                            "halo loaded with ", len(q), "particles and r_vir =", rvir
                        )

                        profile = SymphonyHaloProfile(
                            q[ok], pos, self.mp, rvir, a=self.a[snapshot]
                        )

                        accelerations = profile.getRadialAccelerationProfile(
                            bins=radial_bin_count
                        )

                        cube[halo_index, snapshot, :] = accelerations
            # save cube as compressed numpy array
            np.savez_compressed(write_dir, cube)

    def make_mass_cube(self, write_dir):
    
            if os.path.exists(write_dir):
                print("Found archived enclosed mass cube...")
                self.acc_cube = np.load(write_dir)["arr_0"]
            else:
                radial_bin_count = 100
                cube = (
                    np.zeros((self.rs.shape[0], self.rs.shape[1], radial_bin_count))
                    * np.nan
                )
    
                # loop over each snapshot
                for snapshot in tqdm(range(self.rs.shape[1])):
                    particles = self.particle_class.read(snapshot, mode="all")
    
                    # loop over each halo
                    for halo_index in range(self.rs.shape[0]):
                        # check if that halo is trackable
                        # i.e., is it flagged 'ok' by rockstar?
                        is_tracked = self.rs[halo_index, snapshot]["ok"]
    
                        if is_tracked:
                            # Bulk subhalo properties
                            pos = self.rs[halo_index, snapshot]["x"]
                            vel = self.rs[halo_index, snapshot]["v"]
                            mass = self.rs[halo_index, snapshot]["m"]
    
                            rvir = self.rs[halo_index, snapshot]["rvir"]
    
                            # select for bound particles only
                            q = particles[halo_index]["x"]
                            p = particles[halo_index]["v"]
    
                            dq = q - pos
                            dp = p - vel
    
                            r = np.sqrt(np.sum(dq**2, axis=1))
                            order = np.argsort(r)
    
                            ke = np.sum(dp**2, axis=1) / 2
                            ok = np.ones(len(ke), dtype=bool)
    
                            for _ in range(3):
                                _, vmax, pe, _ = symlib.profile_info(self.params, dq, ok=ok)
                                E = ke + pe * vmax**2
                                ok = E < 0
    
                            print(
                                "halo loaded with ", len(q), "particles and r_vir =", rvir
                            )
    
                            profile = SymphonyHaloProfile(
                                q[ok], pos, self.mp, rvir, a=self.a[snapshot]
                            )
    
                            masses = profile.getMassProfile(
                                bins=radial_bin_count
                            )
    
                            cube[halo_index, snapshot, :] = masses
                # save cube as compressed numpy array
                np.savez_compressed(write_dir, cube)
###########################################################
# Utility functions that I probably should move... ########
def rh_rvir_relation(rvir, addScatter=True):
    # Kravstov 2013
    slope = 0.95
    normalization = 0.015
    scatter = 0.2  # dex

    rand = norm.rvs(loc=0, scale=0.2, size=1) if addScatter else 0.0

    log_rvir = np.log10(rvir)
    log_rh = slope * log_rvir + rand + np.log10(normalization)
    return 10**log_rh

def get_binding_energy(q, p, subhalo_pos, subhalo_vel, params):
    dq = q - subhalo_pos
    dp = p - subhalo_vel

    E = None

    r = np.sqrt(np.sum(dq**2, axis=1))
    ke = np.sum(dp**2, axis=1) / 2
    ok = np.ones(len(ke), dtype=bool)

    for i in range(3):
        _, vmax, pe, _ = symlib.profile_info(params, dq, ok=ok)
        E = ke + pe * vmax**2
        ok = E < 0

    return E


def get_bounded_particles(q, p, subhalo_pos, subhalo_vel, params):
    dq = q - subhalo_pos
    dp = p - subhalo_vel

    r = np.sqrt(np.sum(dq**2, axis=1))
    ke = np.sum(dp**2, axis=1) / 2
    ok = np.ones(len(ke), dtype=bool)

    for i in range(3):
        _, vmax, pe, _ = symlib.profile_info(params, dq, ok=ok)
        E = ke + pe * vmax**2
        ok = E < 0

        if (i == 3) and sort:
            ok = np.argsort(E[ok])

    return q[ok], p[ok]
