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
        self, halo_index, snapshots, mass, disrupt_snapshot, **kwargs
    ):
        self.halo_id = halo_index

        # NOTE: Sentinel value—central galaxy should have infall_snap of -1
        # and should be the first entry.

        self.infall_snap = snapshots
        self.infall_time = self.cosmic_time[snapshots]
        self.infall_a = self.a[snapshots]
        self.infall_mass = mass
        self.disrupt_snap = disrupt_snapshot

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

        # Obtain halo and particle data
        self.rs, self.hist = symlib.read_rockstar(self.sim_dir)
        self.sf, _ = symlib.read_symfind(self.sim_dir)
        self.um = symlib.read_um(self.sim_dir)
        self.part = symlib.Particles(self.sim_dir)

        # Define subhalo infall characteristics
        halo_id = np.arange(0, self.rs.shape[0], dtype=int)
        infall_snaps = self.hist["first_infall_snap"]
        infall_mass = self.um["m_star"][halo_id, infall_snaps]
        infall_halo_mass = self.rs["m"][halo_id, infall_snaps]

        # Get snapshot of subhalo disruption.
        ok = self.rs['ok']
        # Reverse along axis 1
        rev_idx = ok[:, ::-1].argmax(axis=1)
        has_true = ok.any(axis=1)
        # Compute last True index: (N - 1) - rev_idx
        last_true_idx = ok.shape[1] - 1 - rev_idx
        # Only keep valid rows, others set to -1
        disrupt_snap = np.where(has_true, last_true_idx + 1, -1)
        disrupt_snap[disrupt_snap == 236] = -1

        self.infall_snaps = infall_snaps
        self.infall_mass = infall_mass
        self.infall_halo_mass = infall_halo_mass
        self.disrupt_snaps = disrupt_snap

        # Obtain the pre-infall host indices.
        self.preinfall_host_idx = symlib.pre_infall_host(self.hist)

        # Get the galaxy halo model for star tagging process.
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

        # Interface with simulation outputs and generate initial conditions for
        # stream progenitors.
        
        self.assign_particle_tags(
            KGSampler, DwarfGCMF, write_dir=os.path.join(self.halo_label, "./ParticleTags.npz")
        )
        
        # DEPRECATE:
        # The dynamics of a DM particle cannot be fully reconstructed and the
        # streams come out noodle-y.
        # self.track_particles(write_dir=os.path.join(self.halo_label, "./ParticleTracks.npz"))
        
        self.approximate_sph_potential(write_dir=os.path.join(self.halo_label, "./pot_cube.npz"))

        # self.getConvergenceRadii(write_dir=os.path.join(self.halo_label, "./rconv.npz"))

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
        preinfall_host_idx = self.preinfall_host_idx[m]

        halo_indices_list = []
        infall_snap_list = []
        disrupt_snap_list = []
        gc_masses_list = []
        preinfall_host_idx_list = []

        for i, infall_mass in enumerate(tqdm(infall_masses)):
            # obtain individual GC masses for each GC system
            gc_masses = gc_mass_sampler(
            infall_mass,
            system_mass_sampler=system_mass_sampler,
            halo_mass=infall_halo_mass[i]
            )

            if gc_masses is None:
                continue
            else:
                gc_masses = np.array(gc_masses)

            halo_indices_list.append(np.repeat(halo_indices[i], len(gc_masses)))
            infall_snap_list.append(np.repeat(infall_snaps[i], len(gc_masses)))
            disrupt_snap_list.append(np.repeat(disrupt_snaps[i], len(gc_masses)))
            print(len(gc_masses))
            gc_masses_list.append(gc_masses)
            preinfall_host_idx_list.append(np.repeat(preinfall_host_idx[i], len(gc_masses)))

        # Prepare entries for the structured array.
        array_halo_indices = np.hstack(halo_indices_list)  # int
        array_infall_snap = np.hstack(infall_snap_list)  # int
        array_disrupt_snap = np.hstack(disrupt_snap_list)  # int
        array_gc_masses = np.hstack(gc_masses_list)  # float64
        array_preinfall_host_idx = np.hstack(preinfall_host_idx_list)  # int

        # Define structured array
        dtype = np.dtype(
            [
            ("halo_index", int),
            ("infall_snap", int),
            ("disrupt_snap", int),
            ("gc_mass", float),
            ("preinfall_host_idx", int)
            ]
        )

        # Load structured array with entries.
        result_array = np.empty(len(array_halo_indices), dtype=dtype)
        result_array["halo_index"] = array_halo_indices
        result_array["infall_snap"] = array_infall_snap
        result_array["disrupt_snap"] = array_disrupt_snap
        result_array["gc_mass"] = array_gc_masses
        result_array["preinfall_host_idx"] = array_preinfall_host_idx

        return result_array

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
                # particles = self.part.read(snap, mode="stars")

                # define the indices to assign tags to
                
                # only assign tags to halos which are currently infalling at this snapshot
                infall_snap_condition = particle_tag_arr["infall_snap"] == snap
                
                # only assign tags to halos that are infalling onto the host halo
                infall_halo_condition = arr["preinfall_host_idx"] == -1

                indices = np.where(infall_snap_condition & infall_halo_condition)[0]

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
                particles = self.part.read(snapshot, mode="stars")

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

    # DEPRECATE
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
                particles = self.part.read(snapshot, mode="smooth")
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
    
    # DEPRECATE
    def get_subhalo_bfe(self, write_dir):
        if not os.path.exists(write_dir):
            print("Creating halo directory...")
            os.mkdir(write_dir)
        for snapshot in tqdm(range(self.rs.shape[1])):
            snapshot_directory = os.path.join(write_dir, f"bfe_{snapshot}")
                
            if not os.path.exists(snapshot_directory):
                os.mkdir(snapshot_directory)

            print("Trackable subhalos: ", np.where(self.rs[:, snapshot]["ok"])[0])

            particles = self.part.read(snapshot, mode="all")
            
            for halo_index in range(self.rs.shape[0]):
                is_tracked = self.rs[halo_index, snapshot]["ok"]

                if is_tracked:
                
                    coef_write_path = os.path.join(snapshot_directory, f"coef_subhalo_{halo_index}.coef_mul")
        
                    if os.path.exists(coef_write_path):
                        print(f"Found halo {halo_index} coefficient file for snapshot {snapshot}.")
                    else:
                        
                        q = particles[halo_index]["x"]
                        ok = particles[halo_index]["ok"]
                        print("Particle count: ", sum(ok))
                        if np.sum(ok) < 20:
                            continue
    
                        subhalo_pos = self.rs[halo_index, snapshot]["x"]                        
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
    
    def approximate_sph_potential(self, write_dir):
        
        if os.path.exists(write_dir):
            
            print("Found archived potential cube.")
            self.pot_cube = np.load(write_dir)["arr_0"]
        
        else:
            
            # Cube columns:
            # [0-2] = fit parameters,
            # [3] = 0 (Einasto) or 1 (NFW),
            # [4] = log(rh) for baryons
            
            cube = np.zeros((self.rs.shape[0], self.rs.shape[1], 5)) * np.nan

            # Obtain 
            accreting_halos = np.where(self.preinfall_host_idx == -1)[0][1:]
            for s in tqdm(range(self.rs.shape[1])):
                
                # These are particles from the accreting subhalos that will
                # be loaded onto the central halo representation once they
                # become unbound from their host.

                x_stack = []
                v_stack = []

                # load everything in smooth
                particles = self.part.read(s, mode='smooth')
 
                for h in range(1, self.rs.shape[0]):

                    # Check if subhalo has not disrupted
                    intact = self.rs[h, s]["ok"]

                    # `merger_snap`

                    # Check if subhalo infalls onto the main halo
                    infall = True if (s >= self.hist["merger_snap"][h]) else False

                    if not infall:
                        continue

                    ok = is_bound(
                            particles[h]["x"],
                            particles[h]["v"],
                            self.rs[h, s]['x'],
                            self.rs[h, s]["v"],
                            self.params
                        )
                    
                    # Arbitrary particle cut to ensure a good fit
                    # particle_cut = np.sum(ok) > 40

                    if intact:

                        # Load unbound particles into the central halo
                        x_stack.append(particles[h]['x'][~ok])
                        v_stack.append(particles[h]['v'][~ok])

                        h_x = self.rs[h, s]["x"]
                        h_v = self.rs[h, s]["v"]
                        rvir = self.rs[h, s]["rvir"]

                        logrh = np.log10(rh_rvir_relation(rvir, True))

                        q = particles[h]["x"][ok]
                        p = particles[h]["v"][ok]

                        params = None

                        try:
                            profile = SymphonyHaloProfile(
                                q, h_x, self.mp, rvir, a=self.a[s]
                            )

                            l_conv = self.getConvergenceRadius(s)
                            fit_output = profile.fit(l_conv)

                            params = [
                                fit_output["alpha"],
                                fit_output["Rs"],
                                fit_output["logScaleDensity"],
                            ]

                            print("Einasto: ", params)
                            cube[h, s, :3] = params
                            cube[h, s, 3] = 0
                        except:
                            print("Unable to fit Einasto, switching to NFW.")
                            params = [
                                self.rs[h, s]["m"],
                                self.rs[h, s]["rvir"],
                                self.rs[h, s]["cvir"],
                            ]
                            print("NFW: ", params)
                            cube[h, s, :3] = params
                            cube[h, s, 3] = 1

                        cube[h, s, 4] = logrh
                    
                    elif (not intact):
                        # If fully disrupted or insufficient particle count,
                        # dump all particles into the central.
                        print(f"[{h}, {s}]: Fully disrupted/insufficient count, dumping particles into main halo...")
                        x_stack.append(particles[h]['x'])
                        v_stack.append(particles[h]['v'])
                    else:
                        # Do nothing.
                        print(f"[{h}, {s}]: Halos that have not infallen are not tracked.")
                        continue 

                # Perform fit on central halo
                particles = self.part.read(s, mode='smooth')
                
                x_stack.append(particles[0]['x'])
                v_stack.append(particles[0]['x'])

                x_c = np.vstack(x_stack)
                v_c = np.vstack(v_stack)

                try:
                    profile = SymphonyHaloProfile(
                        x_c, self.rs[0, s]["x"], self.mp, self.rs[0, s]["rvir"], a=self.a[s]
                    )

                    l_conv = self.getConvergenceRadius(s)
                    fit_output = profile.fit(l_conv)

                    params = [
                        fit_output["alpha"],
                        fit_output["Rs"],
                        fit_output["logScaleDensity"],
                    ]

                    print("Central Einasto: ", params)
                    cube[0, s, :3] = params
                    cube[0, s, 3] = 0
                except:
                    print("Unable to fit Einasto for central, switching to NFW.")
                    params = [
                        self.rs[0, s]["m"],
                        self.rs[0, s]["rvir"],
                        self.rs[0, s]["cvir"],
                    ]
                    print("Central NFW: ", params)
                    cube[0, s, :3] = params
                    cube[0, s, 3] = 1

                cube[0, s, 4] = np.log10(rh_rvir_relation(self.rs[0, s]["rvir"], True))
            
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
