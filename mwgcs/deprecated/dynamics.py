import numpy as np
import abc

from tqdm import tqdm

from scipy.interpolate import UnivariateSpline, interp1d, LinearNDInterpolator
import symlib
from colossus.cosmology import cosmology


from .fit import NFW, Einasto, Plummer, get_tidal_strength


class Orbit(abc.ABC):
    def __init__(
        self,
        tracking_cube,
        gc_index,
        snapshot_times,
        acc_cube=None,
        pot_cube=None,
        sim_dir=None,
    ):

        self.gc_index = gc_index
        self.acc_cube = acc_cube
        self.pot_cube = pot_cube
        self.sim_dir = sim_dir

        # consider a single GC in the tracking catalog

        self.pos = tracking_cube[:, gc_index, :3]
        self.vel = tracking_cube[:, gc_index, 3:]

        self.snapshot_times = (
            snapshot_times  # this should be in Myr, shape of tracking_c
        )

        self.orbit_snapshots = np.arange(0, tracking_cube.shape[0])[
            ~np.isnan(tracking_cube[:, gc_index, 0])
        ]

        # self.start_snap, self.end_snap, self.snaps = start_snap, end_snap, snaps

        orbit_start = self.snapshot_times[self.orbit_snapshots[0]]
        orbit_end = self.snapshot_times[self.orbit_snapshots[-1]]
        orbit_time = orbit_end - orbit_start  # in Myr

        self.orbit_time = orbit_time
        self.t = None
        self.w = None

    def get_interpolated_times(self, dt=1.0):

        # splits the total orbit time
        nsteps = int(self.orbit_time / dt)
        interp_times = np.linspace(
            self.snapshot_times[self.orbit_snapshots[0]],
            self.snapshot_times[self.orbit_snapshots[-1]],
            nsteps,
        )

        self.t = interp_times
        return interp_times

    def get_gc_phase_space(self, dt=1.0):

        self._check_t()

        orbit_times = self.snapshot_times[self.orbit_snapshots]
        x, y, z = self.pos.T[:, self.orbit_snapshots]

        xs = UnivariateSpline(orbit_times, x, s=0)
        ys = UnivariateSpline(orbit_times, y, s=0)
        zs = UnivariateSpline(orbit_times, z, s=0)

        w_x, w_y, w_z = xs(self.t), ys(self.t), zs(self.t)

        # velocities
        vxs = xs.derivative(1)  # kpc / Myr
        vys = ys.derivative(1)
        vzs = zs.derivative(1)

        w_vx, w_vy, w_vz = vxs(self.t), vys(self.t), vzs(self.t)

        w = np.array([w_x, w_y, w_z, w_vx, w_vy, w_vz])

        self.w = w

        return w

    @abc.abstractmethod
    def get_accelerations(self):
        pass

    def _check_w(self, dt=1.0):
        if self.w is None:
            print("Interpolated phase space not computed. Computing...")
            self.get_gc_phase_space(dt)
            print("Done.")

    def _check_t(self, dt=1.0):
        if self.t is None:
            self.get_interpolated_times(dt)


class SymphonyOrbit(Orbit):

    def __init__(self, tracking_cube, acc_cube, pot_cube, gc_index, sim_dir):

        # symlib
        params = symlib.simulation_parameters(sim_dir)
        self.rs, hist = symlib.read_rockstar(sim_dir)
        self.um = symlib.read_um(sim_dir)
        col_params = symlib.colossus_parameters(params)
        cosmo = cosmology.setCosmology("cosmo", params=col_params)
        scale_factors = np.array(symlib.scale_factors(sim_dir))

        z = (1 / scale_factors) - 1

        snapshot_times = cosmo.hubbleTime(z) * 1000  # in Myr

        # find a way such to pull this from metadata
        self.radial_bins = np.logspace(-3.0, 5.0, 100)

        super().__init__(
            tracking_cube, gc_index, snapshot_times, acc_cube=acc_cube, sim_dir=sim_dir, pot_cube=pot_cube)

        if self.sim_dir is None:
            raise ValueError(
                "Symlib needs to read the stored directory but it has not been provided."
            )

        self.rs = symlib.read_rockstar(self.sim_dir)[0]

    def get_snap_acc(self):

        if self.acc_cube is None:
            raise ValueError("No acceleration catalog provided!")

        q = self.pos[self.orbit_snapshots]
        self.orbit_radii = np.log10(np.sqrt(np.sum(q**2, axis=1)))
        self.orbit_acc = np.zeros(len(self.orbit_radii)) * np.nan

        for i, snapshot in tqdm(enumerate(self.orbit_snapshots)):

            # cube is shaped [halo_id, snapshot, radius]
            central_id = 0

            central_acceleration_profile = self.acc_cube[
                central_id, snapshot, :
            ]  # grab all accelerations
            phys_radii = (
                self.radial_bins * self.rs[central_id, snapshot]["rvir"]
            )  # convert to radii in kpc instead of r/rvir
            nan_mask = np.isnan(central_acceleration_profile)

            interp = interp1d(
                np.log10(phys_radii[~nan_mask]),
                np.log10(central_acceleration_profile[~nan_mask]),
                kind="linear",
                fill_value=0.0,
                bounds_error=False,
            )

            self.orbit_acc[i] = np.power(10.0, interp(self.orbit_radii[i]))

        return self.orbit_acc

    def accelerations(self, q, t):
        central_id = 0
        snapshot_idx = 0
        snap_times = self.snapshot_times[self.orbit_snapshots]

        # check which snapshot we should interpolate from
        if (t > self.snapshot_times[self.orbit_snapshots]).any():
            snapshot_idx = np.where((t > self.snapshot_times[self.orbit_snapshots]))[0][
                -1
            ]

        snapshot = self.orbit_snapshots[snapshot_idx]

        # can't interpolate after end of simulation
        if snapshot == 235:
            return 0.0

        # current radius
        r = np.sqrt(np.sum(q**2, axis=0))

        acc_i = self.acc_cube[central_id, snapshot, :]
        r_i = self.rs[central_id, snapshot]["rvir"] * self.radial_bins
        t_i = self.snapshot_times[snapshot]

        # check which bins the current radius is between
        r_1i_idx = np.where(r >= r_i)[0][-1]
        r_2i_idx = r_1i_idx + 1

        r_1i = r_i[r_1i_idx]
        r_2i = r_i[r_2i_idx]
        a_1i = acc_i[r_1i_idx]
        a_2i = acc_i[r_2i_idx]

        #### do same thing for next snapshot #####

        acc_k = self.acc_cube[central_id, snapshot + 1, :]
        r_k = self.rs[central_id, snapshot + 1]["rvir"] * self.radial_bins
        t_k = self.snapshot_times[snapshot + 1]

        r_1k_idx = np.where(r >= r_k)[0][-1]
        r_2k_idx = r_1k_idx + 1

        r_1k = r_k[r_1k_idx]
        r_2k = r_k[r_2k_idx]
        a_1k = acc_k[r_1k_idx]
        a_2k = acc_k[r_2k_idx]

        # log protect
        if a_1k == 0.0:
            a_1k = 1e-99

        if a_2k == 0.0:
            a_2k = 1e-99

        if a_1i == 0.0:
            a_1i = 1e-99

        if a_2i == 0.0:
            a_2i = 1e-99

        # now get acceleration slopes m
        m_i = get_log_slope(a_2i, a_1i, r_2i, r_1i)
        m_k = get_log_slope(a_2k, a_1k, r_2k, r_1k)

        log_a_i = log_point_slope(r, a_1i, r_1i, m_i)
        log_a_k = log_point_slope(r, a_1k, r_1k, m_k)

        # interpolate across time

        m_t = (log_a_k - log_a_i) / (t_k - t_i)
        evaluated_log_a = semilog_point_slope(t, 10**log_a_k, t_k, m_t)

        return 10**evaluated_log_a

    def get_accelerations(self):
        # calculates the acceleration vector of a test particle

        accs = []

        central_id = 0
        snap_times = self.snapshot_times[self.orbit_snapshots]

        for i, t in enumerate(tqdm(self.t[:-1])):

            snapshot_idx = 0

            if (t > self.snapshot_times[self.orbit_snapshots]).any():
                snapshot_idx = np.where(
                    (t > self.snapshot_times[self.orbit_snapshots])
                )[0][-1]
            snapshot = self.orbit_snapshots[snapshot_idx]

            if snapshot == 235:
                break

            # current radius
            r = np.sqrt(np.sum(self.w[:3, i] ** 2))
            acc_i = self.acc_cube[central_id, snapshot, :]
            r_i = self.rs[central_id, snapshot]["rvir"] * self.radial_bins
            t_i = self.snapshot_times[snapshot]

            # check which bins the current radius is between
            r_1i_idx = np.where(r >= r_i)[0][-1]
            r_2i_idx = r_1i_idx + 1

            r_1i = r_i[r_1i_idx]
            r_2i = r_i[r_2i_idx]
            a_1i = acc_i[r_1i_idx]
            a_2i = acc_i[r_2i_idx]

            #### do same thing for next snapshot #####

            acc_k = self.acc_cube[central_id, snapshot + 1, :]
            r_k = self.rs[central_id, snapshot + 1]["rvir"] * self.radial_bins
            t_k = self.snapshot_times[snapshot + 1]

            r_1k_idx = np.where(r >= r_k)[0][-1]
            r_2k_idx = r_1k_idx + 1

            r_1k = r_k[r_1k_idx]
            r_2k = r_k[r_2k_idx]
            a_1k = acc_k[r_1k_idx]
            a_2k = acc_k[r_2k_idx]

            # log protect
            if a_1k == 0.0:
                a_1k = 1e-99

            if a_2k == 0.0:
                a_2k = 1e-99

            if a_1i == 0.0:
                a_1i = 1e-99

            if a_2i == 0.0:
                a_2i = 1e-99

            # now get acceleration slopes m
            m_i = get_log_slope(a_2i, a_1i, r_2i, r_1i)
            m_k = get_log_slope(a_2k, a_1k, r_2k, r_1k)

            log_a_i = log_point_slope(r, a_1i, r_1i, m_i)
            log_a_k = log_point_slope(r, a_1k, r_1k, m_k)

            # interpolate across time

            m_t = (log_a_k - log_a_i) / (t_k - t_i)
            evaluated_log_a = semilog_point_slope(t, 10**log_a_k, t_k, m_t)
            accs.append(10**evaluated_log_a)

        return accs

    def get_orbital_tidal_strength(self, q, t):
        
        central_id = 0
        
        snap_times = self.snapshot_times[self.orbit_snapshots]

        snapshot_idx = 0

        if (t > self.snapshot_times[self.orbit_snapshots]).any():
            snapshot_idx = np.where((t > self.snapshot_times[self.orbit_snapshots]))[0][
                -1
            ]
        snapshot = self.orbit_snapshots[snapshot_idx]

        if snapshot == 235:
            return 0.0

        # current radius
        r = np.sqrt(np.sum(q**2))

        # let i be the current snapshot, and k = i+1, because I already have
        # variables with numbers

        pot_i_type = self.pot_cube[central_id, snapshot, 3]
        pot_i_params = self.pot_cube[central_id, snapshot, :3]
        pot_i_rh = np.power(10., self.pot_cube[central_id, snapshot, 4])

        pot_k_type = self.pot_cube[central_id, snapshot + 1, 3]
        pot_k_params = self.pot_cube[central_id, snapshot + 1, :3]
        pot_k_rh = np.power(10., self.pot_cube[central_id, snapshot + 1, 4])

        def get_potential(pot_type, pot_params):
            if pot_type == 0:
                alpha, Rs, logScaleDensity = pot_params
                return Einasto(alpha, Rs, logScaleDensity)
            elif pot_type == 1:
                mvir, rvir, cvir = pot_params
                return NFW(mvir, rvir, cvir)
        
        pot_i = get_potential(pot_i_type, pot_i_params)
        pot_k = get_potential(pot_k_type, pot_k_params)
        
        r_i = np.sqrt(np.sum(self.pos[snapshot] ** 2))
        t_i = self.snapshot_times[snapshot]
        r_k = np.sqrt(np.sum(self.pos[snapshot + 1] ** 2))
        t_k = self.snapshot_times[snapshot + 1]

        gal_i = Plummer(self.um[central_id, snapshot]['m_star'], Plummer.half_radius_to_scale(pot_i_rh))
        hess_i = pot_i.hessian(r_i) + gal_i.hessian(r_i)
        lam_i = get_tidal_strength(hess_i)

        gal_k = Plummer(self.um[central_id, snapshot + 1]['m_star'], Plummer.half_radius_to_scale(pot_k_rh))
        self._test_pot_k = pot_k
        self._test_gal_k = gal_k
        self._test_r_k = r_k
        
        hess_k = pot_k.hessian(r_k) + gal_k.hessian(r_k)
        lam_k = get_tidal_strength(hess_k)


        m_t = (lam_k - lam_i) / (t_k - t_i)

        # interpolate across time
        evaluated_strength = m_t * (t - t_k) + lam_k

        return evaluated_strength


########################################################################################################


class Potential(abc.ABC):

    def __init__(self, params):

        self.params = params

    @abc.abstractmethod
    def getTidalStrengthProfile(self, **kwargs):
        pass

    @abc.abstractmethod
    def hessian(self, q, **kwargs):
        return


class HybridPotential(Potential):

    def __init__(self, params, profile_type):

        self.params = params
        self.profile_type = profile_type


########################################################################################################


class MassLossModel(abc.ABC):

    def __init__(self, orbit: Orbit, m0, potential_catalog, **kwargs):

        self.orbit = orbit
        self.m0 = m0  # initial mass in Msun
        self.potential_catalog = potential_catalog  # potential

    @abc.abstractmethod
    def get_interpolated_potential(self, **kwargs):
        pass

    @abc.abstractmethod
    def rate(self, **kwargs):
        # units of Msun/Myr
        pass

    @abc.abstractmethod
    def mass(self, **kwargs):
        # units of Msun
        pass

    @abc.abstractmethod
    def destruction_time(self, **kwargs):
        # units of Myr
        pass


########################################################################################################

# class SymphonyMassLoss(MassLossModel):
#     def __init__(self, qs, ts, m0, potential_catalog):
#         super().__init__(qs, ts, m0, potential_catalog)

#     def get_interpolated_potential(self, **kwargs):
#         # self.potential_catalog

#     def get_interpolated_lambda(self):
#         pass

# class SymphonyMassLoss(MassLossModel):
#     # include some tidal shocking prescription?
#     # include ext_pot
#     pass

########################################################################################################


class Integrator:
    def __init__(self, w0, acc, ts):
        self.w0 = w0  # shape is (6,)
        self.acc = acc
        self.ts = ts
        self.steps = len(ts)

        self.w = np.zeros((6, self.steps))
        self.w[:, 0] = self.w0

    @abc.abstractmethod
    def step(self, step):
        # do something to the current phase space position!
        new_w_i = np.zeros((6,))
        self.w[:, step] = new_w_i


class LeapfrogIntegrator(Integrator):

    def __init__(self, w0, acc, ts):

        super().__init__(w0, acc, ts)
        self.dt = ts[1] - ts[0]

    def get_normal_vectors(self, q):
        normal = -q / np.linalg.norm(q)
        return normal

    def loop(self, nsteps):

        q = self.w0[:3]
        p = self.w0[3:]

        self.w[:3, 0] = q
        self.w[3:, 0] = p

        a0 = self.acc(q, self.ts[0]) * self.get_normal_vectors(q)

        # kick step
        p_half = p + (self.dt / 2.0 * a0)

        for k in tqdm(range(1, nsteps)):

            # drift
            q = q + (self.dt / 2.0 * p_half)
            self.w[:3, k] = q

            t_i = self.ts[k]
            a_i = self.acc(q, t_i) * self.get_normal_vectors(q)

            p_half = p_half + (self.dt * a_i / 2)

            self.w[3:, k] = p_half


########################################################################################################


class Stream(abc.ABC):
    def __init__(self, qs, tstar, m0, pot):
        pass


########################################################################################################


def get_spherical_representation(w):

    x, y, z, vx, vy, vz = w

    w_r = (x**2 + y**2 + z**2) ** 0.5
    w_theta = np.arctan(y / x)
    w_phi = np.arcsin(z / w_r)

    w_vr = (x * vx + v * vy + z * vz) / r
    w_vtheta = (vx * y - x * vy) / (x**2 + y**2)
    w_vphi = (z * (x * vx + y * vy) - (x**2 + y**2) * vx) / (
        w_r**2 * (x**2 + y**2) ** (1 / 2)
    )

    w_sph = np.array([w_r, w_theta, w_phi, w_vr, w_vtheta, w_vphi])

    return w_sph


def get_cartesian_representation(w_sph):

    r, theta, phi, vr, vtheta, vphi = w_sph

    w_x = r * np.cos(phi) * np.cos(theta)
    w_y = r * np.cos(phi) * np.sin(theta)
    w_z = r * np.sin(phi)

    w_vx = (
        np.cos(phi) * np.cos(theta) * vr
        + r * np.cos(phi) * np.sin(theta) * vtheta
        + r * np.sin(phi) * np.cos(theta) * vphi
    )

    w_vy = (
        np.cos(phi) * np.sin(theta) * vr
        - r * np.cos(phi) * np.cos(theta) * vtheta
        + r * np.sin(phi) * np.sin(theta) * vphi
    )

    w_vz = np.sin(phi) * vr - r * np.cos(phi) * vphi

    w_car = np.array([w_x, w_y, w_z, w_vx, w_vy, w_vz])

    return w_car


####


# gets the power law slope between two points (in linear space)
def get_log_slope(y1, y2, x1, x2):
    return (np.log10(y2) - np.log10(y1)) / (np.log10(x2) - np.log10(x1))


# returns the power law given a point on the line and the power law slope
def log_point_slope(x, yk, xk, mk):
    return np.log10(yk) + mk * (np.log10(x) - np.log10(xk))

# returns the power law given that the x-axis is linear scale
def semilog_point_slope(x, yk, xk, mk):
    return np.log10(yk) + mk * (x - xk)
