from .evolve import ClusterMass, calculate_tidal_tensor, evolve_stellar_mass
from .spray import compute_jacobi_radius
import numpy as np
import agama
import os
from tqdm import tqdm
from itertools import chain
import time

agama.setUnits(length=1.0, mass=1.0, velocity=1.0)


class GC:
    def __init__(
        self,
        potential,
        w0,
        t0,
        tf,
        m0,
        npts,
        kappa=4.0,
        self_gravity=True,
        seed=None,
        evolve_mass=True,
        age=0.0,
        imf=None,
        sev=True,
        feh=-2,
        accuracy=1e-8,
        output_prefix="./",
        scaleRadius=6e-3,
        thread_count=4,
        constant=False,
    ):
        self.host_pot = potential
        self.masses = None
        self.seed = seed

        # setup progenitor properties
        self.w0 = w0
        self.t0 = t0
        self.tf = tf
        self.m0 = m0
        self.npts = npts
        self.self_gravity = self_gravity
        self.constant = constant

        os.makedirs(output_prefix, exist_ok=True)

        self.gc_path = os.path.join(output_prefix, "gc.npz")
        self.stream_path = os.path.join(output_prefix, "stream.npz")
        self.prog_dir = os.path.join(output_prefix, "prog")

        self.thread_count = thread_count
        self.accuracy = accuracy

        # mass loss properties
        self.kappa = kappa
        self.rj = None

        traj_size = self.npts

        self.prog_pot = None

        print("Starting GC tracer integration...")

        t_start = time.time()
        with agama.setNumThreads(self.thread_count):
            output = agama.orbit(
                potential=self.host_pot,
                ic=self.w0,
                timestart=self.t0,
                time=self.tf - self.t0,
                trajsize=traj_size,
                accuracy=accuracy,
            )
        t_elapsed = time.time() - t_start
        print(f"Orbit integration took {t_elapsed:.3f} s")

        self.prog_t, self.prog_w = output

        dts = np.diff(self.prog_t)

        t_start = time.time()
        ml = ClusterMass(
            self.m0, age, self.kappa, self.tf, imf=imf, sev=sev, Z=10**feh
        )
        t_elapsed = time.time() - t_start
        print(f"Stellar evolution took {t_elapsed:.3f} s")

        self.tts = []

        if self.masses is None:
            if evolve_mass:
                tts = calculate_tidal_tensor(
                    self.host_pot, self.prog_w[:, :3], t=self.prog_t
                )
                masses = ml.evolve(age, dts, tts)
                self.masses = masses
            else:
                self.masses = np.ones(self.prog_w.shape[0]) * self.m0

        self.rj = compute_jacobi_radius(self.host_pot, self.prog_w, self.masses)
        self.t = self.prog_t

        self.pot = self.host_pot

        self.ml = ml

        np.savez(
            self.gc_path,
            track=self.prog_w,
            m=self.masses,
            t=self.prog_t,
            rj=self.rj,
            # mass loss collected values
            rlx_dmdt=ml.rlx_dmdt,
            sev_dmdt=ml.ev_dmdt,
            strength=ml.strengths,
        )

        self.gen_stream_prog(self.prog_dir, scaleRadius)

    def gen_stream_ic(self):
        """
        Source: Bill Chen
        Generate initial conditions for escaping particles using CG 2024 method.

        Args:
            pot_host:  Agama potential instance for host galaxy.
            sat_orbit: (N, 6) array of satellite positions and velocities.
            sat_mass:  Satellite mass (single value or array of length N).
            ml: MassLoss class object (this holds the initial mass function)

        Returns:
            (2*N, 6) array of stream particle initial conditions.
        """

        N = int(np.sum(self.masses >= 0.0))
        x, y, z, vx, vy, vz = self.prog_w[:N, :].T

        # Compute angular momentum components
        Lx, Ly, Lz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
        r = np.sqrt(x**2 + y**2 + z**2)
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

        # Rotation matrices from host to satellite frame
        R = np.zeros((N, 3, 3))
        R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = x / r, y / r, z / r
        R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = Lx / L, Ly / L, Lz / L
        R[:, 1, 0] = R[:, 0, 2] * R[:, 2, 1] - R[:, 0, 1] * R[:, 2, 2]
        R[:, 1, 1] = R[:, 0, 0] * R[:, 2, 2] - R[:, 0, 2] * R[:, 2, 0]
        R[:, 1, 2] = R[:, 0, 1] * R[:, 2, 0] - R[:, 0, 0] * R[:, 2, 1]

        # Compute second derivative of potential w.r.t. radius
        der = self.host_pot.eval(self.prog_w[:N, :3], der=True)

        d2Phi_dr2 = (
            -(
                x**2 * der[:, 0]
                + y**2 * der[:, 1]
                + z**2 * der[:, 2]
                + 2 * x * y * der[:, 3]
                + 2 * y * z * der[:, 4]
                + 2 * z * x * der[:, 5]
            )
            / r**2
        )

        # Compute Jacobi radius and velocity
        Omega = L / r**2

        rj = (agama.G * self.masses[:N] / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)

        mean = np.array([1.6, -30, 0, 1, 20, 0])

        cov = np.array(
            [
                [0.1225, 0, 0, 0, -4.9, 0],
                [0, 529, 0, 0, 0, 0],
                [0, 0, 144, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-4.9, 0, 0, 0, 400, 0],
                [0, 0, 0, 0, 0, 484],
            ]
        )

        if not self.constant:
            imf = self.ml.spop_init
            # Obtain how much mass lost in a given time bin
            dm = np.abs(np.diff(self.masses))
            dts = np.diff(self.prog_t)

            dm_masses = []

            pool = imf.copy()

            for i, delta_m in enumerate(tqdm(dm)):
                if delta_m < np.min(pool):
                    dm_masses.append([0])
                    continue

                sampled = []

                # Shuffle IMF pool for random sampling without replacement
                np.random.shuffle(pool)
                mu = np.mean(pool)

                N_draw = np.floor(delta_m / mu)

                # Get fractional part
                f_draw = (delta_m / mu) - N_draw

                if np.random.uniform() < f_draw:
                    N_draw += 1

                if N_draw == 0:
                    dm_masses.append([0])
                    continue

                sampled = pool[: int(N_draw)]

                # Remove sampled stars from pool
                pool = np.setdiff1d(pool, sampled, assume_unique=True)

                dm_masses.append(sampled)

                # update
                pool = evolve_stellar_mass(pool, dts[i], self.prog_t[i])

                if len(pool) == 0:
                    break

            ics = []
            release_idx = []

            # Generate initial escape conditions for each particle in each bin
            for i in range(len(dm_masses)):
                count = len(dm_masses[i])
                if count > 0:
                    pq = np.random.multivariate_normal(mean, cov, size=count)
                    Dr = rj[i] * pq[:, 0]
                    v_esc = np.sqrt(2 * agama.G * np.repeat(self.masses[i], count) / Dr)
                    Dv = pq[:, 3] * v_esc

                    phi = pq[:, 1] * np.pi / 180
                    theta = pq[:, 2] * np.pi / 180
                    alpha = pq[:, 4] * np.pi / 180
                    beta = pq[:, 5] * np.pi / 180

                    dx = Dr * np.cos(theta) * np.cos(phi)
                    dy = Dr * np.cos(theta) * np.sin(phi)
                    dz = Dr * np.sin(theta)
                    dvx = Dv * np.cos(beta) * np.cos(alpha)
                    dvy = Dv * np.cos(beta) * np.sin(alpha)
                    dvz = Dv * np.sin(beta)

                    dq = np.column_stack([dx, dy, dz])
                    dp = np.column_stack([dvx, dvy, dvz])

                    # Randomnly assign the Lagrange point

                    lp = np.random.choice([-1, 1], count)

                    ic = np.tile(self.prog_w[i], count).reshape(count, 6)

                    ic[:, :3] += np.einsum("ni,ij->nj", dq * lp[:, None], R[i, :, :])
                    ic[:, 3:6] += np.einsum("ni,ij->nj", dp * lp[:, None], R[i, :, :])

                    # Save the initial conditions
                    ics.append(ic)

                    # Save the temporal index of release
                    release_idx.append(np.repeat(i, count))

            # Reformatting the shape of the ics

            release_idx = np.array(list(chain.from_iterable(release_idx)))
            smasses = np.array(list(chain.from_iterable(dm_masses)))
            ics = np.vstack(ics)

            return ics, release_idx, smasses
        else:
            rj = np.repeat(rj, 2)

            pq = np.random.multivariate_normal(mean, cov, size=2 * N)
            Dr = rj[i] * pq[:, 0]
            v_esc = np.sqrt(2 * agama.G * np.repeat(self.masses[:N], 2) / Dr)
            Dv = pq[:, 3] * v_esc

            phi = pq[:, 1] * np.pi / 180
            theta = pq[:, 2] * np.pi / 180
            alpha = pq[:, 4] * np.pi / 180
            beta = pq[:, 5] * np.pi / 180

            dx = Dr * np.cos(theta) * np.cos(phi)
            dy = Dr * np.cos(theta) * np.sin(phi)
            dz = Dr * np.sin(theta)
            dvx = Dv * np.cos(beta) * np.cos(alpha)
            dvy = Dv * np.cos(beta) * np.sin(alpha)
            dvz = Dv * np.sin(beta)

            dq = np.column_stack([dx, dy, dz])
            dp = np.column_stack([dvx, dvy, dvz])

            ic_stream = np.tile(self.prog_w[:N], 2).reshape(2 * N, 6)

            ic_stream[::2, 0:3] += np.einsum("ni,nij->nj", dq[::2], R)
            ic_stream[::2, 3:6] += np.einsum("ni,nij->nj", dp[::2], R)
            ic_stream[1::2, 0:3] += np.einsum("ni,nij->nj", -dq[1::2], R)
            ic_stream[1::2, 3:6] += np.einsum("ni,nij->nj", -dp[1::2], R)

            return ic_stream

    def gen_stream_prog(self, prog_dir, scaleRadius=6e-3):
        os.makedirs(prog_dir, exist_ok=True)

        for k, mass in enumerate(self.masses):
            filename = os.path.join(prog_dir, f"potential_{k}.ini")

            # Generate plummer approximation of cluster potential
            with open(filename, "w") as f:
                f.write("[Potential]\n")
                f.write("type=Plummer\n")
                f.write(f"mass={mass}\n")
                f.write(f"scaleRadius={scaleRadius}\n")

            # Link potentials together
            with open(os.path.join(prog_dir, "progenitor.ini"), "w") as f:
                f.write("[Potential prog]\n")
                f.write("type=Evolving\n")
                f.write("interpLinear=True\n")
                f.write(f"center=progenitor.txt\n")
                f.write("Timestamps\n")
                for k, t in enumerate(self.prog_t):
                    f.write(f"{t} {prog_dir}/potential_{k}.ini\n")

            # Write orbit centers
            with open(os.path.join(prog_dir, "progenitor.txt"), "w") as f:
                for k, t in enumerate(self.prog_t):
                    f.write(f"{t} {' '.join(map(str, self.prog_w[k, :]))} \n")

    def stream(self, checkpoints=10, t_final=None):
        ws = []

        if self.constant:
            self.ics = self.gen_stream_ic()
        else:
            self.ics, self.idx, self.smasses = self.gen_stream_ic()

        t_final = self.tf if t_final is None else t_final

        t_ej = self.prog_t[self.idx]
        t_start = t_ej[0]

        total_integ_time = t_final - t_start

        dt = total_integ_time / checkpoints

        start_times = np.linspace(
            t_start, t_start + total_integ_time, checkpoints, endpoint=False
        )

        for cp_time in start_times:
            # Particles are active if they have been ejected before or within the current window
            active = (t_ej < cp_time + dt) & (~np.isnan(self.ics).any(axis=1))

            # For each active particle, start integration at max(cp_time, ejection_time)
            t_start_particles = np.maximum(cp_time, t_ej[active])
            t_int = (cp_time + dt) - t_start_particles
            t_int[t_int < 0] = 0.0  # Safety check

            with agama.setNumThreads(self.thread_count):
                output = agama.orbit(
                    potential=self.pot,
                    ic=self.ics[active],
                    time=t_int,
                    timestart=t_start_particles,
                    trajsize=1,
                    accuracy=self.accuracy,
                )

            # extract active positions
            xv = np.vstack(output[:, 1])

            # update active positions
            self.ics[active] = xv

            # store active positions
            ws.append(xv)

        data = {
            "w": np.array(ws, dtype=object),
            "track": self.prog_w,
            "ics": self.ics,
            "m": self.masses,
            "t": self.prog_t,
            "rj": self.rj,
        }

        if not self.constant:
            data["smass"] = self.smasses
            data["t_ej"] = t_ej

        np.savez(self.stream_path, **data)


class StreamConstantSpray:
    def __init__(
        self, sat_xv, sat_m, sat_t, potential, sat_phi, stream_path, npts=500, accuracy=1e-12, thread_count = 4
    ):
        self.prog_w = sat_xv
        self.prog_t = sat_t
        self.masses = sat_m
        self.host_pot = potential
        self.sat_pot = sat_phi
        self.npts = npts
        self.accuracy = accuracy
        self.thread_count = thread_count

        self.rj = compute_jacobi_radius(potential, self.prog_w, self.masses)
        self.stream_path = stream_path

        self.pot = agama.Potential(self.host_pot, self.sat_pot)

    def gen_stream_ic(self):
        """
        Source: Bill Chen
        Generate initial conditions for escaping particles using CG 2024 method.

        Args:
            pot_host:  Agama potential instance for host galaxy.
            sat_orbit: (N, 6) array of satellite positions and velocities.
            sat_mass:  Satellite mass (single value or array of length N).
            ml: MassLoss class object (this holds the initial mass function)

        Returns:
            (2*N, 6) array of stream particle initial conditions.
        """

        N = int(np.sum(self.masses >= 0.0))
        x, y, z, vx, vy, vz = self.prog_w[:N, :].T

        # Compute angular momentum components
        Lx, Ly, Lz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
        r = np.sqrt(x**2 + y**2 + z**2)
        L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

        # Rotation matrices from host to satellite frame
        R = np.zeros((N, 3, 3))
        R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = x / r, y / r, z / r
        R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = Lx / L, Ly / L, Lz / L
        R[:, 1, 0] = R[:, 0, 2] * R[:, 2, 1] - R[:, 0, 1] * R[:, 2, 2]
        R[:, 1, 1] = R[:, 0, 0] * R[:, 2, 2] - R[:, 0, 2] * R[:, 2, 0]
        R[:, 1, 2] = R[:, 0, 1] * R[:, 2, 0] - R[:, 0, 0] * R[:, 2, 1]

        # Compute second derivative of potential w.r.t. radius
        der = self.host_pot.eval(self.prog_w[:N, :3], der=True)

        d2Phi_dr2 = (
            -(
                x**2 * der[:, 0]
                + y**2 * der[:, 1]
                + z**2 * der[:, 2]
                + 2 * x * y * der[:, 3]
                + 2 * y * z * der[:, 4]
                + 2 * z * x * der[:, 5]
            )
            / r**2
        )

        # Compute Jacobi radius and velocity
        Omega = L / r**2

        rj = (agama.G * self.masses[:N] / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)

        mean = np.array([1.6, -30, 0, 1, 20, 0])

        cov = np.array(
            [
                [0.1225, 0, 0, 0, -4.9, 0],
                [0, 529, 0, 0, 0, 0],
                [0, 0, 144, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-4.9, 0, 0, 0, 400, 0],
                [0, 0, 0, 0, 0, 484],
            ]
        )

        rj = np.repeat(rj, 2)
        
        pq = np.random.multivariate_normal(mean, cov, size=2 * N)
        Dr = rj[:(2*N)] * pq[:, 0]
        v_esc = np.sqrt(2 * agama.G * np.repeat(self.masses[:N], 2) / Dr)
        Dv = pq[:, 3] * v_esc

        phi = pq[:, 1] * np.pi / 180
        theta = pq[:, 2] * np.pi / 180
        alpha = pq[:, 4] * np.pi / 180
        beta = pq[:, 5] * np.pi / 180

        dx = Dr * np.cos(theta) * np.cos(phi)
        dy = Dr * np.cos(theta) * np.sin(phi)
        dz = Dr * np.sin(theta)
        dvx = Dv * np.cos(beta) * np.cos(alpha)
        dvy = Dv * np.cos(beta) * np.sin(alpha)
        dvz = Dv * np.sin(beta)

        dq = np.column_stack([dx, dy, dz])
        dp = np.column_stack([dvx, dvy, dvz])

        ic_stream = np.tile(self.prog_w[:N], 2).reshape(2 * N, 6)

        ic_stream[::2, 0:3] += np.einsum("ni,nij->nj", dq[::2], R)
        ic_stream[::2, 3:6] += np.einsum("ni,nij->nj", dp[::2], R)
        ic_stream[1::2, 0:3] += np.einsum("ni,nij->nj", -dq[1::2], R)
        ic_stream[1::2, 3:6] += np.einsum("ni,nij->nj", -dp[1::2], R)

        return ic_stream

    def stream(self, checkpoints=10, t_final=None):
        ws = []

        self.ics = self.gen_stream_ic()

        N = self.ics.shape[0] // 2
        self.idx = np.repeat(np.arange(N), 2)

        t_final = self.tf if t_final is None else t_final

        t_ej = self.prog_t[self.idx]
        t_start = t_ej[0]

        total_integ_time = t_final - t_start

        dt = total_integ_time / checkpoints

        start_times = np.linspace(
            t_start, t_start + total_integ_time, checkpoints, endpoint=False
        )

        for cp_time in start_times:
            # Particles are active if they have been ejected before or within the current window
            active = (t_ej < cp_time + dt) & (~np.isnan(self.ics).any(axis=1))

            # For each active particle, start integration at max(cp_time, ejection_time)
            t_start_particles = np.maximum(cp_time, t_ej[active])
            t_int = (cp_time + dt) - t_start_particles
            t_int[t_int < 0] = 0.0  # Safety check

            with agama.setNumThreads(self.thread_count):
                output = agama.orbit(
                    potential=self.pot,
                    ic=self.ics[active],
                    time=t_int,
                    timestart=t_start_particles,
                    trajsize=1,
                    accuracy=self.accuracy,
                )

            # extract active positions
            xv = np.vstack(output[:, 1])

            # update active positions
            self.ics[active] = xv

            # store active positions
            ws.append(xv)

        data = {
            "w": np.array(ws, dtype=object),
            "track": self.prog_w,
            "ics": self.ics,
            "m": self.masses,
            "t": self.prog_t,
            "rj": self.rj,
        }

        np.savez(self.stream_path, **data)
