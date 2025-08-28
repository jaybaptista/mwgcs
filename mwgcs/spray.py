import numpy as np 
import agama
from tqdm import tqdm
agama.setUnits(length=1.,mass=1.,velocity=1.)
from itertools import chain
import os

from .evolve import evolve_stellar_mass


def compute_jacobi_radius(pot_host, sat_orbit, sat_mass):
    N = int(np.sum(sat_mass > 0.0))
    x, y, z, vx, vy, vz = sat_orbit[:N, :].T

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
    der = pot_host.eval(sat_orbit[:N, :3], der=True)
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
    rj = (agama.G * sat_mass[:N] / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)
    return rj


def gen_stream_ic_dmdt(pot_host, sat_orbit, sat_mass, ml, times, seed=None):
    # This version of the initial condition generator
    # produces particles with masses sampled from its initial mass function.

    # times is the array of times since accretion

    if seed is not None:
        np.random.seed(seed)

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

    N = int(np.sum(sat_mass >= 0.0))
    x, y, z, vx, vy, vz = sat_orbit[:N, :].T

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
    der = pot_host.eval(sat_orbit[:N, :3], der=True)

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
    rj = (agama.G * sat_mass[:N] / (Omega**2 - d2Phi_dr2)) ** (1.0 / 3)

    dts = np.diff(times)

    # Obtain initial mass function
    imf = ml.spop_init

    # Obtain how much mass lost in a given time bin
    # Integrate ml.rlx_dmdt over time to get cumulative mass loss, then compute dm in each interval
    # Use trapezoidal rule for mass loss integration
    dm = np.abs(0.5 * (ml.rlx_dmdt[:-1] + ml.rlx_dmdt[1:]) * dts)

    dm[sat_mass[1:] == 0.0] = 0.0


    dm_masses = []
    initial_masses = []
    lifetimes = []

    pool = imf.copy()
    pool_initial = imf.copy()
    pool_lifetimes = ml.lifetimes

    exp_ndraws = []
    ndraws = []

    for i, delta_m in enumerate(tqdm(dm)):

        assert len(pool) == len(pool_initial) == len(pool_lifetimes), (
                f"pool, pool_initial, and pool_lifetimes must have the same length! "
                f"Got: {len(pool)}, {len(pool_initial)}, {len(pool_lifetimes)}"
            )

        if delta_m < np.min(pool):
            dm_masses.append([0])
            initial_masses.append([0])
            lifetimes.append([0])
            continue

        sampled = []

        # Shuffle IMF pool for random sampling without replacement
        idx = np.arange(len(pool))
        np.random.shuffle(idx)
        pool = pool[idx]
        pool_initial = pool_initial[idx]
        pool_lifetimes = pool_lifetimes[idx]
        
        mu = np.mean(pool)

        N_draw = np.floor(delta_m / mu)

        exp_ndraws.append(delta_m / mu)

        # Get fractional part
        f_draw = (delta_m / mu) - N_draw

        if np.random.uniform() < f_draw:
            N_draw += 1
        
        ndraws.append(N_draw)

        if N_draw == 0:
            dm_masses.append([0])
            initial_masses.append([0])
            lifetimes.append([0])
            continue

        sampled = pool[:int(N_draw)].copy()
        sampled_initial = pool_initial[:int(N_draw)].copy()
        sampled_lifetimes = pool_lifetimes[:int(N_draw)].copy()

        dm_masses.append(sampled.copy())
        initial_masses.append(sampled_initial.copy())
        lifetimes.append(sampled_lifetimes.copy())

        pool = np.delete(pool, np.arange(int(N_draw)))
        pool_initial = np.delete(pool_initial, np.arange(int(N_draw)))
        pool_lifetimes = np.delete(pool_lifetimes, np.arange(int(N_draw)))

        # Appends the released population of stars along with
        # their initial masses (for determining photometry later).
        # update

        pool = evolve_stellar_mass(pool_initial.copy(), pool_lifetimes.copy(), times[i])

        if len(pool) == 0:
            break

        # This assertion passes.
        try:
            assert len(sampled) == len(sampled_initial)
        except:
            print(f"Lengths are inconsistent, {sampled_initial}, {sampled}")
            raise 

    # CG24
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

    ics = []
    release_idx = []

    # Generate initial escape conditions for each particle in each bin
    smasses_unflattened = []
    smasses_init_unflattened = []
    smasses_lifetimes_unflattened = []

    for i in range(len(dm_masses)):

        count = len(dm_masses[i])

        # This assertion fails.
        try:
            assert len(initial_masses[i]) == len(dm_masses[i])
        except:
            print(f"Lengths are inconsistent, {initial_masses[i]}, {dm_masses[i]}")
            raise 

        if count > 1:
            # if np.all(np.array(dm_masses[i]) > 0):

            smasses_unflattened.append(dm_masses[i])
            smasses_init_unflattened.append(initial_masses[i])
            smasses_lifetimes_unflattened.append(lifetimes[i])

            pq = np.random.multivariate_normal(mean, cov, size=count)
            Dr = rj[i] * pq[:, 0]
            Dr[Dr == 0] = 1e-5  # Prevent division by zero

            v_esc = np.sqrt(2 * agama.G * np.repeat(sat_mass[i], count) / Dr)
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

            ic = np.tile(sat_orbit[i], count).reshape(count, 6)
            ic[:, :3] += np.einsum("ni,ij->nj", dq * lp[:, None], R[i, :, :])
            ic[:, 3:6] += np.einsum("ni,ij->nj", dp * lp[:, None], R[i, :, :])
            # Save the initial conditions
            ics.append(ic)
            # Save the temporal index of release
            release_idx.append(np.repeat(i, count))

    # Reformatting the shape of the ics

    # Assert that dm_masses and initial_masses have the same length
    if len(dm_masses) != len(initial_masses):
        raise ValueError(
            f"dm_masses and initial_masses have different lengths!\n"
            f"dm_masses: {dm_masses}\n"
            f"initial_masses: {initial_masses}"
        )

    release_idx = np.array(list(chain.from_iterable(release_idx)))
    smasses = np.array(list(chain.from_iterable(smasses_unflattened)))
    smasses_initial = np.array(list(chain.from_iterable(smasses_init_unflattened)))
    smasses_lifetimes = np.array(list(chain.from_iterable(smasses_lifetimes_unflattened)))

    if len(ics) == 0:
        print("No stars emitted.")
        return None

    ics = np.vstack(ics)

    stars = {
        "eject_mass": smasses,
        "zams_mass": smasses_initial,
        "t_ms": smasses_lifetimes,
        "release_idx": release_idx,
        # values for diagnostics; remove in final:
        "release_mass": dm,
        "exp_draws": exp_ndraws,
        "draws": ndraws,
        "exp_eject_mass": dm,
    }

    return ics, stars


def gen_stream_prog(orbit, time, sat_mass, snapshot_dir="./tmp", scaleRadius=4e-3):
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)


    potential_path = os.path.join(snapshot_dir, "progenitor.ini")
    track_path = os.path.join(snapshot_dir, "progenitor.txt")

    # write potential at each time step
    for k, mass in enumerate(sat_mass):
        filename = os.path.join(snapshot_dir, f"potential_{k}.ini")
        with open(filename, "w") as f:
            f.write("[Potential]\n")
            f.write("type=Plummer\n")
            
            f.write(f"mass={mass}\n")
            f.write(f"scaleRadius={scaleRadius}\n")

    # link potentials together
    with open(potential_path, "w") as f:
        print(f"Progenitor potentials written.")
        f.write("[Potential prog]\n")
        f.write("type=Evolving\n")
        f.write("interpLinear=True\n")
        f.write(f"center=progenitor.txt\n")
        f.write("Timestamps\n")
        for k, t in enumerate(time):
            f.write(f"{t} potential_{k}.ini\n")

    # write orbit track
    with open(track_path, "w") as f:
        print(f"Progenitor track written to {track_path}.")
        for k, t in enumerate(time):
            f.write(f"{t} {' '.join(map(str, orbit[k, :]))} \n")