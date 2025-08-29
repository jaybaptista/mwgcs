import numpy as np
import agama
from scipy.integrate import quad
from tqdm import tqdm


agama.setUnits(mass=1.0, length=1.0, velocity=1.0)


def evolve_stellar_mass(initial_mass, lifetimes, tf):
    """
    Function takes in an array of ZAMS masses [Msun] and main sequence lifetimes [Gyr]
    and returns their evolved mass after some time `tf` [Gyr].
    """

    # Making sure masses are numpy-friendly
    m0 = np.asarray(initial_mass, dtype=np.float64)
    remnant = m0.copy()  # Remnant mass array

    # Select stars that have evolved off main sequence after `tf`
    has_evolved = np.asarray(lifetimes, dtype=np.float64) < tf

    if not np.any(has_evolved):
        return remnant if isinstance(initial_mass, np.ndarray) else remnant.item()

    m = m0[has_evolved]

    # Mass evolution conditions and remnant masses are
    # chosen from Kruissjen 2009
    # [https://arxiv.org/abs/0910.4579]

    conds = [
        (m >= 1.0) & (m <= 8.0),  # white dwarf
        (m > 8.0) & (m <= 30.0),  # neutron star
        (m > 30.0),
    ]  # black hole

    choices = [
        0.109 * m + 0.394,  # white dwarf
        0.03636 * (m - 8.0) + 1.02,  # neutron star
        0.06 * (m - 30.0) + 8.3,
    ]  # black hole

    remnant[has_evolved] = np.select(conds, choices, default=m)
    return remnant if isinstance(initial_mass, np.ndarray) else remnant.item()


def get_evolution_state(m0, lifetimes, tf):
    """
    Returns an array of values that correspond to a star's evolutionary
    state.

    Key:
    0 — Not evolved
    1 — WD
    2 — NS
    3 — BH
    """

    s0 = np.zeros(len(lifetimes))

    has_evolved = lifetimes < tf

    wd_mask = (m0 >= 1.0) & (m0 <= 8.0) & has_evolved
    ns_mask = (m0 > 8.0) & (m0 <= 30.0) & has_evolved
    bh_mask = (m0 > 30.0) & has_evolved

    s0[wd_mask] = 1
    s0[ns_mask] = 2
    s0[bh_mask] = 3
    return s0


def t_main_sequence(mass, Z=0.01):
    """
    Calculates main sequence lifetime [NOTE: Myr] from Hurley+2001 based
    on the stellar mass and the metallicity of the stars
    Source: https://ui.adsabs.harvard.edu/abs/2000MNRAS.315..543H/abstract
    """

    mass = np.asarray(mass, dtype=np.float64)
    zeta = np.log10(Z / 0.02)

    # Coefficients evaluated with fixed zeta
    a1 = np.polyval([0.0, 2.327785e2, 1.231226e3, 2.053038e3, 1.593890e3], zeta)
    a2 = np.polyval([0.0, 7.411230e1, 5.772723e2, 1.483131e3, 2.706708e3], zeta)
    a3 = np.polyval([0.0, -1.391127e1, -6.795374e1, -1.048442e2, 1.466143e2], zeta)
    a4 = np.polyval([0.0, 5.571483e-3, 2.958542e-2, 4.564888e-2, 4.141960e-2], zeta)
    a5 = np.polyval([0.0, 0.0, 0.0, 0.0, 3.426349e-1], zeta)
    a6 = np.polyval([0.0, -4.470533, -6.008212, 1.758178, 1.949814e1], zeta)
    a7 = np.polyval([0.0, 0.0, 0.0, 0.0, 4.903830], zeta)
    a8 = np.polyval([0.0, -2.271549e-3, -2.750074e-3, 3.166411e-2, 5.212154e-2], zeta)
    a9 = np.polyval([0.0, 2.610989e-2, 9.231860e-2, -3.294936e-1, 1.312179], zeta)
    a10 = np.polyval([0.0, 0.0, 0.0, 0.0, 8.073972e-1], zeta)

    # Precompute powers of mass
    mass2 = mass**2
    mass4 = mass**4
    mass5_5 = mass**5.5
    mass7 = mass**7
    ma7 = mass**a7
    ma10 = mass**a10

    # Compute x and mu
    x = np.clip(0.95 - 0.03 * (zeta + 0.30103), 0.95, 0.99)
    mu = np.clip(1.0 - 0.01 * np.maximum(a6 / ma7, a8 + a9 / ma10), 0.5, 1.0)

    # Compute times
    t_BGB = (a1 + a2 * mass4 + a3 * mass5_5 + mass7) / (a4 * mass2 + a5 * mass7)
    t_hook = mu * t_BGB

    return np.maximum(t_hook, x * t_BGB)


# Initial Mass Functions


def sample_salpeter_imf(m_min, m_max, alpha, size):
    """
    Inverse transform sampling from Salpeter IMF: dN/dM ∝ M^-alpha
    """
    x = np.random.uniform(0, 1, size)
    pow = 1.0 - alpha
    return ((m_max**pow - m_min**pow) * x + m_min**pow) ** (1.0 / pow)


def imf(m, A=1):
    """
    Kroupa Initial Mass Function (IMF).
    """
    m1, m2, m3 = 0.08, 0.5, 1.0
    k0 = 1.0
    k1 = k0 * m1 ** (-0.3 + 1.3)
    k2 = k1 * m2 ** (-1.3 + 2.3)
    k3 = k2 * m3 ** (-2.3 + 2.3)

    if 0.01 < m < 0.08:
        return A * k0 * m ** (-0.3)
    elif 0.08 < m < 0.5:
        return A * k1 * m ** (-1.3)
    elif 0.5 < m < 1.0:
        return A * k2 * m ** (-2.3)
    elif 1.0 < m < 150.0:
        return A * k3 * m ** (-2.3)
    return 0.0


# Normalize the IMF
integral, _ = quad(imf, 0.01, 150)
norm_A = 1.0 / integral


def sample_kroupa_imf(N, A=norm_A):
    """
    Vectorized rejection sampling for the Kroupa IMF.
    """
    m_grid = np.logspace(np.log10(0.01), np.log10(150), 1000)
    imf_max = np.max([imf(m, A) for m in m_grid])

    samples = []
    batch_size = max(10000, N)
    while len(samples) < N:
        m = np.random.uniform(0.01, 150.0, batch_size)
        y = np.random.uniform(0, imf_max, batch_size)
        accepted = m[y < np.array([imf(mi, A) for mi in m])]
        samples.extend(accepted.tolist())
    return np.array(samples[:N])


def sample_ssp(total_mass, age, m_min=0.01, m_max=100, alpha=2.35, Z=0.01, imf=None):
    """
    Minimum mass cutoff is 0.394/(1 - 0.394)
    This is so remnant mass of the red dwarf matches the end mass of the
    white dwarf function.
    """

    masses = []
    lifetimes = []

    m = 0.0

    # sample in batches for speed
    batch_size = np.max([10, int(total_mass // 1e4)])

    while m < total_mass:
        if imf == "kroupa":
            sample = sample_kroupa_imf(batch_size)
        else:
            sample = sample_salpeter_imf(m_min, m_max, alpha, batch_size)

        lifetime = t_main_sequence(sample, Z=Z) / 1000
        mask = lifetime > age
        sample = sample[mask]
        lifetime = lifetime[mask]

        masses.extend(sample)
        lifetimes.extend(lifetime)

        m += sample.sum()

    m_init = np.array(masses)
    lifetimes = np.array(lifetimes)

    return m_init, lifetimes


def calculate_tidal_tensor(potential, pos, t=0.0):
    """
    Calculate the tidal tensor (Hessian of the potential) at given position(s).

    potential : an instance of an AGAMA potential
    pos       : position(s) as (3,) or (N,3) numpy array
    t         : time or array of times of length N

    Returns:
        tt : a (3,3) tensor if single pos, or (N,3,3) array if multiple
    """
    pos = np.atleast_2d(pos)  # Ensure pos is (N, 3)
    N = len(pos)

    if np.isscalar(t):
        t = np.full(N, t)
    elif len(t) != N:
        raise ValueError("Length of t must match number of positions")

    # Evaluate derivatives for each (pos[i], t[i])
    derivatives = np.array(
        [potential.eval(p, der=True, t=ti) for p, ti in zip(pos, t)]
    )  # (N, 6)

    # Extract second derivatives
    d2phidx2 = derivatives[:, 0]
    d2phidy2 = derivatives[:, 1]
    d2phidz2 = derivatives[:, 2]
    d2phidxdy = derivatives[:, 3]
    d2phidydz = derivatives[:, 4]
    d2phidzdx = derivatives[:, 5]

    # Construct tidal tensor for each position
    tidal_tensor = np.array(
        [
            [d2phidx2, d2phidxdy, d2phidzdx],
            [d2phidxdy, d2phidy2, d2phidydz],
            [d2phidzdx, d2phidydz, d2phidz2],
        ]
    )  # shape (3, 3, N)

    tidal_tensor = np.moveaxis(tidal_tensor, -1, 0)  # shape (N, 3, 3)

    return tidal_tensor[0] if tidal_tensor.shape[0] == 1 else tidal_tensor


def tidal_strength(tt):
    """
    A simple function that takes a tidal tensor or list of tidal tensors
    and computes the largest eigenvalue, which corresponds to the tidal
    strength.
    """
    if tt.ndim == 3:
        
        # tt has shape (3, 3, N) -> move the third axis to the front
        # so we get an array of shape (N, 3, 3) for vectorized eigval computation
        
        tt = np.moveaxis(tt, -1, 0)
        eigenvalues = np.linalg.eigvals(tt)  # shape (N, 3)

        if eigenvalues.shape[0] == 1:
            return np.max(np.abs(eigenvalues))

        return np.max(np.abs(eigenvalues), axis=1)  # shape (N,)
    
    else:
        
        # input tt has shape (3, 3)

        eigenvalues = np.linalg.eigvals(tt)
        return np.max(np.abs(eigenvalues))


class ClusterMass:

    """
    Class that evolves the stellar mass of a globular cluster by sampling an
    initial mass function, evolving the cluster to a specified age (to account
    for losses from stellar evolution), and incorporates mass loss due to relaxation
    from the tidal field experienced by the tracer cluster.
    """

    def __init__(
        self,
        initial_mass,
        initial_age,
        kappa=1.0,
        t_final=13.9,
        imf=None,
        sev=True,
    ):
        self.kappa = kappa
        self.m0 = initial_mass

        # Sample initial stellar population
        spop, lifetimes = sample_ssp(initial_mass, initial_age, imf=imf)
        self.spop_init = np.copy(spop)
        self.spop = np.copy(self.spop_init) # evolved stellar population
        self.mstar = np.sum(spop) # initial ZAMS stellar mass

        # Get remnant states at end of simulation
        self.remnant_type = get_evolution_state(spop, lifetimes, t_final)
        self.sev = sev

        # Estimate lifetimes of the stellar population
        self.lifetimes = lifetimes

    def evolve(self, t0, dts, tts):
        """
        A routine that evolves the mass of the cluster. All units of time are in
        Gyr.
        
        t0:  age of cluster (i.e., time since ZAMS)
        dts: time between tidal tensor calculations
        tts: a list of tidal tensors evaluated at each position of the cluster
        
        """

        steps = len(dts)
        masses = np.zeros(steps + 1)

        self.rlx_dmdt = np.zeros(steps + 1)
        self.ev_dmdt = np.zeros(steps + 1)
        self.strengths = np.zeros(steps + 1)
        
        masses[0] = self.mstar # set initial stellar mass

        t = t0 + np.cumsum(dts)

        for k in tqdm(range(1, len(masses)), desc="Evolving cluster mass"):
            rate = 0.0

            # Evolve mass function if stellar evolution is enabled
            if self.sev:
                evolved_spop = evolve_stellar_mass(
                    self.spop_init, self.lifetimes, t[k - 1]
                )

                # Calculate change in stellar mass due to remnant mass ejection
                self.ev_dmdt[k - 1] = (
                    np.abs(np.sum(self.spop) - np.sum(evolved_spop)) / dts[k - 1]
                )

                rate += self.ev_dmdt[k - 1]
                
                # Update running mass function
                self.spop = evolved_spop

            # Calculate most negative eigenvalue of the tidal tensor
            lam = tidal_strength(tts[k - 1, :, :])

            self.strengths[k - 1] = lam

            # Evaluate mass lost from two-body relaxation as described in
            # Chen & Gnedin 2023
            
            omega_tid = self.kappa * np.sqrt(1 / 3 * lam)
            t_tid = 10 * (masses[k - 1] / 2e5) ** (2 / 3) / (omega_tid / 100)
            self.rlx_dmdt[k - 1] = masses[k - 1] / t_tid

            rate += self.rlx_dmdt[k - 1]

            m_fin = masses[k - 1] - (dts[k - 1] * rate)

            if m_fin <= 0 or np.isnan(m_fin):
                print(f"Cluster has disrupted at t={t[k-1]}")
                break

            masses[k] = m_fin

        return masses
