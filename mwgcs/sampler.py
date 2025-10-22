import numpy as np
from scipy.integrate import quad
from scipy.stats import uniform
from scipy.interpolate import interp1d

from itertools import chain
"""
Mass-to-light ratios are taken from N-body modeling of globular clusters:
https://arxiv.org/abs/1609.08794
"""


# TODO: Test this function at some point for GC occupation.
def EadieProbGC(stellar_mass, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    """
    Implementation for GC occupation probability as a function of a
    galaxy's stellar mass.

    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac33b0
    """

    p = (1 + np.exp(-1 * (b0 + b1 * np.log10(stellar_mass)))) ** (-1)

    if uniform.rvs() > p:  # i.e., failure to draw
        return 0
    else:
        return 1


def GCS_MASS_LINEAR(stellar_mass, g0=-0.725, g1=0.788):
    """
    Implementation of the linear regression model from Eadie+2022
    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac33b0
    """
    return 10 ** (g0 + g1 * np.log10(stellar_mass))

def GCS_MASS_HARRIS(halo_mass, g0=-0.725, g1=0.788):
    """
    Implementation of the Harris halo mass–GCS mass relation from
    Harris, Blakeslee, & Harris (2017) paper.
    """
    eta = 2.9e-5
    return eta * halo_mass


def GCS_MASS_EADIE(stellar_mass, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    """
    Implementation of the log-hurdle model from Eadie+2022
    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac33b0
    """

    system_mass = (
        1
        / (1 + np.exp(-(b0 + (b1 * np.log10(stellar_mass)))))
        * (g0 + (g1 * np.log10(stellar_mass)))
    )

    return 10**system_mass

def GCS_NUMBER_LINEAR(halo_mass):
    eta_N = 10**(-8.56 - 0.11 * np.log10(halo_mass))
    return eta_N * halo_mass

def KGSampler(halo_mass):
    """
    Equation 6 from Kravstov & Gnedin (2003)
    NOTE: I'm doing the very bad thing of extrapolating to low masses. :(
    Source: https://arxiv.org/abs/astro-ph/0305199
    """
    return 3.2e6 * (halo_mass / 1e11) ** (1.13)


def magnitude_to_luminosity(magnitude, zero_point=5.12):
    """Convert magnitude to luminosity using the zero point."""
    return 10 ** ((zero_point - magnitude) / 2.5)


def luminosity_to_mass(luminosity, ratio=3.0):
    """Convert luminosity to mass using the mass-to-light ratio."""
    return luminosity * ratio

"""
SAMPLER UTILITIES FOR GAUSSIANS
"""

def generic_gcmf_gaussian(M, mu=-7.0, sigma_m=1.0, M_sun=5.12, ml_ratio=2.0, A=1.0):
    jac = 1.0 / (np.log(10) * M)
    sigma_M = sigma_m / 2.5
    C = M_sun + 2.5 * np.log10(
        ml_ratio
    )  # Note, M_gsun is the absolute M-band magnitude, not mass
    M_mu = 10 ** ((C - mu) / 2.5)
    norm = A / (np.sqrt(2 * np.pi) * sigma_M)
    dn_dm = norm * np.exp(
        -(1 / (2 * sigma_M**2)) * (-np.log10(M) + np.log10(M_mu)) ** 2
    )
    return dn_dm * jac

def make_lognormal10_icdf(mu_log10, sigma, 
                          Mmin=1e-2, Mmax=1e8, n_grid=4096):
    """
    Build an inverse CDF for the PDF:
      dN/dM = [1/(ln 10 * M)] * [1/(sqrt(2π) * σ_M)] *
              exp( - (log10 M - μ)^2 / (2 σ_M^2) )
    """
    M = np.logspace(np.log10(Mmin), np.log10(Mmax), n_grid)

    # logspaced grid
    x = np.log10(M)

    norm = np.log(10) * M * (np.sqrt(2*np.pi) * sigma)
    pdf = np.exp(-0.5 * ((x - mu_log10)/sigma)**2) / norm
    pdf = np.clip(pdf, 0.0, np.inf)

    # Integrate with trapezoid rule to get CDF
    cdf = np.zeros_like(M)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(M))

    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PDF integral is non-positive over the chosen range.")
    cdf /= total

    # Ensure strict monotonicity (protect against flat tails)
    cdf = np.maximum.accumulate(cdf)
    cdf[-1] = 1.0

    # Build inverse CDF via interpolation
    icdf_interp = interp1d(cdf, M, kind="linear", bounds_error=False,
                           fill_value=(Mmin, Mmax), assume_sorted=True)

    def icdf(u):
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        return icdf_interp(u)

    return icdf


def sample_generic_gcmf(n, mu, sigma_m, M_sun=5.12, ml_ratio=2.0, A=1.0):

    sigma_M = sigma_m / 2.5
    C = M_sun + 2.5 * np.log10(
        ml_ratio
    )  # Note, M_gsun is the absolute M-band magnitude, not mass
    M_mu = 10 ** ((C - mu) / 2.5)

    icdf = make_lognormal10_icdf(np.log10(M_mu), sigma_M)
    u = np.random.uniform(size=n)
    samples = icdf(u)
    return samples

"""
SAMPLER UTILITIES FOR SCHECTER FUNCTIONS (WIP)
"""

def _schecter_gcmf(M, D, Mc):
    x = (M+D)
    return np.exp(x/Mc) / x**2

def get_schecter_gcmf(Mmin, Mmax, D, Mc):
    norm, _ = quad(_schecter_gcmf, Mmin, Mmax, args=(D, Mc))
    
    def schecter_pdf(M):
        return _schecter_gcmf(M, D, Mc) / norm
    
    return schecter_pdf

def get_schecter_icdf(Mmin, Mmax, D, Mc):
    n_grid = 100
    M_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), n_grid)
    _pdf = get_schecter_gcmf(Mmin, Mmax, D, Mc)
    pdf = _pdf(M_grid)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    icdf = interp1d(cdf, M_grid, bounds_error=False, fill_value=(Mmin, Mmax))
    return icdf

"""
SCHECTER MASS FUNCTION (WIP)
======================
"""

def GCMF_SCHECTER(
    stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_LINEAR,
    halo_mass=1e12,
    allow_nsc=True,
):
    """
    Taken from Jordan+07: https://arxiv.org/abs/astro-ph/0702496
    Implements Equation 7 for evolved Schecter mass function
    """

    M_c = 10 ** (5.9)
    Delta = 5.4

    min_mass = 1e3

    def dN_dM_unnorm(mgc, const=1):
        return const * np.exp(-(mgc + Delta) / M_c) / (mgc + Delta) ** 2

    # Normalize dN/dM over the allowed mass range
    max_mass = system_mass_sampler(stellar_mass)

    norm, _ = quad(dN_dM_unnorm, min_mass, max_mass)

    def dN_dM(mgc):
        return dN_dM_unnorm(mgc) / norm

    gc_mass = []

    while np.sum(gc_mass) <= max_mass:
        u = np.random.uniform()
        # Inverse CDF sampling via rejection (since analytic inverse is hard)
        # Sample mgc in [min_mass, max_mass] and accept with probability proportional to dN_dM
        for _ in range(1000):  # Limit attempts to avoid infinite loop
            mgc_candidate = np.random.uniform(min_mass, max_mass)
            prob = dN_dM(mgc_candidate) / dN_dM(min_mass)
            if np.random.uniform() < prob:
                if (mgc_candidate + np.sum(gc_mass)) > max_mass:
                    prob_last = (max_mass - np.sum(gc_mass)) / mgc_candidate
                    if np.random.rand() > prob_last:
                        break
                gc_mass.append(mgc_candidate)
                break

    return np.array(gc_mass)


"""
    GAUSSIAN LUMINOSITY FUNCTIONS
    =============================
    The GCMF is basically a log-normal distribution
    (scaled by log10 instead of ln)
    
    The mean of the distribution is:
            M_mu * exp(0.5 * (ln 10 * sigma_M)^2)
        where the following conversions are used:
            M_mu = 10^((C - mu)/2.5);
        where
            C = M_sun + 2.5 log10(mass-light ratio)
    """


def GCMF_GEORGIEV(
    stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_LINEAR,
    halo_mass=1e12,
    allow_nsc=True,
    p_gc=False,
):
    """
    Taken from Georgiev+2009 sample for the dSph luminosity functions
    Source: https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..879G/abstract
    => Valid up to galaxy stellar masses of ~1e9 Msun
    """

    # Georgiev Survey Parameters
    mu_V = -7.04
    sigma_V = 1.15
    V_sun = 4.80  # AB mag
    C = V_sun + 2.5 * np.log10(mass_light_ratio)
    M_mu = 10 ** ((C - mu_V) / 2.5)
    sigma_mu = sigma_V / 2.5
    mean_M_mu = M_mu * np.exp(0.5 * (np.log(10) * sigma_mu) ** 2)
    gcs_mass = 0.0

    if (system_mass_sampler == GCS_MASS_EADIE) or (
        system_mass_sampler == GCS_MASS_LINEAR
    ):
        gcs_mass = system_mass_sampler(stellar_mass)
    elif (system_mass_sampler == GCS_NUMBER_LINEAR):
        # If the system mass is a number:
        lam = system_mass_sampler(halo_mass)
        n_draws = np.random.poisson(lam)
        samples = sample_generic_gcmf(
            n_draws, mu_V, sigma_V, M_sun=V_sun, ml_ratio=mass_light_ratio
        )
        
        return np.array(samples)
    else:
        gcs_mass = system_mass_sampler(halo_mass)

    if (gcs_mass == 0.0) or (p_gc and (EadieProbGC(stellar_mass) == 0)):
        return []

    gc_mass = []

    lam = gcs_mass / mean_M_mu

    if (lam <= 0) or np.isnan(lam):
        print(f"ERROR: gcs_mass / mean_gc_mass = {lam}")
        return []

    n_draws = np.random.poisson(lam)

    samples = sample_generic_gcmf(
        n_draws, mu_V, sigma_V, M_sun=V_sun, ml_ratio=mass_light_ratio
    )

    gc_mass.extend(samples)

    if allow_nsc:
        if np.log10(stellar_mass) > 5.5:
            gc_mass.append(10 ** (2.68 + 0.38 * np.log10(stellar_mass)))

    return np.array(gc_mass)


def GCMF_ELVES(
    stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_LINEAR,
    halo_mass=1e12,
    allow_nsc=True,
    p_gc=False,
):
    """
    Returns samples from the dwarf galaxy mass function derived from the GCLF from
    ELVES (Carlsten+22a).

    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac457e


    NOTE: Not implementing a low-mass cut-off at the moment.

    This article would motivate the cutoff
    https://iopscience.iop.org/article/10.3847/1538-4357/abd557

    NOTE: Why did I remove the cutoff?

    Some globular cluster systems only have stellar masses of ~100 Msun
    in the Local Group! (See https://iopscience.iop.org/article/10.3847/1538-4357/ac33b0)
    """

    # ELVES Survey Parameters
    mu_g = -7.02
    sigma_g = 0.57
    g_sun = 5.05  # AB mag
    C = g_sun + 2.5 * np.log10(mass_light_ratio)
    M_mu = 10 ** ((C - mu_g) / 2.5)
    sigma_mu = sigma_g / 2.5
    

    gcs_mass = 0.0

    if (system_mass_sampler == GCS_MASS_EADIE) or (
        system_mass_sampler == GCS_MASS_LINEAR
    ):
        gcs_mass = system_mass_sampler(stellar_mass)
    elif (system_mass_sampler == GCS_NUMBER_LINEAR):
        
        # If the system mass is a number:
        lam = system_mass_sampler(halo_mass)
        n_draws = np.random.poisson(lam)
        samples = sample_generic_gcmf(
            n_draws, mu_g, sigma_g, M_sun=g_sun, ml_ratio=mass_light_ratio
        )
        
        return np.array(samples)
    else:
        gcs_mass = system_mass_sampler(halo_mass)

    if (gcs_mass == 0.0) or (p_gc and (EadieProbGC(stellar_mass) == 0)):
        return []
    
    gc_mass = []

    mean_M_mu = M_mu * np.exp(0.5 * (np.log(10) * sigma_mu) ** 2)
    lam = gcs_mass / mean_M_mu

    if (lam <= 0) or np.isnan(lam):
        print(f"ERROR: gcs_mass / mean_gc_mass = {lam}")
        return []
    
    n_draws = np.random.poisson(lam)

    samples = sample_generic_gcmf(
        n_draws, mu_g, sigma_g, M_sun=g_sun, ml_ratio=mass_light_ratio
    )

    gc_mass.extend(samples)

    if allow_nsc:
        if np.log10(stellar_mass) > 5.5:
            gc_mass.append(10 ** (2.68 + 0.38 * np.log10(stellar_mass)))

    return np.array(gc_mass)
