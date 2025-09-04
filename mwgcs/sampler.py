import numpy as np
from scipy.integrate import quad
from scipy.stats import uniform

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

def GCMF_EADIE(stellar_mass, system_mass_sampler=GCS_MASS_EADIE, b0=-10.31, b1=1.43, mass_light_ratio=1.98):
    gcs_mass = system_mass_sampler(stellar_mass)
    
    # Obtain N_gcs
    n_gcs = np.exp(b0 + b1 * np.log10(gcs_mass))
    
    # Use the ELVES GCMF to sample n_gcs

    gc_mass = []

    while np.sum(gc_mass) <= gcs_mass:
        sampled_magnitude = np.random.normal(-7.02, 0.57)
        sampled_luminosity = magnitude_to_luminosity(sampled_magnitude)
        sampled_mass = luminosity_to_mass(sampled_luminosity, mass_light_ratio)

        if (sampled_mass + np.sum(gc_mass)) > gcs_mass:
            # With probability (gcs_mass - np.sum(gc_mass)) / sampled_mass, keep the last sample
            prob = (gcs_mass - np.sum(gc_mass)) / sampled_mass
            if np.random.rand() > prob:
                break

        gc_mass.append(sampled_mass)

    return np.array(gc_mass)



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
    
def GCMF_VIRGO(stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_EADIE,
    halo_mass=1e12,
    allow_nsc=True):

    """
    Taken from Jordan+07: https://arxiv.org/abs/astro-ph/0702496
    Implements Equation 7 for evolved Schecter mass function
    """

    M_c = 10**(5.9)
    Delta = 5.4

    min_mass = 1e3

    def dN_dM_unnorm(mgc, const=1):
        return const * np.exp(- (mgc + Delta) / M_c) / (mgc + Delta)**2
    
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
    



def GCMF_2009(
    stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_EADIE,
    halo_mass=1e12,
    allow_nsc=True
    ):
    
    """
    Taken from Georgiev+2009 sample for the dSph luminosity functions
    Source: https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..879G/abstract
    => Valid up to galaxy stellar masses of ~1e9 Msun
    """

    gcs_mass = 0.0

    if system_mass_sampler == GCS_MASS_EADIE:
        gcs_mass = system_mass_sampler(stellar_mass)
    else:
        gcs_mass = system_mass_sampler(halo_mass)

    if gcs_mass == 0.0:
        return None

    gc_mass = []

    while np.sum(gc_mass) <= gcs_mass:
        sampled_magnitude = np.random.normal(-7.02, 0.57)
        sampled_luminosity = magnitude_to_luminosity(sampled_magnitude)
        sampled_mass = luminosity_to_mass(sampled_luminosity, mass_light_ratio)

        if (sampled_mass + np.sum(gc_mass)) > gcs_mass:
            # With probability (gcs_mass - np.sum(gc_mass)) / sampled_mass, keep the last sample
            prob = (gcs_mass - np.sum(gc_mass)) / sampled_mass
            if np.random.rand() > prob:
                break

        gc_mass.append(sampled_mass)

    if allow_nsc:
        if np.log10(stellar_mass) > 5.5:
            gc_mass.append(10 ** (2.68 + 0.38 * np.log10(stellar_mass)))

    return np.array(gc_mass)


def GCMF_ELVES(
    stellar_mass,
    mass_light_ratio=1.98,
    system_mass_sampler=GCS_MASS_EADIE,
    halo_mass=1e12,
    allow_nsc=True,
):
    """
    Returns samples from the dwarf galaxy mass function derived from the GCLF from
    ELVES (Carlsten+22a).

    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac457e
    """

    # NOTE: Not implementing a low-mass cut-off at the moment.
    # This article motivates the cutoff
    # https://iopscience.iop.org/article/10.3847/1538-4357/abd557
    # NOTE: Why did I remove the cutoff?
    # Some globular cluster systems only have stellar masses of ~100 Msun
    # in the Local Group!
    # See https://iopscience.iop.org/article/10.3847/1538-4357/ac33b0


    gc_cutoff = 2e4
    gcs_mass = 0.0

    if system_mass_sampler == GCS_MASS_EADIE:
        gcs_mass = system_mass_sampler(stellar_mass)
    else:
        gcs_mass = system_mass_sampler(halo_mass)

    if gcs_mass == 0.0:  # or (gcs_mass < gc_cutoff):
        return None

    gc_mass = []

    while np.sum(gc_mass) <= gcs_mass:
        sampled_magnitude = np.random.normal(-7.02, 0.57)
        sampled_luminosity = magnitude_to_luminosity(sampled_magnitude)
        sampled_mass = luminosity_to_mass(sampled_luminosity, mass_light_ratio)

        if (sampled_mass + np.sum(gc_mass)) > gcs_mass:
            # With probability (gcs_mass - np.sum(gc_mass)) / sampled_mass, keep the last sample
            prob = (gcs_mass - np.sum(gc_mass)) / sampled_mass
            if np.random.rand() > prob:
                break

        gc_mass.append(sampled_mass)

    """
    Making a nuclear star cluster since they might turn into observed
    globular clusters in our sample!
    Source: https://iopscience.iop.org/article/10.3847/1538-4357/ac457e#apjac457es4
    """

    # TODO: This one also looks like a good reference too: https://arxiv.org/pdf/1601.02613

    if allow_nsc:
        if np.log10(stellar_mass) > 5.5:
            gc_mass.append(10 ** (2.68 + 0.38 * np.log10(stellar_mass)))

    return np.array(gc_mass)
