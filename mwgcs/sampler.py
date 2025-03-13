import abc
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm, uniform

#################################################################
# Samplers for the GC System mass, should return a single value for mass

def EadieSampler(stellar_mass, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):

    # GC predictor (put this back in later)
    # p = (1 + np.exp(-1 * (b0 + b1 * np.log10(stellar_mass))))**(-1)

    # if uniform.rvs() > p: # i.e., failure to draw
    #     return 0.
    
    system_mass = (
        1
        / (1 + np.exp(-(b0 + (b1 * np.log10(stellar_mass)))))
        * (g0 + (g1 * np.log10(stellar_mass)))
    )
    
    return 10**system_mass

def KGSampler(halo_mass):
    return 3.2e6 * (halo_mass / 1e11)**(1.13)

#################################################################
# Samplers for the individual GC tags, should return a number of GC masses

def magnitude_to_luminosity(magnitude, zero_point=5.12):
    """Convert magnitude to luminosity using the zero point."""
    return 10 ** ((zero_point - magnitude) / 2.5)

def luminosity_to_mass(luminosity, ratio=3.0):
    """Convert luminosity to mass using the mass-to-light ratio."""
    return luminosity * ratio


def DwarfGCMF(stellar_mass, mass_light_ratio=3.0, system_mass_sampler=EadieSampler, halo_mass=1e12):

    # NOTE—make sure that you have a keyword "system_mass_sampler"
    # which is a function that takes in stellar mass in Msun
    # and outputs the bulk GC system mass.

    # This article motivates the cutoff
    # https://iopscience.iop.org/article/10.3847/1538-4357/abd557
    gc_cutoff = 2e4
    gcs_mass = 0.0

    if system_mass_sampler == EadieSampler:
        gcs_mass = system_mass_sampler(stellar_mass)
    else:
        gcs_mass = system_mass_sampler(halo_mass)

    if (gcs_mass == 0.0) or (gcs_mass < 2e4):
        return None
    gc_mass = []
    while np.sum(gc_mass) <= gcs_mass:
        sampled_magnitude = np.random.normal(-7.02, 0.57)
        sampled_luminosity = magnitude_to_luminosity(sampled_magnitude)
        sampled_mass = luminosity_to_mass(sampled_luminosity, mass_light_ratio)
        
        if sampled_mass > gcs_mass:
            continue
        
        gc_mass.append(sampled_mass)
    
    # ratio = (np.sum(accumulated_mass) - gc_peak_mass) / last_sampled_mass

    # if (np.random.uniform(0, 1) > ratio) & (len(accumulated_mass) > 1):
    #     accumulated_mass = accumulated_mass[:-1]

    return np.array(gc_mass)


#################################################################
# Samplers for the stream phase space distribution

# Chen model

def ChenSampler(r_tidal):
    r_mu = 1.6 * r_tidal
    phi_mu = 30.
    theta_mu = 0.
    alpha_mu = 20.
    beta_mu = 0.

    pearson_R = -0.7

    r_sigma = 0.35 * r_tidal
    phi_sigma = 23.
    theta_sigma = 12.
    alpha_sigma = 20.
    beta_sigma = 22.

    C_r_phi = pearson_R * r_sigma * phi_sigma
    C_r_theta = pearson_R * r_sigma * theta_sigma
    C_r_alpha = pearson_R * r_sigma * alpha_sigma
    C_r_beta = pearson_R * r_sigma * beta_sigma

    cov = np.diag([r_sigma, phi_sigma, theta_sigma, alpha_sigma, beta_sigma])**2

    cov[0, 1] = cov[1, 0] = C_r_phi
    cov[0, 2] = cov[2, 0] = C_r_theta
    cov[0, 3] = cov[3, 0] = C_r_alpha
    cov[0, 4] = cov[4, 0] = C_r_beta

    # sample from the multivariate normal
    return np.random.multivariate_normal([r_mu, phi_mu, theta_mu, alpha_mu, beta_mu], cov)