import abc
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm 

#################################################################
# Samplers for the GC System mass, should return a single value for mass

def EadieSampler(stellar_mass, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    system_mass = 1/(1 + np.exp(-(b0 + (b1 * np.log10(stellar_mass))))) * (g0 + (g1*np.log10(stellar_mass)))
    return system_mass        

#################################################################
# Samplers for the individual GC tags, should return a number of GC masses
def DwarfGCMF(stellar_mass,
              mass_light_ratio = 10.,
              system_mass_sampler = EadieSampler):
    
    # NOTE—make sure that you have a keyword "system_mass_sampler"
    # which is a function that takes in stellar mass in Msun
    # and outputs the bulk GC system mass.
    
    gc_peak_mass = 10**(system_mass_sampler(stellar_mass))

    min_gband = -5.5
    max_gband = -9.5
    
    min_mass = mass_light_ratio * 10**(0.4*(5.03 - max_gband))
    max_mass = mass_light_ratio * 10**(0.4*(5.03 - min_gband))
    
    gc_mass_range = np.logspace(np.log10(min_mass), np.log10(max_mass))

    def _gclf(mass, M_mean=-7., M_sigma=0.7):
        # get magnitude from mass
        mag = 5.03 - 2.5 * np.log10(mass)
        gclf_value = norm.pdf(mag, loc=M_mean, scale=M_sigma)
        return gclf_value
    
    def r(M):
        _a = quad(_gclf, min_mass, M)[0]
        _b = quad(_gclf, min_mass, max_mass)[0]
        return _a / _b
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interp1d(cdf, gc_mass_range, kind="linear")

    accumulated_mass = [inv_cdf(np.random.uniform(0, 1))]
    last_sampled_mass = accumulated_mass[0]

    while np.sum(accumulated_mass) < gc_peak_mass:
        # sample a mass
        sampled_mass = inv_cdf(np.random.uniform(0, 1))

        # add it to the running total
        accumulated_mass.append(sampled_mass)

        # keep track of the last sampled mass
        last_sampled_mass = sampled_mass

    ratio = (np.sum(accumulated_mass) - gc_peak_mass) / last_sampled_mass 
    
    if (np.random.uniform(0, 1) > ratio) & (len(accumulated_mass) > 1):
        accumulated_mass = accumulated_mass[:-1]

    return accumulated_mass