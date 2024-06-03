import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm 

mass_light_ratio = 2.


def lognorm_hurdle(ms, b0=-10.83, b1=1.59, g0=-0.83, g1=0.8):
    """
    ms: stellar mass of the galaxy
    """
    mgc = 1/(1 + np.exp(-(b0 + (b1 * np.log10(ms))))) * (g0 + (g1*np.log10(ms)))
    return mgc

def MilkyWayGCMF(m, logmc=5.4, logDelta=5.9):
        """
        m: mass of the globular cluster

        """
        dM = 10**logDelta
        mc = 10**logmc
        return (m+dM)**(-2) * np.exp(-(m+dM)/mc)

def sampleMilkyWayGCMF(ms, Mmin = 1e3, Mmax = 1e7):
    ms_peak_idx  = np.argmax(ms)
    ms_peak = ms[ms_peak_idx]
    gc_peak_mass = 10**(lognorm_hurdle(ms_peak))
    gc_mass_range = np.logspace(np.log10(Mmin), np.log10(Mmax))
    
    def r(M):
        num = quad(MilkyWayGCMF, Mmin, M)[0]
        den = quad(MilkyWayGCMF, Mmin, Mmax)[0]
        return num / den
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interp1d(cdf, gc_mass_range, kind="linear")
    
    accumulated_mass  = [inv_cdf(np.random.uniform(0, 1))]
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

    return accumulated_mass, gc_mass_range


def DwarfGCMF(mass, M_mean=-7., M_sigma=0.7):
    
    # get magnitude from mass
    
    mag = 5.03 - 2.5 * np.log10(mass)
    gclf_value = norm.pdf(mag, loc=M_mean, scale=M_sigma)
    
    return gclf_value
    
def sampleDwarfGCMF(ms):
    # This will only work for dwarfs between 5.5 and 8.5 log star mass
    
    gc_peak_mass = 10**(lognorm_hurdle(ms))
    
    min_gband = -5.5
    max_gband = -9.5
    min_mass = mass_light_ratio * 10**(0.4*(5.03 - max_gband))
    max_mass = mass_light_ratio * 10**(0.4*(5.03 - min_gband))
    
    gc_mass_range = np.logspace(np.log10(min_mass), np.log10(max_mass))
    
    def r(M):
        num = quad(DwarfGCMF, min_mass, M)[0]
        den = quad(DwarfGCMF, min_mass, max_mass)[0]
        return num / den
    
    cdf           = np.array([r(m_i) for m_i in gc_mass_range])
    inv_cdf       = interp1d(cdf, gc_mass_range, kind="linear")
    
    accumulated_mass  = [inv_cdf(np.random.uniform(0, 1))]
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

    return accumulated_mass, gc_mass_range
