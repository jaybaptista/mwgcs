import numpy as np

# getMassLossRate, getTidalTimescale, getTidalFrequency

def getTidalFrequency(l):
    """
    Returns the tidal frequency in Gyr for a given tidal strength.
    --------
    Parameters:
    l: float
        The tidal strength in Gyr^-2.
    --------
    Returns:
    om: float
        The tidal frequency in Gyr^-1.
    """

    om = np.sqrt(l / 3)
    
    return om


def getTidalTimescale(m, l):
    """
    Returns the tidal timescale in Gyr for a given tidal strength.
    --------
    Parameters:
    m: float
        The mass of the system in Msun.
    l: float
        The tidal strength in Gyr^-2.
    """
    om = getTidalFrequency(l)
    tt = 10. * (m / 2e5)**(2/3) * (100 / om)
    return tt


def getMassLossRate(m, tidalStrength):
    rate = -1. * m / getTidalTimescale(m, tidalStrength)
    return rate