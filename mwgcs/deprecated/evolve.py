import numpy as np
import pyximport

pyximport.install(setup_args={"script_args": ["--verbose"]})
from .cy_evolve import evolveMass, evolveMassGG
import astropy.units as u


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
    tt = 10.0 * (m / 2e5) ** (2 / 3) * (100 / om)
    return tt


def getMassLossRate(m, tidalStrength):
    rate = -1.0 * m / (10.0 * (m / 2e5) ** (2 / 3) * (100 / np.sqrt(tidalStrength / 3)))
    return rate


def CMassLoss(m0, tidalStrength, steps, dt):
    # get tidalStrength into okay units

    tidalStrength = tidalStrength.to(u.Gyr ** (-2)).value
    tidalStrength = np.array(tidalStrength, dtype=np.double)

    dt = dt.to(u.Gyr).value

    masses = np.zeros(steps, dtype=np.float64)
    evolveMass(m0, tidalStrength, steps, dt, masses)

    return masses

def CMassLossGG(m0, tidalStrength, steps, dt):
    # get tidalStrength into okay units

    tidalStrength = tidalStrength.to(u.Gyr ** (-2)).value
    tidalStrength = np.array(tidalStrength, dtype=np.double)

    dt = dt.to(u.Gyr).value

    masses = np.zeros(steps, dtype=np.double)
    evolveMassGG(m0, tidalStrength, steps, dt, masses)

    return masses
    
def mdot_gg23(m, tidalStrength, m_i=3.5e4):
    x = 0.67
    y = 1.33
    _m_ref = -45  # msun/myr
    m_ref = _m_ref * 1000

    rate = (
        m_ref
        * (m / m_i) ** (1 - y)
        * (m / (2e5)) ** (1 - x)
        * getTidalFrequency(tidalStrength)
        / (0.32 * 1000)
    )
    return rate
