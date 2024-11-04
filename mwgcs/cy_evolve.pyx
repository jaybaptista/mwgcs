cimport cython

cimport numpy as np

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolveMass(double m0, double[:] tidalStrength, long steps, double dt, double[:] masses):

    cdef long i
    
    masses[0] = m0

    for i in range(steps-1):
        masses[i+1] = masses[i] - (dt * 30.0 * 1000.0 * (masses[i] / 2e5)**(1.0/3.0) * np.sqrt(tidalStrength[i]) / 3.0 / (0.32 * 1000.0))
        # masses[i+1] = masses[i] - (dt * masses[i] / (10.0 * (masses[i] / 2e5)**(2.0/3.0) * (100.0 / np.sqrt(tidalStrength[i] / 3.0))))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolveMassGG(double m0, double[:] tidalStrength, long steps, double dt, double[:] masses):

    cdef long i
        
    masses[0] = m0

    # assume mu_sev = 1:
    for i in range(steps-1):
        masses[i+1] = masses[i] - (dt * 45.0 * 1000.0 * (masses[i] / 2e5)**(1.0-0.67) * np.sqrt(tidalStrength[i]) / 3.0 / (0.32 * 1000.0))

    # for i in range(steps-1):
    #     masses[i+1] = masses[i] - (dt * 45.0 * 1000.0 * (masses[i] / m0)**(1-1.33) * (masses[i] / 200000.0)**(1.0-0.67) * np.sqrt(tidalStrength[i]) / 3.0 / (0.32 * 1000.0))

    # x = 0.67
    # y = 1.33
    # _m_ref = -45  # msun/myr
    # m_ref = _m_ref * 1000

    # rate = (
    #     m_ref
    #     * (m / m_i) ** (1 - y)
    #     * (m / (2e5)) ** (1 - x)
    #     * getTidalFrequency(tidalStrength)
    #     / (0.32 * 1000)
