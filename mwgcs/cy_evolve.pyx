cimport cython
cimport numpy as cnp
import numpy as np

"""
Functions below may be deprecated
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolveMass(double m0, double[:] tidalStrength, long steps, double dt, double[:] masses):

    cdef long i
    
    masses[0] = m0

    for i in range(steps-1):
        masses[i+1] = masses[i] - (dt * 30.0 * 1000.0 * (masses[i] / 2e5)**(1.0/3.0) * np.sqrt(tidalStrength[i]) / 3.0 / (0.32 * 1000.0))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def evolveMassGG(double m0, double[:] tidalStrength, long steps, double dt, double[:] masses):

    cdef long i
        
    masses[0] = m0

    # assume mu_sev = 1:
    for i in range(steps-1):
        masses[i+1] = masses[i] - (dt * 45.0 * 1000.0 * (masses[i] / 2e5)**(1.0-0.67) * np.sqrt(tidalStrength[i]) / 3.0 / (0.32 * 1000.0))