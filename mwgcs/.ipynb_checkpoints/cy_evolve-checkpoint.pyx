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
        masses[i+1] = masses[i] - (dt * masses[i] / (10.0 * (masses[i] / 2e5)**(2.0/3.0) * (100.0 / np.sqrt(tidalStrength[i] / 3.0))))