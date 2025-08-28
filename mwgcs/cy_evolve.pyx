cimport cython
cimport numpy as cnp
import numpy as np

""""

cdef inline double _remnant_scalar(double m) nogil:
    if m >= 1.0 and m <= 8.0:
        return 0.109 * m + 0.394
    elif m > 8.0 and m <= 30.0:
        return 0.03636 * (m - 8.0) + 1.02
    elif m > 30.0:
        return 0.06 * (m - 30.0) + 8.3
    else:
        return m

cpdef evolve_stellar_mass(object initial_mass, object lifetimes, double tf):
    """
    float or 1D array-like initial_mass, lifetimes (same shape). Returns scalar if input was scalar.
    """
    # --- all declarations first ---
    cdef bint is_scalar
    cdef double m0_s = 0.0
    cdef double lt_s = 0.0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] m0_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] lt_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rem
    cdef double[:] m0
    cdef double[:] lt
    cdef double[:] r
    cdef Py_ssize_t i, n

    # --- scalar fast-path ---
    is_scalar = not hasattr(initial_mass, "__len__")
    if is_scalar:
        m0_s = float(initial_mass)
        lt_s = float(lifetimes)
        if lt_s < tf:
            return _remnant_scalar(m0_s)
        else:
            return m0_s

    # --- array path ---
    m0_arr = np.asarray(initial_mass, dtype=np.float64)
    lt_arr = np.asarray(lifetimes,    dtype=np.float64)

    if m0_arr.shape[0] != lt_arr.shape[0]:
        raise ValueError("initial_mass and lifetimes must have the same length")

    rem = m0_arr.copy()
    m0 = m0_arr
    lt = lt_arr
    r  = rem
    n = m0.shape[0]

    with nogil:
        for i in range(n):
            if lt[i] < tf:
                r[i] = _remnant_scalar(m0[i])

    return rem


"""

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