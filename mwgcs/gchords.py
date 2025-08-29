from .interface import SymphonyInterfacer
from .sampler import GCS_MASS_EADIE, GCMF_ELVES

halo_directory = '/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/SymphonyMilkyWay/Halo023'

OUTPUT            = "output"

"""GC–Halo Connection"""
GC_MASS_FUNCTION  = GCMF_ELVES
GCS_MASS_FUNCTION = GCS_MASS_EADIE
ALLOW_NSC         = True

"""Multipole Approximation"""
LMAX         = 4
LMAX_SUBHALO = 1
RMIN         = 0.001
RMAX         = 250

"""GC Evolution and Integration"""
KAPPA = 1.0
IMF   = "kroupa"

si = SymphonyInterfacer(
    halo_directory,
    gcmf=GC_MASS_FUNCTION,
    gcsysmf=GCS_MASS_FUNCTION,
    output_prefix=OUTPUT,
    allow_nsc=ALLOW_NSC
)

