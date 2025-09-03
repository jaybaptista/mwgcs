import symlib
import numpy as np
import os
import argparse
import agama
import pandas as pd
from tqdm import tqdm

agama.setUnits(mass=1.,length=1.,velocity=1.)

from mwgcs import GCS_MASS_EADIE, GCMF_ELVES, SymphonyInterfacer, GC


base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/"
n_hosts = np.arange(symlib.n_hosts("SymphonyMilkyWay")) # 45 hosts
output_directory = "/sdf/data/kipac/u/jaymarie/gchords_n_gc"

os.makedirs(output_directory, exist_ok=True)

parser = argparse.ArgumentParser(description="Run SymphonyMW for a specific host.")
parser.add_argument("n_halo", type=int, help="Index of the host halo to process")
args = parser.parse_args()

# Knobs!

"""GC–Halo Connection"""
GC_MASS_FUNCTION  = GCMF_ELVES
GCS_MASS_FUNCTION = GCS_MASS_EADIE
ALLOW_NSC         = False

"""Orbit Integration"""
ACCURACY=1e-8
THREAD_COUNT=32

"""Multipole Approximation"""
LMAX         = 4
LMAX_SUBHALO = 1
RMIN         = 0.001
RMAX         = 250

"""GC Evolution and Integration"""
KAPPA = 4.0
IMF   = "kroupa"



def main():

    N_REALIZATIONS = 50

    if args.n_halo >= len(n_hosts):
        raise ValueError(f"n_halo ({args.n_halo}) is out of range. Must be less than {len(n_hosts)}.")

    host_dir = symlib.get_host_directory(base_dir, "SymphonyMilkyWay", args.n_halo)
    host_name = os.path.split(host_dir)[-1]
    output = os.path.join(output_directory, host_name)

    # Creates clusters 
    si = SymphonyInterfacer(
        host_dir,
        gcmf=GC_MASS_FUNCTION,
        gcsysmf=GCS_MASS_FUNCTION,
        output_prefix=output,
        allow_nsc=ALLOW_NSC,
        freeze=True,
        )
    
    for k in tqdm(np.arange(N_REALIZATIONS), desc="Population realization"):
        si.generate_clusters(GCS_MASS_FUNCTION, GC_MASS_FUNCTION, write_path=os.path.join(si.output_dir, f"cr_{k}.csv"), allow_nsc=False)
    
if __name__ == "__main__":
    main()
