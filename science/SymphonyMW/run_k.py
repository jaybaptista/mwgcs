import symlib
import numpy as np
import os
import argparse
import agama
import pandas as pd
from tqdm import tqdm

agama.setUnits(mass=1.,length=1.,velocity=1.)

from mwgcs import GCS_MASS_EADIE, GCMF_ELVES, GCS_NUMBER_LINEAR, SymphonyInterfacer, GC


base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/"
n_hosts = np.arange(symlib.n_hosts("SymphonyMilkyWay")) # 45 hosts
output_directory = "/sdf/data/kipac/u/jaymarie/gchords_1022_k100"

os.makedirs(output_directory, exist_ok=True)

parser = argparse.ArgumentParser(description="Run SymphonyMW for a specific host.")
parser.add_argument("n_halo", type=int, help="Index of the host halo to process")
parser.add_argument("--kappa", type=float, default=4.0, help="kappa (evaporation strength)")
args = parser.parse_args()

# Knobs!

"""GC–Halo Connection"""
GC_MASS_FUNCTION  = GCMF_ELVES
GCS_MASS_FUNCTION = GCS_NUMBER_LINEAR
ALLOW_NSC         = False

"""Orbit Integration"""
ACCURACY=1e-12
THREAD_COUNT=32

"""Multipole Approximation"""
LMAX         = 4
LMAX_SUBHALO = 1
RMIN         = 0.001
RMAX         = 250

"""GC Evolution and Integration"""
# KAPPA = 100.0
IMF   = "kroupa"



def main():

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
        freeze=True
        )
    
    clusters = pd.read_csv(os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster/', "clusters.csv"))
    tracking = np.load(os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster', "particle_tracking.npz"))
    
    # Make potentials
    # /sdf/data/kipac/u/jaymarie/gchords_1021_k4/Halo023/potential

    potential = agama.Potential(file=os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/potential', 'cosmo_potential.dat'))

    # Run globular cluster evolution

    for i_gc in tqdm(np.arange(len(clusters))):

        infall_snapshot = clusters['infall_snap'][i_gc]
        m0 = clusters['gc_mass'][i_gc]
        w0 = tracking['xv'][infall_snapshot, i_gc]
        t0 = si.times_ag[infall_snapshot]
        tf = si.times_ag[-1]
        feh = clusters['feh'][i_gc]

        gc = GC(
            potential, w0, t0, tf, m0,
            feh=feh,
            npts=np.max([250 * int(np.floor(tf - t0)), 500]), # arbitrary
            kappa=args.kappa,
            imf=IMF,
            accuracy=ACCURACY,
            thread_count=THREAD_COUNT,
            output_prefix=os.path.join(output, f"gc_{i_gc}"),
        )

if __name__ == "__main__":
    main()