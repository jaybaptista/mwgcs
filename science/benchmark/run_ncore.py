import symlib
from colossus.cosmology import cosmology
import numpy as np
import os
import argparse
import agama
import pandas as pd
from tqdm import tqdm
import time

agama.setUnits(mass=1.,length=1.,velocity=1.)

from mwgcs import GCS_MASS_EADIE, GCMF_ELVES, GCS_NUMBER_LINEAR, SymphonyInterfacer, GC


base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/"
n_hosts = np.arange(symlib.n_hosts("SymphonyMilkyWay")) # 45 hosts

parser = argparse.ArgumentParser(description="Run SymphonyMW Halo023 for a specific host with a given number of cores.")
parser.add_argument("n_core", type=int, help="Number of cores")
args = parser.parse_args()

# Knobs!

"""GC–Halo Connection"""
GC_MASS_FUNCTION  = GCMF_ELVES
GCS_MASS_FUNCTION = GCS_NUMBER_LINEAR
ALLOW_NSC         = False

"""Orbit Integration"""
ACCURACY=1e-12

"""Multipole Approximation"""
LMAX         = 4
LMAX_SUBHALO = 1
RMIN         = 0.001
RMAX         = 2.0 # This is a multiplier on the virial radius

"""GC Evolution and Integration"""
# KAPPA = 100.0
IMF   = "kroupa"



def main():

    host_dir = symlib.get_host_directory(base_dir, "SymphonyMilkyWay", 0)
    host_name = os.path.split(host_dir)[-1]

    output_directory = f"/sdf/data/kipac/u/jaymarie/gchords_bench_{args.n_core}"
    os.makedirs(output_directory, exist_ok=True)


    output = os.path.join(output_directory, host_name)

    col_params = symlib.colossus_parameters(symlib.simulation_parameters(host_dir))
    cosmo = cosmology.Cosmology(name="", **col_params)

    clusters = pd.read_csv(os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster/', "clusters.csv"))
    tracking = np.load(os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster', "particle_tracking.npz"))
    
    # Make potentials
    # /sdf/data/kipac/u/jaymarie/gchords_1021_k4/Halo023/potential
    print("Loading potential...")
    _t0 = time.time()
    potential = agama.Potential(file=os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_bfe/{host_name}/potential', 'cosmo_potential.dat'))
    _tf = time.time()
    print(f"Potential loaded in {_tf-_t0} seconds.")
    # Run globular cluster evolution

    for i_gc in tqdm(np.arange(len(clusters))):

        infall_snapshot = clusters['infall_snap'][i_gc]
        m0 = clusters['gc_mass'][i_gc]
        w0 = tracking['xv'][infall_snapshot, i_gc]
        t0 = 0.
        tf = 14.
        feh = clusters['feh'][i_gc]

        z_form = 1/clusters['a_form'][i_gc] - 1
        age = cosmo.hubbleTime(z_form)

        gc = GC(
            potential, w0, t0, tf, m0,
            feh=feh,
            age=age,
            npts=np.max([250 * int(np.floor(tf - t0)), 500]), # arbitrary
            kappa=4,
            imf=IMF,
            accuracy=ACCURACY,
            thread_count=args.n_core,
            output_prefix=os.path.join(output, f"gc_{i_gc}"),
        )

if __name__ == "__main__":
    main()
