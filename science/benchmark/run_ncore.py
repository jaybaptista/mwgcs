import symlib
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
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



def main():

    host_dir = symlib.get_host_directory(base_dir, "SymphonyMilkyWay", 0)
    host_name = os.path.split(host_dir)[-1]

    col_params = symlib.colossus_parameters(symlib.simulation_parameters(host_dir))
    cosmo = cosmology.Cosmology(name="", **col_params)
    print("Loading potential...")
    _t0 = time.time()
    potential = agama.Potential(file=os.path.join(f'/sdf/data/kipac/u/jaymarie/gchords_bfe/{host_name}/potential', 'cosmo_potential.dat'))
    _tf = time.time()
    print(f"Potential loaded in {_tf-_t0} seconds.")

    print('Starting GC tracer integration...')

    w0 = np.array([2.29060478e+01, -8.91048355e+01, -8.91237717e+01, 3.14766884e+01, -3.31345329e+01, -4.47778625e+01])

    with agama.setNumThreads(args.n_core):

        t_start = time.time()
        output = agama.orbit(
            potential=potential,
            ic=w0,
            timestart=0,
            time=14,
            trajsize=1,
            accuracy=1e-12,
        )
        t_elapsed = time.time() - t_start
        print(f"Orbit integration took {t_elapsed:.3f} s")

    t, xv = output
    r = np.linalg.norm(xv[:, :3], axis=1)
    fig, ax = plt.subplots(dpi=200, figsize=(4,4))
    ax.plot(t, r, c='k')
    ax.set_xlabel(r'$t\ \mathrm{(Gyr)}$')
    ax.set_ylabel(r'$r\ \mathrm{(kpc)}$')
    plt.savefig(f'~/trajectory_{args.n_core}.png')
    plt.close()



if __name__ == "__main__":
    main()
