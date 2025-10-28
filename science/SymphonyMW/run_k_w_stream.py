import symlib
from colossus.cosmology import cosmology
import numpy as np
import os
import argparse
import agama
import pandas as pd
from tqdm import tqdm

agama.setUnits(mass=1., length=1., velocity=1.)

from mwgcs import GCS_MASS_EADIE, GCMF_ELVES, GCS_NUMBER_LINEAR, SymphonyInterfacer, GC

base_dir = "/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/"
n_hosts = np.arange(symlib.n_hosts("SymphonyMilkyWay"))  # 45 hosts

parser = argparse.ArgumentParser(description="Run SymphonyMW for a specific host (chunked over clusters).")
parser.add_argument("n_halo", type=int, help="Index of the host halo to process")
parser.add_argument("--kappa", type=float, default=4.0, help="kappa (evaporation strength)")
parser.add_argument("--chunk", type=int, default=0, help="0-indexed chunk ID to run (e.g., 0..nchunks-1)")
parser.add_argument("--nchunks", type=int, default=1, help="Total number of chunks to split clusters into")
args = parser.parse_args()

# ---- Knobs! ----
# GC–Halo Connection
GC_MASS_FUNCTION  = GCMF_ELVES
GCS_MASS_FUNCTION = GCS_NUMBER_LINEAR
ALLOW_NSC         = False

# Orbit Integration
ACCURACY     = 1e-12
THREAD_COUNT = 32

# Multipole Approximation
LMAX         = 4
LMAX_SUBHALO = 1
RMIN         = 0.001
RMAX         = 2.0  # multiplier on virial radius

# GC Evolution and Integration
IMF = "kroupa"


def compute_chunk_indices(n_total: int, chunk: int, nchunks: int) -> np.ndarray:
    """Return the global indices of elements belonging to `chunk` when splitting n_total into nchunks."""
    if nchunks < 1:
        raise ValueError(f"nchunks ({nchunks}) must be >= 1.")
    if chunk < 0 or chunk >= nchunks:
        raise ValueError(f"chunk ({chunk}) must be in [0, {nchunks-1}].")
    start = (n_total * chunk) // nchunks
    end   = (n_total * (chunk + 1)) // nchunks
    return np.arange(start, end, dtype=int)


def main():
    if args.n_halo >= len(n_hosts) or args.n_halo < 0:
        raise ValueError(f"n_halo ({args.n_halo}) is out of range. Must be in [0, {len(n_hosts)-1}].")

    host_dir = symlib.get_host_directory(base_dir, "SymphonyMilkyWay", args.n_halo)
    host_name = os.path.split(host_dir)[-1]

    output_directory = f"/sdf/data/kipac/u/jaymarie/streams_kappa_{args.kappa}"
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, host_name)

    col_params = symlib.colossus_parameters(symlib.simulation_parameters(host_dir))
    cosmo = cosmology.Cosmology(name="", **col_params)

    # Interface / inputs
    si = SymphonyInterfacer(
        host_dir,
        gcmf=GC_MASS_FUNCTION,
        gcsysmf=GCS_MASS_FUNCTION,
        output_prefix=output,
        allow_nsc=ALLOW_NSC,
        freeze=True
    )

    clusters_path = os.path.join(f"/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster", "clusters.csv")
    tracking_path = os.path.join(f"/sdf/data/kipac/u/jaymarie/gchords_1021_k4/{host_name}/cluster", "particle_tracking.npz")
    clusters = pd.read_csv(clusters_path)
    tracking = np.load(tracking_path)

    # Potential
    potential_path = os.path.join(f"/sdf/data/kipac/u/jaymarie/gchords_bfe/{host_name}/potential", "cosmo_potential.dat")
    potential = agama.Potential(file=potential_path)

    # Determine which GC indices to run in this chunk
    n_total = len(clusters)
    gc_indices = compute_chunk_indices(n_total, args.chunk, args.nchunks)

    # Helpful log
    print(
        f"[Host {host_name}] Total clusters: {n_total} | "
        f"nchunks={args.nchunks} | chunk={args.chunk} -> "
        f"indices {gc_indices[0] if len(gc_indices)>0 else '[]'}"
        f"{'..'+str(gc_indices[-1]) if len(gc_indices)>1 else ''} "
        f"({len(gc_indices)} clusters)"
    )

    # Run only this chunk's clusters
    for i_gc in tqdm(gc_indices, desc=f"{host_name} [chunk {args.chunk}/{args.nchunks}]"):
        infall_snapshot = clusters.loc[i_gc, "infall_snap"]
        m0   = clusters.loc[i_gc, "gc_mass"]
        w0   = tracking["xv"][infall_snapshot, i_gc]
        t0   = si.times_ag[infall_snapshot]
        tf   = si.times_ag[-1]
        feh  = clusters.loc[i_gc, "feh"]

        z_form = 1.0 / clusters.loc[i_gc, "a_form"] - 1.0
        age    = cosmo.hubbleTime(z_form)

        gc = GC(
            potential, w0, t0, tf, m0,
            feh=feh,
            age=age,
            npts=max(250 * int(np.floor(tf - t0)), 500),  # arbitrary
            kappa=args.kappa,
            imf=IMF,
            accuracy=ACCURACY,
            thread_count=THREAD_COUNT,
            output_prefix=os.path.join(output, f"gc_{i_gc}"),
        )

        gc.stream(20, si.times_ag[-1])


if __name__ == "__main__":
    main()
