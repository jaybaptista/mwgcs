from mwgcs import SymphonyInterface, FiducialGCHaloModel, GChords, SymphonyPotential, FiducialMassLossModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

halo_dir = "/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/SymphonyMilkyWay/Halo023/"

# Initialize the Symphony interface, GC halo model, and GChords
interface = SymphonyInterface(halo_dir)
model = FiducialGCHaloModel()
gchords = GChords(interface, model)

# Generates accreted star cluster samples
gchords.generate_clusters()

# Building an interpolated potential representation
potential = SymphonyPotential(interface, 'potential')
potential.construct_potential(lmax=0)

# Compute the tidal field and cluster masses for each cluster across all snapshots
gchords.compute_cluster_tidal_field(potential)

# Relaxation only mass loss model
mass_loss_model = FiducialMassLossModel()

tidal_field = np.load('tidal_field.npz', allow_pickle=True)

# Compute masses
gchords.compute_cluster_masses(tidal_field, mass_loss_model)