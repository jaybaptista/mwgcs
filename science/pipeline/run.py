import os
import numpy as np
import asdf
from mwgcs import SymphonyInterfacer

mw_sim_path = '/fs/ddn/sdf/group/kipac/g/cosmo/ki21/phil1/simulations/ZoomIns/SymphonyMilkyWay'

sim_names = os.listdir(mw_sim_path)

for sim_name in sim_names:
    print(sim_name)
    sim_dir = os.path.join(mw_sim_path, sim_name)
    interfacer = SymphonyInterfacer(sim_dir)