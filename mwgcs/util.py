import asdf
import astropy.constants as c
import astropy.units as u
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
import symlib
from tqdm import tqdm

import os

from .sym import Simulation
from .form import lognorm_hurdle, sampleMilkyWayGCMF, sampleDwarfGCMF
from .evolve import getMassLossRate, getTidalTimescale, getTidalFrequency

# from .fit import MassProfile

# lookup tables

# class EinastoLookupTable():
#     def __init__(self, sim : Simulation, ):
#         self.sim = sim
#         self.df = asdf.AsdfFile()
#         self.scale = 1.5
#         self.n_sh = self.sim.rs.shape[0]
    
#     def createLookupTable(self, write_dir):
        
#         self.alpha = np.zeros(self.sim.rs.shape) - 1.
#         self.rs = np.zeros(self.sim.rs.shape) - 1.
#         self.logrho = np.zeros(self.sim.rs.shape) - 99.
        
#         for k in tqdm(range(self.n_sh)):
#             ok_snaps = np.where(self.sim.rs[k, :]['ok'])[0]
#             for sn in tqdm(ok_snaps):
#                 prof = MassProfile(self.sim, sn, k)
                
#                 alpha, rs, logrho = np.nan, np.nan, np.nan
                
#                 try:
#                     prof.fit()
#                     alpha, rs, logrho = prof.profile_params
#                 except ValueError:
#                     print("Error: possible infinite residual, ignoring fit.")
                
#                 self.alpha[k, sn] = alpha
#                 self.rs[k, sn] = rs
#                 self.logrho[k, sn] = logrho

#         self.df['alpha'] = self.alpha
#         self.df['rs'] = self.rs
#         self.df['logrho'] = self.logrho

#         self.df.write_to(os.path.join(write_dir , f'einasto_params_{self.sim.getSimulationName()}.asdf'))

########################################################################################################

# LinearNDInterpolatorExtrapolator.py
# from https://gist.github.com/tunnell/90a83b7e1f894e8f882a029caece447b
import scipy # sigh py 
from scipy.interpolate import LinearNDInterpolator as LNDI


class Linear2DInterpolator(object):
    """
    An extension of *LinearNDInterpolator* in 2D that
    extrapolates (linearly) outside of the convex hull.
    Nearest simplex is used for that, which might not be
    the best choice, but it's the simplest one.
    Many bug fixes and modifications for 2D field:
    https://gist.github.com/minzastro/af35e8e9b3a1626e1586f75f96439ebd
    """

    def __init__(self, points, values):
        self.lndi = LNDI(points, values)
        hull = self.lndi.tri.convex_hull.tolist()
        # Build a mask showing if a simplex has a side that
        # is in the convex hull.
        self.is_convex_simplex = np.zeros(len(self.lndi.tri.simplices),
                                          dtype=bool)
        for irow, row in enumerate(self.lndi.tri.simplices):
            rrow = row[[0, 1, 2, 0]]
            for pos in range(3):
                if rrow[pos:pos + 2].tolist() in hull or \
                        rrow[[pos + 1, pos]].tolist() in hull:
                    self.is_convex_simplex[irow] = True

    def __call__(self, xi):
        """
        Predict values at points *xi*
        """
        result = self.lndi(xi)
        mask = np.isnan(result.sum(axis=1))
        if not np.any(mask):
            # All points are within the convex hull - nothing to do more.
            return result
        simplices = []

        # Build a list of neares simplices for points outside
        # the convex hull
        for i, drow in enumerate(self.lndi.tri.plane_distance(xi[mask])):

            mi = np.max(drow[self.is_convex_simplex])  # Fixed if nearested isn't on hull
            w = np.where(drow == mi)[0]

            if len(w) == 1 and self.is_convex_simplex[w]:
                simplices.append(w)
            elif len(w) > 1:
                ww = w[self.is_convex_simplex[w]][0]
                simplices.append(ww)
            else:
                raise ValueError

        simplices = np.array(simplices)
        result_update = np.zeros((mask.sum(), 2))  # fixed
        for simple in np.unique(simplices):
            indices = np.where(simplices == simple)[0]
            result_update[indices] = self._get_simplex_at_point(simple, xi[mask][indices])

        result[mask] = result_update
        return result

    def _get_simplex_at_point(self, ind, point):
        """
        Calculate the value at a point (or points)
        using the plane build on simplex with index *ind*.
        """
        point = np.atleast_2d(np.array(point))
        simplex = self.lndi.tri.simplices[ind]

        plane_points = self.lndi.tri.points[[simplex]]
        values = self.lndi.values[[simplex]]

        point = point - plane_points[0]

        extrapolated_points = []

        for i, value in enumerate(values[0]):
            # These two vectors are in the plane
            v1 = (plane_points[2][0] - plane_points[0][0],
                  plane_points[2][1] - plane_points[0][1],
                  values[2][i] - values[0][i])

            v2 = (plane_points[1][0] - plane_points[0][0],
                  plane_points[1][1] - plane_points[0][1],
                  values[1][i] - values[0][i])

            # the cross product is a vector normal to the plane
            cp = np.cross(v1, v2)

            # z corresponding to plane requires dot with norm = 0
            # (Norm) dot (position with only z unknown) = 0, solve.
            extrapolated_points.append(values[0][i] + (- cp[0] * point[:, 0] - cp[1] * point[:, 1]) / cp[2])

        np.array(extrapolated_points).T

        return np.array(extrapolated_points).T

# Nearest neighbor outside of convex hull
class LinearNDInterpolatorExt(object):
    def __init__(self, points, values):
        self.funcinterp = scipy.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = scipy.interpolate.NearestNDInterpolator(points, values)

    def __call__(self, *args):
        t = self.funcinterp(*args)
        u = self.funcnearest(*args)

        t[np.isnan(t)] = u[np.isnan(t)]
        return t