import numpy as np
import heapq
import symlib


class GlobularClusterRhalf(symlib.RHalfModel):
    """
    A model for the half-light radius of a globular cluster system.
    The assumption is that the rGC (= Rhalf of the GC system) is a fixed multiple of the
    galaxy's effective radius. We adopt the stacked fit from Carlsten+2021 (arxiv.org/abs/2105.03440)
    which, for a local volume sample, fits rGC/r_eff ~ 1.06.
    """

    def __init__(self, ratio=1.06, galaxy_profile=symlib.Jiang2019RHalf()):
        self.ratio = ratio
        self.galaxy_profile = galaxy_profile

    def r_half(self, rvir=None, cvir=None, z=None):
        if rvir is None:
            raise ValueError("rvir not supplied")
        if cvir is None:
            raise ValueError("cvir not supplied")
        if z is None:
            raise ValueError("z not supplied")
        reff_galaxy = self.galaxy_profile.r_half(rvir=rvir, cvir=cvir, z=z)
        return self.ratio * reff_galaxy

    def r_half_is_2d(self):
        return self.galaxy_profile.r_half_is_2d()

    def var_names(self):
        return ["rvir", "cvir", "z"]



"""
Credit: Phil Mansfield
A set of functions to return star weights in the event
that fitting the profile fails.
"""


def tag_energy_cut(p, E, r50, n_min=8, profile="uniform"):
    """tag_energy_cut tags particles according to an energy cut. Invalid
    partcles need to be removed and the particles already need to be centered
    on their subhalos.
    """
    assert len(p) > n_min

    order = np.argsort(E)

    r = np.sqrt(np.sum(p["x"][order] ** 2, axis=1))
    ok = p["smooth"]

    r_med = running_median(r[ok])

    too_small = r_med <= r50
    too_big = r_med > r50

    candidates = np.where(too_small[:-1] & too_big[1:])[0]

    if len(candidates) == 0 or candidates[-1] < n_min:
        n_core = n_min
    else:
        n_core = candidates[-1]

    mp = np.zeros(len(E))

    if profile == "uniform":
        mp[order[:n_core]] = 1 / n_core
    elif profile == "plummer":
        a = r50
        r_core = r[order[:n_core]]
        mp[order[:n_core]] = (
            3 * a**2 / (4 * np.pi * r_core**2 * (r_core**2 + a**2) ** (3 / 2))
        )  # plummer
        mp[order[:n_core]] /= np.sum(mp[order[:n_core]])  # normalize
    else:
        raise ValueError(f"unknown profile: {profile}")

    return mp


def running_median(x):
    if len(x) == 0:
        return np.array([], dtype=x.dtype)
    med = np.zeros(len(x))

    low, high = [], [x[0]]

    med[0] = x[0]

    # This can be sped up with more conditionals that replace separate
    # heappush and heappop calls with a combined heappushpop.
    for i in range(1, len(x)):
        # Grow the correct queue.
        if x[i] <= med[i - 1]:
            heapq.heappush(low, -x[i])
        else:
            heapq.heappush(high, x[i])

        # If one queue is too big, equalise them.
        if len(low) == len(high) + 2:
            xx = heapq.heappop(low)
            heapq.heappush(high, -xx)
        elif len(high) == len(low) + 2:
            xx = heapq.heappop(high)
            heapq.heappush(low, -xx)

        # Evaluate median.
        if len(low) == len(high):
            med[i] = (-low[0] + high[0]) / 2
        elif len(low) == len(high) + 1:
            med[i] = -low[0]
        elif len(high) == len(low) + 1:
            med[i] = high[0]

    return np.array(med)


def r50(p, E, mp):
    ok = p["ok"] & (E < 0)
    p, mp = p[ok], mp[ok]

    r = radius(p["x"])

    order = np.argsort(r)
    r, mp = r[order], mp[order]

    # Compute the mass CDF.
    m_enc = np.cumsum(mp) / np.sum(mp)

    # We need to add zeros to the start to avoid crashing on a degenerate
    # case where the first particle has more than half the mass. Although in
    # such a case, we're probably doomed anyway.
    r, m_enc = np.hstack([[0], r]), np.hstack([[0], m_enc])

    # interpolate
    i = np.searchsorted(m_enc, 0.5)
    return r[i - 1] + (0.5 - m_enc[i - 1]) * (r[i] - r[i - 1]) / (
        m_enc[i] - m_enc[i - 1]
    )


def energy(param, p):
    rmax, vmax, phi_vmax2, _ = symlib.profile_info(param, p["x"])
    return np.sum(p["v"] ** 2, axis=1) / 2 + phi_vmax2 * vmax**2


def radius(x):
    return np.sqrt(np.sum(x**2, axis=1))


def center(p, halo):
    out = np.copy(p)
    out["x"] -= halo["x"]
    out["v"] -= halo["v"]
    return out


def expand(x, ok):
    out = np.ones(len(ok)) * np.nan
    out[ok] = x
    return out
