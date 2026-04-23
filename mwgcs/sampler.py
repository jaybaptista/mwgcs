import abc
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import interp1d
import symlib
from mwgcs.tag import GlobularClusterRhalf

# TODO: kwargs

"""
Mass-to-light ratios are taken from N-body modeling of globular clusters:
https://arxiv.org/abs/1609.08794
"""


def magnitude_to_luminosity(magnitude, zero_point=5.12):
    """Convert magnitude to luminosity using the zero point."""
    return 10 ** ((zero_point - magnitude) / 2.5)


def luminosity_to_mass(luminosity, ratio=3.0):
    """Convert luminosity to mass using the mass-to-light ratio."""
    return luminosity * ratio


class OccupationModel(abc.ABC):
    """
    Base class for GC occupation models
    """

    def __init__(self, seed=None):
        self.kind = None
        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def p_gc(self, **kwargs):
        pass

    @abc.abstractmethod
    def has_gc(self, **kwargs):
        pass


class GCSMassModel(abc.ABC):
    """
    Base class for GC system mass models
    """

    def __init__(self, seed=None):
        self.kind = None
        self.evolving = False
        self.mean_mass = 0

        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def mass(self, **kwargs):
        pass


class GCLuminosityFunction(abc.ABC):
    """
    Base class for GC luminosity functions
    """

    def __init__(self, seed=None):
        self.kind = None
        self.evolving = False

        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def sample(self, n_draws, **kwargs):
        pass


def _lognormal_icdf(logmu, sigma, Mmin=1e-2, Mmax=1e8, n_grid=4096):
    """
    Returns the interpolated inverse CDF for the PDF of
    the form:

    dN/dM = [1/(ln 10 * M)] * [1/(sqrt(2π) * σ_M)] *
              exp( - (log10 M - μ)^2 / (2 σ_M^2) )

    NOTE: logmu is the mean of the distribution in log10
    """
    M = np.logspace(np.log10(Mmin), np.log10(Mmax), n_grid)
    x = np.log10(M)

    norm = np.log(10) * M * (np.sqrt(2 * np.pi) * sigma)
    pdf = np.exp(-0.5 * ((x - logmu) / sigma) ** 2) / norm
    pdf = np.clip(pdf, 0.0, np.inf)

    # Integrate with trapezoid rule to get CDF
    cdf = np.zeros_like(M)
    cdf[1:] = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(M))

    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PDF integral is non-positive over the chosen range.")
    cdf /= total

    # Ensure strict monotonicity (protect against flat tails)
    cdf = np.maximum.accumulate(cdf)
    cdf[-1] = 1.0

    # Build inverse CDF via interpolation
    icdf_interp = interp1d(
        cdf,
        M,
        kind="linear",
        bounds_error=False,
        fill_value=(Mmin, Mmax),
        assume_sorted=True,
    )

    # define function
    def icdf(u):
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        return icdf_interp(u)

    return icdf


class GaussianGCLF(GCLuminosityFunction):
    def __init__(self, mu=-7.0, sigma=1.0, M_sun=5.12, ml_ratio=2.0, seed=None):
        """
        Generic Gaussian GCLF
        """

        super().__init__(seed=seed)

        self.mu = mu
        self.sigma = sigma
        self.M_sun = M_sun
        self.ml_ratio = ml_ratio

        # converting to mass
        C = M_sun + 2.5 * np.log10(ml_ratio)
        self.M_mu = 10 ** ((C - mu) / 2.5)
        self.sigma_M = sigma / 2.5
        self.mean_mass = self.M_mu * np.exp(0.5 * (np.log(10) * self.sigma_M) ** 2)

        # precompute the inverse CDF for sampling
        self.icdf = _lognormal_icdf(np.log10(self.M_mu), self.sigma_M)

        self.kind = None  # Kind of dependence of the GCLF parameters
        self.evolving = False  # Whether the GCLF parameters evolve over redshift

    def var_names(self):
        return ['mu', 'sigma', 'M_sun', 'ml_ratio']

    def sample_mag(self, n_draws):
        """
        Samples magnitudes from the GCLF
        """
        return np.random.normal(self.mu, self.sigma, size=n_draws)

    def sample(self, n_draws, **kwargs):
        """
        Samples masses from the GCLF
        """
        u = np.random.uniform(0, 1, size=n_draws)
        return self.icdf(u)


"""
Occupation models
"""


class EadieOccupationModel(OccupationModel):
    def __init__(self, b0=-10.83, b1=1.59, seed=None):
        super().__init__(seed=seed)
        self.kind = "stellar"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, stellar_mass):
        p = (1 + np.exp(-1 * (self.b0 + self.b1 * np.log10(stellar_mass)))) ** (-1)
        return p

    def has_gc(self, stellar_mass):
        p = self.p_gc(stellar_mass)
        return uniform.rvs() < p


class DornanOccupationModel(OccupationModel):
    def __init__(self, b0=-31.86, b1=3.0, seed=None):
        super().__init__(seed=seed)
        self.kind = "halo"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, halo_mass):
        p = 1 / (1 + np.exp(-(self.b0 + self.b1 * np.log10(halo_mass))))
        return p

    def has_gc(self, halo_mass):
        p = self.p_gc(halo_mass)
        return uniform.rvs() < p


"""
GC system mass models
"""


class GCSMassLinearModel(GCSMassModel):
    def __init__(self, g0=-0.725, g1=0.788, seed=None):
        """
        Implementation of the linear regression model from Eadie+2022
        Source: https://iopscience.iop.org/article/10.3847/153
        """
        super().__init__(seed=seed)

        self.g0 = g0
        self.g1 = g1
        self.kind = "stellar"

    def var_names(self):
        return ["g0", "g1"]

    def mass(self, stellar_mass):
        return 10 ** (self.g0 + self.g1 * np.log10(stellar_mass))


class GCSMassHarrisModel(GCSMassModel):
    def __init__(self, g0=-0.725, g1=0.788, scatter=0.0, seed=None):
        """
        Implementation of the Harris halo mass–GCS mass relation from
        Harris, Blakeslee, & Harris (2017) paper.
        """

        super().__init__(seed=seed)

        self.g0 = g0
        self.g1 = g1
        self.scatter = scatter
        self.kind = "halo"

    def var_names(self):
        return ["g0", "g1", "scatter"]

    def mass(self, halo_mass):
        eta = 2.9e-5
        mhalo = eta * halo_mass

        if self.scatter > 0:
            log_scatter = self.scatter * np.random.normal(0, 1, size=np.shape(mhalo))
            log_eta = np.log10(eta) + log_scatter
            return 10**log_eta * halo_mass
        else:
            return eta * halo_mass


class GCSMassDornanModel(GCSMassModel):
    def __init__(self, slope=0.9257, intercept=-3.5645, scatter=0.3, seed=None):
        """
        Implementation of the Dornan and Harris (2026) halo mass–GCS mass relation
        refit to the Dornan dwarf galaxy catalog (except for Fornax Deep Survey)
        """

        super().__init__(seed=seed)

        self.slope = slope
        self.intercept = intercept
        self.scatter = scatter
        self.kind = "halo"

    def var_names(self):
        return ["slope", "intercept", "scatter"]

    def mass(self, halo_mass, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if self.scatter > 0:
            log_scatter = self.scatter * np.random.normal(
                0, 1, size=np.shape(halo_mass)
            )
            log_mgc = self.intercept + self.slope * np.log10(halo_mass) + log_scatter
            return 10**log_mgc
        else:
            return 10 ** (self.intercept + self.slope * np.log10(halo_mass))


"""
GC luminosity functions
"""


class GCMFGeorgiev(GaussianGCLF):
    def __init__(self, mass_light_ratio=1.98, seed=None):
        """
        Implementation of the GCMF from Georgiev+2009
        Source: https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..879G/abstract
        => Valid up to galaxy stellar masses of ~1e9 Msun
        """
        mu_V = -7.04
        sigma_V = 1.15
        V_sun = 4.80

        super().__init__(
            mu=mu_V, sigma=sigma_V, M_sun=V_sun, ml_ratio=mass_light_ratio, seed=seed
        )


class GCMFElves(GaussianGCLF):
    def __init__(self, mass_light_ratio=1.98, seed=None):
        """
        Implementation of the GCMF from ELVES (Carlsten+22a; https://www.arxiv.org/abs/2105.03440).
        """
        mu_g = -7.02
        sigma_g = 0.57
        g_sun = 5.05

        super().__init__(
            mu=mu_g, sigma=sigma_g, M_sun=g_sun, ml_ratio=mass_light_ratio, seed=seed
        )


class GCMFVillegas(GaussianGCLF):
    def __init__(self, halo_mass, mass_light_ratio=1.98, seed=None):
        """
        Implementation of the GCMF from Villegas+2010, which depends on galaxy mass
        """
        self.kind = "halo"
        mu_g = self._mu_g(np.log10(halo_mass), seed=seed)
        sigma_g = self._sigma_g(np.log10(halo_mass), seed=seed)
        g_sun = 5.05
        super().__init__(
            mu=mu_g, sigma=sigma_g, M_sun=g_sun, ml_ratio=mass_light_ratio, seed=seed
        )
        self.kind = "halo"

    """
    Fits to the Villegas+2010 GCLF parameters as a function of galaxy mass, with scatter
    """

    def _mu_g(self, mpeak, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x0 = 11.8
        y0 = -7.29
        m = -0.09
        sqrtV = 0.18
        scatter = np.random.normal(0, 1) * sqrtV
        return y0 + m * (np.log10(mpeak) - x0) + scatter

    def _sigma_g(self, mpeak, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x0 = 11.8
        y0 = 0.95
        m = 0.34
        sqrtV = 0.15
        scatter = np.random.normal(0, 1) * sqrtV
        return y0 + m * (np.log10(mpeak) - x0) + scatter


class GCHaloModel:
    def __init__(self, occupation_model, mass_model, gclf_model, nimbus_model, seed=None):
        self.occupation_model = occupation_model
        self.mass_model = mass_model
        self.gclf_model = gclf_model
        self.nimbus_model = nimbus_model

        if seed is not None:
            np.random.seed(seed)

        self.required_inputs = ["halo_mass", "stellar_mass"]

    def var_names(self):
        return {
            "occupation_model": self.occupation_model.var_names(),
            "mass_model": self.mass_model.var_names(),
            "gclf_model": self.gclf_model.var_names(),
        }

    def generate(self, **kwargs):
        """
        Returns a tuple (bool: has_gc, int: gc_count, list: gc_masses)
        """

        halo_mass = kwargs.get("halo_mass")
        stellar_mass = kwargs.get("stellar_mass")
        # TODO: add redshift evolution

        if halo_mass is None:
            raise ValueError("halo_mass not supplied")
        if stellar_mass is None:
            raise ValueError("stellar_mass not supplied")

        has_gc = self.occupation_model.has_gc(
            input_mass=halo_mass
            if self.occupation_model.kind == "halo"
            else stellar_mass
        )

        if not has_gc:
            return False, 0, []

        gc_mass = self.mass_model.mass(
            input_mass=halo_mass if self.mass_model.kind == "halo" else stellar_mass
        )

        if gc_mass <= 0:
            return True, 0, None

        lam = gc_mass / self.gclf_model.mean_mass

        if lam <= 0 or np.isnan(lam):
            print(f"ERROR: gc_mass / mean_gc_mass = {lam}")
            return True, 0, None

        n_draws = np.random.poisson(lam)

        gc_masses = self.gclf_model.sample(n_draws)

        return True, n_draws, gc_masses

GC_HALO_MODEL = symlib.GalaxyHaloModel(
    symlib.StellarMassModel(symlib.UniverseMachineMStarFit(), symlib.DarkMatterSFH()),
    symlib.ProfileModel(GlobularClusterRhalf(), symlib.PlummerProfile()),
    symlib.MetalModel(
        symlib.Kirby2013Metallicity(),
        symlib.Kirby2013MDF(model_type="gaussian"),
        symlib.GaussianCoupalaCorrelation(),
    ),
)

class FiducialGCHaloModel(GCHaloModel):
    def __init__(self):
        super().__init__(
            occupation_model=EadieOccupationModel(),
            mass_model=GCSMassLinearModel(),
            gclf_model=GCMFGeorgiev(),
            nimbus_model=GC_HALO_MODEL,
        )
