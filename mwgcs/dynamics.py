import numpy as np
import abc

class InterpolatedPotential(abc.ABC):
    

    def __init__(self, params, ts=None, **kwargs):
        self.params = params
        self.ts = ts

        if ts is not None:
            if self.params.shape[0] != self.ts.shape[0]:
                raise ValueError("Time steps must match the number of snapshots \
                                 suggested by the potential parameters.")

    @abc.abstractmethod
    def hessian(self, **kwargs):
        pass

    @abc.abstractmethod
    def acceleration(self, **kwargs):
        # units of km/s^2
        pass

    @abc.abstractmethod
    def potential(self, **kwargs):
        # units of km^2/s^2
        pass

class MassLossModel(abc.ABC):

    def __init__(self, qs, ts, m0, pot, **kwargs):
        self.qs = qs # positions in kpc
        self.ts = ts # time steps in Myr
        self.m0 = m0 # initial mass in Msun
        self.pot = pot # potential

        if self.qs.shape[0] != self.ts.shape[0]:
            raise ValueError("Time steps must match the number of snapshots \
                             suggested by the positions.")
    
        # check if potential is an instance of the Potential class and 
        # if the time steps match the number of snapshots suggested by the potential parameters
        if not isinstance(self.pot, Potential):
            raise ValueError("Potential must be an instance of the Potential class.")
        
        if self.pot.ts is not None:
            if self.ts.shape[0] != self.pot.ts.shape[0]:
                raise ValueError("Time steps must match the number of snapshots \
                                 suggested by the potential")

    @abc.abstractmethod
    def rate(self, **kwargs):
        # units of Msun/Myr
        pass

    @abc.abstractmethod
    def mass(self, **kwargs):
        # units of Msun
        pass

    @abc.abstractmethod
    def destruction_time(self, **kwargs):
        # units of Myr
        pass


from scipy.integrate import quad
import jax.scipy as jsc
import jax.numpy as jnp
import astropy.constants as c
import astropy.units as u

class EinastoPotential(InterpolatedPotential):
    
    def __init__(self, qs, params, ts=None, **kwargs):
        """
        Initialize the interpolated Einasto potential class.

        Parameters
        ----------

        qs : array
            Positions of the test particles in kpc

        params : array
            Array of potential parameters: alpha, scaleRadius, logScaleDensity
        
        ts : array
            Time steps in Myr
        """
        super().__init__(params, ts, **kwargs)
        alpha, scaleRadius, logScaleDensity = params

        if ts is not None:
            if (len(ts) != len(alpha)) or (len(ts) != len(scaleRadius)) or (len(ts) != len(logScaleDensity)):
                raise ValueError("Time steps must match the number of snapshots \
                                 suggested by the potential parameters.")
        
        self.ts = ts

        xs, ys, zs = qs
        
        self.xs = jsc.interpolate.UnivariateSpline(ts, xs, s=0)
        self.ys = jsc.interpolate.UnivariateSpline(ts, ys, s=0)
        self.zs = jsc.interpolate.UnivariateSpline(ts, zs, s=0)

        self.alpha = jsc.interpolate.UnivariateSpline(ts, alpha, s=0)
        self.scaleRadius = jsc.interpolate.UnivariateSpline(ts, scaleRadius, s=0)
        self.logScaleDensity = jsc.interpolate.UnivariateSpline(ts, logScaleDensity, s=0)
        
    # setup 

    def A(self, alpha):
        """
        Assuming an Einasto profile, the radius at which the density curve is maximal is A times the scale radius Rs.

        Returns
        -------
        float
            A parameter for the Einasto profile
        """
        return 1.715 * (alpha**(-.00183)) * (alpha + 0.0817)**(-.179488)

    
    # numbers
    def hessian(self, **kwargs):
        pass
        

    def potential(self, t, **kwargs):
        """
        Returns the potential of the test particles at time t.

        Parameters
        ----------
        q : array
            Position at which to evaluate the potential
        
        Returns
        -------
        float
            Potential at q in units of km^2/s^2
        """

        r = jnp.sqrt(self.xs(t)**2 + self.ys(t)**2 + self.zs(t)**2) # in kpc
        _a = self.alpha(self.alpha(t))
        _g = 4.498502151469554e-06 # units of kpc^3/Gyr^2/Msun
        
        scaleDensity = 10**self.logScaleDensity(t)

        def lowerIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammainc(a, x) * jsc.special.gamma(a)
            _tilde = (_a * self.Rs(t)**_a / 2)**a
            
            if tilde:
                return base * _tilde
            else:
                return base
        
        def upperIncompleteGamma(a, x, tilde=False):
            base = jsc.special.gammaincc(a, x) * jsc.special.gamma(a)
            _tilde = (_a * self.Rs**_a(t) / 2)**a
            if tilde:
                return base * _tilde
            else:
                return base

        _sr = 2 * r**_a / (_a * self.Rs(t)**_a)

        tmp1 = 4*np.pi*_g*scaleDensity*jnp.exp(2/_a) / _a
        tmp2 = (1/r) * lowerIncompleteGamma(3/_a, _sr, tilde=True)
        tmp3 = upperIncompleteGamma(2/_a, _sr, tilde=True)
        
        # convert pot to km^2/s^2
        factor = 0.9560776287794536
        pot = tmp1 * (tmp2 - tmp3)
        pot *= factor

        return pot
    
    def acceleration(self, t, **kwargs):

        pot = self.potential(t)

        # calculate acceleration from potential

        return -jnp.array(jnp.gradient(pot, [self.xs(t), self.ys(t), self.zs(t)]))