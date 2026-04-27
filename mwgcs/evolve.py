from abc import ABC, abstractmethod
import numpy as np
import agama
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator

agama.setUnits(mass=1.0, length=1.0, velocity=1.0)

class MassLossModel(ABC):
    def __init__(self, relaxation_model, stellar_evolution_model):
        self.relaxation_model = relaxation_model
        self.stellar_evolution_model = stellar_evolution_model
        self.has_closed_form = False

    @abstractmethod    
    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False):
        if integrated and not self.has_closed_form:
            raise ValueError("This mass loss model does not have a closed-form solution for the integrated mass loss. Set `integrated=False`.")

class MassLossComponent(ABC):
    @abstractmethod
    def mass_loss_rate(self, cluster_mass, tidal_strength):
        pass
class StellarEvolutionMassLossModel(MassLossComponent):
    def __init__(self):
        pass

class RelaxationMassLossModel(MassLossComponent):
    def __init__(self):
        pass

class NoStellarEvolutionModel(StellarEvolutionMassLossModel):
    '''
    A mass loss model that does not include stellar evolution effects.
    Valid if you assume that all cluster masses in your simulation have "accounted" for
    the mass loss from stellar evolution (i.e., using some prescription for the 'evolved' GCLF after instantaneous stellar evolution). 
    '''
    def mass_loss_rate(self, cluster_mass, tidal_strength):
        return 0.0
    
class FiducialRelaxationMassLossModel(RelaxationMassLossModel):
    """
    Original equation:
    M(t) = \left(M_0^{2/3} - \frac{2A}{3} \int_{t_0}^{t} \sqrt{\frac{\kappa}{3} \lambda (\tau)} d\tau\right)^{3/2}

    I absorb sqrt(kappa / 3) into the A term:
    M(t) = \left(M_0^{2/3} - \frac{2A}{3} \int_{t_0}^{t} \sqrt{\lambda (\tau)} d\tau\right)^{3/2}

    And I save the spline integral of sqrt(lambda(tau)) dtau as:
    X(t)=\int_{t_0}^{t} \sqrt{\lambda (\tau)} d\tau

    M(t) = \left(M_0^{2/3} - \frac{2AX(t)}{3} \right)^{3/2}    
    """

    def __init__(self, kappa=1.0):
        self.kappa = kappa
    
    def mass_loss_rate(self, m_cl, tidal_strength):
        return - m_cl / ( 10 * (m_cl / 2e5)**(2.0/3.0) / (np.sqrt(self.kappa * tidal_strength / 3) / 100.0))

class FiducialMassLossModel(MassLossModel):
    def __init__(self, kappa=1.0):
        super().__init__(relaxation_model=FiducialRelaxationMassLossModel(kappa=kappa), stellar_evolution_model=NoStellarEvolutionModel())
        self.has_closed_form = True

    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False):
        
        A = np.sqrt(self.relaxation_model.kappa / 3.0) * (2e5)**(2/3) / 1000

        if integrated:
            # `tidal_strength` is the time-integrated tidal strength
            x = np.maximum(0.0, initial_mass**(2/3) - (2 * A / 3) * tidal_strength)
            return x**(3/2)
        else:
            # time is the time when the tidal field is evaluated, and `tidal_strength` is the instantaneous tidal strength at that time
            # numerically integrate the mass loss rate over time to get the evolved mass at the given time

            interpolated_strength = PchipInterpolator(time, tidal_strength)

            def dMdt(t, m_cl):
                return self.relaxation_model.mass_loss_rate(m_cl, interpolated_strength(t))

            sol = solve_ivp(dMdt, (time[0], time[-1]), [initial_mass], t_eval=[time[-1]])
            return sol.y[0]
