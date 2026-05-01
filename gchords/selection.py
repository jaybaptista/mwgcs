from abc import ABC, abstractmethod
import numpy as np
import os

class SelectionFunction(ABC):
    def __init__(self, frame):
        self.frame # observer frame
    
    @abstractmethod
    def select_particles(self):
        pass

    @abstractmethod
    def create_observable(self):
        pass


class SimpleGaiaSelectionFunction(SelectionFunction):
    def select_particles(self, particles):
        pass

    def get_gaia_g(self, mstar, mass_to_light_ratio=2.0):
        '''
        returns absolute Gaia G magnitude from luminosity 
        Solar G band is roughly 4.7 (https://arxiv.org/pdf/1904.04841)
        '''
        M_sun_G = 4.7
        luminosity = mstar / mass_to_light_ratio
        M_G = M_sun_G - 2.5 * np.log10(luminosity)
        return M_G