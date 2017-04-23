import numpy as np


class PhonionSource(ParameterSet):

    def __init__(self):
        pass

    def draw(self, ndraw):
        raise(NotImplementedError)

    def transform(self):
        
        pass

    def get_locations(self):
        pass

    def get_location_gradients(self):
        pass
        

class Sersic(PhonionSource):

    




def rgrid_flux(ngrid, nsersic):
    """Returns r sampled uniformly in cumulative luminosity.
    """
    clf = np.linspace(0, 1, ngrid)
    z = gammaincinv(2. * nsersic, clf)
    return z**nsersic
