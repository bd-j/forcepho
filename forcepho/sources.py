
import numpy as np


__all__ = ["Scene", "Star", "Galaxy"]
   

# ------
# The setup here is that we are setting all the source parameters at once,
# including a vector for fluxes.  However, this requires having the flux
# attribute of Galaxies and Stars be a vector, and having convert_to_gaussians
# and get_gaussian_gradients look for the appropriate element of the flux
# array.
# ------


class Scene(object):
    """The Scene holds the sources and provides the mapping between a giant 1-d
    array of parameters and the parameters of each source in each band/image
    """

    def __init__(self, sources=[]):
        self.sources = sources
        self.identify_sources()

    def param_indices(self, sid, filtername=None):
        """Get the indices of the relevant parameters in the giant Theta vector.
        
        :param sid:
            Source ID

        :param bid:
            Band ID


        :returns theta:
            An array with elements [flux, (shape_params)]
        """
        npar_per_source = [s.nparam for s in self.sources[:sid]]
        # get all the shape parameters
        # TODO: nshape (and use_gradients) should probably be an attribute of the source
        source = self.sources[sid]
        start = int(np.sum(npar_per_source))
        # indices of the shape and position parameters
        #print(start, source.nband, source.nparam)
        inds = range(start + source.nband, start + source.nparam)
        # put in the flux for this source in this band
        inds.insert(0, start + source.filter_index(filtername))
        return inds

    def set_all_source_params(self, Theta):
        start = 0
        for source in self.sources:
            end = start + source.nparam
            source.set_params(Theta[start:end])
            start += source.nparam

    def identify_sources(self):
        for i, source in enumerate(self.sources):
            source.id = i


class Source(object):
    """Parameters describing a source in the celestial plane. For each galaxy
    there are 7 parameters, only some of which may be relevant for changing the
    apparent flux:
      * flux: total flux
      * ra: right ascension (degrees)
      * dec: declination (degrees)
      * q, pa: axis ratio squared and position angle (might be parameterized differently in future)
      * n: sersic index
      * r: half-light radius (arcsec)

    Methods are provided to return the amplitudes and covariance matrices of
    the constituent gaussians, as well as derivatives of the amplitudes with
    respect to sersic index and half light radius.
    """

    id = 0
    fixed = False
    radii = np.zeros(1)
    
    # Parameters
    flux = 0.     # flux.  This will get rewritten on instantiation to have a length that is the number of bands
    ra = 0.
    dec = 0.
    q = 1.        # sqrt of the axis ratio, i.e.  (b/a)^0.5
    pa = 0.       # postion angle (N of E)
    sersic = 0.   # sersic index
    rh = 0.       # half light radius

    use_gradients = slice(None)

    def __init__(self, filters=['dummy'], radii=None):
        self.filternames = filters
        self.flux = np.zeros(len(self.filternames))
        if radii is not None:
            self.radii = radii

    @property
    def nband(self):
        return len(self.filternames)

    @property
    def nparam(self):
        return self.npos + self.nshape + self.nband

    @property
    def ngauss(self):
        return len(self.radii)

    @property
    def use_gradients(self):
        """Which of the 7 gradients (d/dFlux, d/dRA, d/dDec, d/dq, d/dpa,
        d/dsersic, d/drh) will you actually use?
        """
        return slice(0, 1 + self.npos + self.nshape)

    def filter_index(self, filtername):
        return self.filternames.index(filtername)

    @property
    def covariances(self):
        raise(NotImplementedError)

    @property
    def amplitudes(self):
        """Code here for getting amplitudes from a splined look-up table
        (dependent on self.n and self.r).  Placeholder code gives them all
        equal amplitudes.
        """
        return np.ones(self.ngauss) / (self.ngauss * 1.0)

    @property
    def damplitude_dsersic(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.sersic and self.rh)
        """
        # ngauss array of da/dsersic
        return np.zeros(self.ngauss)

    @property
    def damplitude_drh(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.sersic and self.rh)
        """
        # ngauss array of da/drh
        return np.zeros(self.ngauss)


class Star(Source):
    """This is a represenation of a point source in terms of Scene (on-sky)
    parameters.  Only 3 of the 7 full Source parameters are relevant:
      * flux: total flux
      * ra: right ascension (degrees)
      * dec: declination (degrees)
    """

    radii = np.zeros(1)

    # PointSources only have two position parameters and a single flux parameter
    npos = 2
    nshape = 0
    use_gradients = slice(0, 3)

    def set_params(self, theta, filtername=None):
        """Set the parameters from a theta array
        """
        if filtername is not None:
            nflux = 1
            flux_inds = self.filter_index(filtername)
        else:
            nflux = self.nband
            flux_inds = slice(None)
        assert len(theta) == nflux + 2, "The length of the parameter vector is not appropriate for this source"
        self.flux[flux_inds] = theta[:nflux]
        self.ra  = theta[nflux]
        self.dec = theta[nflux + 1]


    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the radii
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return np.zeros([1, 2, 2])


class SimpleGalaxy(Source):
    """Parameters describing a simple gaussian galaxy in the celestial plane (i.e. the Scene parameters)
    Only 5 of the possible 7 Source parameters are relevant:
      * flux: total flux
      * ra: right ascension (degrees)
      * dec: declination (degrees)
      * q, pa: axis ratio squared and position angle (might be parameterized differently in future)
    """

    radii = np.ones(1)
    
    # Galaxies have two position parameters, 2  or 4 shape parameters (pa and q) and nband flux parameters
    npos = 2
    nshape = 2

    def set_params(self, theta, filtername=None):
        """Set the parameters from a theta array
        """
        if filtername is not None:
            nflux = 1
            flux_inds = self.filter_index(filtername)
        else:
            nflux = self.nband
            flux_inds = slice(None)
        assert len(theta) == nflux + 4, "The length of the parameter vector is not appropriate for this source"
        self.flux[flux_inds] = theta[:nflux]
        self.ra  = theta[nflux]
        self.dec = theta[nflux + 1]
        self.q   = theta[nflux + 2]
        self.pa  = theta[nflux + 3]

    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the fixed
        radii used in approximating the galaxies.
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return (self.radii**2)[:, None, None] * np.eye(2)


class Galaxy(Source):
    """Parameters describing a gaussian galaxy in the celestial plane (i.e. the Scene parameters)
    All 7 Source parameters are relevant:
      * flux: total flux
      * ra: right ascension (degrees)
      * dec: declination (degrees)
      * q, pa: axis ratio squared and position angle (might be parameterized differently in future)
      * n: sersic index
      * r: half-light radius (arcsec)

    Methods are provided to return the amplitudes and covariance matrices of
    the constituent gaussians, as well as derivatives of the amplitudes with
    respect to sersic index and half light radius.
    """

    radii = np.ones(1)
    
    # Galaxies have two position parameters, 2  or 4 shape parameters (pa and q) and nband flux parameters
    npos = 2
    nshape = 2

    def __init__(self, filters=['dummy'], radii=None, splines=None):
        super(Galaxy, self).__init__(filters=filters, radii=radii)
        if splines is not None:
            self.nshape = 4

    def set_params(self, theta, filtername=None):
        """Set the parameters from a theta array
        """
        if filtername is not None:
            nflux = 1
            flux_inds = self.filter_index(filtername)
        else:
            nflux = self.nband
            flux_inds = slice(None)
        assert len(theta) == nflux + 6, "The length of the parameter vector is not appropriate for this source"
        self.flux[flux_inds] = theta[:nflux]
        self.ra  = theta[nflux]
        self.dec = theta[nflux + 1]
        self.q   = theta[nflux + 2]
        self.pa  = theta[nflux + 3]
        self.sersic = theta[nflux + 4]
        self.rh = theta[nflux + 5]

    @property
    def covariances(self):
        """This just constructs a set of covariance matrices based on the fixed
        radii used in approximating the galaxies.
        """
        # ngauss x 2 x 2
        # this has no derivatives, since the radii are fixed.
        return (self.radii**2)[:, None, None] * np.eye(2)

    @property
    def amplitudes(self):
        """Code here for getting amplitudes from a splined look-up table
        (dependent on self.n and self.r).  Placeholder code gives them all
        equal amplitudes.
        """
        return np.ones(self.ngauss) / (self.ngauss * 1.0)

    @property
    def damplitude_dsersic(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/dsersic
        return np.zeros(self.ngauss)

    @property
    def damplitude_drh(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/drh
        return np.zeros(self.ngauss)


    
def scale_matrix(q):
    #return np.array([[q**(-0.5), 0],
    #                [0, q**(0.5)]])
    return np.array([[1./q, 0],  #use q=(b/a)^0.5
                    [0, q]])


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def scale_matrix_deriv(q):
    #return np.array([[-0.5 * q**(-1.5), 0],
    #                [0, 0.5 * q**(-0.5)]])
    return np.array([[-1./q**2, 0], # use q=(b/a)**2
                    [0, 1]])


def rotation_matrix_deriv(theta):
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta), -np.sin(theta)]])
