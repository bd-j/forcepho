import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import h5py

__all__ = ["Scene", "Source",
           "Star", "SimpleGalaxy", "Galaxy"]


class Scene(object):
    """The Scene holds the sources and provides the mapping between a giant 1-d
    array of parameters and the parameters of each source in each band/image
    """

    def __init__(self, sources=[]):
        self.sources = sources
        self.identify_sources()

    def param_indices(self, sid, filtername=None):
        """Get the indices of the relevant parameters in the giant Theta
        vector.  Assumes that the order of parameters for each source is
        is [flux1, flux2...fluxN, pos1, pos2, shape1, shape2, ..., shapeN]

        :param sid:
            Source ID

        :param filtername: (optional, default: None)
            The name of the filter for which you want the corresponding flux
            parameter index.  If None (default) then indices for all fluxes are
            returned

        :returns theta:
            An array with elements [flux, (shape_params)]
        """
        npar_per_source = [s.nparam for s in self.sources[:sid]]
        # get all the shape parameters
        source = self.sources[sid]
        start = int(np.sum(npar_per_source))
        # indices of the shape and position parameters
        inds = range(start + source.nband, start + source.nparam)
        # put in the flux for this source in this band
        inds.insert(0, start + source.filter_index(filtername))
        return inds

    def set_all_source_params(self, Theta):
        """Loop over sources in the scene, setting the parameters in each
        source based on the relevant subset of Theta parameters.
        """
        start = 0
        for source in self.sources:
            end = start + source.nparam
            source.set_params(Theta[start:end])
            start += source.nparam

    def get_all_source_params(self):
        """Get the total scene parameter vector
        """
        plist = [s.get_param_vector() for s in self.sources if not s.fixed]
        params = np.concatenate(plist)
        return params

    @property
    def parameter_names(self):
        """Get names for all the parameters in the scene
        """
        return np.concatenate([s.parameter_names for s in self.sources])

    def identify_sources(self):
        """Give each source in the scene a unique identification number.
        """
        for i, source in enumerate(self.sources):
            source.id = i


class Source(object):
    """Parameters describing a source in the celestial plane. For each galaxy
    there are 7 parameters, only some of which may be relevant for changing the
    apparent flux:
      * flux: total flux (possibly a vector)
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
    flux = 0.     # flux.  This will get rewritten on instantiation to have
                  #        a length that is the number of bands
    ra = 0.
    dec = 0.
    q = 1.        # sqrt of the axis ratio, i.e.  (b/a)^0.5
    pa = 0.       # postion angle (N of E)
    sersic = 0.   # sersic index
    rh = 0.       # half light radius

    def __init__(self, filters=['dummy'], radii=None):
        """
        :param filters:
            A list of strings giving the filternames for which you want fluxes.
            These names should correspond to values of the `filtername`
            attribute of PostageStamps, since they will be used to choose the
            appropriate element of the `flux` vector for generating model pixel
            values.

            The length of the supplied `filters` list will determine the length
            of the `flux` vector, accessible via the `nband` attribute.
        """
        assert type(filters) == list
        self.filternames = filters
        self.flux = np.zeros(len(self.filternames))
        if radii is not None:
            self.radii = radii

    @property
    def nband(self):
        """Number of elements of the flux vector (corresponding to the filters
        in `filternames`)
        """
        return len(self.filternames)

    @property
    def nparam(self):
        """Total number of source parameters, including position, shape, and
        flux(es)
        """
        return self.npos + self.nshape + self.nband

    @property
    def ngauss(self):
        """Total number of gaussians used to describe the source.
        """
        return len(self.radii)

    @property
    def use_gradients(self):
        """Which of the 7 gradients (d/dFlux, d/dRA, d/dDec, d/dq, d/dpa,
        d/dsersic, d/drh) will you actually use?
        """
        return slice(0, 1 + self.npos + self.nshape)

    @property
    def parameter_names(self):
        names = self.filternames + ["ra", "dec"] + ["q", "pa", "n", "r"][:self.nshape]
        names = ["{}_{}".format(n, self.id) for n in names]
        return names

    def filter_index(self, filtername):
        """Returns the index of the element of the `flux` array that
        corresponds to the supplied `filtername`.

        :param filtername:
            String giving the name of the filter for which you want the corresponding `flux` vector index.

        :returns index:
            An integer index that when used to subscript the `flux` attribute
            gives the source flux in `filtername`
        """
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

    def set_params(self, theta, filtername=None):
        """Set the parameters (flux(es), ra, dec) from a theta array.  Assumes
        that the order of parameters in the theta vector is [flux1,
        flux2...fluxN, ra, dec]

        :param theta:
            The source parameter values that are to be set.  Sequence of length
            either `nband + 2` (if `filtername` is `None`) or 3.

        :param filtername: (optional, default: None)
            If supplied, the theta vector is assumed to be 3-element (fluxI,
            ra, dec) where fluxI is the source flux through the filter given by
            `filtername`.  If `None` then the theta vector is assumed to be of
            length `Source().nband + 2`, where the first `nband` elements
            correspond to the fluxes.
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

    def get_param_vector(self, filtername=None):
        """Get the relevant source parameters as a simple 1-D ndarray.
        """
        if filtername is not None:
            flux = [self.flux[self.filter_index(filtername)]]
        else:
            flux = self.flux
        params = np.concatenate([flux, [self.ra], [self.dec]])
        return params

    @property
    def covariances(self):
        """This just constructs a set of source gaussian covariance matrices
        based on the radii.  For point sources these are all zeros, since a
        point source has no size.
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

    The radial profile is assumed to be a sum of equally weighted gaussians
    with radii given by SimpleGalaxy.radii
    """

    radii = np.ones(1)
    
    # Galaxies have two position parameters, 2  or 4 shape parameters (pa and q) and nband flux parameters
    npos = 2
    nshape = 2

    def set_params(self, theta, filtername=None):
        """Set the parameters (flux(es), ra, dec, q, pa) from a theta array.
        Assumes that the order of parameters in the theta vector is [flux1,
        flux2...fluxN, ra, dec, q, pa]

        :param theta:
            The source parameter values that are to be set.  Sequence of length
            either `nband + 4` (if `filtername` is `None`) or 5 (if a filter is
            specified)

        :param filtername: (optional, default: None)
            If supplied, the theta vector is assumed to be 5-element (fluxI,
            ra, dec, q, pa) where fluxI is the source flux through the filter
            given by `filtername`.  If `None` then the theta vector is assumed
            to be of length `Source().nband + 4`, where the first `nband`
            elements correspond to the fluxes.
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

    def get_param_vector(self, filtername=None):
        """Get the relevant source parameters as a simple 1-D ndarray.
        """
        if filtername is not None:
            flux = [self.flux[self.filter_index(filtername)]]
        else:
            flux = self.flux
        params = np.concatenate([flux, [self.ra], [self.dec],
                                 [self.q], [self.pa]])
        return params

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

    The amplitudes and the derivatives of the amplitudes with respect to the
    sersic index and half-light radius are based on splines.
    """

    radii = np.ones(1)
    
    # Galaxies have 2 position parameters,
    #    2 or 4 shape parameters (pa and q),
    #    and nband flux parameters
    npos = 2
    nshape = 4

    def __init__(self, filters=['dummy'], radii=None, splinedata=None, free_sersic=True):
        self.filternames = filters
        self.flux = np.zeros(len(self.filternames))
        if radii is not None:
            self.radii = radii
        if splinedata is None:
            raise ValueError, "Galaxies must have information to make A(r, n) bivariate splines"
            #self.splines = [dummy_spline] * self.ngauss
        else:
            self.initialize_splines(splinedata)

        if not free_sersic:
            # Fix the sersic parameters n_sersic and r_h
            self.nshape = 2

    def set_params(self, theta, filtername=None):
        """Set the parameters (flux(es), ra, dec, q, pa, n_sersic, r_h) from a
        theta array.  Assumes that the order of parameters in the theta vector
        is [flux1, flux2...fluxN, ra, dec, q, pa, sersic, rh]

        :param theta:
            The source parameter values that are to be set.  Sequence of length
            either `nband + npos + nshape` (if `filtername` is `None`) or `1 +
            npos + nshape` (if a filter is specified)

        :param filtername: (optional, default: None)
            If supplied, the theta vector is assumed to be 7-element (fluxI,
            ra, dec, q, pa) where fluxI is the source flux through the filter
            given by `filtername`.  If `None` then the theta vector is assumed
            to be of length `Source().nband + 6`, where the first `nband`
            elements correspond to the fluxes.
        """
        if filtername is not None:
            nflux = 1
            flux_inds = self.filter_index(filtername)
        else:
            nflux = self.nband
            flux_inds = slice(None)
        assert len(theta) == nflux + self.npos + self.nshape, "The length of the parameter vector is not appropriate for this source"
        self.flux[flux_inds] = theta[:nflux]
        self.ra  = theta[nflux]
        self.dec = theta[nflux + 1]
        self.q   = theta[nflux + 2]
        self.pa  = theta[nflux + 3]
        if self.nshape > 2:
            self.sersic = theta[nflux + 4]
            self.rh = theta[nflux + 5]

    def get_param_vector(self, filtername=None):
        """Get the relevant source parameters as a simple 1-D ndarray.
        """
        if filtername is not None:
            flux = [self.flux[self.filter_index(filtername)]]
        else:
            flux = self.flux
        params = np.concatenate([flux, [self.ra, self.dec, self.q, self.pa]])
        if self.nshape > 2:
            params = np.concatenate([params, [self.sersic, self.rh]])
        return params

    def initialize_splines(self, splinedata, spline_smoothing=None):
        """Initialize Bivariate Splines used to interpolate and get derivatives
        for gaussian amplitudes as a function of sersic and rh
        """
        with h5py.File(splinedata, "r") as data:
            n = data["nsersic"][:]
            r = data["rh"][:]
            A = data["amplitudes"][:]
            self.radii = data["radii"][:]

        nm, ng = A.shape
        self.splines = [SmoothBivariateSpline(n, r, A[:, i], s=spline_smoothing) for i in range(ng)]
        self.rh_range = (r.min(), r.max())
        self.sersic_range = (n.min(), n.max())
        
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
        return np.squeeze(np.array([spline(self.sersic, self.rh) for spline in self.splines]))

    @property
    def damplitude_dsersic(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/dsersic
        return np.array([spline(self.sersic, self.rh, dx=1) for spline in self.splines])

    @property
    def damplitude_drh(self):
        """Code here for getting amplitude derivatives from a splined look-up
        table (dependent on self.n and self.r)
        """
        # ngauss array of da/drh
        return np.array([spline(self.sersic, self.rh, dy=1) for spline in self.splines])

    
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


def dummy_spline(x, y, dx=0, dy=0):
    if (dx > 0) | (dy > 0):
        return 0.
    else:
        return 1.
