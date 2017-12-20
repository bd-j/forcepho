# ----------
# Script to fit a single point source in multiple (same band) simulated images.
#-----------

import sys, os
from copy import deepcopy
from functools import partial as argfix

import numpy as np
import matplotlib.pyplot as pl
import astropy.io.fits as fits
from astropy import wcs as apy_wcs

from forcepho import paths
from forcepho.data import PostageStamp
from forcepho import psf as pointspread
from forcepho.gaussmodel import Star, convert_to_gaussians, get_gaussian_gradients, compute_gaussian
from forcepho.likelihood import WorkPlan

# ------
# The setup here is that we are pulling the source parameters from the total
# scene Theta vector and setting them one at a time (with a different flux
# depepnding on the desired filter.  Ideally this would be modified so that we
# would set all the source parameters at once, including a vector for fluxes.
# However, this requires having the flux attribute of Galaxies and Stars be a
# vector, and having convert_to_gaussians and get_gaussian_gradients look for
# the appropriate element of the flux array.
# ------


class Scene(object):
    """The Scene holds the sources and provides the mapping between a giant 1-d
    array of parameters and the parameters of each source in each band/image
    """

    nfilters = 1
    filterloc = {'F090W': 0}
    # point sources
    nshape = 2
    use_gradients = slice(0, 3)

    
    def param_indices(self, sourceid, filterid):
        """Get the indices of the relevant parameters in the giant Theta vector.
        
        :returns theta:
            An array with elements [flux, (shape_params)]
        """
        start = sourceid * (self.nshape + self.nfilters)
        # get all the shape parameters
        # TODO: nshape (and use_gradients) should probably be an attribute of the source
        inds = range(start, start + self.nshape)
        # put in the flux for this source in this band
        inds.insert(0, start + self.nshape + filterid)
        return inds

    def set_source_params(self, theta, source, filterid=None):
        """Set the parameters of a source
        """
        t = np.array(theta).copy()
        if len(t) == 3:
            # Star
            t = np.append(t, np.array([1., 0., 0., 0.]))
        elif len(t) == 5:
            # Galaxy
            t = np.append(np.array(t), np.array([0., 0.]))
        else:
            print("theta vector {} not a valid length: {}".format(theta, len(theta)))
        flux, ra, dec, q, pa, sersic, rh = t
        # if allowing sources to hold the multiband fluxes you'd do this line
        # instead.  Or something even smarter since probably want to update all
        # sources and fluxes at once.
        #source.flux[filterid] = flux
        source.flux = flux
        source.ra = ra
        source.dec = dec
        source.q = q
        source.pa = pa
        source.sersic = sersic
        source.rh = rh

    def set_params(self, Theta, filterid=None):
        """Set all source parameters at once.
        """
        for source in self.sources:
            inds = self.param_indices(source.id, filterid)
            self.set_source_params(Theta[inds], source, filterid)


def lnlike_multistamp_nograd(Theta, scene=None, stamps=None):

    lnp = 0.0
    lnp_grad = np.zeros(len(Theta))
    for k, stamp in enumerate(stamps):
        # Create the workplan
        active = []
        inds = []
        for source in scene.sources:
            sourceinds = scene.param_indices(source.id, stamp.filter)
            scene.set_source_params(Theta[sourceinds], source, stamp.filter)
            gig = convert_to_gaussians(source, stamp)
            gig = get_gaussian_gradients(source, stamp, gig)
            active.append(gig)
            inds += sourceinds
        # TODO: don't reintialize the workplan every call
        wp = WorkPlan(stamp, active)
        lnp_stamp, lnp_stamp_grad = wp.lnlike(active)
        #print(k, inds, lnp_stamp, lnp_stamp_grad)
        lnp += lnp_stamp
        # TODO: test that flatten does the right thing here
        lnp_grad[inds] += lnp_stamp_grad[:, scene.use_gradients].flatten()

    return lnp


def make_stamp(imname, center=(None, None), size=(None, None),
               center_type='pixels', psfname=None, fwhm=1.0):
    """Make a postage stamp around the given position using the given image name
    """
    data = fits.getdata(imname)
    hdr = fits.getheader(imname)
    crpix = np.array([hdr['CRPIX1'], hdr['CRPIX2']])
    crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                   [hdr['CD2_1'], hdr['CD2_2']]])

    # Pull slices and transpose to get to an axis order that makes sense to me
    # and corresponds with the wcs keyword ordering
    im = data[0, :, :].T
    err = data[1, :, :].T

    # ---- Extract subarray -----
    center = np.array(center)
    # here we get the center coordinates in pixels (accounting for the transpose above)
    if center_type == 'celestial':
        world = np.append(center, 0)
        #hdr.update(NAXIS=2)
        ast = apy_wcs.WCS(hdr)
        center = ast.wcs_world2pix(world[None, :], 0)[0, :2]
    # --- here is much mystery ---
    size = np.array(size)
    lo, hi = (center - 0.5 * size).astype(int), (center + 0.5 * size).astype(int)

    xinds = slice(int(lo[0]), int(hi[0]))
    yinds = slice(int(lo[1]), int(hi[1]))
    crpix_stamp = np.floor(0.5 * size)
    crval_stamp = crpix_stamp + lo
    W = np.eye(2)
    if center_type == 'celestial':
        crval_stamp = ast.wcs_pix2world(np.append(crval_stamp, 0.)[None,:], 0)[0, :2]
        W[0, 0] = np.cos(np.deg2rad(crval_stamp[-1]))

    # --- Add image and uncertainty data to Stamp ----
    stamp = PostageStamp()
    stamp.pixel_values = im[xinds, yinds]
    stamp.ierr = 1./err[xinds, yinds]

    bad = ~np.isfinite(stamp.ierr)
    stamp.pixel_values[bad] = 0.0
    stamp.ierr[bad] = 0.0
    stamp.ierr = stamp.ierr.flatten()

    stamp.nx, stamp.ny = stamp.pixel_values.shape
    stamp.npix = stamp.nx * stamp.ny
    # note the inversion of x and y order in the meshgrid call here
    stamp.ypix, stamp.xpix = np.meshgrid(np.arange(stamp.ny), np.arange(stamp.nx))

    # --- Add WCS info to Stamp ---
    stamp.crpix = crpix_stamp
    stamp.crval = crval_stamp
    stamp.scale = np.matmul(np.linalg.inv(CD), W)
    stamp.pixcenter_in_full = center
    stamp.lo = lo
    stamp.CD = CD
    stamp.W = W

    # --- Add the PSF ---
    if psfname is not None:
        import pickle
        with open(psfname, 'rb') as pf:
            pdat = pickle.load(pf)

        oversample, center = 8, 504 - 400
        answer = pdat[6][2]
        stamp.psf = pointspread.make_psf(answer, oversample=oversample, center=center)
    else:
        stamp.psf = pointspread.PointSpreadFunction()
        stamp.psf.covaraniaces *= fwhm/2.355
    
    # --- Add extra information ---
    stamp.full_header = dict(hdr)    
    return stamp


imnames = ['sim_cube_F090W_487_001.slp.fits', 'sim_cube_F090W_487_008.slp.fits']
imnames = [os.path.join(paths.starsims, im) for im in imnames]
psfname = os.path.join(paths.psfmixture, 'f090_ng6_em_random.p')

def fit_source(ra=53.115295, dec=-27.803501, dofit=True, nlive=100):
    
    
    # --- Build the postage stamp ----
    ra_init, dec_init = ra, dec
    pos_init = (ra_init, dec_init)
    stamps = [make_stamp(im, pos_init, center_type='celestial', size=(50, 50), psfname=psfname)
              for im in imnames]

    # override the psf to reflect in both directions
    T = -1.0 * np.eye(2)
    for s in stamps:
        s.psf.covariances = np.matmul(T, np.matmul(s.psf.covariances, T.T))
        s.psf.means = np.matmul(s.psf.means, T)

    # --- get the Scene ---
    scene = Scene()
    sources = [Star()]
    sources[0].id = 0
    scene.sources = sources
    label = ['RA', 'Dec', 'Counts']

    lnlike = argfix(lnlike_multistamp_nograd, scene=scene, stamps=stamps)

    # --- Initialize ---
    theta_init = np.array([ra_init, dec_init, stamps[0].pixel_values.sum() * 1.0])
    # a rough measure of dpix/dcoordinate
    plate_scale, _ = np.linalg.eig(np.linalg.inv(stamps[0].scale))
    # make the prior ~10 pixels wide, and 1000 counts wide
    theta_width = np.array([10 * plate_scale[0], 10 * plate_scale[1], 1000])

    # --- Nested sampling ---
    ndim = 3

    def prior_transform(unit_coords):
        # convert to uniform -1 to 1
        u = (2 * unit_coords - 1.)
        # now scale and shift
        theta = theta_init + theta_width * u
        return theta

    if dofit:
        import dynesty, time
        
        # "Standard" nested sampling.
        sampler = dynesty.NestedSampler(lnlike, prior_transform, ndim, nlive=nlive, bootstrap=0)
        t0 = time.time()
        sampler.run_nested()
        dur = time.time() - t0
        results = sampler.results
        results['duration'] = dur
        indmax = results['logl'].argmax()
        theta_max = results['samples'][indmax, :]
        
        truths = np.array([53.115299, -27.803508, 4562.19])
        #print(dur)
        #print(theta_max - truths)
        #print(plate_scale)

    else:
        results = None
        theta_max = np.zeros(3)
        stamps = None

    return results, theta_max, stamps, scene

if __name__ == "__main__":


    # ---- Read the input catalog -----
    catname = os.path.join(paths.starsims, 'stars_f090w.cat')
    dt = np.dtype([(n, np.float) for n in ['ra', 'dec', 'x', 'y', 'mag', 'counts', 'flux1', 'flux2']])
    cat = np.genfromtxt(catname, usecols=np.arange(1, 9), dtype=dt)
    cat = cat[:100]

    # --- setup output ---
    fn = 'output_pointsource_dynesty.dat'
    pn = 'pointsource_resid.pdf'
    out = open(fn, 'w')
    strfmt = "{}  {:11.8f}   {:11.8f}  {:10.2f}  {:10.2f}  {:14.6f}   {} \n"
    dt2 = np.dtype([(n, np.float) for n in ['id', 'ra', 'dec', 'counts', 'sum', 'lnp', 'ncall']])
    out.write("# i    ra    dec   counts    imsum    lnp   ncall\n")
    #pdf = PdfPages(pn)

    all_results, all_pos = [], []
    for i, c in enumerate(cat[:10]):
        blob = fit_source(ra=c['ra'], dec=c['dec'], nlive=50)

        result, vals, stamps, scene = blob
        all_results.append(result)

        #from dynesty import plotting as dyplot
        #cfig, caxes = dyplot.cornerplot(result, fig=pl.subplots(3,3, figsize=(13., 10)),
        #                                labels=label, show_titles=True, title_fmt='.8f')
        #tfig, taxes = dyplot.traceplot(result, fig=pl.subplots(3, 2, figsize=(13., 13.)),
        #                               labels=label)
        #axes[0].text(0.05, 0.82,
        #             'obj={}\ncounts={:5.1f}\nx={:4.2f}, y={:4.2f}'.format(i+1, c['counts'], c['x'], c['y']),
        #             transform=axes[0].transAxes, size=10)
        #[ax.set_xlim(35, 65) for ax in axes.flat]
        #[ax.set_ylim(35, 65) for ax in axes.flat]
        #pdf.savefig(fig)
        #pl.close(fig)
        
        #counts = vals[-1]
        #center = np.array(stamp.pixcenter_in_full)
        #lo = (center - 0.5 * size).astype(int)
        #x, y = lo + vals[:2]
        #all_pos.append((x, y))

        print('----')
        print(vals)
        print(c['counts'])
        out.write(strfmt.format(i, vals[0], vals[1], vals[2], result['logl'].max(), stamps[0].pixel_values.sum(), result['ncall']))

    out.close()
    ocat = np.genfromtxt(fn, dtype=dt2)
    
