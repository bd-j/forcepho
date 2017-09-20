import numpy as np
import matplotlib.pyplot as pl


from forcepho.gaussmodel import Star
from demo_utils import Scene, make_stamp, negative_lnlike_stamp, negative_lnlike_nograd, make_image



if __name__ == "__main__":
    psfname = '/Users/bjohnson/Codes/image/forcepho/data/psf_mixtures/f090_ng6_em_random.p'
    size = (100, 100)
        
    stamp = make_stamp(size, psfname=psfname)
    # make the pixels tiny
    #stamp.distortion = np.eye(2) * 8.0
    oversample = 8
    stamp.psf.covariances *= oversample**2
    #stamp.psf.covariances[:, 0, 0] /= oversample
    #stamp.psf.covariances[:, 1, 1] /= oversample
    stamp.psf.means *= oversample
    stamp.crpix = np.array([stamp.nx/2., stamp.ny/2.])
    stamp.crval = np.zeros(2)

    T = -1.0 * np.eye(2)
    stamp.psf.covariances = np.matmul(T, np.matmul(stamp.psf.covariances, T.T))
    stamp.psf.means = np.matmul(stamp.psf.means, T)

    # --- get the Scene ---
    scene = Scene()
    sources = [Star()]
    scene.sources = sources

    # --- Get the mock image ----
    label = ['flux', 'x', 'y']
    theta = np.array([100., 0.0, 0.0])
    ptrue = theta
    stamp.pixel_values = make_image(ptrue, scene, stamp)[0]

    fig, ax = pl.subplots()
    ax.imshow(stamp.pixel_values.T, origin='lower')
    
    pl.show()
