import numpy as np
import matplotlib.pyplot as pl

def test_second_order(oversample=10, sigma=2):

    from proto import counts_pg_native
    from proto import scale_matrix, rotation_matrix
    var = np.diag([sigma**2, sigma**2])
    F = np.linalg.inv(var)
    
    from itertools import product
    # min and max pixel centers
    pmin, pmax = -9.0, 10.0
    d = pmax - pmin
    dx = 1
    nx = int(d/dx) + 1
    # set up the big pixel centers
    x = np.linspace(pmin, pmax, nx)
    pixels = np.array(list(product(x, x)))
    # now oversample
    dxs = 1.0 / oversample
    ns = int(d/dxs) + 1
    # set up the small pixel centers
    epsilon = dxs/2.0
    xs = np.linspace(pmin, pmax, ns) + epsilon
    subpixels = np.array(list(product(xs, xs)))

    # calculate pixel center fluxes
    im1, gr1 = counts_pg_native(pixels[:,0], pixels[:,1],
                                F[0,0], F[1,1], F[1,0], 1.0)
    im2, gr2 = counts_pg_native(pixels[:,0], pixels[:,1],
                                F[0,0], F[1,1], F[1,0], 1.0,
                                second_order=False)

    ims, grs = counts_pg_native(subpixels[:,0], subpixels[:,1],
                                F[0,0], F[1,1], F[1,0], 1.0,
                                second_order=False)

    #g = ((subpixels[:,0] <= 1.0) & (subpixels[:,0] > -0.5) &
    #     (pixels[:,1] < 0.5) & (pixels[:,1] > -0.5))

    bins = [pmin - 0.5] + (x + 0.5).tolist()
    h, bx, by = np.histogram2d(subpixels[:, 0],
                             subpixels[:, 1],
                             weights=ims, bins=bins)

    him = h*dxs**2
    xs0 = np.argmin(np.abs(xs))
    x0 = np.argmin(np.abs(x))

    fig, ax = pl.subplots()
    ax.plot(x, im1.reshape(nx, nx)[:, x0], label='Taylor expansion')
    ax.plot(x, im2.reshape(nx, nx)[:, x0], label='simple')
    ax.plot(x, him[:, x0], label='subsampling')
    ax.plot(xs, ims.reshape(ns, ns)[:, xs0], label='subpixels')
    ax.legend()
    fig, ax = pl.subplots()
    #ax.plot(xs, ims.reshape(ns, ns)[:, xs0], label='subpixels')
    ax.plot(x, im1.reshape(nx, nx)[:, x0]/him[:, x0], label='Taylor expansion/subsampling')
    ax.plot(x, im2.reshape(nx, nx)[:, x0]/him[:, x0], label='simple/subsampling')
    #ax.plot(x, him[:, x0], label='subsampling')
    ax.legend()


def test_single_central_pixel():

    from proto import counts_pg_native
    var = np.diag([sigma**2, sigma**2])
    F = np.linalg.inv(var)

    im1, gr1 = counts_pg_native(0., 0., F[0,0], F[1,1], F[1,0], 1.0)
    im2, gr1 = counts_pg_native(0., 0., F[0,0], F[1,1], F[1,0], 1.0,
                                second_order=False)
    ns = 4
    dx = 1.0/ns
    xs = np.linspace(-0.5 + dx/2., 0.5-dx/2., ns)
    subpixels = np.array(list(product(xs, xs)))
    ims, grs = counts_pg_native(subpixels[:,0], subpixels[:,1],
                                F[0,0], F[1,1], F[1,0], 1.0,
                                second_order=False)

    ims3 = ims.sum() * dx**2
