# Script to make a psf gaussian mixture.  many things hardcoded that shouldn't be
import sys
from forcepho.mixtures import fit_jwst_psfs
from forcepho.paths import psfims

try:
    band = sys.argv[1]
except(IndexError):
    band = 'f150w'

pname = psfims + 'PSF_NIRCam_{}.fits'.format(band.upper())
start, stop = 400, 600
nmix = 6
ans = fit_jwst_psfs.fit_jwst_psf(pname, band, nmix=nmix, start=start, stop=stop, nrepeat=3)

import cPickle as pickle
with open('{}_ng{}_em_random.p'.format(band, int(nmix)), 'wb') as out:
    pickle.dump(ans, out)
