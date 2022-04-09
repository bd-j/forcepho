# Demos

Demonstrations of some basic capabilities.  Each directory contains scripts to
make mock images using GalSim and to fit them with forcepho.  Code is also
included for plotting some results. Each demo has a readme that describes the
contents and the aim, and how to run the scripts. It would be nice to show

- [x] `demo_basic/` - Basic: Trace, residuals, and corner plot for a single
   source in a single band.  Show posterior PDFs as a function of S/N

- [x] `demo_pair/` - Pairs: The covariances between adjacent sources as a
    function of the ratio of distance to rhalf and the flux ratios.

- [x] `demo_color/` - Colors: The difference in posteriors between fitting
    multiple (different resolution) bands separately versus simultaneously.

- [x] `demo_mosaic/` - Mosaics: The impact of pixel covariances by fitting at
    the exposure level and then at the mosaic level.

- [ ] `demo_scene/` - Demonstrating the use of superscenes to sample smaller
    scenes in a Gibbs-like fashion.

- [ ] Bulge/Disk - fit two sources with a common center but different profiles
  to a composite object.

- [ ] Biases due to color gradients, or unmodeled substructure.
