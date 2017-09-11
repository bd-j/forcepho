Coordinate conventions
======

Because we represent images with numpy arrays, our coordinates must match accordingly.
Numpy (and PyFITS) assume that two-dimensional images are indexed by row and columns (abbreviated to either (row, col) or (r, c)), with the lowest element (0, 0) at the top-left corner.
However, this is annoying and the inverse of how, say, IDL treats images.
We therefore distinguish this from (x, y), which commonly denote standard Cartesian coordinates, where x is the horizontal coordinate, y the vertical, and the origin is on the bottom left.

The cartesian coordinate convention is what we assume in the code.
For these reasons, when reading FITS data using `pyfits.getdata`, we *always* simply transpose the resulting 2-d array.
Futhermore, when using ```matplotlib.pyplot.imshow``` we must then use
```python
imshow(image.T, origin='lower')
```
in order to get a 2D image with x increasing from left to right and y increasing from bottom to top (i.e. cartesian convention)
