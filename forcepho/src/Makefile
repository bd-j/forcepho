c++ -O3 -Wall -shared -std=c++11 \
  -undefined dynamic_lookup $(python -m pybind11 --includes) compute_gaussians_kernel.cc \
  -o compute_gaussians_kernel$(python3-config --extension-suffix)