// Here are the headers for the CUDA code

#ifndef HEADER_INCLUDE
#define HEADER_INCLUDE


#ifdef __CUDACC__
#define CUDA_MEMBER __host__ __device__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda.h>
#else
#define CUDA_MEMBER
#define CUDA_CALLABLE_MEMBER
#endif


#include "kernel_limits.h"
#include <stdio.h>
#include <cstdint>
#include <math.h>
#include "matrix22.cc"

typedef float PixFloat;


#endif
