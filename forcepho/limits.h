// This file contains numerical limits that we have to hardwire to 
// ease fixed-sized allocations on the GPU

// The maximum number of bands we're allowed to use
#define MAXBANDS 30     

// The maximum number of active sources that the GPU can use
#define MAXSOURCES 30   

// The number of on-sky parameters per band that yield derivatives
#define NPARAMS 7

// The maximum square distance in a Gaussian evaluation before we no-op.
// Note that this refers to Y in exp(-0.5*Y)
#define MAX_EXP_ARG 36.0

// The number of separate accumulators in each GPU block.
// Using more will consume more memory.
// It is a requirement that NUMACCUMS divide blockDim.x evenly, yielding a
// multiple of 32.
#define NUMACCUMS 1

