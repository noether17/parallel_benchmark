#include "multiply/cuda_multiply.cuh"
#include "multiply/multiply.hpp"

void cuda_device_multiply_impl(int threads_per_block, int n,
                               double const* dev_a, double const* dev_b,
                               double* dev_c) {
  int blocks_per_grid = quotient_ceiling(n, threads_per_block);
  cuda_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(n, dev_a, dev_b,
                                                               dev_c);
}
