#include "multiply/cuda_multiply.cuh"
#include "multiply/multiply.hpp"

void cuda_device_multiply_impl(std::size_t threads_per_block, std::size_t n,
                               double const* dev_a, double const* dev_b,
                               double* dev_c) {
  std::size_t blocks_per_grid = quotient_ceiling(n, threads_per_block);
  cuda_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(n, dev_a, dev_b,
                                                               dev_c);
}
