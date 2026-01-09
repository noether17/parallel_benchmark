#include "multiply/cuda_multiply.cuh"
#include "multiply/multiply.hpp"

void cuda_multiply_impl(std::size_t threads_per_block, std::size_t n,
                        double const* a, double const* b, double* c) {
  double* dev_a = nullptr;
  cudaMalloc(&dev_a, n * sizeof(double));
  cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
  double* dev_b = nullptr;
  cudaMalloc(&dev_b, n * sizeof(double));
  cudaMemcpy(dev_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
  double* dev_c = nullptr;
  cudaMalloc(&dev_c, n * sizeof(double));

  std::size_t blocks_per_grid = quotient_ceiling(n, threads_per_block);
  cuda_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(n, dev_a, dev_b,
                                                               dev_c);

  cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_b);
  cudaFree(dev_a);
}
