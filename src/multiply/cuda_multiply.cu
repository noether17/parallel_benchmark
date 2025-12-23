#include "multiply/multiply.hpp"

__global__ void cuda_multiply_kernel(int n, double const* a, double const* b,
                                     double* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += gridDim.x * blockDim.x) {
    c[i] = a[i] * b[i];
  }
}

constexpr int bpg(int n, int threads_per_block) {
  return (n + threads_per_block - 1) / threads_per_block;
}

void cuda_host_multiply_impl(int threads_per_block, int n, double const* a,
                             double const* b, double* c) {
  double* dev_a = nullptr;
  cudaMalloc(&dev_a, n * sizeof(double));
  cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
  double* dev_b = nullptr;
  cudaMalloc(&dev_b, n * sizeof(double));
  cudaMemcpy(dev_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
  double* dev_c = nullptr;
  cudaMalloc(&dev_c, n * sizeof(double));

  int blocks_per_grid = bpg(n, threads_per_block);
  cuda_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(n, dev_a, dev_b,
                                                               dev_c);

  cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_b);
  cudaFree(dev_a);
}

void cuda_device_multiply_impl(int threads_per_block, int n, double const* a,
                               double const* b, double* c) {
  int blocks_per_grid = bpg(n, threads_per_block);
  cuda_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(n, a, b, c);
}
