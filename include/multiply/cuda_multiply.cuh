#pragma once

#include <concepts>

/* Provides the ceiling of the non-truncated quotient of two integers, i.e., the
 * result of dividend / divisor if the remainder is zero or dividend / divisor +
 * 1 otherwise. Assumes positive inputs. Intended to be used to compute blocks
 * per grid given total grid size and threads per block.
 */
inline constexpr std::size_t quotient_ceiling(std::size_t dividend,
                                              std::size_t divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <std::floating_point T>
__global__ void cuda_multiply_kernel(std::size_t n, T const* dev_a,
                                     T const* dev_b, T* dev_c) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += gridDim.x * blockDim.x) {
    dev_c[i] = dev_a[i] * dev_b[i];
  }
}

template <typename CudaMultiplier>
void cuda_multiply_host_data(CudaMultiplier& multiplier, std::size_t n,
                             double const* a, double const* b, double* c) {
  auto dev_a = static_cast<double*>(nullptr);
  auto dev_b = static_cast<double*>(nullptr);
  auto dev_c = static_cast<double*>(nullptr);
  cudaMalloc(&dev_a, n * sizeof(double));
  cudaMalloc(&dev_b, n * sizeof(double));
  cudaMalloc(&dev_c, n * sizeof(double));
  cudaMemcpy(dev_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

  multiplier(n, dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_b);
  cudaFree(dev_a);
}
