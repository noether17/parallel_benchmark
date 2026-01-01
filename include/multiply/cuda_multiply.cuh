#pragma once

#include <concepts>

/* Provides the ceiling of the non-truncated quotient of two integers, i.e., the
 * result of dividend / divisor if the remainder is zero or dividend / divisor +
 * 1 otherwise. Assumes positive inputs. Intended to be used to compute blocks
 * per grid given total grid size and threads per block.
 */
inline constexpr int quotient_ceiling(int dividend, int divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <std::floating_point T>
__global__ void cuda_multiply_kernel(int n, T const* dev_a, T const* dev_b,
                                     T* dev_c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (; i < n; i += gridDim.x * blockDim.x) {
    dev_c[i] = dev_a[i] * dev_b[i];
  }
}
