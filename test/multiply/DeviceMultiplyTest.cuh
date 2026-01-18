#pragma once

#include "MultiplyTest.hpp"

class DeviceMultiplyTest : public MultiplyTest {
 protected:
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  DeviceMultiplyTest() {
    cudaMalloc(&dev_a, N * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMalloc(&dev_c, N * sizeof(double));
    cudaMemcpy(dev_a, a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  }

  ~DeviceMultiplyTest() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  template <typename Multiplier>
  void multiply() {
    auto multiplier = Multiplier{};
    multiplier(N, dev_a, dev_b, dev_c);
    cudaMemcpy(c.data(), dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
  }
};
