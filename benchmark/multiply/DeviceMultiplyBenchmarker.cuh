#pragma once

#include "MultiplyBenchmarker.hpp"
#include "REPEAT.hpp"

template <auto multiplier>
struct DeviceMultiplyBenchmarker : MultiplyBenchmarker<multiplier> {
  using MultiplyBenchmarker<multiplier>::a;
  using MultiplyBenchmarker<multiplier>::b;
  using MultiplyBenchmarker<multiplier>::c;
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  explicit DeviceMultiplyBenchmarker(int n)
      : MultiplyBenchmarker<multiplier>(n) {
    cudaMalloc(&dev_a, n * sizeof(double));
    cudaMalloc(&dev_b, n * sizeof(double));
    cudaMalloc(&dev_c, n * sizeof(double));
    cudaMemcpy(dev_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  }

  ~DeviceMultiplyBenchmarker() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void multiply() { multiplier(a.size(), dev_a, dev_b, dev_c); }

  void repeat_multiply() {
    REPEAT({ multiply(); });
    cudaDeviceSynchronize();
  }

  static auto n_repeat() { return N_REPEAT; }
};
