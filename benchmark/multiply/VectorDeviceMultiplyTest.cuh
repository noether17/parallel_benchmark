#pragma once

#include <algorithm>
#include <vector>

template <auto multiplier>
struct VectorDeviceMultiplyTest {
  std::vector<double> a{};
  std::vector<double> b{};
  std::vector<double> c{};
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  VectorDeviceMultiplyTest(int n) : a(n), b(n), c(n) {
    for (auto i = 0; auto& x : a) {
      x = static_cast<double>(i++) / static_cast<double>(n);
    }
    std::ranges::copy(a, std::begin(b));
    cudaMalloc(&dev_a, n * sizeof(double));
    cudaMalloc(&dev_b, n * sizeof(double));
    cudaMalloc(&dev_c, n * sizeof(double));
    cudaMemcpy(dev_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  }

  ~VectorDeviceMultiplyTest() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void multiply() { multiplier(a.size(), dev_a, dev_b, dev_c); }
};
