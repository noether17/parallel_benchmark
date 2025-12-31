#include <gtest/gtest.h>

#include <array>
#include <random>

#include "multiply/multiply.hpp"

class CudaMultiplyTest : public testing::Test {
 protected:
  static constexpr auto N = 1 << 10;
  std::array<double, N> a{};
  std::array<double, N> b{};
  std::array<double, N> c{};
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  CudaMultiplyTest() {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    for (auto& x : a) {
      x = dist(gen);
    }
    for (auto& x : b) {
      x = dist(gen);
    }
    cudaMalloc(&dev_a, N * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMalloc(&dev_c, N * sizeof(double));
  }

  ~CudaMultiplyTest() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void copy_input_to_device() {
    cudaMemcpy(dev_a, a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  }

  void copy_output_to_host() {
    cudaMemcpy(c.data(), dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
  }
};

TEST_F(CudaMultiplyTest, CudaDeviceMultiply32Test) {
  copy_input_to_device();
  cuda_device_multiply<32>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(CudaMultiplyTest, CudaDeviceMultiply64Test) {
  copy_input_to_device();
  cuda_device_multiply<64>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(CudaMultiplyTest, CudaDeviceMultiply128Test) {
  copy_input_to_device();
  cuda_device_multiply<128>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(CudaMultiplyTest, CudaDeviceMultiply256Test) {
  copy_input_to_device();
  cuda_device_multiply<256>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(CudaMultiplyTest, CudaDeviceMultiply512Test) {
  copy_input_to_device();
  cuda_device_multiply<512>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(CudaMultiplyTest, CudaDeviceMultiply1024Test) {
  copy_input_to_device();
  cuda_device_multiply<1024>(N, dev_a, dev_b, dev_c);
  copy_output_to_host();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
