#include "multiply/multiply.hpp"

#include <gtest/gtest.h>

#include <array>
#include <random>

class MultiplyTest : public testing::Test {
 protected:
  static constexpr auto N = 1 << 10;
  std::array<double, N> a{};
  std::array<double, N> b{};
  std::array<double, N> c{};

  MultiplyTest() {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    for (auto& x : a) {
      x = dist(gen);
    }
    for (auto& x : b) {
      x = dist(gen);
    }
  }
};

TEST_F(MultiplyTest, SingleThreadedMultiplyTest) {
  single_threaded_multiply(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, StdTransformMultiplyTest) {
  std_transform_multiply(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, ParStdTransformMultiplyTest) {
  par_std_transform_multiply(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply32Test) {
  cuda_multiply<32>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply64Test) {
  cuda_multiply<64>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply128Test) {
  cuda_multiply<128>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply256Test) {
  cuda_multiply<256>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply512Test) {
  cuda_multiply<512>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

TEST_F(MultiplyTest, CudaMultiply1024Test) {
  cuda_multiply<1024>(N, a.data(), b.data(), c.data());
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
