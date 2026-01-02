#include "multiply/multiply.hpp"

#include "MultiplyTest.hpp"

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
