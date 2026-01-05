#include "ParExecDeviceMultiplyTest.cuh"
#include "ParallelExecutor/CudaExecutor.cuh"

using CudaExecutorTest = ParExecDeviceMultiplyTest<ParODE::CudaExecutor<256>>;
TEST_F(CudaExecutorTest, CudaExecutorMultiplyTest) {
  multiply();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
