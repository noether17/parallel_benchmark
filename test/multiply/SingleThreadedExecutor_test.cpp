#include "ParallelExecutor/SingleThreadedExecutor.hpp"

#include "ParExecMultiplyTest.hpp"

using SingleThreadedExecutorTest =
    ParExecMultiplyTest<ParODE::SingleThreadedExecutor>;
TEST_F(SingleThreadedExecutorTest, SingleThreadedExecutorMultiplyTest) {
  multiply();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
