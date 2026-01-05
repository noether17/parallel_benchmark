#include "ParExecMultiplyTest.hpp"
#include "multiply/ThreadPoolExecutorTemplate.hpp"

using ThreadPoolExecutorTest1 =
    ParExecMultiplyTest<ThreadPoolExecutorTemplate<1>>;
TEST_F(ThreadPoolExecutorTest1, ThreadPoolExecutorMultiplyTest) {
  multiply();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

using ThreadPoolExecutorTest4 =
    ParExecMultiplyTest<ThreadPoolExecutorTemplate<4>>;
TEST_F(ThreadPoolExecutorTest4, ThreadPoolExecutorMultiplyTest) {
  multiply();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}

using ThreadPoolExecutorTest16 =
    ParExecMultiplyTest<ThreadPoolExecutorTemplate<16>>;
TEST_F(ThreadPoolExecutorTest16, ThreadPoolExecutorMultiplyTest) {
  multiply();
  for (auto i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(a[i] * b[i], c[i]);
  }
}
