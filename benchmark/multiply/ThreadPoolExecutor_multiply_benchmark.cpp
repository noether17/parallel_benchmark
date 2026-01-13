#include "BM_Multiply.hpp"
#include "ParExecMultiplyBenchmarker.hpp"
#include "ThreadPoolExecutorTemplate.hpp"

BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<1>>);
BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<2>>);
BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<4>>);
BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<8>>);
BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<16>>);
BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ThreadPoolExecutorTemplate<32>>);
