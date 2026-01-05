#include "BM_Multiply.hpp"
#include "ParExecMultiplyBenchmarker.hpp"
#include "ParallelExecutor/SingleThreadedExecutor.hpp"

BM_MULTIPLY_SET(ParExecMultiplyBenchmarker<ParODE::SingleThreadedExecutor>);
