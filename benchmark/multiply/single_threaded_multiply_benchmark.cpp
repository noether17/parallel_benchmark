#include "BM_Multiply.hpp"
#include "MultiplyBenchmarker.hpp"
#include "multiply/multiply.hpp"

BM_MULTIPLY_SET(MultiplyBenchmarker<single_threaded_multiply>);
