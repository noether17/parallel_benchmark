#include "BM_Multiply.hpp"
#include "MultiplyBenchmarker.hpp"
#include "multiply/multiply.hpp"

BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<32>>);
BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<64>>);
BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<128>>);
BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<256>>);
BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<512>>);
BM_MULTIPLY_SET(MultiplyBenchmarker<cuda_multiply<1024>>);
