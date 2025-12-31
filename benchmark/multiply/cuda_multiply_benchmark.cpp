#include "VectorMultiplyTest.hpp"
#include "bm_multiply.hpp"
#include "multiply/multiply.hpp"

BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<32>>);
BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<64>>);
BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<128>>);
BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<256>>);
BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<512>>);
BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<1024>>);
