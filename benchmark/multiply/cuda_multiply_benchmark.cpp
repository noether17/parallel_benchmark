#include "VectorMultiplyTest.hpp"
#include "bm_multiply.hpp"
#include "multiply/multiply.hpp"

BM_MULTIPLY(VectorMultiplyTest<cuda_multiply<256>>);
