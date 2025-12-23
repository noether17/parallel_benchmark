#include "NoOp.hpp"
#include "VectorMultiplyTest.hpp"
#include "bm_multiply.hpp"

BM_MULTIPLY(VectorMultiplyTest<no_op_binary>);
