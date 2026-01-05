#include "BM_Multiply.hpp"
#include "ParExecDeviceMultiplyBenchmarker.cuh"
#include "ParallelExecutor/CudaExecutor.cuh"

BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<32>>);
BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<64>>);
BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<128>>);
BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<256>>);
BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<512>>);
BM_MULTIPLY_SET(ParExecDeviceMultiplyBenchmarker<ParODE::CudaExecutor<1024>>);
