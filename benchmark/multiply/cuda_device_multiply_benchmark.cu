#include "BM_Multiply.hpp"
#include "DeviceMultiplyBenchmarker.cuh"
#include "multiply/multiply.hpp"

BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<32>>);
BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<64>>);
BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<128>>);
BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<256>>);
BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<512>>);
BM_MULTIPLY_SET(DeviceMultiplyBenchmarker<cuda_device_multiply<1024>>);
