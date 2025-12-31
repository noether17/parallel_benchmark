#include "VectorDeviceMultiplyTest.cuh"
#include "bm_device_multiply.cuh"
#include "multiply/multiply.hpp"

BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<32>>);
BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<64>>);
BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<128>>);
BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<256>>);
BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<512>>);
BM_DEVICE_MULTIPLY(VectorDeviceMultiplyTest<cuda_device_multiply<1024>>);
