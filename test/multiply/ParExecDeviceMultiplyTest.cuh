#pragma once

#include "ParExecMultiplyTest.hpp"
#include "ParallelExecutor/ParallelExecutor.hpp"
#include "multiply/par_exec_multiply.hpp"

template <typename ParExec>
class ParExecDeviceMultiplyTest : public ParExecMultiplyTest<ParExec> {
 protected:
  using ParExecMultiplyTest<ParExec>::N;
  using ParExecMultiplyTest<ParExec>::a;
  using ParExecMultiplyTest<ParExec>::b;
  using ParExecMultiplyTest<ParExec>::c;
  using ParExecMultiplyTest<ParExec>::par_exec;
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  ParExecDeviceMultiplyTest() {
    cudaMalloc(&dev_a, N * sizeof(double));
    cudaMalloc(&dev_b, N * sizeof(double));
    cudaMalloc(&dev_c, N * sizeof(double));
    cudaMemcpy(dev_a, a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  }

  ~ParExecDeviceMultiplyTest() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void multiply() {
    ParODE::call_kernel<multiply_kernel>(par_exec, N, dev_a, dev_b, dev_c);
    cudaMemcpy(c.data(), dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
  }
};
