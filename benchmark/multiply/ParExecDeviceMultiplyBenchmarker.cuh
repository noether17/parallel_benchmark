#pragma once

#include "ParExecMultiplyBenchmarker.hpp"
#include "ParallelExecutor/ParallelExecutor.hpp"
#include "REPEAT.hpp"
#include "multiply/par_exec_multiply.hpp"

template <typename ParExec>
struct ParExecDeviceMultiplyBenchmarker : ParExecMultiplyBenchmarker<ParExec> {
  using ParExecMultiplyBenchmarker<ParExec>::a;
  using ParExecMultiplyBenchmarker<ParExec>::b;
  using ParExecMultiplyBenchmarker<ParExec>::c;
  using ParExecMultiplyBenchmarker<ParExec>::par_exec;
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};

  explicit ParExecDeviceMultiplyBenchmarker(int n)
      : ParExecMultiplyBenchmarker<ParExec>(n) {
    cudaMalloc(&dev_a, n * sizeof(double));
    cudaMalloc(&dev_b, n * sizeof(double));
    cudaMalloc(&dev_c, n * sizeof(double));
    cudaMemcpy(dev_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  }

  ~ParExecDeviceMultiplyBenchmarker() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void multiply() {
    ParODE::call_kernel<multiply_kernel>(par_exec, a.size(), dev_a, dev_b,
                                         dev_c);
  }

  void repeat_multiply() {
    REPEAT({ multiply(); });
    cudaDeviceSynchronize();
  }

  static auto n_repeat() { return N_REPEAT; }
};
