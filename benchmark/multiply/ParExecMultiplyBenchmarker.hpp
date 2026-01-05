#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <vector>

#include "ParallelExecutor/ParallelExecutor.hpp"
#include "REPEAT.hpp"
#include "multiply/par_exec_multiply.hpp"

template <typename ParExec>
struct ParExecMultiplyBenchmarker {
  std::vector<double> a{};
  std::vector<double> b{};
  std::vector<double> c{};
  ParExec par_exec{};

  explicit ParExecMultiplyBenchmarker(int n) : a(n), b(n), c(n) {
    for (auto i = 0; auto& x : a) {
      x = static_cast<double>(i++) / static_cast<double>(n);
    }
    std::ranges::copy(a, std::begin(b));
  }

  void multiply() {
    ParODE::call_kernel<multiply_kernel>(par_exec, a.size(), a.data(), b.data(),
                                         c.data());
  }

  void repeat_multiply() {
    REPEAT({
      auto result_ptr = c.data();
      multiply();
      benchmark::DoNotOptimize(result_ptr);
      benchmark::ClobberMemory();
    });
  }

  static auto n_repeat() { return N_REPEAT; }
};
