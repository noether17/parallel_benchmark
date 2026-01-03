#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <vector>

#include "REPEAT.hpp"

template <auto multiplier>
struct MultiplyBenchmarker {
  std::vector<double> a{};
  std::vector<double> b{};
  std::vector<double> c{};

  explicit MultiplyBenchmarker(int n) : a(n), b(n), c(n) {
    for (auto i = 0; auto& x : a) {
      x = static_cast<double>(i++) / static_cast<double>(n);
    }
    std::ranges::copy(a, std::begin(b));
  }

  void multiply() { multiplier(a.size(), a.data(), b.data(), c.data()); }

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
