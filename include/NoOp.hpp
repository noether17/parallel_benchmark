#pragma once

#include <benchmark/benchmark.h>

inline constexpr void warmup_binary(int, double const*, double const*,
                                    double const*) {}

inline constexpr void no_op_binary(int n, double const* a, double const* b,
                                   double const* c) {
  for (auto i = 0; i < n; ++i) {
    auto a_i = a[i];
    auto b_i = b[i];
    auto c_i = c[i];
    benchmark::DoNotOptimize(a_i);
    benchmark::DoNotOptimize(b_i);
    benchmark::DoNotOptimize(c_i);
    benchmark::ClobberMemory();
  }
}
