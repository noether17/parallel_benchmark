#pragma once

#include <benchmark/benchmark.h>

template <typename Benchmarker>
static void BM_Multiply(benchmark::State& state) {
  auto const N = state.range(0);
  auto bm = Benchmarker(N);

  for (auto _ : state) {
    bm.repeat_multiply();
  }

  state.SetItemsProcessed(state.iterations() * N * bm.n_repeat());
}

#define BM_MULTIPLY_SET(Benchmarker)           \
  BENCHMARK_TEMPLATE(BM_Multiply, Benchmarker) \
      ->MeasureProcessCPUTime()                \
      ->UseRealTime()                          \
      ->RangeMultiplier(2)                     \
      ->Range(1 << 5, 1 << 10);
