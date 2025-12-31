#pragma once

#include <benchmark/benchmark.h>

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(REPEAT2(X))
#define REPEAT16(X) REPEAT4(REPEAT4(X))
#define REPEAT(X) REPEAT16(REPEAT16(X))

inline constexpr auto n_repetitions = 256;

template <typename MultTest>
static void BM_device_multiply(benchmark::State& state) {
  auto const N = state.range(0);
  auto test = MultTest(N);

  for (auto _ : state) {
    REPEAT(test.multiply(););
    cudaDeviceSynchronize();
  }

  state.SetItemsProcessed(state.iterations() * N * n_repetitions);
}

#define BM_DEVICE_MULTIPLY(MultiplyTest)               \
  BENCHMARK_TEMPLATE(BM_device_multiply, MultiplyTest) \
      ->MeasureProcessCPUTime()                        \
      ->UseRealTime()                                  \
      ->RangeMultiplier(2)                             \
      ->Range(1 << 10, 1 << 20);
