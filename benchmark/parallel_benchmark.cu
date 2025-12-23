#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "multiply.cuh"

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(REPEAT2(X))
#define REPEAT16(X) REPEAT4(REPEAT4(X))
#define REPEAT(X) REPEAT16(REPEAT16(X))

static constexpr auto n_repetitions = 256;

template <auto multiply> static void BM_multiply(benchmark::State &state) {
  auto const N = state.range(0);
  auto const a = [N] {
    auto v = std::vector<double>(N);
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<>{0.0, 1.0};
    for (auto &x : v) {
      x = dist(gen);
    }
    return v;
  }();
  auto const b = a;
  auto c = std::vector<double>(N);
  auto data = c.data();

  for (auto _ : state) {
    REPEAT(multiply(a, b, c); benchmark::DoNotOptimize(data);
           benchmark::ClobberMemory(););
  }

  state.SetItemsProcessed(state.iterations() * N * n_repetitions);
}

#define BM_MULTIPLY(MultiplyFunction)                                          \
  BENCHMARK_TEMPLATE(BM_multiply, MultiplyFunction)                            \
      ->MeasureProcessCPUTime()                                                \
      ->UseRealTime()                                                          \
      ->RangeMultiplier(2)                                                     \
      ->Range(1 << 10, 1 << 28);

BM_MULTIPLY(single_threaded_multiply);
BM_MULTIPLY(std_transform_par_multiply);
BM_MULTIPLY(cuda_multiply<256>);
