#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>
#include <vector>

#include "Multiplier.hpp"
#include "NoOp.hpp"

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(REPEAT2(X))
#define REPEAT16(X) REPEAT4(REPEAT4(X))
#define REPEAT(X) REPEAT16(REPEAT16(X))

static constexpr auto n_repetitions = 256;

template <auto multiplier>
struct VectorMultiplyTest {
  std::vector<double> a{};
  std::vector<double> b{};
  std::vector<double> c{};

  VectorMultiplyTest(int n) : a(n), b(n), c(n) {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<>{0.0, 1.0};
    for (auto& x : a) {
      x = dist(gen);
    }
    std::ranges::copy(a, std::begin(b));
  }

  void multiply() { multiplier(a.size(), a.data(), b.data(), c.data()); }

  double* result_ptr() { return c.data(); }
};

template <typename MultTest>
static void BM_multiply(benchmark::State& state) {
  auto const N = state.range(0);
  auto test = MultTest(N);
  auto* data = test.result_ptr();

  for (auto _ : state) {
    REPEAT(test.multiply(); data = test.result_ptr();
           benchmark::DoNotOptimize(data); benchmark::ClobberMemory(););
  }

  state.SetItemsProcessed(state.iterations() * N * n_repetitions);
}

#define BM_MULTIPLY(MultiplyTest)               \
  BENCHMARK_TEMPLATE(BM_multiply, MultiplyTest) \
      ->MeasureProcessCPUTime()                 \
      ->UseRealTime()                           \
      ->RangeMultiplier(2)                      \
      ->Range(1 << 10, 1 << 28);

BM_MULTIPLY(VectorMultiplyTest<warmup_binary>);
BM_MULTIPLY(VectorMultiplyTest<no_op_binary>);
BM_MULTIPLY(VectorMultiplyTest<single_threaded_multiply>);
BM_MULTIPLY(VectorMultiplyTest<std_transform_multiply>);
BM_MULTIPLY(VectorMultiplyTest<std_transform_par_multiply>);
