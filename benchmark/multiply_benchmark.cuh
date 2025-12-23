#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(REPEAT2(X))
#define REPEAT16(X) REPEAT4(REPEAT4(X))
#define REPEAT(X) REPEAT16(REPEAT16(X))

static constexpr auto n_repetitions = 256;

template <auto multiplier>
struct CudaMultiplyTest {
  double* dev_a{};
  double* dev_b{};
  double* dev_c{};
  int size{};

  CudaMultiplyTest(int n) : size{n} {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<>{0.0, 1.0};
    auto host_vec = std::vector<double>(n);
    for (auto& x : host_vec) {
      x = dist(gen);
    }

    cudaMalloc(&dev_a, n * sizeof(double));
    cudaMemcpy(dev_a, host_vec.data(), n * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMalloc(&dev_b, n * sizeof(double));
    cudaMemcpy(dev_b, host_vec.data(), n * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMalloc(&dev_c, n * sizeof(double));
  }

  ~CudaMultiplyTest() {
    cudaFree(dev_c);
    cudaFree(dev_b);
    cudaFree(dev_a);
  }

  void multiply() { multiplier(size, dev_a, dev_b, dev_c); }

  double* result_ptr() { return dev_c; }
};

template <typename MultTest>
static void BM_multiply(benchmark::State& state) {
  auto const N = state.range(0);
  auto test = MultTest(N);
  auto* data = test.result_ptr();

  for (auto _ : state) {
    REPEAT(test.multiply(), data = test.result_ptr();
           benchmark::DoNotOptimize(data); benchmark::ClobberMemory(););
  }

  state.SetItemsProcessed(state.iterations() * N * n_repetitions);
}
