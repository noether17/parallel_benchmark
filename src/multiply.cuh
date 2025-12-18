#include <execution>
#include <functional>
#include <span>

inline void single_threaded_multiply(std::span<double const> a,
                                     std::span<double const> b,
                                     std::span<double> c) {
  for (auto i = 0; i < std::ssize(a); ++i) {
    c[i] = a[i] * b[i];
  }
}

inline void std_transform_par_multiply(std::span<double const> a,
                                       std::span<double const> b,
                                       std::span<double> c) {
  std::transform(std::execution::par, a.begin(), a.end(), b.begin(), c.begin(),
                 std::multiplies<double>{});
}

__global__ void cuda_multiply_kernel(unsigned n, double const *a,
                                     double const *b, double *c) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n) {
    c[i] = a[i] * b[i];
    i += gridDim.x * blockDim.x;
  }
}

template <int tpb>
inline void cuda_multiply(std::span<double const> a, std::span<double const> b,
                          std::span<double> c) {
  auto const n = a.size();
  double *dev_a = nullptr;
  cudaMalloc(&dev_a, n * sizeof(double));
  cudaMemcpy(dev_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  double *dev_b = nullptr;
  cudaMalloc(&dev_b, n * sizeof(double));
  cudaMemcpy(dev_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  double *dev_c = nullptr;
  cudaMalloc(&dev_c, n * sizeof(double));

  auto const bpg = (n + tpb - 1) / tpb;
  cuda_multiply_kernel<<<bpg, tpb>>>(n, dev_a, dev_b, dev_c);

  cudaMemcpy(c.data(), dev_c, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_c);
  cudaFree(dev_b);
  cudaFree(dev_a);
}
