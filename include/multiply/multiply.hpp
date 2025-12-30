#pragma once

inline constexpr void single_threaded_multiply(int n, double const* a,
                                               double const* b, double* c) {
  for (auto i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}

void std_transform_multiply(int n, double const* a, double const* b, double* c);
void par_std_transform_multiply(int n, double const* a, double const* b,
                                double* c);

void cuda_multiply_impl(int threads_per_block, int n, double const* a,
                        double const* b, double* c);
template <int threads_per_block>
void cuda_multiply(int n, double const* a, double const* b, double* c) {
  cuda_multiply_impl(threads_per_block, n, a, b, c);
}

// void cuda_device_multiply_impl(int threads_per_block, int n, double const* a,
//                                double const* b, double* c);
// template <int threads_per_block>
// void cuda_device_multiply(int n, double const* a, double const* b, double* c)
// {
//   cuda_device_multiply_impl(threads_per_block, n, a, b, c);
// }
