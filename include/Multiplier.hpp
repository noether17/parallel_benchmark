#pragma once

#include <algorithm>
#include <execution>
#include <functional>

inline constexpr void single_threaded_multiply(int n, double const* a,
                                               double const* b, double* c) {
  for (auto i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}

inline constexpr void std_transform_multiply(int n, double const* a,
                                             double const* b, double* c) {
  std::transform(a, a + n, b, c, std::multiplies<double>{});
}

inline void std_transform_par_multiply(int n, double const* a, double const* b,
                                       double* c) {
  std::transform(std::execution::par, a, a + n, b, c,
                 std::multiplies<double>{});
}
