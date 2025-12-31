#pragma once

#include <algorithm>
#include <vector>

template <auto multiplier>
struct VectorMultiplyTest {
  std::vector<double> a{};
  std::vector<double> b{};
  std::vector<double> c{};

  VectorMultiplyTest(int n) : a(n), b(n), c(n) {
    for (auto i = 0; auto& x : a) {
      x = static_cast<double>(i++) / static_cast<double>(n);
    }
    std::ranges::copy(a, std::begin(b));
  }

  void multiply() { multiplier(a.size(), a.data(), b.data(), c.data()); }

  double* result_ptr() { return c.data(); }
};
