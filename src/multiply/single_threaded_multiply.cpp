#include "multiply/multiply.hpp"

void single_threaded_multiply(int n, double const* a, double const* b, double* c) {
  for (auto i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}
