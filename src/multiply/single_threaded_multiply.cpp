#include "multiply/multiply.hpp"

void single_threaded_multiply::operator()(std::size_t n, double const* a,
                                          double const* b, double* c) {
  for (auto i = 0ul; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}
