#pragma once

#include "ParD/ParallelDoer.hpp"

inline constexpr void multiply_kernel(std::size_t i, double const* a,
                                      double const* b, double* c) {
  c[i] = a[i] * b[i];
}
template <typename ParallelExecutor>
void par_exec_multiply(ParallelExecutor& par_exec, std::size_t n,
                       double const* a, double const* b, double* c) {
  ParD::call_kernel<multiply_kernel>(par_exec, n, a, b, c);
}
