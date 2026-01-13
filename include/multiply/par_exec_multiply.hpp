#pragma once

#include "ParallelExecutor/ParallelExecutor.hpp"

inline constexpr void multiply_kernel(int i, double const* a, double const* b,
                                      double* c) {
  c[i] = a[i] * b[i];
}
template <typename ParallelExecutor>
void par_exec_multiply(ParallelExecutor& par_exec, int n, double const* a,
                       double const* b, double* c) {
  ParODE::call_kernel<multiply_kernel>(par_exec, n, a, b, c);
}
