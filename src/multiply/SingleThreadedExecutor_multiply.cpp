#include "ParallelExecutor/SingleThreadedExecutor.hpp"
#include "multiply/multiply.hpp"
#include "multiply/par_exec_multiply.hpp"

void SingleThreadedExecutor_multiply(std::size_t n, const double *a,
                                     const double *b, double *c) {
  static auto exe = ParODE::SingleThreadedExecutor{};
  par_exec_multiply(exe, n, a, b, c);
}
