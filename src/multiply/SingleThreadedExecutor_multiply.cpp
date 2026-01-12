#include "ParallelExecutor/SingleThreadedExecutor.hpp"
#include "multiply/par_exec_multiply.hpp"

void SingleThreadedExecutor_multiply(ParODE::SingleThreadedExecutor &exe,
                                     std::size_t n, const double *a,
                                     const double *b, double *c) {
  par_exec_multiply(exe, n, a, b, c);
}
