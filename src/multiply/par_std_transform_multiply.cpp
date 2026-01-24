#include <algorithm>
#include <execution>
#include <functional>

#include "par_std_transform_multiply.hpp"

void par_std_transform_multiply::operator()(std::size_t n, double const* a,
                                            double const* b, double* c) {
  std::transform(std::execution::par, a, a + n, b, c,
                 std::multiplies<double>{});
}
