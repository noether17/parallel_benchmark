#include <algorithm>
#include <functional>

#include "std_transform_multiply.hpp"

void std_transform_multiply::operator()(std::size_t n, double const* a,
                                        double const* b, double* c) {
  std::transform(a, a + n, b, c, std::multiplies<double>{});
}
