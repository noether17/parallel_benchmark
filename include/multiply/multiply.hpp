#pragma once

#include <cstddef>

void single_threaded_multiply(std::size_t n, double const* a, double const* b,
                              double* c);

void std_transform_multiply(std::size_t n, double const* a, double const* b,
                            double* c);
void par_std_transform_multiply(std::size_t n, double const* a, double const* b,
                                double* c);

void cuda_multiply_impl(std::size_t threads_per_block, std::size_t n,
                        double const* a, double const* b, double* c);
template <std::size_t threads_per_block>
void cuda_multiply(std::size_t n, double const* a, double const* b, double* c) {
  cuda_multiply_impl(threads_per_block, n, a, b, c);
}

void cuda_device_multiply_impl(std::size_t threads_per_block, std::size_t n,
                               double const* dev_a, double const* dev_b,
                               double* dev_c);
template <std::size_t threads_per_block>
void cuda_device_multiply(std::size_t n, double const* dev_a,
                          double const* dev_b, double* dev_c) {
  cuda_device_multiply_impl(threads_per_block, n, dev_a, dev_b, dev_c);
}
