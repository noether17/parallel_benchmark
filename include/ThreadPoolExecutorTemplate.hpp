#pragma once

#include "ParallelExecutor/ThreadPoolExecutor.hpp"

template <int num_threads>
struct ThreadPoolExecutorTemplate : ParODE::ThreadPoolExecutor {
  ThreadPoolExecutorTemplate() : ParODE::ThreadPoolExecutor(num_threads) {}
};
