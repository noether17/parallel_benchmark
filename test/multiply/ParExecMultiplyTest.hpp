#pragma once

#include <gtest/gtest.h>

#include <array>
#include <random>

#include "ParallelExecutor/ParallelExecutor.hpp"
#include "multiply/par_exec_multiply.hpp"

template <typename ParExec>
class ParExecMultiplyTest : public testing::Test {
 protected:
  static constexpr auto N = 1 << 10;
  std::array<double, N> a{};
  std::array<double, N> b{};
  std::array<double, N> c{};
  ParExec par_exec{};

  ParExecMultiplyTest() {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    for (auto& x : a) {
      x = dist(gen);
    }
    for (auto& x : b) {
      x = dist(gen);
    }
  }

  void multiply() {
    ParODE::call_kernel<multiply_kernel>(par_exec, N, a.data(), b.data(),
                                         c.data());
  }
};
