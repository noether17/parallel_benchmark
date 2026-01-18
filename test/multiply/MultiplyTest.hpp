#pragma once

#include <gtest/gtest.h>

#include <array>
#include <random>

class MultiplyTest : public testing::Test {
 protected:
  static constexpr auto N = 1 << 10;
  std::array<double, N> a{};
  std::array<double, N> b{};
  std::array<double, N> c{};

  MultiplyTest() {
    auto gen = std::mt19937{};
    auto dist = std::uniform_real_distribution<double>{0.0, 1.0};
    for (auto& x : a) {
      x = dist(gen);
    }
    for (auto& x : b) {
      x = dist(gen);
    }
  }

  template <typename Multiplier>
  void multiply() {
    auto multiplier = Multiplier{};
    multiplier(N, a.data(), b.data(), c.data());
  }
};
