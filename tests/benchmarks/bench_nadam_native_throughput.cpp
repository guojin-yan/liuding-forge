// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/optim/nadam_adapter.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::int64_t kParameterCount = 4096;
constexpr std::int64_t kSteps = 2000;
constexpr std::size_t kWarmups = 2;
constexpr std::size_t kSamples = 5;

using Clock = std::chrono::steady_clock;

std::int64_t measure_native_adapter() {
  auto parameter = torch::ones(
      {kParameterCount},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto gradient = torch::full_like(parameter, 0.001);
  jyppx::optim::NAdamOptimizerAdapter optimizer(
      "optimizer.benchmark",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.002}}});

  const auto begin = Clock::now();
  for (std::int64_t step = 0; step < kSteps; ++step) {
    parameter.mutable_grad() = gradient;
    optimizer.step();
  }
  const auto end = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
      .count();
}

double median(std::vector<std::int64_t> values) {
  std::sort(values.begin(), values.end());
  return static_cast<double>(values[values.size() / 2]);
}

}  // namespace

int main() {
  for (std::size_t warmup = 0; warmup < kWarmups; ++warmup) {
    static_cast<void>(measure_native_adapter());
  }

  std::vector<std::int64_t> samples;
  samples.reserve(kSamples);
  std::cout
      << "implementation,sample,steps,parameters,elapsed_ns,ns_per_step\n";
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const auto elapsed = measure_native_adapter();
    samples.push_back(elapsed);
    std::cout << "jyppx_native," << sample << ',' << kSteps << ','
              << kParameterCount << ',' << elapsed << ','
              << std::setprecision(10) << static_cast<double>(elapsed) / kSteps
              << '\n';
  }

  std::cout << "summary,median_ns_per_step," << std::setprecision(10)
            << median(samples) / kSteps << '\n';
  return 0;
}
