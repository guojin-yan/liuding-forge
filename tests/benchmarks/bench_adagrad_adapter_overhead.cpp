// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/optim/adagrad_adapter.hpp>

#include <torch/optim/adagrad.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::int64_t kParameterCount = 4096;
constexpr std::int64_t kSteps = 2000;
constexpr std::size_t kWarmups = 2;
constexpr std::size_t kSamples = 5;

using Clock = std::chrono::steady_clock;

std::int64_t measure_native() {
  auto parameter = torch::ones(
      {kParameterCount},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto gradient = torch::full_like(parameter, 0.001);
  torch::optim::Adagrad optimizer({parameter},
                                  torch::optim::AdagradOptions(0.01)
                                      .lr_decay(0.001)
                                      .weight_decay(0.01)
                                      .initial_accumulator_value(0.1));

  const auto begin = Clock::now();
  for (std::int64_t step = 0; step < kSteps; ++step) {
    parameter.mutable_grad() = gradient;
    optimizer.step();
  }
  const auto end = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
      .count();
}

std::int64_t measure_adapter() {
  auto parameter = torch::ones(
      {kParameterCount},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto gradient = torch::full_like(parameter, 0.001);
  jyppx::liuding::optim::AdagradOptimizerAdapter optimizer(
      "optimizer.benchmark",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.01,
                    .learning_rate_decay = 0.001,
                    .weight_decay = 0.01,
                    .initial_accumulator_value = 0.1}}});

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

void print_sample(std::string_view implementation, std::size_t sample,
                  std::int64_t elapsed_nanoseconds) {
  std::cout << implementation << ',' << sample << ',' << kSteps << ','
            << kParameterCount << ',' << elapsed_nanoseconds << ','
            << std::setprecision(10)
            << static_cast<double>(elapsed_nanoseconds) / kSteps << '\n';
}

}  // namespace

int main() {
  for (std::size_t warmup = 0; warmup < kWarmups; ++warmup) {
    static_cast<void>(measure_native());
    static_cast<void>(measure_adapter());
  }

  std::vector<std::int64_t> native_samples;
  std::vector<std::int64_t> adapter_samples;
  native_samples.reserve(kSamples);
  adapter_samples.reserve(kSamples);
  std::cout
      << "implementation,sample,steps,parameters,elapsed_ns,ns_per_step\n";
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const auto native = measure_native();
    const auto adapter = measure_adapter();
    native_samples.push_back(native);
    adapter_samples.push_back(adapter);
    print_sample("native", sample, native);
    print_sample("adapter", sample, adapter);
  }

  const auto native_median = median(native_samples);
  const auto adapter_median = median(adapter_samples);
  std::cout << "summary,adapter_overhead_ratio," << std::setprecision(10)
            << adapter_median / native_median << '\n';
  return 0;
}
