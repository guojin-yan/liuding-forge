// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/nn/loss/contrastive_loss.hpp>
#include <jyppx/nn/loss/dice_loss.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::int64_t kSteps = 200;
constexpr std::size_t kWarmups = 2;
constexpr std::size_t kSamples = 5;
using Clock = std::chrono::steady_clock;

template <typename Loss>
std::int64_t measure(torch::Tensor input, Loss&& loss) {
  const auto begin = Clock::now();
  for (std::int64_t step = 0; step < kSteps; ++step) {
    if (input.grad().defined()) {
      input.mutable_grad().zero_();
    }
    loss(input).backward();
  }
  const auto end = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
      .count();
}

std::int64_t measure_label_smoothing() {
  torch::manual_seed(20261021);
  auto logits = torch::randn(
      {64, 32},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto target = torch::randint(0, 32, {64}, torch::kLong);
  return measure(logits, [&](const torch::Tensor& value) {
    return jyppx::nn::functional::label_smoothing_cross_entropy(value, target,
                                                                0.1, "mean");
  });
}

std::int64_t measure_focal() {
  torch::manual_seed(20261022);
  auto logits = torch::randn(
      {64, 32},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto target = torch::randint(0, 2, {64, 32}, torch::kFloat32);
  return measure(logits, [&](const torch::Tensor& value) {
    return jyppx::nn::functional::focal_loss(value, target, 0.25, 2.0, "mean");
  });
}

std::int64_t measure_dice() {
  torch::manual_seed(20261023);
  auto logits = torch::randn(
      {16, 4, 32, 32},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto target = torch::randint(0, 2, {16, 4, 32, 32}, torch::kFloat32);
  jyppx::nn::DiceLossOptions options;
  options.mode = jyppx::nn::DiceMode::multilabel;
  options.average = jyppx::nn::DiceAverage::macro;
  return measure(logits, [&](const torch::Tensor& value) {
    return jyppx::nn::functional::dice_loss(value, target, options);
  });
}

std::int64_t measure_contrastive() {
  torch::manual_seed(20261024);
  auto x1 = torch::randn(
      {128, 64},
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  const auto x2 = torch::randn({128, 64}, torch::kFloat32);
  const auto target = torch::randint(0, 2, {128}, torch::kLong);
  return measure(x1, [&](const torch::Tensor& value) {
    return jyppx::nn::functional::contrastive_loss(value, x2, target, 1.0,
                                                   "mean");
  });
}

template <typename Measure>
void run_case(std::string_view name, Measure&& measure_case) {
  for (std::size_t warmup = 0; warmup < kWarmups; ++warmup) {
    static_cast<void>(measure_case());
  }
  std::vector<std::int64_t> samples;
  samples.reserve(kSamples);
  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const auto elapsed = measure_case();
    samples.push_back(elapsed);
    std::cout << name << ',' << sample << ',' << kSteps << ',' << elapsed << ','
              << std::setprecision(12) << static_cast<double>(elapsed) / kSteps
              << '\n';
  }
  std::sort(samples.begin(), samples.end());
  std::cout << "summary," << name << ",median_ns_per_step,"
            << std::setprecision(12)
            << static_cast<double>(samples[samples.size() / 2]) / kSteps
            << '\n';
}

}  // namespace

int main() {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  std::cout << "loss,sample,steps,elapsed_ns,ns_per_step\n";
  run_case("label_smoothing", measure_label_smoothing);
  run_case("focal_binary", measure_focal);
  run_case("dice_multilabel", measure_dice);
  run_case("contrastive", measure_contrastive);
  return 0;
}
