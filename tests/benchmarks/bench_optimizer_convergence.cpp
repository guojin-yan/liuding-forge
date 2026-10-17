// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Optimizer convergence benchmark. Output is CSV for comparison with an
// equivalent PyTorch script using the same seed and generated tensors.

#include <jyppx/optim/adam.hpp>
#include <jyppx/optim/adamw.hpp>
#include <jyppx/optim/sgd_adapter.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {

template <typename OptimizerT, typename OptionsT>
void benchmark_legacy(const std::string& name, const OptionsT& options) {
  torch::manual_seed(20261012);
  auto first =
      torch::randn({32, 16}, torch::TensorOptions().requires_grad(true));
  auto second =
      torch::randn({16, 1}, torch::TensorOptions().requires_grad(true));
  const auto inputs = torch::randn({128, 32});
  const auto targets = torch::randn({128, 1});
  auto optimizer =
      std::make_shared<OptimizerT>(jyppx::TensorList{first, second}, options);

  std::cout << name << ",step,loss\n";
  for (int step = 0; step < 100; ++step) {
    optimizer->zero_grad();
    const auto prediction = inputs.matmul(first).matmul(second);
    const auto loss = torch::mse_loss(prediction, targets);
    loss.backward();
    optimizer->step();
    std::cout << name << ',' << step << ',' << std::setprecision(10)
              << loss.item<double>() << '\n';
  }
}

void benchmark_sgd() {
  torch::manual_seed(20261012);
  auto first =
      torch::randn({32, 16}, torch::TensorOptions().requires_grad(true));
  auto second =
      torch::randn({16, 1}, torch::TensorOptions().requires_grad(true));
  const auto inputs = torch::randn({128, 32});
  const auto targets = torch::randn({128, 1});
  jyppx::optim::SgdOptimizerAdapter optimizer(
      "optimizer.benchmark",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first},
                       {.parameter_id = "second", .parameter = second}},
        .options = {.learning_rate = 1e-2}}});

  std::cout << "SGD,step,loss\n";
  for (int step = 0; step < 100; ++step) {
    optimizer.zero_grad();
    const auto prediction = inputs.matmul(first).matmul(second);
    const auto loss = torch::mse_loss(prediction, targets);
    loss.backward();
    optimizer.step();
    std::cout << "SGD," << step << ',' << std::setprecision(10)
              << loss.item<double>() << '\n';
  }
}

}  // namespace

int main() {
  benchmark_sgd();
  benchmark_legacy<jyppx::optim::Adam>("Adam",
                                       jyppx::optim::AdamOptions{.lr = 1e-3});
  benchmark_legacy<jyppx::optim::AdamW>("AdamW",
                                        jyppx::optim::AdamWOptions{.lr = 1e-3});
}
