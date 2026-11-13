// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace optim = jyppx::liuding::optim;
using jyppx::liuding::Tensor;
using jyppx::liuding::training::amp::GradScaler;

std::shared_ptr<optim::OptimizerAdapter> make_optimizer(
    std::string id, const Tensor& parameter) {
  return std::make_shared<optim::SgdOptimizerAdapter>(
      std::move(id),
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "group.main",
           .parameters = {{.parameter_id = "parameter.main",
                           .parameter = parameter}},
           .options = {.learning_rate = 0.125}}});
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void single_cycle(bool overflow) {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto optimizer = make_optimizer("optimizer.single", parameter);
  GradScaler scaler({.init_scale = 8.0, .growth_interval = 1});
  auto compute_loss = [&] {
    return overflow
               ? (parameter * std::numeric_limits<double>::infinity()).sum()
               : parameter.square().sum();
  };

  optimizer->zero_grad();
  auto loss = compute_loss();
  auto scaled = scaler.scale(loss);
  scaled.backward();
  scaler.unscale(optimizer);
  const bool applied = scaler.step_allowed(optimizer);
  if (applied) {
    optimizer->step();
  }
  scaler.mark_step(optimizer, applied);
  scaler.update();

  require(applied == !overflow, "Unexpected single-optimizer decision");
  require(parameter.item<double>() == (overflow ? 1.0 : 0.75),
          "Unexpected single-optimizer parameter");
  require(optimizer->completed_steps() == (overflow ? 0U : 1U),
          "Unexpected single-optimizer update count");
  require(scaler.get_scale() == (overflow ? 4.0 : 16.0) &&
              scaler.found_inf() == overflow && scaler.checkpoint_ready(),
          "Incomplete single-optimizer boundary");
  std::cout << "mode=" << (overflow ? "overflow" : "single")
            << " parameter=" << parameter.item<double>()
            << " applied=" << applied << " scale=" << scaler.get_scale()
            << " steps=" << optimizer->completed_steps() << '\n';
}

void group_cycle(bool overflow) {
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true);
  auto encoder = torch::ones({1}, options);
  auto head = torch::full({1}, 2.0, options);
  auto encoder_optimizer = make_optimizer("optimizer.encoder", encoder);
  auto head_optimizer = make_optimizer("optimizer.head", head);
  GradScaler scaler({.init_scale = 8.0, .growth_interval = 1});
  auto compute_loss = [&] {
    auto head_loss = overflow
                         ? (head * std::numeric_limits<double>::infinity()).sum()
                         : head.square().sum();
    return encoder.square().sum() + head_loss;
  };

  encoder_optimizer->zero_grad();
  head_optimizer->zero_grad();
  auto loss = compute_loss();
  auto scaled = scaler.scale(loss);
  scaled.backward();
  scaler.unscale(encoder_optimizer);
  scaler.unscale(head_optimizer);
  const bool apply_group = scaler.cycle_allowed();
  if (apply_group) {
    encoder_optimizer->step();
    head_optimizer->step();
  }
  scaler.mark_step(encoder_optimizer, apply_group);
  scaler.mark_step(head_optimizer, apply_group);
  scaler.update();

  require(apply_group == !overflow, "Unexpected optimizer-group decision");
  require(encoder.item<double>() == (overflow ? 1.0 : 0.75) &&
              head.item<double>() == (overflow ? 2.0 : 1.5),
          "Optimizer group updated partially or used scaled gradients");
  require(encoder_optimizer->completed_steps() == (overflow ? 0U : 1U) &&
              head_optimizer->completed_steps() == (overflow ? 0U : 1U),
          "Unexpected optimizer-group update counts");
  require(scaler.get_scale() == (overflow ? 4.0 : 16.0) &&
              scaler.found_inf() == overflow && scaler.checkpoint_ready(),
          "Incomplete optimizer-group boundary");
  std::cout << "mode=" << (overflow ? "group-overflow" : "group")
            << " encoder=" << encoder.item<double>()
            << " head=" << head.item<double>() << " applied=" << apply_group
            << " scale=" << scaler.get_scale()
            << " encoder_steps=" << encoder_optimizer->completed_steps()
            << " head_steps=" << head_optimizer->completed_steps() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const std::string_view mode = argc == 2 ? argv[1] : "";
    if (mode == "single" || mode == "overflow") {
      single_cycle(mode == "overflow");
    } else if (mode == "group" || mode == "group-overflow") {
      group_cycle(mode == "group-overflow");
    } else {
      throw std::invalid_argument(
          "Usage: liuding-forge-example-grad-scaler "
          "single|overflow|group|group-overflow");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
