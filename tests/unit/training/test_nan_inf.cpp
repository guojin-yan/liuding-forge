// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/tensor_utils.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>

#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {

using jyppx::liuding::Tensor;
using jyppx::liuding::optim::SgdOptimizerAdapter;
using jyppx::liuding::training::amp::GradScaler;
using jyppx::liuding::training::amp::GradScalerOptions;

std::shared_ptr<SgdOptimizerAdapter> make_optimizer(
    const Tensor& parameter, double learning_rate = 0.01) {
  return std::make_shared<SgdOptimizerAdapter>(
      "optimizer.nan_inf",
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "group.main",
           .parameters = {{.parameter_id = "parameter.main",
                           .parameter = parameter}},
           .options = {.learning_rate = learning_rate}}});
}

bool apply_scaler_step(
    GradScaler& scaler,
    const std::shared_ptr<jyppx::liuding::optim::OptimizerAdapter>& optimizer) {
  scaler.unscale(optimizer);
  const bool applied = scaler.step_allowed(optimizer);
  if (applied) {
    optimizer->step();
  }
  scaler.mark_step(optimizer, applied);
  return applied;
}

TEST(NaNInfTest, NaNGradientIsDetectedAndOptimizerStepIsSkipped) {
  auto parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto before = parameter.clone();
  parameter.mutable_grad() =
      torch::tensor({std::numeric_limits<double>::quiet_NaN(), 1.0},
                    torch::TensorOptions().dtype(torch::kFloat64));
  auto optimizer = make_optimizer(parameter, 0.25);
  GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});

  EXPECT_FALSE(apply_scaler_step(scaler, optimizer));
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 8.0);
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_FALSE(jyppx::liuding::is_finite(parameter.grad()));
}

TEST(NaNInfTest, InfGradientIsDetectedAndScaleBacksOff) {
  auto parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  parameter.mutable_grad() =
      torch::tensor({std::numeric_limits<double>::infinity(), -1.0},
                    torch::TensorOptions().dtype(torch::kFloat64));
  auto optimizer = make_optimizer(parameter);
  GradScaler scaler({.init_scale = 16.0, .backoff_factor = 0.25});

  EXPECT_FALSE(apply_scaler_step(scaler, optimizer));
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 16.0);
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(NaNInfTest, FiniteGradientUpdatesParameterAndGrowsAfterInterval) {
  auto parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  parameter.mutable_grad() = torch::full_like(parameter, 4.0);
  auto optimizer = make_optimizer(parameter, 0.25);
  GradScaler scaler(
      {.init_scale = 2.0, .growth_factor = 2.0, .growth_interval = 1});

  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  scaler.update();
  EXPECT_FALSE(scaler.found_inf());
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
  EXPECT_TRUE(jyppx::liuding::is_finite(parameter));
  EXPECT_TRUE(
      torch::allclose(parameter, torch::full({2}, 0.5, parameter.options())));
}

}  // namespace
