// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "test_utils/jyppx_test_helpers.hpp"

#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/training/amp/autocast.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace jyppx;
using namespace jyppx_test;
using jyppx::training::amp::AutocastGuard;
using jyppx::training::amp::GradScaler;
using jyppx::training::amp::GradScalerOptions;

std::shared_ptr<optim::SgdOptimizerAdapter> make_optimizer(
    const std::vector<Tensor>& parameters, double learning_rate = 0.1,
    std::string optimizer_id = "optimizer.amp") {
  std::vector<optim::OptimizerParameterBinding> bindings;
  bindings.reserve(parameters.size());
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    bindings.push_back({.parameter_id = "parameter." + std::to_string(index),
                        .parameter = parameters[index]});
  }
  return std::make_shared<optim::SgdOptimizerAdapter>(
      std::move(optimizer_id),
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "group.main",
           .parameters = std::move(bindings),
           .options = {.learning_rate = learning_rate}}});
}

bool apply_scaler_step(
    GradScaler& scaler,
    const std::shared_ptr<optim::OptimizerAdapter>& optimizer) {
  scaler.unscale(optimizer);
  const bool applied = scaler.step_allowed(optimizer);
  if (applied) {
    optimizer->step();
  }
  scaler.mark_step(optimizer, applied);
  return applied;
}

TEST(AutocastGuardTest, RestoresCpuStateAndRejectsUnsupportedDtype) {
  const bool previous_enabled =
      at::autocast::is_autocast_enabled(c10::DeviceType::CPU);
  const auto previous_dtype =
      at::autocast::get_autocast_dtype(c10::DeviceType::CPU);
  at::autocast::set_autocast_enabled(c10::DeviceType::CPU, false);
  at::autocast::set_autocast_dtype(c10::DeviceType::CPU, at::kBFloat16);

  {
    AutocastGuard guard(c10::DeviceType::CPU, at::kBFloat16);
    EXPECT_TRUE(at::autocast::is_autocast_enabled(c10::DeviceType::CPU));
    EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CPU),
              at::kBFloat16);
  }
  EXPECT_FALSE(at::autocast::is_autocast_enabled(c10::DeviceType::CPU));
  EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CPU),
            at::kBFloat16);
  EXPECT_THROW(AutocastGuard(c10::DeviceType::CPU, at::kHalf),
               std::invalid_argument);

  at::autocast::set_autocast_dtype(c10::DeviceType::CPU, previous_dtype);
  at::autocast::set_autocast_enabled(c10::DeviceType::CPU, previous_enabled);
}

TEST(AutocastGuardCudaTest, NestedGuardRestoresOuterDtype) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const bool previous_enabled =
      at::autocast::is_autocast_enabled(c10::DeviceType::CUDA);
  const auto previous_dtype =
      at::autocast::get_autocast_dtype(c10::DeviceType::CUDA);
  {
    AutocastGuard outer(c10::DeviceType::CUDA, at::kHalf);
    {
      AutocastGuard inner(c10::DeviceType::CUDA, at::kBFloat16);
      EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
                at::kBFloat16);
    }
    EXPECT_TRUE(at::autocast::is_autocast_enabled(c10::DeviceType::CUDA));
    EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
              at::kHalf);
  }
  EXPECT_EQ(at::autocast::is_autocast_enabled(c10::DeviceType::CUDA),
            previous_enabled);
  EXPECT_EQ(at::autocast::get_autocast_dtype(c10::DeviceType::CUDA),
            previous_dtype);
}

TEST(GradScalerTest, InitialScaleMatchesOptions) {
  GradScaler scaler;
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 65536.0);

  GradScaler custom({.init_scale = 128.0});
  EXPECT_DOUBLE_EQ(custom.get_scale(), 128.0);
}

TEST(GradScalerTest, ScaleMultipliesLoss) {
  GradScaler scaler({.init_scale = 4.0});
  auto loss = torch::tensor(1.5, torch::TensorOptions().dtype(torch::kFloat32));
  EXPECT_NEAR(scaler.scale(loss).item<double>(), 6.0, 1e-6);
}

TEST(GradScalerTest, DisabledScalerPassesThrough) {
  GradScaler scaler({.init_scale = 4.0, .enabled = false});
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);

  auto loss = torch::tensor(1.5, torch::TensorOptions().dtype(torch::kFloat32));
  EXPECT_NEAR(scaler.scale(loss).item<double>(), 1.5, 1e-6);

  // Disabled mode still owns an explicit cycle but leaves gradients unscaled.
  torch::manual_seed(1);
  auto params = make_params({{2}});
  const auto initial = params[0].detach().clone();
  auto optimizer = make_optimizer(params);
  params[0].mutable_grad() =
      torch::full({2}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  EXPECT_TRUE(torch::allclose(params[0], initial - 0.8, 1e-12, 1e-12));
  scaler.update();
}

TEST(GradScalerTest, StepUnscalesGradientsBeforeStepping) {
  torch::manual_seed(2);
  auto params = make_params({{4}});
  const auto initial = params[0].detach().clone();
  auto optimizer = make_optimizer(params);

  GradScaler scaler({.init_scale = 8.0});
  // Simulate a scaled gradient of 8.0; unscale must divide it back to 1.0.
  params[0].mutable_grad() =
      torch::full({4}, 8.0, torch::TensorOptions().dtype(torch::kFloat64));
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));

  // param = initial - lr * (8 / 8)
  const auto expected = initial - 0.1;
  EXPECT_TRUE(torch::allclose(params[0], expected, 1e-12, 1e-12));
  EXPECT_FALSE(scaler.found_inf());
  scaler.update();
}

TEST(GradScalerTest, InfGradientSkipsStepAndBacksOff) {
  torch::manual_seed(3);
  auto params = make_params({{4}});
  const auto initial = params[0].detach().clone();
  auto optimizer = make_optimizer(params);

  GradScaler scaler({.init_scale = 2.0, .backoff_factor = 0.5});
  params[0].mutable_grad() =
      torch::full({4}, std::numeric_limits<double>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat64));

  EXPECT_FALSE(apply_scaler_step(scaler, optimizer));
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_TRUE(torch::equal(params[0], initial));  // update skipped
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 2.0);      // update owns scale changes
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // 2.0 * 0.5
}

TEST(GradScalerTest, OverflowSkipsTheCompleteMultiOptimizerCycle) {
  auto finite_params = make_params({{2}});
  auto overflow_params = make_params({{2}});
  const auto finite_initial = finite_params[0].detach().clone();
  const auto overflow_initial = overflow_params[0].detach().clone();
  auto finite_optimizer =
      make_optimizer(finite_params, 0.1, "optimizer.finite");
  auto overflow_optimizer =
      make_optimizer(overflow_params, 0.1, "optimizer.overflow");

  GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});
  finite_params[0].mutable_grad() = torch::full_like(finite_params[0], 8.0);
  overflow_params[0].mutable_grad() = torch::full_like(
      overflow_params[0], std::numeric_limits<double>::infinity());

  scaler.unscale(finite_optimizer);
  scaler.unscale(overflow_optimizer);
  EXPECT_TRUE(scaler.step_allowed(finite_optimizer));
  EXPECT_FALSE(scaler.step_allowed(overflow_optimizer));
  EXPECT_FALSE(scaler.cycle_allowed());
  scaler.mark_step(finite_optimizer, false);
  scaler.mark_step(overflow_optimizer, false);
  EXPECT_TRUE(torch::equal(finite_params[0], finite_initial));
  EXPECT_TRUE(torch::equal(overflow_params[0], overflow_initial));
  EXPECT_TRUE(scaler.found_inf());
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 8.0);
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerCudaTest, MultiOptimizerOverflowSkipsAllCudaUpdates) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options =
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::vector<Tensor> finite_params{
      torch::ones({2}, options).requires_grad_(true)};
  std::vector<Tensor> overflow_params{
      torch::ones({2}, options).requires_grad_(true)};
  auto finite_optimizer =
      make_optimizer(finite_params, 0.1, "optimizer.cuda.finite");
  auto overflow_optimizer =
      make_optimizer(overflow_params, 0.1, "optimizer.cuda.overflow");

  GradScaler scaler({.init_scale = 8.0, .backoff_factor = 0.5});
  finite_params[0].mutable_grad() = torch::full_like(finite_params[0], 8.0);
  overflow_params[0].mutable_grad() = torch::full_like(
      overflow_params[0], std::numeric_limits<float>::infinity());

  scaler.unscale(finite_optimizer);
  scaler.unscale(overflow_optimizer);
  EXPECT_FALSE(scaler.cycle_allowed());
  scaler.mark_step(finite_optimizer, false);
  scaler.mark_step(overflow_optimizer, false);
  scaler.update();

  EXPECT_TRUE(torch::equal(finite_params[0].cpu(), torch::ones({2})));
  EXPECT_TRUE(torch::equal(overflow_params[0].cpu(), torch::ones({2})));
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, UnscaleBeforeClipAndStepIsIdempotent) {
  torch::manual_seed(4);
  auto params = make_params({{4}});
  const auto initial = params[0].detach().clone();
  auto optimizer = make_optimizer(params);

  GradScaler scaler({.init_scale = 4.0});
  params[0].mutable_grad() =
      torch::full({4}, 4.0, torch::TensorOptions().dtype(torch::kFloat64));

  // Trainer-style flow: explicit unscale before clipping and strategy update.
  scaler.unscale(optimizer);
  EXPECT_NEAR(params[0].grad().max().item<double>(), 1.0, 1e-12);
  EXPECT_TRUE(scaler.step_allowed(optimizer));
  optimizer->step();
  scaler.mark_step(optimizer, true);
  scaler.update();

  const auto expected = initial - 0.1;
  EXPECT_TRUE(torch::allclose(params[0], expected, 1e-12, 1e-12));
}

TEST(GradScalerTest, UpdateGrowsScaleAfterInterval) {
  GradScaler scaler(
      {.init_scale = 1.0, .growth_factor = 2.0, .growth_interval = 2});
  auto params = make_params({{1}});
  auto optimizer = make_optimizer(params);
  const auto successful_step = [&] {
    params[0].mutable_grad() = torch::ones_like(params[0]);
    EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
    scaler.update();
  };

  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // interval not reached
  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 2.0);  // growth applied

  // Successful steps keep counting; the tracker restarts after growth.
  successful_step();
  successful_step();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, UnscaleSkipsParametersWithoutGradient) {
  auto params = make_params({{2}});
  auto optimizer = make_optimizer(params);
  GradScaler scaler({.init_scale = 4.0});

  // A parameter may legitimately have no gradient in a partial model update.
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  EXPECT_FALSE(scaler.found_inf());
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 4.0);
}

TEST(GradScalerTest, GrowthSaturatesAtFiniteMaximum) {
  GradScaler scaler({.init_scale = std::numeric_limits<double>::max(),
                     .growth_factor = 2.0,
                     .growth_interval = 1});
  auto params = make_params({{1}});
  auto optimizer = make_optimizer(params);
  params[0].mutable_grad() = torch::ones_like(params[0]);

  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), std::numeric_limits<double>::max());
}

TEST(GradScalerTest, OverflowingStepsDoNotCountTowardGrowth) {
  torch::manual_seed(5);
  auto params = make_params({{2}});
  auto optimizer = make_optimizer(params);

  GradScaler scaler({.init_scale = 1.0, .growth_interval = 2});

  // One overflowing step resets the tracker.
  params[0].mutable_grad() =
      torch::full({2}, std::numeric_limits<double>::infinity(),
                  torch::TensorOptions().dtype(torch::kFloat64));
  EXPECT_FALSE(apply_scaler_step(scaler, optimizer));
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 0.5);
  EXPECT_EQ(scaler.capture_state().growth_tracker, 0U);

  // Two healthy steps reach the interval again.
  params[0].mutable_grad() = torch::ones_like(params[0]);
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  scaler.update();
  params[0].mutable_grad() = torch::ones_like(params[0]);
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  scaler.update();
  EXPECT_DOUBLE_EQ(scaler.get_scale(), 1.0);  // 0.5 * 2.0 after two clean steps
}

TEST(GradScalerTest, TypedStateRoundTrip) {
  GradScaler scaler({.init_scale = 100.0, .growth_interval = 3});
  auto params = make_params({{1}});
  auto optimizer = make_optimizer(params);
  for (int i = 0; i < 2; ++i) {
    params[0].mutable_grad() = torch::ones_like(params[0]);
    EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
    scaler.update();
  }

  GradScaler restored({.init_scale = 100.0, .growth_interval = 3});
  restored.restore_state(scaler.capture_state());
  EXPECT_DOUBLE_EQ(restored.get_scale(), scaler.get_scale());
  EXPECT_EQ(restored.capture_state().growth_tracker,
            scaler.capture_state().growth_tracker);
}

TEST(GradScalerTest, InvalidOptionsThrow) {
  EXPECT_THROW(GradScaler({.init_scale = 0.0}), InvalidArgument);
  EXPECT_THROW(GradScaler({.growth_factor = 1.0}), InvalidArgument);
  EXPECT_THROW(GradScaler({.backoff_factor = 1.0}), InvalidArgument);
  EXPECT_THROW(GradScaler({.growth_interval = 0}), InvalidArgument);
  EXPECT_THROW(
      GradScaler({.init_scale = std::numeric_limits<double>::quiet_NaN()}),
      InvalidArgument);
  EXPECT_THROW(GradScaler({.growth_factor =
                               std::numeric_limits<double>::quiet_NaN()}),
               InvalidArgument);
  EXPECT_THROW(GradScaler({.backoff_factor =
                               std::numeric_limits<double>::quiet_NaN()}),
               InvalidArgument);
  EXPECT_THROW(GradScaler({.backoff_factor = 0.0}), InvalidArgument);
  EXPECT_THROW(GradScaler({.scaler_id = ""}), InvalidArgument);
  EXPECT_THROW(GradScaler({.scaler_id = std::string{"bad\0id", 6}}),
               InvalidArgument);
}

TEST(GradScalerTest, EnforcesStepUpdateStateMachineAndValidatesState) {
  auto params = make_params({{1}});
  auto optimizer = make_optimizer(params);
  GradScaler scaler({.init_scale = 2.0});
  EXPECT_THROW(scaler.update(), NotReady);

  params[0].mutable_grad() = torch::full_like(params[0], 2.0);
  EXPECT_TRUE(apply_scaler_step(scaler, optimizer));
  EXPECT_FALSE(scaler.checkpoint_ready());
  EXPECT_THROW(scaler.capture_state(), NotReady);
  EXPECT_THROW(scaler.step_allowed(optimizer), InvalidArgument);
  EXPECT_THROW(scaler.unscale(optimizer), InvalidArgument);
  scaler.update();

  auto invalid = scaler.capture_state();
  invalid.scale = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(scaler.restore_state(invalid), SerializationError);
  invalid = scaler.capture_state();
  invalid.growth_tracker = invalid.growth_interval;
  EXPECT_THROW(scaler.restore_state(invalid), SerializationError);

  GradScaler disabled({.enabled = false});
  auto disabled_state = disabled.capture_state();
  disabled_state.scale = 2.0;
  EXPECT_THROW(disabled.restore_state(disabled_state), SerializationError);

  GradScaler incompatible({.init_scale = 4.0});
  EXPECT_THROW(incompatible.restore_state(scaler.capture_state()),
               CompatibilityError);

  auto wrong_identity = scaler.capture_state();
  wrong_identity.scaler_id = "grad_scaler.other";
  EXPECT_THROW(scaler.restore_state(wrong_identity), CompatibilityError);
}

TEST(GradScalerTest, ErrorsPreserveStableCodesAndContext) {
  try {
    GradScaler({.init_scale = 0.0});
    FAIL() << "invalid options must throw";
  } catch (const InvalidArgument& error) {
    EXPECT_EQ(error.code(), ErrorCode::invalid_argument);
    EXPECT_EQ(error.details().component, "training.amp.grad_scaler");
    EXPECT_EQ(error.details().operation, "init_scale");
    EXPECT_FALSE(error.details().expected.empty());
    EXPECT_FALSE(error.details().hint.empty());
  }

  GradScaler scaler;
  try {
    scaler.update();
    FAIL() << "an optimizer cycle is required for update";
  } catch (const NotReady& error) {
    EXPECT_EQ(error.code(), ErrorCode::not_ready);
    EXPECT_EQ(error.details().operation, "update");
  }

  auto incompatible = scaler.capture_state();
  incompatible.format_version++;
  try {
    scaler.restore_state(incompatible);
    FAIL() << "an unsupported state schema must throw";
  } catch (const CompatibilityError& error) {
    EXPECT_EQ(error.code(), ErrorCode::compatibility);
    EXPECT_EQ(error.details().operation, "format_version");
  }
}

TEST(GradScalerTest, NullOptimizerThrows) {
  GradScaler scaler;
  std::shared_ptr<optim::OptimizerAdapter> null_optimizer;
  EXPECT_THROW(scaler.unscale(null_optimizer), InvalidArgument);
  EXPECT_THROW(scaler.step_allowed(null_optimizer), InvalidArgument);
  EXPECT_THROW(scaler.mark_step(null_optimizer, false), InvalidArgument);
}

}  // namespace
