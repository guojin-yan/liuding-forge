// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/optim/adadelta_adapter.hpp>
#include <jyppx/liuding/optim/adagrad_adapter.hpp>
#include <jyppx/liuding/optim/adam_adapter.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/lamb_adapter.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/nadam_adapter.hpp>
#include <jyppx/liuding/optim/rmsprop_adapter.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace lf = jyppx::liuding;
namespace opt = lf::optim;
namespace amp = lf::training::amp;

class MemoryStore final : public lf::StateWriter, public lf::StateReader {
 public:
  explicit MemoryStore(lf::StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs_[std::string(name)] = {bytes.begin(), bytes.end()};
  }
  const lf::StateManifest& manifest() const noexcept override {
    return manifest_;
  }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& bytes = blobs_.at(std::string(name));
    if (bytes.size() > maximum)
      throw std::runtime_error("Matrix checkpoint read exceeds bound");
    return bytes;
  }

 private:
  lf::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

struct Optimizer {
  std::shared_ptr<opt::OptimizerAdapter> adapter;
  std::unique_ptr<lf::CheckpointParticipant> checkpoint;
};

template <class Adapter, class Group, class Participant>
Optimizer optimizer(const lf::Tensor& weight, std::string identity) {
  Group group{.group_id = "main",
              .parameters = {{.parameter_id = "weight", .parameter = weight}}};
  group.options.learning_rate = 0.01;
  if constexpr (std::is_same_v<Adapter, opt::SgdOptimizerAdapter>)
    group.options.momentum = 0.8;
  auto adapter =
      std::make_shared<Adapter>(std::move(identity), std::vector<Group>{group});
  return {adapter, std::make_unique<Participant>(adapter)};
}

Optimizer optimizer(int kind, const lf::Tensor& weight,
                    std::string identity = "matrix.optimizer") {
  switch (kind) {
    case 0:
      return optimizer<opt::SgdOptimizerAdapter, opt::SgdParameterGroup,
                       opt::SgdOptimizerCheckpointParticipant>(weight,
                                                               identity);
    case 1:
      return optimizer<opt::AdamOptimizerAdapter, opt::AdamParameterGroup,
                       opt::AdamOptimizerCheckpointParticipant>(weight,
                                                                identity);
    case 2:
      return optimizer<opt::AdamWOptimizerAdapter, opt::AdamWParameterGroup,
                       opt::AdamWOptimizerCheckpointParticipant>(weight,
                                                                 identity);
    case 3:
      return optimizer<opt::AdagradOptimizerAdapter, opt::AdagradParameterGroup,
                       opt::AdagradOptimizerCheckpointParticipant>(weight,
                                                                   identity);
    case 4:
      return optimizer<opt::AdadeltaOptimizerAdapter,
                       opt::AdadeltaParameterGroup,
                       opt::AdadeltaOptimizerCheckpointParticipant>(weight,
                                                                    identity);
    case 5:
      return optimizer<opt::RmspropOptimizerAdapter, opt::RmspropParameterGroup,
                       opt::RmspropOptimizerCheckpointParticipant>(weight,
                                                                   identity);
    case 6:
      return optimizer<opt::NAdamOptimizerAdapter, opt::NAdamParameterGroup,
                       opt::NAdamOptimizerCheckpointParticipant>(weight,
                                                                 identity);
    case 7:
      return optimizer<opt::LambOptimizerAdapter, opt::LambParameterGroup,
                       opt::LambOptimizerCheckpointParticipant>(weight,
                                                                identity);
  }
  throw std::runtime_error("Unknown matrix optimizer");
}

std::shared_ptr<opt::LrScheduler> scheduler(
    int kind, const std::shared_ptr<opt::OptimizerAdapter>& optimizer) {
  switch (kind) {
    case 0:
      return std::make_shared<opt::StepLrScheduler>(
          optimizer, opt::StepLrSchedulerOptions{.step_size = 3, .gamma = 0.5});
    case 1:
      return std::make_shared<opt::ExponentialLrScheduler>(
          optimizer, opt::ExponentialLrSchedulerOptions{.gamma = 0.95});
    case 2:
      return std::make_shared<opt::MultiStepLrScheduler>(
          optimizer,
          opt::MultiStepLrSchedulerOptions{.milestones = {3, 7}, .gamma = 0.5});
    case 3:
      return std::make_shared<opt::CosineAnnealingLrScheduler>(
          optimizer, opt::CosineAnnealingLrSchedulerOptions{.t_max = 12,
                                                            .eta_min = 0.001});
    case 4:
      return std::make_shared<opt::CosineAnnealingWarmRestartsLrScheduler>(
          optimizer, opt::CosineAnnealingWarmRestartsLrSchedulerOptions{
                         .t_0 = 3, .t_mult = 2, .eta_min = 0.001});
    case 5:
      return std::make_shared<opt::LinearLrScheduler>(
          optimizer, opt::LinearLrSchedulerOptions{.total_iters = 9});
    case 6:
      return std::make_shared<opt::ConstantLrScheduler>(
          optimizer, opt::ConstantLrSchedulerOptions{.total_iters = 7});
    case 7:
      return std::make_shared<opt::PolynomialLrScheduler>(
          optimizer,
          opt::PolynomialLrSchedulerOptions{.total_iters = 12, .power = 2.0});
    case 8:
      return std::make_shared<opt::WarmupLrScheduler>(
          optimizer, opt::WarmupLrSchedulerOptions{.warmup_iters = 7});
    case 9:
      return std::make_shared<opt::ReduceLrOnPlateauScheduler>(
          optimizer, opt::ReduceLrOnPlateauSchedulerOptions{
                         .factor = 0.5, .patience = 1, .cooldown = 1});
    case 10:
      return std::make_shared<opt::OneCycleLrScheduler>(
          optimizer, opt::OneCycleLrSchedulerOptions{
                         .max_learning_rates = {0.02}, .total_steps = 12});
    case 11:
      return std::make_shared<opt::CyclicLrScheduler>(
          optimizer,
          opt::CyclicLrSchedulerOptions{.base_learning_rates = {0.001},
                                        .max_learning_rates = {0.02},
                                        .step_size_up = 3});
    case 12:
      return std::make_shared<opt::LambdaLrScheduler>(
          optimizer, opt::LambdaLrSchedulerOptions{
                         .callable_id = "matrix.half_every_four.v1",
                         .functions = {[](std::uint64_t step) {
                           return step < 4 ? 1.0 : step < 8 ? 0.5 : 0.25;
                         }}});
    case 13:
    case 14: {
      std::vector<std::unique_ptr<opt::LrScheduler>> children;
      children.push_back(std::make_unique<opt::StepLrScheduler>(
          optimizer,
          opt::StepLrSchedulerOptions{.scheduler_id = "matrix.child.step",
                                      .step_size = 3,
                                      .gamma = 0.9}));
      children.push_back(std::make_unique<opt::ExponentialLrScheduler>(
          optimizer, opt::ExponentialLrSchedulerOptions{
                         .scheduler_id = "matrix.child.exp", .gamma = 0.95}));
      if (kind == 13)
        return std::make_shared<opt::SequentialLrScheduler>(
            optimizer, std::move(children),
            opt::SequentialLrSchedulerOptions{.milestones = {6}});
      return std::make_shared<opt::ChainedLrScheduler>(optimizer,
                                                       std::move(children));
    }
  }
  throw std::runtime_error("Unknown matrix scheduler");
}

constexpr std::array<const char*, 8> optimizers{
    "SGD", "Adam", "AdamW", "Adagrad", "Adadelta", "RMSprop", "NAdam", "LAMB"};
constexpr std::array<const char*, 15> schedulers{
    "Step",     "Exponential", "MultiStep",  "Cosine",     "CosineRestarts",
    "Linear",   "Constant",    "Polynomial", "Warmup",     "Plateau",
    "OneCycle", "Cyclic",      "Lambda",     "Sequential", "Chained"};
constexpr std::array<float, 8> input_values{0.25F, 0.5F,  0.75F,  -0.5F,
                                            1.0F,  0.25F, -0.25F, 0.75F};

float rounded(float value, c10::ScalarType dtype) {
  if (dtype == torch::kFloat16) return static_cast<float>(c10::Half(value));
  if (dtype == torch::kBFloat16)
    return static_cast<float>(c10::BFloat16(value));
  return value;
}

// GEMM may round products/reductions in its input precision. Use a scalar
// dot-product error bound, including cancellation, instead of exact rounding
// assumptions about a particular BLAS kernel. Continuation below stays bitwise.
template <std::size_t Size>
void expect_dot(const lf::Tensor& actual,
                const std::array<float, Size>& expected,
                const std::array<float, Size>& magnitudes,
                c10::ScalarType dtype, int terms) {
  const auto values =
      actual.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  ASSERT_EQ(values.numel(), Size);
  ASSERT_TRUE(torch::isfinite(values).all().item<bool>());
  const float unit = dtype == torch::kFloat16    ? std::ldexp(1.0F, -11)
                     : dtype == torch::kBFloat16 ? std::ldexp(1.0F, -8)
                                                 : 0.0F;
  const float gamma = 2 * terms * unit / (1 - 2 * terms * unit);
  for (std::size_t index = 0; index < Size; ++index) {
    const float bound = dtype == torch::kFloat32
                            ? 1e-6F + 1e-5F * std::abs(expected[index])
                            : 1e-6F + gamma * magnitudes[index] +
                                  2 * unit * std::abs(expected[index]);
    EXPECT_NEAR(values.const_data_ptr<float>()[index], expected[index], bound)
        << "element=" << index;
  }
}

void step(const lf::Tensor& weight, Optimizer& optimizer,
          const std::shared_ptr<opt::LrScheduler>& scheduler,
          amp::GradScaler& scaler, c10::ScalarType dtype, int schedule,
          int index, const lf::Tensor& reference_weight = {},
          Optimizer* reference = nullptr,
          const std::shared_ptr<opt::LrScheduler>& reference_scheduler = {}) {
  SCOPED_TRACE(index);
  optimizer.adapter->zero_grad();
  auto input_storage = input_values;
  auto input = torch::from_blob(input_storage.data(), {4, 2}, torch::kFloat32)
                   .clone()
                   .to(weight.device());
  lf::Tensor output;
  {
    std::unique_ptr<amp::AutocastGuard> autocast;
    if (dtype != torch::kFloat32)
      autocast =
          std::make_unique<amp::AutocastGuard>(c10::DeviceType::CUDA, dtype);
    output = torch::matmul(input, weight);
  }
  ASSERT_EQ(output.scalar_type(), dtype);
  const auto cpu_weight = weight.detach().cpu().contiguous();
  const auto* values = cpu_weight.data_ptr<float>();
  std::array<float, 8> expected_output{};
  std::array<float, 8> output_magnitudes{};
  std::array<float, 4> expected_gradient{};
  std::array<float, 4> gradient_magnitudes{};
  for (int batch = 0; batch < 4; ++batch)
    for (int col = 0; col < 2; ++col) {
      float sum = 0;
      for (int row = 0; row < 2; ++row) {
        const float product = input_values[batch * 2 + row] *
                              rounded(values[row * 2 + col], dtype);
        sum += product;
        output_magnitudes[batch * 2 + col] += std::abs(product);
      }
      expected_output[batch * 2 + col] = rounded(sum, dtype);
    }
  ASSERT_NO_FATAL_FAILURE(
      expect_dot(output, expected_output, output_magnitudes, dtype, 2));
  const auto cpu_output =
      output.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  const auto* output_values = cpu_output.const_data_ptr<float>();
  const auto scale = static_cast<float>(scaler.get_scale());
  for (int row = 0; row < 2; ++row)
    for (int col = 0; col < 2; ++col) {
      float sum = 0;
      for (int batch = 0; batch < 4; ++batch) {
        const float product =
            input_values[batch * 2 + row] *
            rounded(output_values[batch * 2 + col] * (scale / 4.0F), dtype);
        sum += product;
        gradient_magnitudes[row * 2 + col] += std::abs(product) / scale;
      }
      expected_gradient[row * 2 + col] = rounded(sum, dtype) / scale;
    }
  scaler.scale(output.to(torch::kFloat32).square().mean()).backward();
  scaler.unscale(optimizer.adapter);
  ASSERT_TRUE(scaler.step_allowed(optimizer.adapter));
  ASSERT_NO_FATAL_FAILURE(expect_dot(weight.grad(), expected_gradient,
                                     gradient_magnitudes, dtype, 4));
  if (reference) {
    reference->adapter->zero_grad();
    reference_weight.mutable_grad() = weight.grad().cpu().clone();
    reference->adapter->step();
  }
  optimizer.adapter->step();
  scaler.mark_step(optimizer.adapter, true);
  scaler.update();
  const auto advance = [&](const std::shared_ptr<opt::LrScheduler>& current) {
    if (schedule == 9)
      current->step(index < 3 ? 1.0 / (index + 1) : 0.5);
    else
      current->step();
  };
  advance(scheduler);
  if (reference) {
    advance(reference_scheduler);
    EXPECT_EQ(scheduler->learning_rates(),
              reference_scheduler->learning_rates());
    EXPECT_TRUE(torch::allclose(weight.detach().cpu(),
                                reference_weight.detach(), 1e-5, 1e-6));
  }
  EXPECT_EQ(optimizer.adapter->completed_steps(),
            static_cast<std::uint64_t>(index + 1));
}

struct MatrixCase {
  int kind;
  c10::ScalarType dtype;
};

void PrintTo(const MatrixCase& value, std::ostream* stream) {
  *stream << optimizers[value.kind] << "_"
          << (value.dtype == torch::kFloat32   ? "FP32"
              : value.dtype == torch::kFloat16 ? "FP16"
                                               : "BF16");
}

std::vector<MatrixCase> matrix_cases() {
  std::vector<MatrixCase> result;
  for (int kind = 0; kind < static_cast<int>(optimizers.size()); ++kind)
    for (const auto dtype :
         {torch::kFloat32, torch::kFloat16, torch::kBFloat16})
      result.push_back({kind, dtype});
  return result;
}

class AcceleratorOptimizerMatrix : public ::testing::TestWithParam<MatrixCase> {
};

TEST_P(AcceleratorOptimizerMatrix,
       AllSchedulersPreserveNumericsAndExactAmpContinuation) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "No CUDA-compatible GPU available";
  const auto [kind, dtype] = GetParam();
  for (int schedule = 0; schedule < static_cast<int>(schedulers.size());
       ++schedule) {
    SCOPED_TRACE(schedulers[schedule]);
    auto initial = torch::tensor({0.25F, -0.5F, 0.75F, 1.0F}).reshape({2, 2});
    auto weight =
        initial.to(torch::Device(torch::kCUDA, 0)).set_requires_grad(true);
    auto reference_weight = initial.clone().set_requires_grad(true);
    auto resumed_weight = weight.detach().clone().set_requires_grad(true);
    auto source = optimizer(kind, weight);
    auto reference = optimizer(kind, reference_weight);
    auto resumed = optimizer(kind, resumed_weight);
    auto source_schedule = scheduler(schedule, source.adapter);
    auto reference_schedule = scheduler(schedule, reference.adapter);
    auto resumed_schedule = scheduler(schedule, resumed.adapter);
    amp::GradScalerOptions options{.init_scale = 8,
                                   .growth_interval = 4,
                                   .enabled = dtype == torch::kFloat16};
    amp::GradScaler scaler(options), resumed_scaler(options);
    for (int index = 0; index < 6; ++index)
      ASSERT_NO_FATAL_FAILURE(step(weight, source, source_schedule, scaler,
                                   dtype, schedule, index, reference_weight,
                                   &reference, reference_schedule));
    MemoryStore optimizer_state(source.checkpoint->describe_state());
    source.checkpoint->save_state(optimizer_state);
    opt::LrSchedulerCheckpointParticipant source_state(source_schedule),
        resumed_state(resumed_schedule);
    MemoryStore scheduler_state(source_state.describe_state());
    source_state.save_state(scheduler_state);
    {
      torch::NoGradGuard guard;
      resumed_weight.copy_(weight);
    }
    resumed.checkpoint->load_state(optimizer_state,
                                   lf::RestorePolicy::exact_only);
    resumed_state.load_state(scheduler_state, lf::RestorePolicy::exact_only);
    resumed_scaler.restore_state(scaler.capture_state());
    for (int index = 6; index < 12; ++index) {
      ASSERT_NO_FATAL_FAILURE(step(weight, source, source_schedule, scaler,
                                   dtype, schedule, index, reference_weight,
                                   &reference, reference_schedule));
      ASSERT_NO_FATAL_FAILURE(step(resumed_weight, resumed, resumed_schedule,
                                   resumed_scaler, dtype, schedule, index));
      ASSERT_TRUE(torch::equal(weight, resumed_weight));
      ASSERT_TRUE(torch::equal(weight.grad(), resumed_weight.grad()));
      EXPECT_EQ(source_schedule->learning_rates(),
                resumed_schedule->learning_rates());
      EXPECT_EQ(source_schedule->completed_steps(),
                resumed_schedule->completed_steps());
      EXPECT_EQ(scaler.get_scale(), resumed_scaler.get_scale());
      EXPECT_EQ(scaler.capture_state().growth_tracker,
                resumed_scaler.capture_state().growth_tracker);
    }
    EXPECT_FALSE(torch::equal(weight.detach().cpu(), initial));
  }
}

std::string matrix_name(
    const ::testing::TestParamInfo<AcceleratorOptimizerMatrix::ParamType>&
        info) {
  return ::testing::PrintToString(info.param);
}
TEST(AcceleratorOptimizerCycle, SharedOverflowSkipsAllAlgorithmsThenRecovers) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "No CUDA-compatible GPU available";
  const auto options =
      torch::TensorOptions().device(torch::Device(torch::kCUDA, 0));
  std::vector<lf::Tensor> weights;
  std::vector<Optimizer> algorithms;
  for (int kind = 0; kind < static_cast<int>(optimizers.size()); ++kind) {
    weights.push_back(torch::ones({2, 2}, options).set_requires_grad(true));
    algorithms.push_back(optimizer(kind, weights.back(),
                                   std::string("cycle.") + optimizers[kind]));
  }
  amp::GradScaler scaler({.init_scale = 16, .growth_interval = 2});
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (std::size_t index = 0; index < weights.size(); ++index) {
      algorithms[index].adapter->zero_grad();
      lf::Tensor output;
      {
        amp::AutocastGuard autocast(c10::DeviceType::CUDA, torch::kFloat16);
        output = torch::matmul(torch::eye(2, options), weights[index]);
      }
      ASSERT_EQ(output.scalar_type(), torch::kFloat16);
      scaler.scale(output.to(torch::kFloat32).square().mean()).backward();
    }
    if (cycle == 0)
      weights.back().mutable_grad().fill_(
          std::numeric_limits<float>::infinity());
    for (auto& algorithm : algorithms) scaler.unscale(algorithm.adapter);
    const bool allowed = scaler.cycle_allowed();
    ASSERT_EQ(allowed, cycle == 1);
    for (std::size_t index = 0; index < weights.size(); ++index) {
      SCOPED_TRACE(optimizers[index]);
      if (allowed) algorithms[index].adapter->step();
      scaler.mark_step(algorithms[index].adapter, allowed);
      EXPECT_EQ(algorithms[index].adapter->completed_steps(),
                static_cast<std::uint64_t>(cycle));
      EXPECT_EQ(torch::equal(weights[index].detach(),
                             torch::ones_like(weights[index])),
                !allowed);
      EXPECT_TRUE(torch::isfinite(weights[index]).all().item<bool>());
    }
    scaler.update();
    EXPECT_EQ(scaler.get_scale(), 8);
    EXPECT_EQ(scaler.capture_state().growth_tracker,
              static_cast<std::uint64_t>(cycle));
  }
}

INSTANTIATE_TEST_SUITE_P(Precision, AcceleratorOptimizerMatrix,
                         ::testing::ValuesIn(matrix_cases()), matrix_name);
}  // namespace
