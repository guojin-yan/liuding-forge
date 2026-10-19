// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/training/single_device_strategy.hpp>

#include <torch/cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  /** @brief Creates an empty in-memory participant state. */
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one complete field into test-owned memory. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  /** @brief Returns the immutable state manifest. */
  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a field is available. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns one field under the requested byte bound. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto& data = blobs_.at(std::string(name));
    if (data.size() > max_bytes) {
      throw jyppx::ResourceExhausted("test state exceeds read limit");
    }
    return data;
  }

  /**
   * @brief Replaces one byte at an absolute field offset.
   * @param name Existing field name.
   * @param offset Zero-based byte offset within the field.
   * @param value Replacement byte.
   */
  void overwrite_at(std::string_view name, std::size_t offset,
                    std::byte value) {
    blobs_.at(std::string(name)).at(offset) = value;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Participant field contract.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields by stable name.
};

jyppx::TensorSchema tensor_schema(std::int64_t second_dimension) {
  jyppx::TensorSchema schema;
  schema.dtype = torch::kFloat32;
  schema.rank = 2;
  schema.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, second_dimension};
  return schema;
}

struct TrainingDomain {
  /** @brief Builds a one-weight linear model and matching delegated optimizer.
   */
  TrainingDomain()
      : linear(torch::nn::LinearOptions(1, 1).bias(false)),
        model(jyppx::nn::adapt_tensor_holder(linear, tensor_schema(1),
                                             tensor_schema(1))) {
    torch::NoGradGuard guard;
    linear->weight.fill_(1.0);
    std::vector<jyppx::optim::OptimizerParameterBinding> bindings;
    for (const auto& item : model.state().named_parameters()) {
      bindings.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
    }
    optimizer = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
        "optimizer.main", std::vector<jyppx::optim::SgdParameterGroup>{
                              {.group_id = "main",
                               .parameters = std::move(bindings),
                               .options = {.learning_rate = 0.1}}});
  }

  torch::nn::Linear linear;       ///< Shared concrete model holder.
  jyppx::nn::ModelAdapter model;  ///< Schema-checked invocation boundary.
  std::shared_ptr<jyppx::optim::SgdOptimizerAdapter>
      optimizer;  ///< Matching delegated optimizer.
};

/** @brief Prepares one strategy with a single optimizer span. */
void prepare_strategy(jyppx::training::SingleDeviceStrategy& strategy,
                      jyppx::nn::ModelAdapter& model,
                      jyppx::optim::OptimizerAdapter& optimizer) {
  jyppx::optim::OptimizerAdapter* optimizers[] = {&optimizer};
  strategy.prepare(model, optimizers);
}

jyppx::training::StrategyStepOutcome train_once(
    TrainingDomain& domain, jyppx::training::SingleDeviceStrategy& strategy,
    const jyppx::Device& device = jyppx::Device(torch::kCPU)) {
  strategy.zero_grad(*domain.optimizer);
  const auto input =
      torch::full({1, 1}, 2.0, torch::TensorOptions().dtype(torch::kFloat32))
          .to(device);
  const auto output =
      domain.model.invoke({.positional = {input}}).value.toTensor();
  const auto loss = output.pow(2).mean();
  strategy.backward(loss);
  return strategy.optimizer_step(*domain.optimizer);
}

TEST(SingleDeviceStrategyTest, MatchesDirectSingleWeightSgdUpdate) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  prepare_strategy(strategy, domain.model, *domain.optimizer);

  const auto outcome = train_once(domain, strategy);

  EXPECT_EQ(outcome.status, jyppx::training::StrategyStepStatus::applied);
  EXPECT_EQ(outcome.optimizer_id, "optimizer.main");
  EXPECT_EQ(outcome.optimizer_completed_steps, 1U);
  EXPECT_EQ(strategy.completed_backward_calls(), 1U);
  EXPECT_EQ(strategy.completed_optimizer_steps("optimizer.main"), 1U);
  EXPECT_NEAR(domain.linear->weight.item<double>(), 0.2, 1e-6);
}

TEST(SingleDeviceStrategyTest,
     PreparesDisjointOptimizersInCanonicalIdentityOrder) {
  auto linear = torch::nn::Linear(torch::nn::LinearOptions(1, 1));
  jyppx::nn::ModelAdapter model(jyppx::nn::adapt_tensor_holder(
      linear, tensor_schema(1), tensor_schema(1)));
  std::map<std::string, jyppx::Tensor> parameters;
  for (const auto& item : model.state().named_parameters()) {
    parameters.emplace(item.key(), item.value());
  }
  auto weight = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.weight",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "weight",
           .parameters = {{.parameter_id = "weight",
                           .parameter = parameters.at("weight")}},
           .options = {.learning_rate = 0.1}}});
  auto bias = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.bias",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "bias",
           .parameters = {{.parameter_id = "bias",
                           .parameter = parameters.at("bias")}},
           .options = {.learning_rate = 0.1}}});
  jyppx::optim::OptimizerAdapter* optimizers[] = {weight.get(), bias.get()};
  jyppx::training::SingleDeviceStrategy strategy;
  strategy.prepare(model, optimizers);
  strategy.zero_grad(*bias);
  strategy.zero_grad(*weight);
  const auto loss = model.invoke({.positional = {torch::ones({1, 1})}})
                        .value.toTensor()
                        .pow(2)
                        .mean();
  strategy.backward(loss);
  EXPECT_EQ(strategy.optimizer_step(*bias).optimizer_id, "optimizer.bias");
  EXPECT_EQ(strategy.optimizer_step(*weight).optimizer_id, "optimizer.weight");
  EXPECT_EQ(strategy.completed_optimizer_steps("optimizer.bias"), 1U);
  EXPECT_EQ(strategy.completed_optimizer_steps("optimizer.weight"), 1U);
  EXPECT_THROW(strategy.completed_optimizer_steps("optimizer.missing"),
               jyppx::InvalidArgument);

  jyppx::optim::OptimizerAdapter* reordered[] = {bias.get(), weight.get()};
  EXPECT_NO_THROW(strategy.prepare(model, reordered));
}

TEST(SingleDeviceStrategyTest, RejectsOverlappingOptimizerParameters) {
  TrainingDomain domain;
  const auto parameter =
      domain.model.state().named_parameters().begin()->value();
  auto overlapping = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.overlap",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "overlap",
           .parameters = {{.parameter_id = "weight", .parameter = parameter}},
           .options = {.learning_rate = 0.1}}});
  jyppx::optim::OptimizerAdapter* optimizers[] = {domain.optimizer.get(),
                                                  overlapping.get()};
  jyppx::training::SingleDeviceStrategy strategy;
  EXPECT_THROW(strategy.prepare(domain.model, optimizers),
               jyppx::InvalidArgument);
  EXPECT_FALSE(strategy.prepared());
}

TEST(SingleDeviceStrategyTest, RejectsIncompleteOptimizerParameterCoverage) {
  auto linear = torch::nn::Linear(torch::nn::LinearOptions(1, 1));
  jyppx::nn::ModelAdapter model(jyppx::nn::adapt_tensor_holder(
      linear, tensor_schema(1), tensor_schema(1)));
  const auto named_parameters = model.state().named_parameters();
  auto weight_only = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.weight",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "weight",
           .parameters = {{.parameter_id = "weight",
                           .parameter = named_parameters["weight"]}},
           .options = {.learning_rate = 0.1}}});
  jyppx::optim::OptimizerAdapter* optimizers[] = {weight_only.get()};
  jyppx::training::SingleDeviceStrategy strategy;

  EXPECT_THROW(strategy.prepare(model, optimizers), jyppx::CompatibilityError);
  EXPECT_FALSE(strategy.prepared());
}

TEST(SingleDeviceStrategyTest,
     RejectsMismatchedParameterIdentityBeforePrepare) {
  TrainingDomain domain;
  const auto parameter = domain.linear->weight;
  jyppx::optim::SgdOptimizerAdapter mismatched(
      "optimizer.other",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "different", .parameter = parameter}},
        .options = {.learning_rate = 0.1}}});
  jyppx::training::SingleDeviceStrategy strategy;

  EXPECT_THROW(prepare_strategy(strategy, domain.model, mismatched),
               jyppx::CompatibilityError);
  EXPECT_FALSE(strategy.prepared());
  EXPECT_TRUE(domain.model.state().parameters().front().device().is_cpu());
}

TEST(SingleDeviceStrategyTest, RejectsInvalidLossAndDifferentOptimizer) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  prepare_strategy(strategy, domain.model, *domain.optimizer);
  const auto parameter = domain.linear->weight;
  jyppx::optim::SgdOptimizerAdapter other(
      "optimizer.other",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.1}}});

  EXPECT_THROW(strategy.backward(torch::ones({1}, torch::kFloat32)),
               jyppx::InvalidArgument);
  auto non_finite = torch::full(
      {}, std::numeric_limits<float>::quiet_NaN(),
      torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  EXPECT_THROW(strategy.backward(non_finite), jyppx::NumericalError);
  EXPECT_THROW(strategy.optimizer_step(other), jyppx::CompatibilityError);
  EXPECT_EQ(strategy.completed_backward_calls(), 0U);
  EXPECT_EQ(strategy.completed_optimizer_steps("optimizer.main"), 0U);
}

TEST(SingleDeviceStrategyTest, DetectsOptimizerUpdatesOutsideStrategy) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  prepare_strategy(strategy, domain.model, *domain.optimizer);
  static_cast<void>(domain.optimizer->step());
  MemoryStateStore state(strategy.describe_state());

  EXPECT_THROW(strategy.zero_grad(*domain.optimizer),
               jyppx::CompatibilityError);
  EXPECT_THROW(strategy.save_state(state), jyppx::CompatibilityError);
}

TEST(SingleDeviceStrategyTest, ValidatesMetricsCancellationAndClose) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  prepare_strategy(strategy, domain.model, *domain.optimizer);

  EXPECT_EQ(strategy.reduce_metrics({{"loss", 1.25}}),
            (jyppx::training::StrategyMetricValues{{"loss", 1.25}}));
  EXPECT_THROW(strategy.reduce_metrics(
                   {{"loss", std::numeric_limits<double>::infinity()}}),
               jyppx::InvalidArgument);
  std::stop_source cancellation;
  cancellation.request_stop();
  EXPECT_THROW(
      strategy.barrier(jyppx::training::StrategyBarrierReason::checkpoint,
                       cancellation.get_token()),
      jyppx::Cancelled);
  strategy.close();
  strategy.close();
  EXPECT_TRUE(strategy.closed());
  EXPECT_THROW(strategy.zero_grad(*domain.optimizer), jyppx::NotReady);
}

TEST(SingleDeviceStrategyTest, RestoresCountersAfterOptimizerState) {
  TrainingDomain source_domain;
  jyppx::training::SingleDeviceStrategy source;
  prepare_strategy(source, source_domain.model, *source_domain.optimizer);
  static_cast<void>(train_once(source_domain, source));
  jyppx::optim::SgdOptimizerCheckpointParticipant source_optimizer_state(
      source_domain.optimizer);
  MemoryStateStore optimizer_state(source_optimizer_state.describe_state());
  source_optimizer_state.save_state(optimizer_state);
  MemoryStateStore strategy_state(source.describe_state());
  source.save_state(strategy_state);

  TrainingDomain restored_domain;
  jyppx::training::SingleDeviceStrategy restored;
  prepare_strategy(restored, restored_domain.model, *restored_domain.optimizer);
  jyppx::optim::SgdOptimizerCheckpointParticipant restored_optimizer_state(
      restored_domain.optimizer);
  restored_optimizer_state.load_state(optimizer_state,
                                      jyppx::RestorePolicy::exact_only);
  restored.load_state(strategy_state, jyppx::RestorePolicy::exact_only);

  EXPECT_EQ(restored.completed_backward_calls(), 1U);
  EXPECT_EQ(restored.completed_optimizer_steps("optimizer.main"), 1U);
  EXPECT_EQ(restored_domain.optimizer->completed_steps(), 1U);
  const auto outcome = train_once(restored_domain, restored);
  EXPECT_EQ(outcome.optimizer_completed_steps, 2U);
  EXPECT_EQ(restored.completed_optimizer_steps("optimizer.main"), 2U);
}

TEST(SingleDeviceStrategyTest, CorruptRestoreHasStrongGuarantee) {
  TrainingDomain source_domain;
  jyppx::training::SingleDeviceStrategy source;
  prepare_strategy(source, source_domain.model, *source_domain.optimizer);
  MemoryStateStore valid_state(source.describe_state());
  source.save_state(valid_state);
  constexpr std::string_view kControlMagic =
      "JYPPX-SINGLE-DEVICE-STRATEGY-CONTROL-V2";
  for (const auto invalid_closed : {std::byte{1}, std::byte{2}}) {
    auto state = valid_state;
    state.overwrite_at("control", sizeof(std::uint64_t) + kControlMagic.size(),
                       invalid_closed);

    TrainingDomain target_domain;
    jyppx::training::SingleDeviceStrategy target;
    prepare_strategy(target, target_domain.model, *target_domain.optimizer);
    EXPECT_THROW(target.load_state(state, jyppx::RestorePolicy::exact_only),
                 jyppx::SerializationError);
    EXPECT_FALSE(target.closed());
    EXPECT_EQ(target.completed_backward_calls(), 0U);
    EXPECT_EQ(target.completed_optimizer_steps("optimizer.main"), 0U);
  }
}

TEST(SingleDeviceStrategyTest, RestoreOrderDoesNotDependOnParticipantIdentity) {
  TrainingDomain source_domain;
  jyppx::training::SingleDeviceStrategy source;
  prepare_strategy(source, source_domain.model, *source_domain.optimizer);
  static_cast<void>(train_once(source_domain, source));
  jyppx::optim::SgdOptimizerCheckpointParticipant source_optimizer_state(
      source_domain.optimizer);
  MemoryStateStore optimizer_state(source_optimizer_state.describe_state());
  source_optimizer_state.save_state(optimizer_state);
  MemoryStateStore strategy_state(source.describe_state());
  source.save_state(strategy_state);

  TrainingDomain target_domain;
  jyppx::training::SingleDeviceStrategy target;
  prepare_strategy(target, target_domain.model, *target_domain.optimizer);
  EXPECT_NO_THROW(
      target.load_state(strategy_state, jyppx::RestorePolicy::exact_only));
  EXPECT_EQ(target.completed_backward_calls(), 1U);
  EXPECT_EQ(target.completed_optimizer_steps("optimizer.main"), 1U);
  EXPECT_THROW(target.zero_grad(*target_domain.optimizer),
               jyppx::CompatibilityError);

  jyppx::optim::SgdOptimizerCheckpointParticipant target_optimizer_state(
      target_domain.optimizer);
  target_optimizer_state.load_state(optimizer_state,
                                    jyppx::RestorePolicy::exact_only);
  EXPECT_NO_THROW(target.zero_grad(*target_domain.optimizer));
  EXPECT_EQ(train_once(target_domain, target).optimizer_completed_steps, 2U);
}

TEST(SingleDeviceStrategyTest, RejectsRestoreIntoClosedInstance) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  prepare_strategy(strategy, domain.model, *domain.optimizer);
  MemoryStateStore state(strategy.describe_state());
  strategy.save_state(state);
  strategy.close();

  EXPECT_THROW(strategy.save_state(state), jyppx::NotReady);
  EXPECT_THROW(strategy.load_state(state, jyppx::RestorePolicy::exact_only),
               jyppx::NotReady);
  EXPECT_TRUE(strategy.closed());
}

TEST(SingleDeviceStrategyTest, RunsOnRealCudaWhenAvailable) {
  if (!torch::cuda::is_available() || torch::cuda::device_count() == 0) {
    GTEST_SKIP() << "CUDA capability is unavailable";
  }
  TrainingDomain domain;
  const auto device = jyppx::Device(torch::kCUDA, 0);
  jyppx::training::SingleDeviceStrategy strategy(
      {.strategy_id = "strategy.cuda", .device = device});
  jyppx::optim::SgdOptimizerCheckpointParticipant optimizer_state(
      domain.optimizer);
  prepare_strategy(strategy, domain.model, *domain.optimizer);
  EXPECT_NO_THROW(static_cast<void>(optimizer_state.describe_state()));

  const auto outcome = train_once(domain, strategy, device);
  EXPECT_EQ(outcome.optimizer_completed_steps, 1U);
  EXPECT_TRUE(domain.linear->weight.device().is_cuda());
  EXPECT_TRUE(torch::isfinite(domain.linear->weight).all().item<bool>());
  EXPECT_NO_THROW(static_cast<void>(optimizer_state.describe_state()));
}

}  // namespace
