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
   * @brief Replaces one byte counted from a field's end.
   * @param name Existing field name.
   * @param trailing_offset Zero selects the final byte.
   * @param value Replacement byte.
   */
  void overwrite_from_end(std::string_view name, std::size_t trailing_offset,
                          std::byte value) {
    auto& data = blobs_.at(std::string(name));
    data.at(data.size() - 1U - trailing_offset) = value;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Participant field contract.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields by stable name.
};

jyppx::nn::TensorSchema tensor_schema(std::int64_t second_dimension) {
  jyppx::nn::TensorSchema schema;
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
  strategy.prepare(domain.model, *domain.optimizer);

  const auto outcome = train_once(domain, strategy);

  EXPECT_EQ(outcome.status, jyppx::training::StrategyStepStatus::applied);
  EXPECT_EQ(outcome.optimizer_id, "optimizer.main");
  EXPECT_EQ(outcome.optimizer_completed_steps, 1U);
  EXPECT_EQ(strategy.completed_backward_calls(), 1U);
  EXPECT_EQ(strategy.completed_optimizer_steps(), 1U);
  EXPECT_NEAR(domain.linear->weight.item<double>(), 0.2, 1e-6);
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

  EXPECT_THROW(strategy.prepare(domain.model, mismatched),
               jyppx::CompatibilityError);
  EXPECT_FALSE(strategy.prepared());
  EXPECT_TRUE(domain.model.state().parameters().front().device().is_cpu());
}

TEST(SingleDeviceStrategyTest, RejectsInvalidLossAndDifferentOptimizer) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  strategy.prepare(domain.model, *domain.optimizer);
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
  EXPECT_EQ(strategy.completed_optimizer_steps(), 0U);
}

TEST(SingleDeviceStrategyTest, DetectsOptimizerUpdatesOutsideStrategy) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  strategy.prepare(domain.model, *domain.optimizer);
  static_cast<void>(domain.optimizer->step());
  MemoryStateStore state(strategy.describe_state());

  EXPECT_THROW(strategy.zero_grad(*domain.optimizer),
               jyppx::CompatibilityError);
  EXPECT_THROW(strategy.save_state(state), jyppx::CompatibilityError);
}

TEST(SingleDeviceStrategyTest, ValidatesMetricsCancellationAndClose) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  strategy.prepare(domain.model, *domain.optimizer);

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
  source.prepare(source_domain.model, *source_domain.optimizer);
  static_cast<void>(train_once(source_domain, source));
  jyppx::optim::SgdOptimizerCheckpointParticipant source_optimizer_state(
      source_domain.optimizer);
  MemoryStateStore optimizer_state(source_optimizer_state.describe_state());
  source_optimizer_state.save_state(optimizer_state);
  MemoryStateStore strategy_state(source.describe_state());
  source.save_state(strategy_state);

  TrainingDomain restored_domain;
  jyppx::training::SingleDeviceStrategy restored;
  restored.prepare(restored_domain.model, *restored_domain.optimizer);
  jyppx::optim::SgdOptimizerCheckpointParticipant restored_optimizer_state(
      restored_domain.optimizer);
  restored_optimizer_state.load_state(optimizer_state,
                                      jyppx::RestorePolicy::exact_only);
  restored.load_state(strategy_state, jyppx::RestorePolicy::exact_only);

  EXPECT_EQ(restored.completed_backward_calls(), 1U);
  EXPECT_EQ(restored.completed_optimizer_steps(), 1U);
  EXPECT_EQ(restored_domain.optimizer->completed_steps(), 1U);
  const auto outcome = train_once(restored_domain, restored);
  EXPECT_EQ(outcome.optimizer_completed_steps, 2U);
  EXPECT_EQ(restored.completed_optimizer_steps(), 2U);
}

TEST(SingleDeviceStrategyTest, CorruptRestoreHasStrongGuarantee) {
  TrainingDomain source_domain;
  jyppx::training::SingleDeviceStrategy source;
  source.prepare(source_domain.model, *source_domain.optimizer);
  MemoryStateStore state(source.describe_state());
  source.save_state(state);
  state.overwrite_from_end("control", 24, std::byte{2});

  TrainingDomain target_domain;
  jyppx::training::SingleDeviceStrategy target;
  target.prepare(target_domain.model, *target_domain.optimizer);
  EXPECT_THROW(target.load_state(state, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_FALSE(target.closed());
  EXPECT_EQ(target.completed_backward_calls(), 0U);
  EXPECT_EQ(target.completed_optimizer_steps(), 0U);
}

TEST(SingleDeviceStrategyTest, RequiresOptimizerRestoreBeforeStrategyRestore) {
  TrainingDomain source_domain;
  jyppx::training::SingleDeviceStrategy source;
  source.prepare(source_domain.model, *source_domain.optimizer);
  static_cast<void>(train_once(source_domain, source));
  MemoryStateStore state(source.describe_state());
  source.save_state(state);

  TrainingDomain target_domain;
  jyppx::training::SingleDeviceStrategy target;
  target.prepare(target_domain.model, *target_domain.optimizer);
  EXPECT_THROW(target.load_state(state, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_EQ(target.completed_backward_calls(), 0U);
  EXPECT_EQ(target.completed_optimizer_steps(), 0U);
}

TEST(SingleDeviceStrategyTest, RejectsRestoreIntoClosedInstance) {
  TrainingDomain domain;
  jyppx::training::SingleDeviceStrategy strategy;
  strategy.prepare(domain.model, *domain.optimizer);
  MemoryStateStore state(strategy.describe_state());
  strategy.save_state(state);
  strategy.close();

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
  strategy.prepare(domain.model, *domain.optimizer);

  const auto outcome = train_once(domain, strategy, device);
  EXPECT_EQ(outcome.optimizer_completed_steps, 1U);
  EXPECT_TRUE(domain.linear->weight.device().is_cuda());
  EXPECT_TRUE(torch::isfinite(domain.linear->weight).all().item<bool>());
}

}  // namespace
