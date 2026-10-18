// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/lamb_adapter.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using jyppx::StateManifest;
using jyppx::Tensor;
using jyppx::optim::LambAdapterOptions;
using jyppx::optim::LambOptimizerAdapter;
using jyppx::optim::LambOptimizerCheckpointParticipant;
using jyppx::optim::LambParameterGroup;

/** @brief In-memory participant store used for storage-neutral state tests. */
class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  /** @brief Creates an empty store for the supplied state contract. */
  explicit MemoryStateStore(StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one participant blob into stable owned storage. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    blobs_[std::string(name)] = {data.begin(), data.end()};
  }

  /** @brief Returns the descriptor associated with the stored blobs. */
  const StateManifest& manifest() const noexcept override { return manifest_; }

  /** @brief Reports whether the named blob has been written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns one complete blob after enforcing the caller limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto iterator = blobs_.find(std::string(name));
    if (iterator == blobs_.end()) {
      throw std::runtime_error("missing in-memory state field");
    }
    if (iterator->second.size() > max_bytes) {
      throw std::runtime_error("in-memory state field exceeds limit");
    }
    return iterator->second;
  }

  /** @brief Replaces one field to construct malformed-state test cases. */
  void replace(std::string name, std::vector<std::byte> bytes) {
    blobs_[std::move(name)] = std::move(bytes);
  }

 private:
  StateManifest manifest_;  ///< Stored participant state contract.
  std::map<std::string, std::vector<std::byte>> blobs_;  ///< Owned blobs.
};

/** @brief Independent vector state used to verify the tensor-level ratio. */
struct ReferenceState {
  std::vector<double> parameters;     ///< Current parameter elements.
  std::vector<double> first_moment;   ///< First-moment elements.
  std::vector<double> second_moment;  ///< Second-moment elements.
  std::int64_t step = 0;  ///< Number of updates with a defined gradient.
};

Tensor make_parameter(std::initializer_list<std::int64_t> shape,
                      c10::ScalarType dtype = torch::kFloat64,
                      c10::Device device = torch::kCPU) {
  return torch::randn(
      shape,
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
}

LambParameterGroup make_group(std::string group_id, std::string parameter_id,
                              const Tensor& parameter,
                              LambAdapterOptions options = {}) {
  return {.group_id = std::move(group_id),
          .parameters = {{.parameter_id = std::move(parameter_id),
                          .parameter = parameter}},
          .options = options};
}

ReferenceState make_reference(const Tensor& parameter) {
  const auto values = parameter.detach().to(torch::kCPU).contiguous();
  ReferenceState state;
  state.parameters.assign(values.data_ptr<double>(),
                          values.data_ptr<double>() + values.numel());
  state.first_moment.resize(state.parameters.size());
  state.second_moment.resize(state.parameters.size());
  return state;
}

void update_reference(ReferenceState& state,
                      const std::vector<double>& gradients,
                      const LambAdapterOptions& options) {
  ++state.step;
  std::vector<double> direction(state.parameters.size());
  for (std::size_t index = 0; index < state.parameters.size(); ++index) {
    const auto gradient =
        options.maximize ? -gradients[index] : gradients[index];
    state.first_moment[index] = options.beta1 * state.first_moment[index] +
                                (1.0 - options.beta1) * gradient;
    state.second_moment[index] = options.beta2 * state.second_moment[index] +
                                 (1.0 - options.beta2) * gradient * gradient;
    const auto corrected_first =
        state.first_moment[index] /
        (1.0 - std::pow(options.beta1, static_cast<double>(state.step)));
    const auto corrected_second =
        state.second_moment[index] /
        (1.0 - std::pow(options.beta2, static_cast<double>(state.step)));
    direction[index] =
        corrected_first / (std::sqrt(corrected_second) + options.epsilon) +
        options.weight_decay * state.parameters[index];
  }
  double parameter_squared_norm = 0.0;
  double direction_squared_norm = 0.0;
  for (std::size_t index = 0; index < state.parameters.size(); ++index) {
    parameter_squared_norm += state.parameters[index] * state.parameters[index];
    direction_squared_norm += direction[index] * direction[index];
  }
  const auto parameter_norm = std::min(std::sqrt(parameter_squared_norm), 10.0);
  const auto direction_norm = std::sqrt(direction_squared_norm);
  const auto trust_ratio = parameter_norm == 0.0 || direction_norm == 0.0
                               ? 1.0
                               : parameter_norm / direction_norm;
  for (std::size_t index = 0; index < state.parameters.size(); ++index) {
    state.parameters[index] -=
        options.learning_rate * trust_ratio * direction[index];
  }
}

void assign_gradient(const Tensor& parameter, std::int64_t step,
                     double phase = 0.0) {
  parameter.mutable_grad() = torch::sin(
      parameter.detach() * 0.17 + static_cast<double>(step) * 0.03 + phase);
}

TEST(LambOptimizerAdapterTest, MatchesIndependentLongRunReferenceAndGroups) {
  const LambAdapterOptions regularized{.learning_rate = 0.012,
                                       .beta1 = 0.82,
                                       .beta2 = 0.973,
                                       .epsilon = 2e-8,
                                       .weight_decay = 0.025};
  const LambAdapterOptions maximizing{.learning_rate = 0.006,
                                      .beta1 = 0.91,
                                      .beta2 = 0.995,
                                      .epsilon = 3e-7,
                                      .weight_decay = 0.01,
                                      .maximize = true};
  auto first = torch::tensor({0.75, -1.25, 2.0, -0.125},
                             torch::TensorOptions().dtype(torch::kFloat64))
                   .set_requires_grad(true);
  auto second = torch::tensor({-0.5, 1.5, 0.25},
                              torch::TensorOptions().dtype(torch::kFloat64))
                    .set_requires_grad(true);
  auto first_reference = make_reference(first);
  auto second_reference = make_reference(second);
  LambOptimizerAdapter optimizer(
      "optimizer.lamb",
      {make_group("regularized", "weight", first, regularized),
       make_group("maximizing", "bias", second, maximizing)});

  for (std::int64_t step = 0; step < 512; ++step) {
    std::vector<double> first_gradients(first_reference.parameters.size());
    std::vector<double> second_gradients(second_reference.parameters.size());
    for (std::size_t index = 0; index < first_gradients.size(); ++index) {
      first_gradients[index] =
          std::sin((static_cast<double>(step) + 1.0) * 0.031 +
                   (static_cast<double>(index) + 1.0) * 0.17);
    }
    for (std::size_t index = 0; index < second_gradients.size(); ++index) {
      second_gradients[index] =
          std::cos((static_cast<double>(step) + 1.0) * 0.019 +
                   (static_cast<double>(index) + 1.0) * 0.23);
    }
    first.mutable_grad() = torch::tensor(first_gradients, torch::kFloat64);
    second.mutable_grad() = torch::tensor(second_gradients, torch::kFloat64);
    update_reference(first_reference, first_gradients, regularized);
    update_reference(second_reference, second_gradients, maximizing);
    optimizer.step();
  }

  for (std::size_t index = 0; index < first_reference.parameters.size();
       ++index) {
    EXPECT_NEAR(first[static_cast<std::int64_t>(index)].item<double>(),
                first_reference.parameters[index], 3e-11);
  }
  for (std::size_t index = 0; index < second_reference.parameters.size();
       ++index) {
    EXPECT_NEAR(second[static_cast<std::int64_t>(index)].item<double>(),
                second_reference.parameters[index], 3e-11);
  }
  EXPECT_EQ(optimizer.completed_steps(), 512U);
  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{0.012, 0.006}));
}

TEST(LambOptimizerAdapterTest, FreezesTrustRatioClampAndZeroNormFallback) {
  auto zero = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto large = torch::full(
      {1}, 20.0,
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto stationary = torch::full(
      {1}, 2.0,
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const LambAdapterOptions options{.learning_rate = 0.01, .epsilon = 1e-6};
  LambOptimizerAdapter optimizer(
      "optimizer.lamb",
      {make_group("zero", "zero", zero, options),
       make_group("large", "large", large, options),
       make_group("stationary", "stationary", stationary, options)});
  zero.mutable_grad() = torch::ones_like(zero);
  large.mutable_grad() = torch::ones_like(large);
  stationary.mutable_grad() = torch::zeros_like(stationary);

  optimizer.step();

  EXPECT_NEAR(zero.item<double>(), -0.01 / (1.0 + 1e-6), 1e-12);
  EXPECT_NEAR(large.item<double>(), 19.9, 1e-12);
  EXPECT_DOUBLE_EQ(stationary.item<double>(), 2.0);
}

TEST(LambOptimizerAdapterTest, RejectsSparseGradientsBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  LambOptimizerAdapter optimizer(
      "optimizer.lamb",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first},
                       {.parameter_id = "second", .parameter = second}}}});
  first.mutable_grad() = torch::ones_like(first);
  second.mutable_grad() =
      torch::sparse_coo_tensor(torch::tensor({{0, 2}}, torch::kInt64),
                               torch::tensor({1.0, 1.0}, torch::kFloat64), {4});

  EXPECT_THROW(optimizer.step(), jyppx::NotImplemented);
  EXPECT_TRUE(torch::equal(first, before_first));
  EXPECT_TRUE(torch::equal(second, before_second));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(LambOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  LambOptimizerAdapter optimizer("optimizer.lamb",
                                 {make_group("main", "weight", parameter)});

  EXPECT_THROW(optimizer.step([&parameter] {
    parameter.mutable_grad() =
        torch::sparse_coo_tensor(torch::tensor({{1}}, torch::kInt64),
                                 torch::tensor({2.0}, torch::kFloat64), {4});
    return torch::tensor(1.0);
  }),
               jyppx::NotImplemented);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(LambOptimizerAdapterTest, CheckpointResumesIndependentMomentsExactly) {
  torch::manual_seed(20261018);
  auto first = make_parameter({6});
  auto second = make_parameter({4});
  auto resumed_first = first.detach().clone().set_requires_grad(true);
  auto resumed_second = second.detach().clone().set_requires_grad(true);
  auto make_two_parameter_group = [](const Tensor& first_parameter,
                                     const Tensor& second_parameter) {
    return LambParameterGroup{
        .group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first_parameter},
                       {.parameter_id = "second",
                        .parameter = second_parameter}},
        .options = {.learning_rate = 0.008,
                    .beta1 = 0.87,
                    .beta2 = 0.991,
                    .epsilon = 4e-7,
                    .weight_decay = 0.01}};
  };
  auto uninterrupted = std::make_shared<LambOptimizerAdapter>(
      "optimizer.lamb",
      std::vector<LambParameterGroup>{make_two_parameter_group(first, second)});
  auto resumed = std::make_shared<LambOptimizerAdapter>(
      "optimizer.lamb",
      std::vector<LambParameterGroup>{
          make_two_parameter_group(resumed_first, resumed_second)});
  LambOptimizerCheckpointParticipant uninterrupted_state(uninterrupted);
  LambOptimizerCheckpointParticipant resumed_state(resumed);
  const auto assign_pair = [](const Tensor& first_parameter,
                              const Tensor& second_parameter,
                              std::int64_t step) {
    if (step % 4 != 0) {
      assign_gradient(first_parameter, step, 0.2);
    } else {
      first_parameter.mutable_grad() = Tensor();
    }
    if (step % 7 != 0) {
      assign_gradient(second_parameter, step, -0.3);
    } else {
      second_parameter.mutable_grad() = Tensor();
    }
  };

  for (std::int64_t step = 0; step < 37; ++step) {
    assign_pair(first, second, step);
    resumed_first.mutable_grad() =
        first.grad().defined() ? first.grad().clone() : Tensor();
    resumed_second.mutable_grad() =
        second.grad().defined() ? second.grad().clone() : Tensor();
    uninterrupted->step();
    resumed->step();
  }
  MemoryStateStore store(uninterrupted_state.describe_state());
  uninterrupted_state.save_state(store);
  resumed->set_learning_rates({0.5});
  resumed_first.mutable_grad() = torch::full_like(resumed_first, 4.0);
  resumed_second.mutable_grad() = torch::full_like(resumed_second, -3.0);
  resumed->step();
  {
    torch::NoGradGuard no_grad;
    resumed_first.copy_(first);
    resumed_second.copy_(second);
  }
  resumed_state.load_state(store, jyppx::RestorePolicy::exact_only);

  for (std::int64_t step = 37; step < 91; ++step) {
    assign_pair(first, second, step);
    resumed_first.mutable_grad() =
        first.grad().defined() ? first.grad().clone() : Tensor();
    resumed_second.mutable_grad() =
        second.grad().defined() ? second.grad().clone() : Tensor();
    uninterrupted->step();
    resumed->step();
  }
  EXPECT_TRUE(torch::equal(first, resumed_first));
  EXPECT_TRUE(torch::equal(second, resumed_second));
  EXPECT_EQ(uninterrupted->completed_steps(), resumed->completed_steps());
  EXPECT_EQ(uninterrupted->learning_rates(), resumed->learning_rates());
}

TEST(LambOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  auto optimizer = std::make_shared<LambOptimizerAdapter>(
      "optimizer.lamb",
      std::vector<LambParameterGroup>{make_group("main", "weight", parameter)});
  LambOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter)});
  LambOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  const auto before_parameter = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();

  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("exp_avg_sq.0000000000", {std::byte{0x01}});
  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before_parameter));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  assign_gradient(parameter, 2);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  EXPECT_TRUE(torch::equal(parameter, reference_parameter));
}

TEST(LambOptimizerAdapterTest, FreshCheckpointUsesLazyEmptyState) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<LambOptimizerAdapter>(
      "optimizer.lamb",
      std::vector<LambParameterGroup>{make_group("main", "weight", parameter)});
  LambOptimizerCheckpointParticipant participant(optimizer);
  MemoryStateStore store(participant.describe_state());

  participant.save_state(store);

  EXPECT_TRUE(store.contains("exp_avg.0000000000"));
  EXPECT_TRUE(store.contains("exp_avg_sq.0000000000"));
  EXPECT_TRUE(store.read_blob("exp_avg.0000000000", 1024).empty());
  EXPECT_TRUE(store.read_blob("exp_avg_sq.0000000000", 1024).empty());
}

TEST(LambOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<LambOptimizerAdapter>(
      "optimizer.lamb",
      std::vector<LambParameterGroup>{make_group("main", "weight", parameter)});
  LambOptimizerCheckpointParticipant participant(optimizer);

  parameter.set_data(parameter.to(torch::kFloat32));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(), jyppx::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(LambOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  LambOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 0.005})});
  cpu.mutable_grad() = torch::ones_like(cpu);
  EXPECT_NO_THROW(cpu_optimizer.step());
  EXPECT_TRUE(torch::isfinite(cpu).all().item<bool>());

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto cuda = make_parameter({8}, dtype, torch::Device(torch::kCUDA, 0));
    LambOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 0.005})});
    cuda.mutable_grad() = torch::ones_like(cuda);
    EXPECT_NO_THROW(cuda_optimizer.step());
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_TRUE(torch::isfinite(cuda).all().item<bool>());
  }
}

TEST(LambOptimizerAdapterTest, ValidatesOptionsIdentityAndRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(LambOptimizerAdapter(
                   "optimizer.lamb",
                   {make_group("main", "weight", parameter, {.beta1 = 1.0})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(LambOptimizerAdapter(
                   "optimizer.lamb",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(LambOptimizerAdapter(
                   "optimizer.lamb",
                   {make_group("main", "weight", parameter,
                               {.weight_decay =
                                    std::numeric_limits<double>::infinity()})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(LambOptimizerAdapter(
                   "optimizer.lamb",
                   {{.group_id = "main",
                     .parameters = {{.parameter_id = "duplicate",
                                     .parameter = parameter},
                                    {.parameter_id = "duplicate",
                                     .parameter = make_parameter({2})}}}}),
               jyppx::InvalidArgument);

  LambOptimizerAdapter optimizer("optimizer.lamb",
                                 {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}), jyppx::InvalidArgument);
  EXPECT_THROW(optimizer.set_learning_rates({0.1, 0.2}),
               jyppx::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
