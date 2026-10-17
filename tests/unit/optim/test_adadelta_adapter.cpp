// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/adadelta_adapter.hpp>

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
using jyppx::optim::AdadeltaAdapterOptions;
using jyppx::optim::AdadeltaOptimizerAdapter;
using jyppx::optim::AdadeltaOptimizerCheckpointParticipant;
using jyppx::optim::AdadeltaParameterGroup;

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

/** @brief Scalar state for an independent Adadelta reference trajectory. */
struct ScalarAdadeltaState {
  double parameter = 0.0;   ///< Current scalar parameter value.
  double square_avg = 0.0;  ///< Running average of squared gradients.
  double acc_delta = 0.0;   ///< Running average of squared parameter deltas.
  std::int64_t step = 0;    ///< Number of updates with a defined gradient.
};

Tensor make_parameter(std::initializer_list<std::int64_t> shape,
                      c10::ScalarType dtype = torch::kFloat64,
                      c10::Device device = torch::kCPU) {
  return torch::randn(
      shape,
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
}

AdadeltaParameterGroup make_group(std::string group_id,
                                  std::string parameter_id,
                                  const Tensor& parameter,
                                  AdadeltaAdapterOptions options = {}) {
  return {.group_id = std::move(group_id),
          .parameters = {{.parameter_id = std::move(parameter_id),
                          .parameter = parameter}},
          .options = options};
}

void assign_gradient(const Tensor& parameter, std::int64_t step,
                     double phase = 0.0) {
  parameter.mutable_grad() = torch::sin(
      parameter.detach() * 0.17 + static_cast<double>(step) * 0.03 + phase);
}

void update_reference(ScalarAdadeltaState& state, double gradient,
                      const AdadeltaAdapterOptions& options) {
  ++state.step;
  if (options.maximize) {
    gradient = -gradient;
  }
  if (options.weight_decay != 0.0) {
    gradient += options.weight_decay * state.parameter;
  }
  state.square_avg = state.square_avg * options.rho +
                     gradient * gradient * (1.0 - options.rho);
  const auto delta = std::sqrt(state.acc_delta + options.epsilon) /
                     std::sqrt(state.square_avg + options.epsilon) * gradient;
  state.acc_delta =
      state.acc_delta * options.rho + delta * delta * (1.0 - options.rho);
  state.parameter -= options.learning_rate * delta;
}

TEST(AdadeltaOptimizerAdapterTest,
     MatchesIndependentLongRunReferenceAndGroups) {
  const AdadeltaAdapterOptions regularized{.learning_rate = 0.75,
                                           .rho = 0.87,
                                           .epsilon = 2e-7,
                                           .weight_decay = 0.025};
  const AdadeltaAdapterOptions maximizing{.learning_rate = 1.2,
                                          .rho = 0.96,
                                          .epsilon = 3e-6,
                                          .weight_decay = 0.015,
                                          .maximize = true};
  auto first = torch::tensor({0.75, -1.25, 2.0, -0.125},
                             torch::TensorOptions().dtype(torch::kFloat64))
                   .set_requires_grad(true);
  auto second = torch::tensor({-0.5, 1.5, 0.25},
                              torch::TensorOptions().dtype(torch::kFloat64))
                    .set_requires_grad(true);
  AdadeltaOptimizerAdapter optimizer(
      "optimizer.adadelta",
      {make_group("regularized", "weight", first, regularized),
       make_group("maximizing", "bias", second, maximizing)});
  std::vector<ScalarAdadeltaState> first_reference;
  std::vector<ScalarAdadeltaState> second_reference;
  const auto first_values = first.detach().contiguous();
  const auto second_values = second.detach().contiguous();
  for (std::int64_t index = 0; index < first_values.numel(); ++index) {
    first_reference.push_back(
        {.parameter = first_values.data_ptr<double>()[index]});
  }
  for (std::int64_t index = 0; index < second_values.numel(); ++index) {
    second_reference.push_back(
        {.parameter = second_values.data_ptr<double>()[index]});
  }

  for (std::int64_t step = 0; step < 512; ++step) {
    auto first_gradient = torch::empty_like(first);
    auto second_gradient = torch::empty_like(second);
    for (std::size_t index = 0; index < first_reference.size(); ++index) {
      const auto gradient = std::sin((static_cast<double>(step) + 1.0) * 0.031 +
                                     (static_cast<double>(index) + 1.0) * 0.17);
      first_gradient[static_cast<std::int64_t>(index)] = gradient;
      update_reference(first_reference[index], gradient, regularized);
    }
    for (std::size_t index = 0; index < second_reference.size(); ++index) {
      const auto gradient = std::cos((static_cast<double>(step) + 1.0) * 0.019 +
                                     (static_cast<double>(index) + 1.0) * 0.23);
      second_gradient[static_cast<std::int64_t>(index)] = gradient;
      update_reference(second_reference[index], gradient, maximizing);
    }
    first.mutable_grad() = std::move(first_gradient);
    second.mutable_grad() = std::move(second_gradient);
    optimizer.step();
  }

  for (std::size_t index = 0; index < first_reference.size(); ++index) {
    EXPECT_NEAR(first[static_cast<std::int64_t>(index)].item<double>(),
                first_reference[index].parameter, 2e-11);
  }
  for (std::size_t index = 0; index < second_reference.size(); ++index) {
    EXPECT_NEAR(second[static_cast<std::int64_t>(index)].item<double>(),
                second_reference[index].parameter, 2e-11);
  }
  EXPECT_EQ(optimizer.completed_steps(), 512U);
  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{0.75, 1.2}));
}

TEST(AdadeltaOptimizerAdapterTest, CoupledWeightDecayAffectsZeroGradient) {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const AdadeltaAdapterOptions options{
      .learning_rate = 1.0, .rho = 0.9, .epsilon = 1e-6, .weight_decay = 0.1};
  AdadeltaOptimizerAdapter optimizer(
      "optimizer.adadelta", {make_group("main", "weight", parameter, options)});
  ScalarAdadeltaState expected{.parameter = 1.0};
  update_reference(expected, 0.0, options);
  parameter.mutable_grad() = torch::zeros_like(parameter);

  optimizer.step();

  EXPECT_NEAR(parameter.item<double>(), expected.parameter, 1e-14);
  EXPECT_LT(parameter.item<double>(), 1.0);
}

TEST(AdadeltaOptimizerAdapterTest, RejectsSparseGradientsBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  AdadeltaOptimizerAdapter optimizer(
      "optimizer.adadelta",
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

TEST(AdadeltaOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  AdadeltaOptimizerAdapter optimizer("optimizer.adadelta",
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

TEST(AdadeltaOptimizerAdapterTest, CheckpointResumesBothAccumulatorsExactly) {
  torch::manual_seed(20261018);
  auto first = make_parameter({6});
  auto second = make_parameter({4});
  auto resumed_first = first.detach().clone().set_requires_grad(true);
  auto resumed_second = second.detach().clone().set_requires_grad(true);
  auto make_two_group = [](const Tensor& first_parameter,
                           const Tensor& second_parameter) {
    return AdadeltaParameterGroup{
        .group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first_parameter},
                       {.parameter_id = "second",
                        .parameter = second_parameter}},
        .options = {.learning_rate = 0.8,
                    .rho = 0.93,
                    .epsilon = 4e-7,
                    .weight_decay = 0.01}};
  };
  auto uninterrupted = std::make_shared<AdadeltaOptimizerAdapter>(
      "optimizer.adadelta",
      std::vector<AdadeltaParameterGroup>{make_two_group(first, second)});
  auto resumed = std::make_shared<AdadeltaOptimizerAdapter>(
      "optimizer.adadelta", std::vector<AdadeltaParameterGroup>{
                                make_two_group(resumed_first, resumed_second)});
  AdadeltaOptimizerCheckpointParticipant uninterrupted_state(uninterrupted);
  AdadeltaOptimizerCheckpointParticipant resumed_state(resumed);

  auto assign_pair = [](const Tensor& first_parameter,
                        const Tensor& second_parameter, std::int64_t step) {
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

TEST(AdadeltaOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  auto optimizer = std::make_shared<AdadeltaOptimizerAdapter>(
      "optimizer.adadelta", std::vector<AdadeltaParameterGroup>{
                                make_group("main", "weight", parameter)});
  AdadeltaOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter)});
  AdadeltaOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  const auto before_parameter = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();

  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("acc_delta.0000000000", {std::byte{0x01}});
  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before_parameter));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
  assign_gradient(parameter, 2);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  EXPECT_TRUE(torch::equal(parameter, reference_parameter));
}

TEST(AdadeltaOptimizerAdapterTest, FreshCheckpointUsesLazyEmptyState) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<AdadeltaOptimizerAdapter>(
      "optimizer.adadelta", std::vector<AdadeltaParameterGroup>{
                                make_group("main", "weight", parameter)});
  AdadeltaOptimizerCheckpointParticipant participant(optimizer);
  MemoryStateStore store(participant.describe_state());

  participant.save_state(store);

  EXPECT_TRUE(store.contains("square_avg.0000000000"));
  EXPECT_TRUE(store.contains("acc_delta.0000000000"));
  EXPECT_TRUE(store.read_blob("square_avg.0000000000", 1024).empty());
  EXPECT_TRUE(store.read_blob("acc_delta.0000000000", 1024).empty());
}

TEST(AdadeltaOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<AdadeltaOptimizerAdapter>(
      "optimizer.adadelta", std::vector<AdadeltaParameterGroup>{
                                make_group("main", "weight", parameter)});
  AdadeltaOptimizerCheckpointParticipant participant(optimizer);

  parameter.set_data(parameter.to(torch::kFloat32));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(), jyppx::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(AdadeltaOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  AdadeltaOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 0.5})});
  cpu.mutable_grad() = torch::ones_like(cpu);
  EXPECT_NO_THROW(cpu_optimizer.step());
  EXPECT_TRUE(torch::isfinite(cpu).all().item<bool>());

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto cuda = make_parameter({8}, dtype, torch::Device(torch::kCUDA, 0));
    AdadeltaOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 0.5})});
    cuda.mutable_grad() = torch::ones_like(cuda);
    EXPECT_NO_THROW(cuda_optimizer.step());
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_TRUE(torch::isfinite(cuda).all().item<bool>());
  }
}

TEST(AdadeltaOptimizerAdapterTest, ValidatesOptionsIdentityAndRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(AdadeltaOptimizerAdapter(
                   "optimizer.adadelta",
                   {make_group("main", "weight", parameter, {.rho = 1.1})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(AdadeltaOptimizerAdapter(
                   "optimizer.adadelta",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(AdadeltaOptimizerAdapter(
                   "optimizer.adadelta",
                   {make_group("main", "weight", parameter,
                               {.weight_decay =
                                    std::numeric_limits<double>::infinity()})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(AdadeltaOptimizerAdapter(
                   "optimizer.adadelta",
                   {{.group_id = "main",
                     .parameters = {{.parameter_id = "duplicate",
                                     .parameter = parameter},
                                    {.parameter_id = "duplicate",
                                     .parameter = make_parameter({2})}}}}),
               jyppx::InvalidArgument);

  AdadeltaOptimizerAdapter optimizer("optimizer.adadelta",
                                     {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}), jyppx::InvalidArgument);
  EXPECT_THROW(optimizer.set_learning_rates({0.1, 0.2}),
               jyppx::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
