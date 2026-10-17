// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/adam_adapter.hpp>

#include <torch/optim/adam.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using jyppx::StateManifest;
using jyppx::Tensor;
using jyppx::optim::AdamAdapterOptions;
using jyppx::optim::AdamOptimizerAdapter;
using jyppx::optim::AdamOptimizerCheckpointParticipant;
using jyppx::optim::AdamParameterGroup;

/** @brief In-memory participant store used to exercise storage-neutral I/O. */
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

Tensor make_parameter(std::initializer_list<std::int64_t> shape,
                      c10::ScalarType dtype = torch::kFloat64,
                      c10::Device device = torch::kCPU) {
  return torch::randn(
      shape,
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
}

AdamParameterGroup make_group(std::string group_id, std::string parameter_id,
                              const Tensor& parameter,
                              AdamAdapterOptions options = {}) {
  return {.group_id = std::move(group_id),
          .parameters = {{.parameter_id = std::move(parameter_id),
                          .parameter = parameter}},
          .options = options};
}

void assign_gradient(const Tensor& parameter, std::int64_t step) {
  parameter.mutable_grad() =
      torch::sin(parameter.detach() * 0.17 + static_cast<double>(step) * 0.03);
}

TEST(AdamOptimizerAdapterTest, MatchesLibTorchExactlyForLongRunAndGroups) {
  torch::manual_seed(20261017);
  auto actual_first = make_parameter({7, 5});
  auto actual_second = make_parameter({5});
  auto expected_first = actual_first.detach().clone().set_requires_grad(true);
  auto expected_second = actual_second.detach().clone().set_requires_grad(true);

  AdamOptimizerAdapter actual("optimizer.adam",
                              {make_group("decay", "weight", actual_first,
                                          {.learning_rate = 3e-3,
                                           .beta1 = 0.83,
                                           .beta2 = 0.97,
                                           .epsilon = 1e-7,
                                           .weight_decay = 0.02,
                                           .amsgrad = true}),
                               make_group("bias", "bias", actual_second,
                                          {.learning_rate = 7e-4,
                                           .beta1 = 0.91,
                                           .beta2 = 0.995,
                                           .epsilon = 1e-9})});
  std::vector<torch::optim::OptimizerParamGroup> native_groups;
  native_groups.emplace_back(std::vector<Tensor>{expected_first});
  native_groups.emplace_back(std::vector<Tensor>{expected_second});
  torch::optim::Adam expected(native_groups, torch::optim::AdamOptions(3e-3));
  expected.param_groups()[0].set_options(
      std::make_unique<torch::optim::AdamOptions>(
          torch::optim::AdamOptions(3e-3)
              .betas({0.83, 0.97})
              .eps(1e-7)
              .weight_decay(0.02)
              .amsgrad(true)));
  expected.param_groups()[1].set_options(
      std::make_unique<torch::optim::AdamOptions>(
          torch::optim::AdamOptions(7e-4).betas({0.91, 0.995}).eps(1e-9)));

  for (std::int64_t step = 0; step < 512; ++step) {
    assign_gradient(actual_first, step);
    assign_gradient(actual_second, step + 11);
    expected_first.mutable_grad() = actual_first.grad().clone();
    expected_second.mutable_grad() = actual_second.grad().clone();
    actual.step();
    expected.step();
  }

  EXPECT_TRUE(torch::equal(actual_first, expected_first));
  EXPECT_TRUE(torch::equal(actual_second, expected_second));
  EXPECT_EQ(actual.completed_steps(), 512U);
  EXPECT_EQ(actual.learning_rates(), (std::vector<double>{3e-3, 7e-4}));
}

TEST(AdamOptimizerAdapterTest, RejectsSparseGradientBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  AdamOptimizerAdapter optimizer(
      "optimizer.adam",
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

TEST(AdamOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  AdamOptimizerAdapter optimizer("optimizer.adam",
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

TEST(AdamOptimizerAdapterTest, CheckpointResumesExactlyWithMissingGradients) {
  torch::manual_seed(81);
  auto uninterrupted = make_parameter({6, 3});
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  auto first = std::make_shared<AdamOptimizerAdapter>(
      "optimizer.adam", std::vector<AdamParameterGroup>{make_group(
                            "main", "weight", uninterrupted,
                            {.learning_rate = 2e-3, .amsgrad = true})});
  auto second = std::make_shared<AdamOptimizerAdapter>(
      "optimizer.adam", std::vector<AdamParameterGroup>{make_group(
                            "main", "weight", resumed,
                            {.learning_rate = 2e-3, .amsgrad = true})});
  AdamOptimizerCheckpointParticipant first_state(first);
  AdamOptimizerCheckpointParticipant second_state(second);

  for (std::int64_t step = 0; step < 37; ++step) {
    if (step % 5 != 0) {
      assign_gradient(uninterrupted, step);
      resumed.mutable_grad() = uninterrupted.grad().clone();
    } else {
      uninterrupted.mutable_grad() = torch::Tensor();
      resumed.mutable_grad() = torch::Tensor();
    }
    first->step();
    second->step();
  }
  MemoryStateStore store(first_state.describe_state());
  first_state.save_state(store);
  second->set_learning_rates({0.5});
  for (std::int64_t step = 0; step < 3; ++step) {
    resumed.mutable_grad() = torch::full_like(resumed, 7.0 + step);
    second->step();
  }
  {
    torch::NoGradGuard no_grad;
    resumed.copy_(uninterrupted);
  }
  second_state.load_state(store, jyppx::RestorePolicy::exact_only);

  for (std::int64_t step = 37; step < 91; ++step) {
    assign_gradient(uninterrupted, step);
    resumed.mutable_grad() = uninterrupted.grad().clone();
    first->step();
    second->step();
  }
  EXPECT_TRUE(torch::equal(uninterrupted, resumed));
  EXPECT_EQ(first->completed_steps(), second->completed_steps());
}

TEST(AdamOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  auto optimizer = std::make_shared<AdamOptimizerAdapter>(
      "optimizer.adam",
      std::vector<AdamParameterGroup>{make_group("main", "weight", parameter)});
  AdamOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter)});
  AdamOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  const auto before_parameter = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();

  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("exp_avg_sq.0000000000", {std::byte{0x01}});
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

TEST(AdamOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  AdamOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 1e-3})});
  cpu.mutable_grad() = torch::ones_like(cpu);
  EXPECT_NO_THROW(cpu_optimizer.step());

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto cuda = make_parameter({8}, dtype, torch::Device(torch::kCUDA, 0));
    AdamOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 1e-3})});
    cuda.mutable_grad() = torch::ones_like(cuda);
    EXPECT_NO_THROW(cuda_optimizer.step());
    EXPECT_TRUE(cuda.is_cuda());
  }
}

TEST(AdamOptimizerAdapterTest, ValidatesOptionsAndLearningRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(AdamOptimizerAdapter(
                   "optimizer.adam",
                   {make_group("main", "weight", parameter, {.beta1 = 1.0})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(AdamOptimizerAdapter(
                   "optimizer.adam",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::InvalidArgument);
  AdamOptimizerAdapter optimizer("optimizer.adam",
                                 {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}), jyppx::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
