// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/adamw_adapter.hpp>
#include <jyppx/optim/rmsprop_adapter.hpp>

#include <torch/optim/rmsprop.h>
#include <torch/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <initializer_list>
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
using jyppx::optim::RmspropAdapterOptions;
using jyppx::optim::RmspropOptimizerAdapter;
using jyppx::optim::RmspropOptimizerCheckpointOptions;
using jyppx::optim::RmspropOptimizerCheckpointParticipant;
using jyppx::optim::RmspropParameterGroup;

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

  /** @brief Returns the stored byte count for one field. */
  std::size_t blob_size(std::string_view name) const {
    const auto iterator = blobs_.find(std::string(name));
    if (iterator == blobs_.end()) {
      throw std::runtime_error("missing in-memory state field");
    }
    return iterator->second.size();
  }

  /** @brief Replaces one field to construct malformed-state test cases. */
  void replace(std::string name, std::vector<std::byte> bytes) {
    blobs_[std::move(name)] = std::move(bytes);
  }

  /** @brief Removes one field to exercise required-field migration checks. */
  void erase(std::string_view name) { blobs_.erase(std::string(name)); }

  /** @brief Replaces the stored manifest without changing owned field bytes. */
  void set_manifest(StateManifest manifest) { manifest_ = std::move(manifest); }

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

RmspropParameterGroup make_group(std::string group_id, std::string parameter_id,
                                 const Tensor& parameter,
                                 RmspropAdapterOptions options = {}) {
  return {.group_id = std::move(group_id),
          .parameters = {{.parameter_id = std::move(parameter_id),
                          .parameter = parameter}},
          .options = options};
}

void assign_gradient(const Tensor& parameter, std::int64_t step) {
  parameter.mutable_grad() =
      torch::sin(parameter.detach() * 0.19 + static_cast<double>(step) * 0.04);
}

StateManifest make_schema_v1_manifest(const StateManifest& current) {
  StateManifest legacy = current;
  legacy.schema_version = 1;
  const auto producer =
      std::find_if(legacy.fields.begin(), legacy.fields.end(),
                   [](const jyppx::StateFieldDescriptor& field) {
                     return field.name == "producer";
                   });
  EXPECT_NE(producer, legacy.fields.end());
  if (producer != legacy.fields.end()) {
    legacy.fields.erase(producer);
  }
  return legacy;
}

void copy_parameter(const Tensor& source, Tensor& destination) {
  torch::NoGradGuard no_grad;
  destination.copy_(source);
}

std::vector<std::byte> read_base64_fixture(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string encoded;
  std::string trailing;
  if (!(input >> encoded) || (input >> trailing) || encoded.empty() ||
      encoded.size() % 4 != 0) {
    throw std::runtime_error("invalid RMSprop checkpoint fixture: " +
                             path.string());
  }
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid RMSprop checkpoint fixture byte: " +
                               path.string());
    }
    return static_cast<std::uint32_t>(position);
  };
  std::vector<std::byte> decoded;
  decoded.reserve((encoded.size() / 4) * 3);
  for (std::size_t offset = 0; offset < encoded.size(); offset += 4) {
    const auto final_group = offset + 4 == encoded.size();
    const auto third_padding = encoded[offset + 2] == '=';
    const auto fourth_padding = encoded[offset + 3] == '=';
    if (encoded[offset] == '=' || encoded[offset + 1] == '=' ||
        (!final_group && (third_padding || fourth_padding)) ||
        (third_padding && !fourth_padding)) {
      throw std::runtime_error("invalid RMSprop checkpoint fixture padding: " +
                               path.string());
    }
    const auto packed =
        (decode(encoded[offset]) << 18U) |
        (decode(encoded[offset + 1]) << 12U) |
        (third_padding ? 0U : decode(encoded[offset + 2]) << 6U) |
        (fourth_padding ? 0U : decode(encoded[offset + 3]));
    decoded.push_back(
        std::byte{static_cast<unsigned char>((packed >> 16U) & 0xFFU)});
    if (!third_padding) {
      decoded.push_back(
          std::byte{static_cast<unsigned char>((packed >> 8U) & 0xFFU)});
    }
    if (!fourth_padding) {
      decoded.push_back(std::byte{static_cast<unsigned char>(packed & 0xFFU)});
    }
  }
  return decoded;
}

MemoryStateStore load_canonical_schema_v1_fixture(
    const RmspropOptimizerCheckpointParticipant& participant) {
  MemoryStateStore fixture(
      make_schema_v1_manifest(participant.describe_state()));
  const auto fixture_dir = std::filesystem::path(JYPPX_TEST_FIXTURE_ROOT) /
                           "checkpoint" / "rmsprop_schema_v1_libtorch_2.5.1";
  for (const auto& field : fixture.manifest().fields) {
    fixture.replace(field.name, read_base64_fixture(fixture_dir /
                                                    (field.name + ".bin.b64")));
  }
  return fixture;
}

TEST(RmspropOptimizerAdapterTest, MatchesLibTorchExactlyForLongRunAndGroups) {
  torch::manual_seed(20261018);
  auto actual_first = make_parameter({7, 5});
  auto actual_second = make_parameter({5});
  auto expected_first = actual_first.detach().clone().set_requires_grad(true);
  auto expected_second = actual_second.detach().clone().set_requires_grad(true);

  RmspropOptimizerAdapter actual(
      "optimizer.rmsprop",
      {make_group("centered", "weight", actual_first,
                  {.learning_rate = 3e-3,
                   .alpha = 0.93,
                   .epsilon = 1e-7,
                   .weight_decay = 0.02,
                   .momentum = 0.7,
                   .centered = true}),
       make_group("plain", "bias", actual_second,
                  {.learning_rate = 7e-4, .alpha = 0.995, .epsilon = 1e-9})});
  std::vector<torch::optim::OptimizerParamGroup> native_groups;
  native_groups.emplace_back(std::vector<Tensor>{expected_first});
  native_groups.emplace_back(std::vector<Tensor>{expected_second});
  torch::optim::RMSprop expected(native_groups,
                                 torch::optim::RMSpropOptions(3e-3));
  expected.param_groups()[0].set_options(
      std::make_unique<torch::optim::RMSpropOptions>(
          torch::optim::RMSpropOptions(3e-3)
              .alpha(0.93)
              .eps(1e-7)
              .weight_decay(0.02)
              .momentum(0.7)
              .centered(true)));
  expected.param_groups()[1].set_options(
      std::make_unique<torch::optim::RMSpropOptions>(
          torch::optim::RMSpropOptions(7e-4).alpha(0.995).eps(1e-9)));

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

TEST(RmspropOptimizerAdapterTest, RejectsSparseGradientBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  RmspropOptimizerAdapter optimizer(
      "optimizer.rmsprop",
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

TEST(RmspropOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  RmspropOptimizerAdapter optimizer("optimizer.rmsprop",
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

TEST(RmspropOptimizerAdapterTest,
     CheckpointResumesExactlyWithMissingGradients) {
  torch::manual_seed(83);
  auto uninterrupted = make_parameter({6, 3});
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const RmspropAdapterOptions options{
      .learning_rate = 2e-3, .alpha = 0.95, .momentum = 0.8, .centered = true};
  auto first = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{make_group(
                               "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{
                               make_group("main", "weight", resumed, options)});
  RmspropOptimizerCheckpointParticipant first_state(first);
  RmspropOptimizerCheckpointParticipant second_state(second);

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

TEST(RmspropOptimizerAdapterTest, PersistsOnlyEnabledOptionalState) {
  const std::vector<RmspropAdapterOptions> option_matrix = {
      {},
      {.momentum = 0.7},
      {.centered = true},
      {.momentum = 0.7, .centered = true},
  };
  for (std::size_t index = 0; index < option_matrix.size(); ++index) {
    auto parameter = make_parameter({4});
    auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
        "optimizer.rmsprop." + std::to_string(index),
        std::vector<RmspropParameterGroup>{
            make_group("main", "weight", parameter, option_matrix[index])});
    RmspropOptimizerCheckpointParticipant participant(optimizer);
    parameter.mutable_grad() = torch::ones_like(parameter);
    optimizer->step();
    MemoryStateStore store(participant.describe_state());
    participant.save_state(store);

    EXPECT_GT(store.blob_size("square_avg.0000000000"), 0U);
    EXPECT_EQ(store.blob_size("momentum_buffer.0000000000") > 0,
              option_matrix[index].momentum != 0.0);
    EXPECT_EQ(store.blob_size("grad_avg.0000000000") > 0,
              option_matrix[index].centered);
  }
}

TEST(RmspropOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  const RmspropAdapterOptions options{.momentum = 0.8, .centered = true};
  auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{make_group(
                               "main", "weight", parameter, options)});
  RmspropOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter, options)});
  RmspropOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  const auto before_parameter = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();

  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("momentum_buffer.0000000000", {std::byte{0x01}});
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

TEST(RmspropOptimizerAdapterTest,
     MigratesSchemaV1WithoutProducerAndContinuesExactly) {
  torch::manual_seed(20261019);
  auto uninterrupted = make_parameter({6, 3});
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const RmspropAdapterOptions options{.learning_rate = 2e-3,
                                      .alpha = 0.94,
                                      .epsilon = 1e-7,
                                      .weight_decay = 0.01,
                                      .momentum = 0.75,
                                      .centered = true};
  auto first = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{make_group(
                               "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{
                               make_group("main", "weight", resumed, options)});
  RmspropOptimizerCheckpointParticipant first_state(first);
  RmspropOptimizerCheckpointParticipant second_state(second);

  for (std::int64_t step = 0; step < 19; ++step) {
    assign_gradient(uninterrupted, step);
    first->step();
  }
  MemoryStateStore store(first_state.describe_state());
  first_state.save_state(store);
  const auto current_manifest = store.manifest();
  store.erase("producer");
  store.set_manifest(make_schema_v1_manifest(current_manifest));

  copy_parameter(uninterrupted, resumed);
  second_state.load_state(store, jyppx::RestorePolicy::exact_only);
  EXPECT_EQ(second->completed_steps(), first->completed_steps());
  EXPECT_EQ(second->learning_rates(), first->learning_rates());

  for (std::int64_t step = 19; step < 57; ++step) {
    assign_gradient(uninterrupted, step);
    resumed.mutable_grad() = uninterrupted.grad().clone();
    first->step();
    second->step();
  }
  EXPECT_TRUE(torch::equal(uninterrupted, resumed));
  EXPECT_EQ(second->completed_steps(), first->completed_steps());
}

TEST(RmspropOptimizerAdapterTest,
     MigratesCanonicalSchemaV1FixtureFromLibtorch251) {
  if (TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 5 ||
      TORCH_VERSION_PATCH != 1) {
    GTEST_SKIP() << "canonical fixture requires exact LibTorch 2.5.1";
  }
  const RmspropAdapterOptions options{.learning_rate = 0.012,
                                      .alpha = 0.91,
                                      .epsilon = 1e-7,
                                      .weight_decay = 0.03,
                                      .momentum = 0.65,
                                      .centered = true};
  auto expected_parameter = torch::tensor(
      {1.25, -0.5, 0.75},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto expected = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop.fixture",
      std::vector<RmspropParameterGroup>{
          make_group("main", "weight", expected_parameter, options)});
  for (std::int64_t step = 1; step <= 2; ++step) {
    expected_parameter.mutable_grad() =
        torch::tensor({0.2 * step, -0.1 * step, 0.05 * step}, torch::kFloat64);
    expected->step();
  }
  expected->set_learning_rates({0.004});

  auto resumed_parameter =
      expected_parameter.detach().clone().set_requires_grad(true);
  auto resumed = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop.fixture",
      std::vector<RmspropParameterGroup>{make_group("main", "weight",
                                                    resumed_parameter,
                                                    {.learning_rate = 0.9,
                                                     .alpha = 0.91,
                                                     .epsilon = 1e-7,
                                                     .weight_decay = 0.03,
                                                     .momentum = 0.65,
                                                     .centered = true})});
  RmspropOptimizerCheckpointParticipant resumed_participant(resumed);
  const auto fixture = load_canonical_schema_v1_fixture(resumed_participant);
  resumed_participant.load_state(fixture, jyppx::RestorePolicy::exact_only);
  EXPECT_EQ(resumed->completed_steps(), 2U);
  EXPECT_EQ(resumed->learning_rates(), (std::vector<double>{0.004}));

  const auto next_gradient =
      torch::tensor({-0.25, 0.6, -0.15}, torch::kFloat64);
  expected_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  expected->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(expected_parameter, resumed_parameter, 1e-12, 1e-12));
}

TEST(RmspropOptimizerAdapterTest, RejectsFutureSchemaAndRelabelledV1) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop",
      std::vector<RmspropParameterGroup>{make_group(
          "main", "weight", parameter, {.momentum = 0.7, .centered = true})});
  RmspropOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  optimizer->step();
  const auto before = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);

  auto future = store.manifest();
  future.schema_version = 3;
  store.set_manifest(future);
  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);

  auto relabelled = participant.describe_state();
  relabelled.schema_version = 1;
  store.set_manifest(relabelled);
  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(RmspropOptimizerAdapterTest,
     RejectsMalformedProducerAndLeavesStateUnchanged) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop",
      std::vector<RmspropParameterGroup>{make_group(
          "main", "weight", parameter, {.momentum = 0.7, .centered = true})});
  RmspropOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  optimizer->step();
  const auto before = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("producer", {std::byte{0x01}});

  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(RmspropOptimizerAdapterTest, EnforcesProducerResourceLimit) {
  auto parameter = make_parameter({2});
  auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{
                               make_group("main", "weight", parameter)});
  RmspropOptimizerCheckpointOptions options;
  options.max_producer_bytes = 1;
  EXPECT_THROW(RmspropOptimizerCheckpointParticipant(optimizer, options),
               jyppx::ResourceExhausted);
}

TEST(RmspropOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{
                               make_group("main", "weight", parameter)});
  RmspropOptimizerCheckpointParticipant participant(optimizer);

  parameter.set_data(parameter.to(torch::kFloat32));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(), jyppx::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(RmspropOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  auto expected_cpu = cpu.detach().clone().set_requires_grad(true);
  RmspropOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 1e-3})});
  torch::optim::RMSprop native_cpu({expected_cpu},
                                   torch::optim::RMSpropOptions(1e-3));
  cpu.mutable_grad() = torch::ones_like(cpu);
  expected_cpu.mutable_grad() = cpu.grad().clone();
  cpu_optimizer.step();
  native_cpu.step();
  EXPECT_TRUE(torch::equal(cpu, expected_cpu));

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto cuda = make_parameter({8}, dtype, torch::Device(torch::kCUDA, 0));
    auto expected_cuda = cuda.detach().clone().set_requires_grad(true);
    RmspropOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 1e-3})});
    torch::optim::RMSprop native_cuda({expected_cuda},
                                      torch::optim::RMSpropOptions(1e-3));
    cuda.mutable_grad() = torch::ones_like(cuda);
    expected_cuda.mutable_grad() = cuda.grad().clone();
    cuda_optimizer.step();
    native_cuda.step();
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_TRUE(torch::equal(cuda, expected_cuda));
  }
}

TEST(RmspropOptimizerAdapterCudaTest,
     MigratesSchemaV1StateForExactGpuContinuation) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto uninterrupted =
      make_parameter({8}, torch::kFloat32, torch::Device(torch::kCUDA, 0));
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const RmspropAdapterOptions options{.learning_rate = 1e-3,
                                      .alpha = 0.95,
                                      .epsilon = 1e-7,
                                      .weight_decay = 0.015,
                                      .momentum = 0.7,
                                      .centered = true};
  auto first = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop.cuda", std::vector<RmspropParameterGroup>{make_group(
                                    "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop.cuda", std::vector<RmspropParameterGroup>{make_group(
                                    "main", "weight", resumed, options)});
  RmspropOptimizerCheckpointParticipant first_state(first);
  RmspropOptimizerCheckpointParticipant second_state(second);
  for (std::int64_t step = 0; step < 13; ++step) {
    assign_gradient(uninterrupted, step);
    first->step();
  }
  MemoryStateStore store(first_state.describe_state());
  first_state.save_state(store);
  const auto current_manifest = store.manifest();
  store.erase("producer");
  store.set_manifest(make_schema_v1_manifest(current_manifest));
  copy_parameter(uninterrupted, resumed);
  second_state.load_state(store, jyppx::RestorePolicy::exact_only);

  for (std::int64_t step = 13; step < 29; ++step) {
    assign_gradient(uninterrupted, step);
    resumed.mutable_grad() = uninterrupted.grad().clone();
    first->step();
    second->step();
  }
  torch::cuda::synchronize();
  EXPECT_TRUE(torch::equal(uninterrupted, resumed));
  EXPECT_EQ(second->completed_steps(), first->completed_steps());
}

TEST(RmspropOptimizerAdapterTest,
     RejectsTopologyMismatchAndEnforcesResourceLimits) {
  auto source_parameter = make_parameter({5});
  auto target_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto source = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{make_group(
                               "main", "weight", source_parameter,
                               {.momentum = 0.8, .centered = true})});
  auto target = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.rmsprop", std::vector<RmspropParameterGroup>{make_group(
                               "main", "weight", target_parameter,
                               {.momentum = 0.7, .centered = false})});
  RmspropOptimizerCheckpointParticipant source_state(source);
  RmspropOptimizerCheckpointParticipant target_state(target);
  MemoryStateStore store(source_state.describe_state());
  source_state.save_state(store);

  EXPECT_THROW(target_state.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  EXPECT_THROW(
      RmspropOptimizerCheckpointParticipant(source, {.max_groups = 1,
                                                     .max_parameters = 1,
                                                     .max_identifier_bytes = 32,
                                                     .max_topology_bytes = 1024,
                                                     .max_control_bytes = 1024,
                                                     .max_tensor_bytes = 1,
                                                     .max_total_bytes = 4096}),
      jyppx::ResourceExhausted);
}

TEST(RmspropOptimizerAdapterTest, RejectsAdamWCheckpointWithMatchingIdentity) {
  auto adamw_parameter = make_parameter({5});
  auto rmsprop_parameter =
      adamw_parameter.detach().clone().set_requires_grad(true);
  auto adamw = std::make_shared<jyppx::optim::AdamWOptimizerAdapter>(
      "optimizer.shared",
      std::vector<jyppx::optim::AdamWParameterGroup>{
          {.group_id = "main",
           .parameters = {
               {.parameter_id = "weight", .parameter = adamw_parameter}}}});
  auto rmsprop = std::make_shared<RmspropOptimizerAdapter>(
      "optimizer.shared", std::vector<RmspropParameterGroup>{
                              make_group("main", "weight", rmsprop_parameter)});
  jyppx::optim::AdamWOptimizerCheckpointParticipant adamw_state(adamw);
  RmspropOptimizerCheckpointParticipant rmsprop_state(rmsprop);

  assign_gradient(adamw_parameter, 1);
  adamw->step();
  MemoryStateStore store(adamw_state.describe_state());
  adamw_state.save_state(store);
  const auto before_parameter = rmsprop_parameter.detach().clone();
  const auto before_steps = rmsprop->completed_steps();
  const auto before_rates = rmsprop->learning_rates();

  EXPECT_THROW(
      rmsprop_state.load_state(store, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  EXPECT_TRUE(torch::equal(rmsprop_parameter, before_parameter));
  EXPECT_EQ(rmsprop->completed_steps(), before_steps);
  EXPECT_EQ(rmsprop->learning_rates(), before_rates);
}

TEST(RmspropOptimizerAdapterTest, ValidatesOptionsAndLearningRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(RmspropOptimizerAdapter(
                   "optimizer.rmsprop",
                   {make_group("main", "weight", parameter, {.alpha = 1.1})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(RmspropOptimizerAdapter(
                   "optimizer.rmsprop",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(RmspropOptimizerAdapter("optimizer.rmsprop",
                                       {make_group("main", "weight", parameter,
                                                   {.momentum = -0.1})}),
               jyppx::InvalidArgument);
  EXPECT_THROW(RmspropOptimizerAdapter(
                   "optimizer.rmsprop",
                   {make_group("main", "weight", parameter,
                               {.weight_decay =
                                    std::numeric_limits<double>::infinity()})}),
               jyppx::InvalidArgument);
  RmspropOptimizerAdapter optimizer("optimizer.rmsprop",
                                    {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}), jyppx::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
