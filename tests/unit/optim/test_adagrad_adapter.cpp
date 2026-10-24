// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/adagrad_adapter.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>

#include <torch/optim/adagrad.h>
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

using jyppx::liuding::StateManifest;
using jyppx::liuding::Tensor;
using jyppx::liuding::optim::AdagradAdapterOptions;
using jyppx::liuding::optim::AdagradOptimizerAdapter;
using jyppx::liuding::optim::AdagradOptimizerCheckpointOptions;
using jyppx::liuding::optim::AdagradOptimizerCheckpointParticipant;
using jyppx::liuding::optim::AdagradParameterGroup;

/** @brief In-memory participant store used to exercise storage-neutral I/O. */
class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
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

AdagradParameterGroup make_group(std::string group_id, std::string parameter_id,
                                 const Tensor& parameter,
                                 AdagradAdapterOptions options = {}) {
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
                   [](const jyppx::liuding::StateFieldDescriptor& field) {
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
    throw std::runtime_error("invalid Adagrad checkpoint fixture: " +
                             path.string());
  }
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid Adagrad checkpoint fixture byte: " +
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
      throw std::runtime_error("invalid Adagrad checkpoint fixture padding: " +
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
    const AdagradOptimizerCheckpointParticipant& participant) {
  MemoryStateStore fixture(
      make_schema_v1_manifest(participant.describe_state()));
  const auto fixture_dir = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                           "checkpoint" / "adagrad_schema_v1_libtorch_2.5.1";
  for (const auto& field : fixture.manifest().fields) {
    fixture.replace(field.name, read_base64_fixture(fixture_dir /
                                                    (field.name + ".bin.b64")));
  }
  return fixture;
}

TEST(AdagradOptimizerAdapterTest, MatchesLibTorchExactlyForLongRunAndGroups) {
  torch::manual_seed(20261018);
  auto actual_first = make_parameter({7, 5});
  auto actual_second = make_parameter({5});
  auto expected_first = actual_first.detach().clone().set_requires_grad(true);
  auto expected_second = actual_second.detach().clone().set_requires_grad(true);

  AdagradOptimizerAdapter actual("optimizer.adagrad",
                                 {make_group("decayed", "weight", actual_first,
                                             {.learning_rate = 3e-2,
                                              .learning_rate_decay = 2e-3,
                                              .weight_decay = 0.02,
                                              .initial_accumulator_value = 0.1,
                                              .epsilon = 1e-8}),
                                  make_group("plain", "bias", actual_second,
                                             {.learning_rate = 7e-3,
                                              .initial_accumulator_value = 0.1,
                                              .epsilon = 1e-9})});
  std::vector<torch::optim::OptimizerParamGroup> native_groups;
  native_groups.emplace_back(std::vector<Tensor>{expected_first});
  native_groups.emplace_back(std::vector<Tensor>{expected_second});
  torch::optim::Adagrad expected(
      native_groups,
      torch::optim::AdagradOptions(3e-2).initial_accumulator_value(0.1));
  expected.param_groups()[0].set_options(
      std::make_unique<torch::optim::AdagradOptions>(
          torch::optim::AdagradOptions(3e-2)
              .lr_decay(2e-3)
              .weight_decay(0.02)
              .initial_accumulator_value(0.1)
              .eps(1e-8)));
  expected.param_groups()[1].set_options(
      std::make_unique<torch::optim::AdagradOptions>(
          torch::optim::AdagradOptions(7e-3).initial_accumulator_value(0.1).eps(
              1e-9)));

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
  EXPECT_EQ(actual.learning_rates(), (std::vector<double>{3e-2, 7e-3}));
}

TEST(AdagradOptimizerAdapterTest, HonorsNonzeroInitialAccumulator) {
  auto actual = make_parameter({8});
  auto expected = actual.detach().clone().set_requires_grad(true);
  AdagradOptimizerAdapter optimizer(
      "optimizer.adagrad",
      {make_group("main", "weight", actual,
                  {.learning_rate = 0.03, .initial_accumulator_value = 0.75})});
  torch::optim::Adagrad native(
      {expected},
      torch::optim::AdagradOptions(0.03).initial_accumulator_value(0.75));
  actual.mutable_grad() = torch::full_like(actual, 0.25);
  expected.mutable_grad() = actual.grad().clone();

  optimizer.step();
  native.step();

  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(AdagradOptimizerAdapterTest, RejectsSparseGradientBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  AdagradOptimizerAdapter optimizer(
      "optimizer.adagrad",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first},
                       {.parameter_id = "second", .parameter = second}}}});
  first.mutable_grad() = torch::ones_like(first);
  second.mutable_grad() =
      torch::sparse_coo_tensor(torch::tensor({{0, 2}}, torch::kInt64),
                               torch::tensor({1.0, 1.0}, torch::kFloat64), {4});

  EXPECT_THROW(optimizer.step(), jyppx::liuding::NotImplemented);
  EXPECT_TRUE(torch::equal(first, before_first));
  EXPECT_TRUE(torch::equal(second, before_second));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(AdagradOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  AdagradOptimizerAdapter optimizer("optimizer.adagrad",
                                    {make_group("main", "weight", parameter)});

  EXPECT_THROW(optimizer.step([&parameter] {
    parameter.mutable_grad() =
        torch::sparse_coo_tensor(torch::tensor({{1}}, torch::kInt64),
                                 torch::tensor({2.0}, torch::kFloat64), {4});
    return torch::tensor(1.0);
  }),
               jyppx::liuding::NotImplemented);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(AdagradOptimizerAdapterTest,
     CheckpointResumesPerParameterStepsWithMissingGradients) {
  torch::manual_seed(83);
  auto first_weight = make_parameter({6, 3});
  auto first_bias = make_parameter({3});
  auto second_weight = first_weight.detach().clone().set_requires_grad(true);
  auto second_bias = first_bias.detach().clone().set_requires_grad(true);
  const AdagradAdapterOptions options{.learning_rate = 2e-2,
                                      .learning_rate_decay = 3e-3,
                                      .initial_accumulator_value = 0.2};
  auto first = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{
          {.group_id = "main",
           .parameters = {{.parameter_id = "weight", .parameter = first_weight},
                          {.parameter_id = "bias", .parameter = first_bias}},
           .options = options}});
  auto second = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{
          {.group_id = "main",
           .parameters = {{.parameter_id = "weight",
                           .parameter = second_weight},
                          {.parameter_id = "bias", .parameter = second_bias}},
           .options = options}});
  AdagradOptimizerCheckpointParticipant first_state(first);
  AdagradOptimizerCheckpointParticipant second_state(second);

  for (std::int64_t step = 0; step < 37; ++step) {
    if (step % 5 != 0) {
      assign_gradient(first_weight, step);
      second_weight.mutable_grad() = first_weight.grad().clone();
    } else {
      first_weight.mutable_grad() = Tensor();
      second_weight.mutable_grad() = Tensor();
    }
    if (step % 3 != 0) {
      assign_gradient(first_bias, step + 7);
      second_bias.mutable_grad() = first_bias.grad().clone();
    } else {
      first_bias.mutable_grad() = Tensor();
      second_bias.mutable_grad() = Tensor();
    }
    first->step();
    second->step();
  }
  MemoryStateStore store(first_state.describe_state());
  first_state.save_state(store);
  second->set_learning_rates({0.5});
  for (std::int64_t step = 0; step < 3; ++step) {
    second_weight.mutable_grad() = torch::full_like(second_weight, 7.0 + step);
    second_bias.mutable_grad() = torch::full_like(second_bias, 9.0 + step);
    second->step();
  }
  {
    torch::NoGradGuard no_grad;
    second_weight.copy_(first_weight);
    second_bias.copy_(first_bias);
  }
  second_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

  for (std::int64_t step = 37; step < 91; ++step) {
    assign_gradient(first_weight, step);
    assign_gradient(first_bias, step + 7);
    second_weight.mutable_grad() = first_weight.grad().clone();
    second_bias.mutable_grad() = first_bias.grad().clone();
    first->step();
    second->step();
  }
  EXPECT_TRUE(torch::equal(first_weight, second_weight));
  EXPECT_TRUE(torch::equal(first_bias, second_bias));
  EXPECT_EQ(first->completed_steps(), second->completed_steps());
}

TEST(AdagradOptimizerAdapterTest, FreshCheckpointReplacesStaleEagerState) {
  auto fresh_parameter = make_parameter({5});
  auto resumed_parameter =
      fresh_parameter.detach().clone().set_requires_grad(true);
  const AdagradAdapterOptions options{.initial_accumulator_value = 0.4};
  auto fresh = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{make_group(
                               "main", "weight", fresh_parameter, options)});
  auto resumed = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{make_group(
                               "main", "weight", resumed_parameter, options)});
  AdagradOptimizerCheckpointParticipant fresh_state(fresh);
  AdagradOptimizerCheckpointParticipant resumed_state(resumed);
  MemoryStateStore store(fresh_state.describe_state());
  fresh_state.save_state(store);

  ASSERT_EQ(store.manifest().fields.size(), 4U);
  EXPECT_EQ(store.manifest().fields[0].name, "topology");
  EXPECT_EQ(store.manifest().fields[1].name, "producer");
  EXPECT_EQ(store.manifest().fields[2].name, "control");
  EXPECT_EQ(store.manifest().fields[3].name, "sum.0000000000");
  EXPECT_EQ(store.blob_size("sum.0000000000"),
            static_cast<std::size_t>(fresh_parameter.numel() *
                                     fresh_parameter.element_size()));
  resumed_parameter.mutable_grad() = torch::ones_like(resumed_parameter);
  resumed->step();
  {
    torch::NoGradGuard no_grad;
    resumed_parameter.copy_(fresh_parameter);
  }
  resumed_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(resumed->completed_steps(), 0U);

  fresh_parameter.mutable_grad() = torch::full_like(fresh_parameter, 0.25);
  resumed_parameter.mutable_grad() = fresh_parameter.grad().clone();
  fresh->step();
  resumed->step();
  EXPECT_TRUE(torch::equal(fresh_parameter, resumed_parameter));
}

TEST(AdagradOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  const AdagradAdapterOptions options{.initial_accumulator_value = 0.2};
  auto optimizer = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{make_group(
                               "main", "weight", parameter, options)});
  AdagradOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter, options)});
  AdagradOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  const auto before_parameter = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();

  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("sum.0000000000", {std::byte{0x01}});
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before_parameter));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
  assign_gradient(parameter, 2);
  reference_parameter.mutable_grad() = parameter.grad().clone();
  optimizer->step();
  reference.step();
  EXPECT_TRUE(torch::equal(parameter, reference_parameter));
}

TEST(AdagradOptimizerAdapterTest,
     MigratesSchemaV1WithoutProducerAndContinuesExactly) {
  torch::manual_seed(20261019);
  auto uninterrupted = make_parameter({6, 3});
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const AdagradAdapterOptions options{.learning_rate = 2e-2,
                                      .learning_rate_decay = 2e-3,
                                      .weight_decay = 0.01,
                                      .initial_accumulator_value = 0.2,
                                      .epsilon = 1e-7};
  auto first = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{make_group(
                               "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{
                               make_group("main", "weight", resumed, options)});
  AdagradOptimizerCheckpointParticipant first_state(first);
  AdagradOptimizerCheckpointParticipant second_state(second);

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
  second_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
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

TEST(AdagradOptimizerAdapterTest,
     MigratesCanonicalSchemaV1FixtureFromLibtorch251) {
  if (TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 5 ||
      TORCH_VERSION_PATCH != 1) {
    GTEST_SKIP() << "canonical fixture requires exact LibTorch 2.5.1";
  }
  const AdagradAdapterOptions options{.learning_rate = 0.012,
                                      .learning_rate_decay = 0.003,
                                      .weight_decay = 0.03,
                                      .initial_accumulator_value = 0.2,
                                      .epsilon = 1e-7};
  auto expected_parameter = torch::tensor(
      {1.25, -0.5, 0.75},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto expected = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad.fixture",
      std::vector<AdagradParameterGroup>{
          make_group("main", "weight", expected_parameter, options)});
  for (std::int64_t step = 1; step <= 2; ++step) {
    expected_parameter.mutable_grad() =
        torch::tensor({0.2 * step, -0.1 * step, 0.05 * step}, torch::kFloat64);
    expected->step();
  }
  expected->set_learning_rates({0.004});

  auto resumed_parameter =
      expected_parameter.detach().clone().set_requires_grad(true);
  auto resumed = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad.fixture",
      std::vector<AdagradParameterGroup>{
          make_group("main", "weight", resumed_parameter,
                     {.learning_rate = 0.9,
                      .learning_rate_decay = 0.003,
                      .weight_decay = 0.03,
                      .initial_accumulator_value = 0.2,
                      .epsilon = 1e-7})});
  AdagradOptimizerCheckpointParticipant resumed_participant(resumed);
  const auto fixture = load_canonical_schema_v1_fixture(resumed_participant);
  resumed_participant.load_state(fixture, jyppx::liuding::RestorePolicy::exact_only);
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

TEST(AdagradOptimizerAdapterTest, RejectsFutureSchemaAndRelabelledV1) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{make_group(
          "main", "weight", parameter, {.initial_accumulator_value = 0.2})});
  AdagradOptimizerCheckpointParticipant participant(optimizer);
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
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);

  auto relabelled = participant.describe_state();
  relabelled.schema_version = 1;
  store.set_manifest(relabelled);
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(AdagradOptimizerAdapterTest,
     RejectsMalformedProducerAndLeavesStateUnchanged) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{make_group(
          "main", "weight", parameter, {.initial_accumulator_value = 0.2})});
  AdagradOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  optimizer->step();
  const auto before = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("producer", {std::byte{0x01}});

  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(AdagradOptimizerAdapterTest, EnforcesProducerResourceLimit) {
  auto parameter = make_parameter({2});
  auto optimizer = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{
                               make_group("main", "weight", parameter)});
  AdagradOptimizerCheckpointOptions options;
  options.max_producer_bytes = 1;
  EXPECT_THROW(AdagradOptimizerCheckpointParticipant(optimizer, options),
               jyppx::liuding::ResourceExhausted);
}

TEST(AdagradOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad", std::vector<AdagradParameterGroup>{
                               make_group("main", "weight", parameter)});
  AdagradOptimizerCheckpointParticipant participant(optimizer);

  parameter.set_data(parameter.to(torch::kFloat32));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(), jyppx::liuding::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(AdagradOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  auto expected_cpu = cpu.detach().clone().set_requires_grad(true);
  AdagradOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 1e-2})});
  torch::optim::Adagrad native_cpu({expected_cpu},
                                   torch::optim::AdagradOptions(1e-2));
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
    AdagradOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 1e-2})});
    torch::optim::Adagrad native_cuda({expected_cuda},
                                      torch::optim::AdagradOptions(1e-2));
    cuda.mutable_grad() = torch::ones_like(cuda);
    expected_cuda.mutable_grad() = cuda.grad().clone();
    cuda_optimizer.step();
    native_cuda.step();
    EXPECT_TRUE(cuda.is_cuda());
    EXPECT_TRUE(torch::equal(cuda, expected_cuda));
  }
}

TEST(AdagradOptimizerAdapterCudaTest,
     MigratesSchemaV1StateForExactGpuContinuation) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto uninterrupted =
      make_parameter({8}, torch::kFloat32, torch::Device(torch::kCUDA, 0));
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const AdagradAdapterOptions options{.learning_rate = 1e-2,
                                      .learning_rate_decay = 2e-3,
                                      .weight_decay = 0.015,
                                      .initial_accumulator_value = 0.2,
                                      .epsilon = 1e-7};
  auto first = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad.cuda", std::vector<AdagradParameterGroup>{make_group(
                                    "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad.cuda", std::vector<AdagradParameterGroup>{make_group(
                                    "main", "weight", resumed, options)});
  AdagradOptimizerCheckpointParticipant first_state(first);
  AdagradOptimizerCheckpointParticipant second_state(second);
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
  second_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

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

TEST(AdagradOptimizerAdapterTest,
     RejectsTopologyMismatchAndEnforcesResourceLimits) {
  auto source_parameter = make_parameter({5});
  auto target_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto source = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{make_group(
          "main", "weight", source_parameter,
          {.learning_rate_decay = 0.1, .initial_accumulator_value = 0.2})});
  auto target = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.adagrad",
      std::vector<AdagradParameterGroup>{make_group(
          "main", "weight", target_parameter,
          {.learning_rate_decay = 0.2, .initial_accumulator_value = 0.3})});
  AdagradOptimizerCheckpointParticipant source_state(source);
  AdagradOptimizerCheckpointParticipant target_state(target);
  MemoryStateStore store(source_state.describe_state());
  source_state.save_state(store);

  EXPECT_THROW(target_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_THROW(
      AdagradOptimizerCheckpointParticipant(source, {.max_groups = 1,
                                                     .max_parameters = 1,
                                                     .max_identifier_bytes = 32,
                                                     .max_topology_bytes = 1024,
                                                     .max_control_bytes = 1024,
                                                     .max_tensor_bytes = 1,
                                                     .max_total_bytes = 4096}),
      jyppx::liuding::ResourceExhausted);
}

TEST(AdagradOptimizerAdapterTest, RejectsAdamWCheckpointWithMatchingIdentity) {
  auto adamw_parameter = make_parameter({5});
  auto adagrad_parameter =
      adamw_parameter.detach().clone().set_requires_grad(true);
  auto adamw = std::make_shared<jyppx::liuding::optim::AdamWOptimizerAdapter>(
      "optimizer.shared",
      std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
          {.group_id = "main",
           .parameters = {
               {.parameter_id = "weight", .parameter = adamw_parameter}}}});
  auto adagrad = std::make_shared<AdagradOptimizerAdapter>(
      "optimizer.shared", std::vector<AdagradParameterGroup>{
                              make_group("main", "weight", adagrad_parameter)});
  jyppx::liuding::optim::AdamWOptimizerCheckpointParticipant adamw_state(adamw);
  AdagradOptimizerCheckpointParticipant adagrad_state(adagrad);

  assign_gradient(adamw_parameter, 1);
  adamw->step();
  MemoryStateStore store(adamw_state.describe_state());
  adamw_state.save_state(store);
  const auto before_parameter = adagrad_parameter.detach().clone();
  const auto before_steps = adagrad->completed_steps();
  const auto before_rates = adagrad->learning_rates();

  EXPECT_THROW(
      adagrad_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(adagrad_parameter, before_parameter));
  EXPECT_EQ(adagrad->completed_steps(), before_steps);
  EXPECT_EQ(adagrad->learning_rates(), before_rates);
}

TEST(AdagradOptimizerAdapterTest, ValidatesOptionsAndLearningRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(
      AdagradOptimizerAdapter("optimizer.adagrad",
                              {make_group("main", "weight", parameter,
                                          {.learning_rate_decay = -0.1})}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(AdagradOptimizerAdapter(
                   "optimizer.adagrad",
                   {make_group("main", "weight", parameter,
                               {.initial_accumulator_value = -0.1})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(AdagradOptimizerAdapter(
                   "optimizer.adagrad",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(AdagradOptimizerAdapter(
                   "optimizer.adagrad",
                   {make_group("main", "weight", parameter,
                               {.weight_decay =
                                    std::numeric_limits<double>::infinity()})}),
               jyppx::liuding::InvalidArgument);
  AdagradOptimizerAdapter optimizer("optimizer.adagrad",
                                    {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
