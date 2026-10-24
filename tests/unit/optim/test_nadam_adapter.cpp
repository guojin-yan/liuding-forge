// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/nadam_adapter.hpp>

#include <torch/version.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

using jyppx::liuding::StateManifest;
using jyppx::liuding::Tensor;
using jyppx::liuding::optim::NAdamAdapterOptions;
using jyppx::liuding::optim::NAdamOptimizerAdapter;
using jyppx::liuding::optim::NAdamOptimizerCheckpointOptions;
using jyppx::liuding::optim::NAdamOptimizerCheckpointParticipant;
using jyppx::liuding::optim::NAdamParameterGroup;

/** @brief In-memory participant store used for storage-neutral state tests. */
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

/** @brief Scalar state for a reference NAdam parameter trajectory. */
struct ScalarNAdamState {
  double parameter = 0.0;      ///< Current scalar parameter value.
  double first_moment = 0.0;   ///< Exponential first moment.
  double second_moment = 0.0;  ///< Exponential second moment.
  std::int64_t step = 0;       ///< Number of updates with a defined gradient.
  float momentum_product = 1.0F;  ///< PyTorch-compatible product precision.
};

Tensor make_parameter(std::initializer_list<std::int64_t> shape,
                      c10::ScalarType dtype = torch::kFloat64,
                      c10::Device device = torch::kCPU) {
  return torch::randn(
      shape,
      torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
}

NAdamParameterGroup make_group(std::string group_id, std::string parameter_id,
                               const Tensor& parameter,
                               NAdamAdapterOptions options = {}) {
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
    throw std::runtime_error("invalid NAdam checkpoint fixture: " +
                             path.string());
  }
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid NAdam checkpoint fixture byte: " +
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
      throw std::runtime_error("invalid NAdam checkpoint fixture padding: " +
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

std::vector<NAdamParameterGroup> make_fixture_groups(
    const Tensor& coupled_parameter, const Tensor& decoupled_parameter,
    double coupled_rate = 0.012, double decoupled_rate = 0.004) {
  return {make_group("coupled", "weight", coupled_parameter,
                     {.learning_rate = coupled_rate,
                      .beta1 = 0.83,
                      .beta2 = 0.97,
                      .epsilon = 1e-7,
                      .weight_decay = 0.03,
                      .momentum_decay = 0.006}),
          make_group("decoupled", "bias", decoupled_parameter,
                     {.learning_rate = decoupled_rate,
                      .beta1 = 0.91,
                      .beta2 = 0.995,
                      .epsilon = 1e-9,
                      .weight_decay = 0.02,
                      .momentum_decay = 0.002,
                      .decoupled_weight_decay = true,
                      .maximize = true})};
}

void apply_fixture_gradients(const Tensor& coupled_parameter,
                             const Tensor& decoupled_parameter,
                             std::int64_t step) {
  coupled_parameter.mutable_grad() = torch::tensor(
      {0.2 * static_cast<double>(step), -0.1 * static_cast<double>(step),
       0.05 * static_cast<double>(step)},
      coupled_parameter.options().requires_grad(false));
  if (step == 2) {
    decoupled_parameter.mutable_grad() = Tensor();
  } else {
    decoupled_parameter.mutable_grad() = torch::tensor(
        {-0.15 * static_cast<double>(step), 0.08 * static_cast<double>(step),
         0.12 * static_cast<double>(step)},
        decoupled_parameter.options().requires_grad(false));
  }
}

MemoryStateStore load_canonical_schema_v1_fixture(
    const NAdamOptimizerCheckpointParticipant& participant) {
  MemoryStateStore fixture(
      make_schema_v1_manifest(participant.describe_state()));
  const auto fixture_dir =
      std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "checkpoint" /
      "nadam_schema_v1_libtorch_2.5.1";
  for (const auto& field : fixture.manifest().fields) {
    fixture.replace(field.name, read_base64_fixture(fixture_dir /
                                                    (field.name + ".bin.b64")));
  }
  return fixture;
}

void update_reference(ScalarNAdamState& state, double gradient,
                      const NAdamAdapterOptions& options) {
  ++state.step;
  if (options.maximize) {
    gradient = -gradient;
  }
  if (options.weight_decay != 0.0) {
    if (options.decoupled_weight_decay) {
      state.parameter *= 1.0 - options.learning_rate * options.weight_decay;
    } else {
      gradient += options.weight_decay * state.parameter;
    }
  }

  const auto numeric_step = static_cast<double>(state.step);
  const auto momentum =
      options.beta1 *
      (1.0 - 0.5 * std::pow(0.96, numeric_step * options.momentum_decay));
  const auto next_momentum =
      options.beta1 * (1.0 - 0.5 * std::pow(0.96, (numeric_step + 1.0) *
                                                      options.momentum_decay));
  state.momentum_product =
      static_cast<float>(state.momentum_product * static_cast<float>(momentum));
  state.first_moment += (gradient - state.first_moment) * (1.0 - options.beta1);
  state.second_moment = state.second_moment * options.beta2 +
                        gradient * gradient * (1.0 - options.beta2);

  const auto denominator =
      std::sqrt(state.second_moment /
                (1.0 - std::pow(options.beta2, numeric_step))) +
      options.epsilon;
  const auto product = static_cast<double>(state.momentum_product);
  state.parameter -= options.learning_rate * (1.0 - momentum) * gradient /
                     ((1.0 - product) * denominator);
  state.parameter -= options.learning_rate * next_momentum *
                     state.first_moment /
                     ((1.0 - product * next_momentum) * denominator);
}

TEST(NAdamOptimizerAdapterTest, MatchesIndependentLongRunReferenceAndGroups) {
  const NAdamAdapterOptions coupled{.learning_rate = 3e-3,
                                    .beta1 = 0.83,
                                    .beta2 = 0.97,
                                    .epsilon = 1e-7,
                                    .weight_decay = 0.02,
                                    .momentum_decay = 0.006};
  const NAdamAdapterOptions decoupled_maximize{.learning_rate = 7e-4,
                                               .beta1 = 0.91,
                                               .beta2 = 0.995,
                                               .epsilon = 1e-9,
                                               .weight_decay = 0.03,
                                               .momentum_decay = 0.002,
                                               .decoupled_weight_decay = true,
                                               .maximize = true};
  auto first = torch::tensor({0.75, -1.25, 2.0, -0.125},
                             torch::TensorOptions().dtype(torch::kFloat64))
                   .set_requires_grad(true);
  auto second = torch::tensor({-0.5, 1.5, 0.25},
                              torch::TensorOptions().dtype(torch::kFloat64))
                    .set_requires_grad(true);
  NAdamOptimizerAdapter optimizer(
      "optimizer.nadam",
      {make_group("coupled", "weight", first, coupled),
       make_group("decoupled", "bias", second, decoupled_maximize)});
  std::vector<ScalarNAdamState> first_reference;
  std::vector<ScalarNAdamState> second_reference;
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
      update_reference(first_reference[index], gradient, coupled);
    }
    for (std::size_t index = 0; index < second_reference.size(); ++index) {
      const auto gradient = std::cos((static_cast<double>(step) + 1.0) * 0.019 +
                                     (static_cast<double>(index) + 1.0) * 0.23);
      second_gradient[static_cast<std::int64_t>(index)] = gradient;
      update_reference(second_reference[index], gradient, decoupled_maximize);
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
  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{3e-3, 7e-4}));
}

TEST(NAdamOptimizerAdapterTest, FirstStepIncludesNextMomentumSchedule) {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  NAdamOptimizerAdapter optimizer("optimizer.nadam",
                                  {make_group("main", "weight", parameter)});
  parameter.mutable_grad() = torch::full_like(parameter, 0.75);
  ScalarNAdamState expected{.parameter = 1.0};
  update_reference(expected, 0.75, {});

  optimizer.step();

  EXPECT_NEAR(parameter.item<double>(), expected.parameter, 1e-13);
}

TEST(NAdamOptimizerAdapterTest, RejectsSparseGradientsBeforeAnyMutation) {
  auto first = make_parameter({4});
  auto second = make_parameter({4});
  const auto before_first = first.detach().clone();
  const auto before_second = second.detach().clone();
  NAdamOptimizerAdapter optimizer(
      "optimizer.nadam",
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

TEST(NAdamOptimizerAdapterTest, ClosureGradientFailureIsAtomic) {
  auto parameter = make_parameter({4});
  const auto before = parameter.detach().clone();
  NAdamOptimizerAdapter optimizer("optimizer.nadam",
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

TEST(NAdamOptimizerAdapterTest, CheckpointResumesPerParameterScheduleExactly) {
  torch::manual_seed(20261018);
  auto first = make_parameter({6});
  auto second = make_parameter({4});
  auto resumed_first = first.detach().clone().set_requires_grad(true);
  auto resumed_second = second.detach().clone().set_requires_grad(true);
  auto make_two_group = [](const Tensor& first_parameter,
                           const Tensor& second_parameter) {
    return NAdamParameterGroup{
        .group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first_parameter},
                       {.parameter_id = "second",
                        .parameter = second_parameter}},
        .options = {.learning_rate = 2e-3, .momentum_decay = 0.007}};
  };
  auto uninterrupted = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam",
      std::vector<NAdamParameterGroup>{make_two_group(first, second)});
  auto resumed = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_two_group(resumed_first, resumed_second)});
  NAdamOptimizerCheckpointParticipant uninterrupted_state(uninterrupted);
  NAdamOptimizerCheckpointParticipant resumed_state(resumed);

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
  resumed_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

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

TEST(NAdamOptimizerAdapterTest, MalformedCheckpointLeavesStateUnchanged) {
  auto parameter = make_parameter({5});
  auto reference_parameter = parameter.detach().clone().set_requires_grad(true);
  auto optimizer = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", parameter)});
  NAdamOptimizerAdapter reference(
      "optimizer.reference",
      {make_group("main", "weight", reference_parameter)});
  NAdamOptimizerCheckpointParticipant participant(optimizer);
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
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
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

TEST(NAdamOptimizerAdapterTest,
     MigratesSchemaV1MomentumProductAndContinuesExactly) {
  torch::manual_seed(20261019);
  auto uninterrupted = make_parameter({6, 3});
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const NAdamAdapterOptions options{.learning_rate = 2e-3,
                                    .beta1 = 0.87,
                                    .beta2 = 0.98,
                                    .epsilon = 1e-7,
                                    .weight_decay = 0.015,
                                    .momentum_decay = 0.007,
                                    .decoupled_weight_decay = true,
                                    .maximize = true};
  auto first = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{make_group(
                             "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", resumed, options)});
  NAdamOptimizerCheckpointParticipant first_state(first);
  NAdamOptimizerCheckpointParticipant second_state(second);

  for (std::int64_t step = 0; step < 19; ++step) {
    assign_gradient(uninterrupted, step, 0.4);
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
    assign_gradient(uninterrupted, step, 0.4);
    resumed.mutable_grad() = uninterrupted.grad().clone();
    first->step();
    second->step();
  }
  EXPECT_TRUE(torch::equal(uninterrupted, resumed));
  EXPECT_EQ(second->completed_steps(), first->completed_steps());
}

TEST(NAdamOptimizerAdapterTest,
     MigratesCanonicalSchemaV1FixtureFromLibtorch251) {
  if (TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 5 ||
      TORCH_VERSION_PATCH != 1) {
    GTEST_SKIP() << "canonical fixture requires exact LibTorch 2.5.1";
  }
  auto expected_coupled = torch::tensor(
      {1.25, -0.5, 0.75},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto expected_decoupled = torch::tensor(
      {-0.75, 0.25, 1.5},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto expected = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam.fixture",
      make_fixture_groups(expected_coupled, expected_decoupled));
  for (std::int64_t step = 1; step <= 3; ++step) {
    apply_fixture_gradients(expected_coupled, expected_decoupled, step);
    expected->step();
  }
  expected->set_learning_rates({0.005, 0.003});

  auto resumed_coupled =
      expected_coupled.detach().clone().set_requires_grad(true);
  auto resumed_decoupled =
      expected_decoupled.detach().clone().set_requires_grad(true);
  auto resumed = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam.fixture",
      make_fixture_groups(resumed_coupled, resumed_decoupled, 0.9, 0.8));
  NAdamOptimizerCheckpointParticipant resumed_participant(resumed);
  const auto fixture = load_canonical_schema_v1_fixture(resumed_participant);
  resumed_participant.load_state(fixture,
                                 jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(resumed->completed_steps(), 3U);
  EXPECT_EQ(resumed->learning_rates(), (std::vector<double>{0.005, 0.003}));

  apply_fixture_gradients(expected_coupled, expected_decoupled, 4);
  resumed_coupled.mutable_grad() = expected_coupled.grad().clone();
  resumed_decoupled.mutable_grad() = expected_decoupled.grad().clone();
  expected->step();
  resumed->step();
  EXPECT_TRUE(torch::equal(expected_coupled, resumed_coupled));
  EXPECT_TRUE(torch::equal(expected_decoupled, resumed_decoupled));
}

TEST(NAdamOptimizerAdapterTest, RejectsFutureSchemaAndRelabelledV1) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", parameter)});
  NAdamOptimizerCheckpointParticipant participant(optimizer);
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
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);

  auto relabelled = participant.describe_state();
  relabelled.schema_version = 1;
  store.set_manifest(relabelled);
  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(NAdamOptimizerAdapterTest,
     RejectsMalformedProducerAndLeavesStateUnchanged) {
  auto parameter = make_parameter({4});
  auto optimizer = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", parameter)});
  NAdamOptimizerCheckpointParticipant participant(optimizer);
  assign_gradient(parameter, 1);
  optimizer->step();
  const auto before = parameter.detach().clone();
  const auto before_steps = optimizer->completed_steps();
  const auto before_rates = optimizer->learning_rates();
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  store.replace("producer", {std::byte{0x01}});

  EXPECT_THROW(
      participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer->completed_steps(), before_steps);
  EXPECT_EQ(optimizer->learning_rates(), before_rates);
}

TEST(NAdamOptimizerAdapterTest, EnforcesProducerResourceLimit) {
  auto parameter = make_parameter({2});
  auto optimizer = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", parameter)});
  NAdamOptimizerCheckpointOptions options;
  options.max_producer_bytes = 1;
  EXPECT_THROW(NAdamOptimizerCheckpointParticipant(optimizer, options),
               jyppx::liuding::ResourceExhausted);
}

TEST(NAdamOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = make_parameter({3});
  auto optimizer = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam", std::vector<NAdamParameterGroup>{
                             make_group("main", "weight", parameter)});
  NAdamOptimizerCheckpointParticipant participant(optimizer);

  parameter.set_data(parameter.to(torch::kFloat32));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(),
               jyppx::liuding::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(NAdamOptimizerAdapterCudaTest,
     MigratesSchemaV1StateForExactGpuContinuation) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto uninterrupted =
      make_parameter({8}, torch::kFloat32, torch::Device(torch::kCUDA, 0));
  auto resumed = uninterrupted.detach().clone().set_requires_grad(true);
  const NAdamAdapterOptions options{.learning_rate = 1e-3,
                                    .beta1 = 0.86,
                                    .beta2 = 0.98,
                                    .epsilon = 1e-7,
                                    .weight_decay = 0.02,
                                    .momentum_decay = 0.006,
                                    .decoupled_weight_decay = true,
                                    .maximize = true};
  auto first = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam.cuda", std::vector<NAdamParameterGroup>{make_group(
                                  "main", "weight", uninterrupted, options)});
  auto second = std::make_shared<NAdamOptimizerAdapter>(
      "optimizer.nadam.cuda", std::vector<NAdamParameterGroup>{make_group(
                                  "main", "weight", resumed, options)});
  NAdamOptimizerCheckpointParticipant first_state(first);
  NAdamOptimizerCheckpointParticipant second_state(second);
  for (std::int64_t step = 0; step < 13; ++step) {
    assign_gradient(uninterrupted, step, 0.25);
    first->step();
  }
  MemoryStateStore store(first_state.describe_state());
  first_state.save_state(store);
  const auto current_manifest = store.manifest();
  store.erase("producer");
  store.set_manifest(make_schema_v1_manifest(current_manifest));
  copy_parameter(uninterrupted, resumed);
  second_state.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

  for (std::int64_t step = 13; step < 37; ++step) {
    assign_gradient(uninterrupted, step, 0.25);
    resumed.mutable_grad() = uninterrupted.grad().clone();
    first->step();
    second->step();
  }
  EXPECT_TRUE(torch::equal(uninterrupted, resumed));
  EXPECT_EQ(second->completed_steps(), first->completed_steps());
  EXPECT_TRUE(uninterrupted.is_cuda());
  EXPECT_TRUE(resumed.is_cuda());
}

TEST(NAdamOptimizerAdapterTest, SupportsCpuBfloat16AndCudaCapability) {
  auto cpu = make_parameter({8}, torch::kBFloat16);
  NAdamOptimizerAdapter cpu_optimizer(
      "optimizer.cpu",
      {make_group("main", "cpu", cpu, {.learning_rate = 1e-3})});
  cpu.mutable_grad() = torch::ones_like(cpu);
  EXPECT_NO_THROW(cpu_optimizer.step());

  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto cuda = make_parameter({8}, dtype, torch::Device(torch::kCUDA, 0));
    NAdamOptimizerAdapter cuda_optimizer(
        dtype == torch::kFloat32 ? "optimizer.cuda.float"
                                 : "optimizer.cuda.half",
        {make_group("main", "cuda", cuda, {.learning_rate = 1e-3})});
    cuda.mutable_grad() = torch::ones_like(cuda);
    EXPECT_NO_THROW(cuda_optimizer.step());
    EXPECT_TRUE(cuda.is_cuda());
  }
}

TEST(NAdamOptimizerAdapterTest, ValidatesOptionsIdentityAndRatesAtomically) {
  auto parameter = make_parameter({2});
  EXPECT_THROW(NAdamOptimizerAdapter(
                   "optimizer.nadam",
                   {make_group("main", "weight", parameter, {.beta1 = 1.0})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(NAdamOptimizerAdapter(
                   "optimizer.nadam",
                   {make_group("main", "weight", parameter, {.epsilon = 0.0})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(NAdamOptimizerAdapter("optimizer.nadam",
                                     {make_group("main", "weight", parameter,
                                                 {.momentum_decay = -0.001})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(NAdamOptimizerAdapter(
                   "optimizer.nadam",
                   {make_group("main", "weight", parameter,
                               {.weight_decay =
                                    std::numeric_limits<double>::infinity()})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(NAdamOptimizerAdapter(
                   "optimizer.nadam",
                   {{.group_id = "main",
                     .parameters = {{.parameter_id = "duplicate",
                                     .parameter = parameter},
                                    {.parameter_id = "duplicate",
                                     .parameter = make_parameter({2})}}}}),
               jyppx::liuding::InvalidArgument);

  NAdamOptimizerAdapter optimizer("optimizer.nadam",
                                  {make_group("main", "weight", parameter)});
  const auto before = optimizer.learning_rates();
  EXPECT_THROW(optimizer.set_learning_rates({-1.0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(optimizer.set_learning_rates({0.1, 0.2}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), before);
}

}  // namespace
