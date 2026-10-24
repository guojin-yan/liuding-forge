// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <torch/optim/sgd.h>

#include <algorithm>
#include <array>
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

using jyppx::liuding::cuda_is_available;

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end()) {
      throw jyppx::liuding::SerializationError("unknown SGD test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized SGD test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate SGD test field");
    }
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw jyppx::liuding::SerializationError("missing SGD test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("SGD test read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  jyppx::liuding::StateManifest& mutable_manifest() noexcept {
    return manifest_;
  }

  void erase_blob(std::string_view name) { blobs_.erase(std::string(name)); }

  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

 private:
  jyppx::liuding::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

jyppx::liuding::optim::SgdParameterGroup make_group(
    std::string group_id, std::string parameter_id, const torch::Tensor& tensor,
    jyppx::liuding::optim::SgdAdapterOptions options) {
  return {.group_id = std::move(group_id),
          .parameters = {{.parameter_id = std::move(parameter_id),
                          .parameter = tensor}},
          .options = options};
}

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_optimizer(
    const torch::Tensor& tensor,
    jyppx::liuding::optim::SgdAdapterOptions options = {.learning_rate = 0.05,
                                                        .momentum = 0.9}) {
  return std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      "optimizer.main",
      std::vector{make_group("main", "layer.weight", tensor, options)});
}

MemoryStateStore save_participant(
    const jyppx::liuding::optim::SgdOptimizerCheckpointParticipant&
        participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

MemoryStateStore make_schema_v1_fixture(const MemoryStateStore& current) {
  MemoryStateStore fixture = current;
  auto& manifest = fixture.mutable_manifest();
  manifest.schema_version = 1;
  std::erase_if(manifest.fields,
                [](const jyppx::liuding::StateFieldDescriptor& field) {
                  return field.name == "producer";
                });
  fixture.erase_blob("producer");
  jyppx::liuding::validate_state_manifest(manifest);
  return fixture;
}

std::vector<std::byte> read_base64_fixture(const std::filesystem::path& path) {
  std::ifstream input(path);
  std::string encoded;
  std::string trailing;
  if (!(input >> encoded) || (input >> trailing)) {
    throw std::runtime_error("invalid base64 checkpoint fixture: " +
                             path.string());
  }
  if (encoded.empty() || encoded.size() % 4 != 0) {
    throw std::runtime_error("invalid base64 checkpoint fixture length: " +
                             path.string());
  }

  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto decode = [&](char value) -> std::uint32_t {
    const auto position = alphabet.find(value);
    if (position == std::string_view::npos) {
      throw std::runtime_error("invalid base64 checkpoint fixture byte: " +
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
      throw std::runtime_error("invalid base64 checkpoint fixture padding: " +
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

MemoryStateStore load_canonical_schema_v1_fixture() {
  MemoryStateStore fixture({
      .participant_id = "optimizer.main",
      .schema_version = 1,
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .fields = {{.name = "topology",
                  .media_type = "application/vnd.jyppx.sgd-topology-v1",
                  .max_bytes = 128,
                  .required = true},
                 {.name = "control",
                  .media_type = "application/vnd.jyppx.sgd-control-v1",
                  .max_bytes = 37,
                  .required = true},
                 {.name = "momentum.0000000000",
                  .media_type = "application/vnd.jyppx.tensor-raw-v1",
                  .max_bytes = 16,
                  .required = true}},
      .degradation_reasons = {},
  });
  const auto fixture_dir =
      std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "checkpoint" /
      "sgd_schema_v1_libtorch_2.5.1";
  for (const auto& field : fixture.manifest().fields) {
    const auto bytes =
        read_base64_fixture(fixture_dir / (field.name + ".bin.b64"));
    fixture.write_blob(field.name, bytes);
  }
  return fixture;
}

TEST(SgdOptimizerAdapterTest, DelegatesUpdatesAndTracksSuccessfulSteps) {
  auto adapted_parameter = torch::tensor(
      {1.0, -2.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto native_parameter =
      adapted_parameter.detach().clone().set_requires_grad(true);
  const jyppx::liuding::optim::SgdAdapterOptions options{.learning_rate = 0.03,
                                                         .momentum = 0.9,
                                                         .dampening = 0.0,
                                                         .weight_decay = 0.01,
                                                         .nesterov = true};
  jyppx::liuding::optim::SgdOptimizerAdapter adapted(
      "optimizer.main",
      {make_group("main", "weight", adapted_parameter, options)});
  torch::optim::SGD native({native_parameter},
                           torch::optim::SGDOptions(options.learning_rate)
                               .momentum(options.momentum)
                               .dampening(options.dampening)
                               .weight_decay(options.weight_decay)
                               .nesterov(options.nesterov));

  for (std::int64_t step = 1; step <= 4; ++step) {
    const auto gradient =
        torch::tensor({0.25 * step, -0.5 * step}, torch::kFloat64);
    adapted_parameter.mutable_grad() = gradient.clone();
    native_parameter.mutable_grad() = gradient.clone();
    const auto result = adapted.step();
    native.step();
    EXPECT_FALSE(result.defined());
    EXPECT_TRUE(
        torch::allclose(adapted_parameter, native_parameter, 1e-12, 1e-12));
  }
  EXPECT_EQ(adapted.completed_steps(), 4U);
  EXPECT_THROW(adapted.step([]() -> torch::Tensor {
    throw std::runtime_error("closure failure");
  }),
               std::runtime_error);
  EXPECT_EQ(adapted.completed_steps(), 4U);

  adapted.zero_grad(false);
  ASSERT_TRUE(adapted_parameter.grad().defined());
  EXPECT_TRUE(torch::equal(adapted_parameter.grad(),
                           torch::zeros_like(adapted_parameter)));
  adapted.zero_grad(true);
  EXPECT_FALSE(adapted_parameter.grad().defined());
}

TEST(SgdOptimizerAdapterTest,
     SynchronizesFreshInPlacePlacementAndThenFreezesIt) {
  auto parameter = torch::ones(
      {3}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto optimizer = make_optimizer(parameter);
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant participant(
      optimizer);

  parameter.set_data(parameter.to(torch::kFloat64));
  EXPECT_NO_THROW(optimizer->synchronize_parameter_placement());
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
  parameter.mutable_grad() = torch::ones_like(parameter);
  optimizer->step();

  EXPECT_THROW(optimizer->synchronize_parameter_placement(),
               jyppx::liuding::NotReady);
  EXPECT_NO_THROW(static_cast<void>(participant.describe_state()));
}

TEST(SgdOptimizerAdapterTest,
     MatchesLibTorchAcrossLongRunsAndOptionCombinations) {
  const std::array<jyppx::liuding::optim::SgdAdapterOptions, 4> option_matrix =
      {jyppx::liuding::optim::SgdAdapterOptions{.learning_rate = 0.03},
       jyppx::liuding::optim::SgdAdapterOptions{
           .learning_rate = 0.02, .momentum = 0.85, .dampening = 0.15},
       jyppx::liuding::optim::SgdAdapterOptions{.learning_rate = 0.015,
                                                .momentum = 0.9,
                                                .weight_decay = 0.02,
                                                .nesterov = true},
       jyppx::liuding::optim::SgdAdapterOptions{
           .learning_rate = 0.0, .momentum = 0.75, .weight_decay = 0.1}};

  for (const auto& options : option_matrix) {
    auto adapted_parameter = torch::tensor(
        {1.0, -2.0, 0.5},
        torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    auto native_parameter =
        adapted_parameter.detach().clone().set_requires_grad(true);
    jyppx::liuding::optim::SgdOptimizerAdapter adapted(
        "optimizer.long_run",
        {make_group("main", "weight", adapted_parameter, options)});
    torch::optim::SGD native({native_parameter},
                             torch::optim::SGDOptions(options.learning_rate)
                                 .momentum(options.momentum)
                                 .dampening(options.dampening)
                                 .weight_decay(options.weight_decay)
                                 .nesterov(options.nesterov));

    for (std::int64_t step = 0; step < 512; ++step) {
      const auto phase = static_cast<double>(step + 1);
      const auto gradient =
          torch::tensor({std::sin(phase * 0.013), std::cos(phase * 0.017),
                         std::sin(phase * 0.019) * std::cos(phase * 0.023)},
                        torch::kFloat64);
      adapted_parameter.mutable_grad() = gradient.clone();
      native_parameter.mutable_grad() = gradient.clone();
      adapted.step();
      native.step();
    }

    EXPECT_EQ(adapted.completed_steps(), 512U);
    EXPECT_TRUE(torch::equal(adapted_parameter, native_parameter));
  }
}

TEST(SgdOptimizerAdapterTest,
     RejectsSparseGradientsBeforeAnyParameterMutation) {
  auto first = torch::tensor(
      {1.0, 2.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto second = torch::tensor(
      {-1.0, 3.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto first_before = first.detach().clone();
  const auto second_before = second.detach().clone();
  jyppx::liuding::optim::SgdOptimizerAdapter optimizer(
      "optimizer.sparse",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "first", .parameter = first},
                       {.parameter_id = "second", .parameter = second}},
        .options = {.learning_rate = 0.1, .momentum = 0.9}}});

  first.mutable_grad() = torch::ones_like(first);
  const auto indices = torch::tensor({{0, 1}}, torch::kLong);
  const auto values = torch::tensor({0.5, -0.25}, torch::kFloat64);
  second.mutable_grad() =
      torch::sparse_coo_tensor(indices, values, second.sizes()).coalesce();

  EXPECT_THROW(optimizer.step(), jyppx::liuding::NotImplemented);
  EXPECT_TRUE(torch::equal(first, first_before));
  EXPECT_TRUE(torch::equal(second, second_before));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(SgdOptimizerAdapterTest,
     ValidatesClosureProducedGradientsBeforeAnyParameterMutation) {
  auto parameter = torch::tensor(
      {1.0, -2.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto before = parameter.detach().clone();
  jyppx::liuding::optim::SgdOptimizerAdapter optimizer(
      "optimizer.closure",
      {make_group("main", "weight", parameter, {.learning_rate = 0.1})});
  const auto indices = torch::tensor({{0, 1}}, torch::kLong);
  const auto values = torch::tensor({0.5, -0.25}, torch::kFloat64);

  EXPECT_THROW(optimizer.step([&]() {
    parameter.mutable_grad() =
        torch::sparse_coo_tensor(indices, values, parameter.sizes()).coalesce();
    return torch::tensor(1.0);
  }),
               jyppx::liuding::NotImplemented);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(SgdOptimizerAdapterTest,
     RejectsMismatchedDenseGradientBeforeParameterMutation) {
  auto parameter = torch::tensor(
      {1.0, -2.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto before = parameter.detach().clone();
  jyppx::liuding::optim::SgdOptimizerAdapter optimizer(
      "optimizer.gradient_descriptor",
      {make_group("main", "weight", parameter, {.learning_rate = 0.1})});
  parameter.mutable_grad() = torch::ones({3}, torch::kFloat64);

  EXPECT_THROW(optimizer.step(), jyppx::liuding::InvalidArgument);
  EXPECT_TRUE(torch::equal(parameter, before));
  EXPECT_EQ(optimizer.completed_steps(), 0U);
}

TEST(SgdOptimizerAdapterTest, MatchesLibTorchForCpuBfloat16) {
  auto adapted_parameter = torch::tensor(
      {1.0, -2.0, 0.5},
      torch::TensorOptions().dtype(torch::kBFloat16).requires_grad(true));
  auto native_parameter =
      adapted_parameter.detach().clone().set_requires_grad(true);
  const jyppx::liuding::optim::SgdAdapterOptions options{
      .learning_rate = 0.02, .momentum = 0.9, .weight_decay = 0.01};
  jyppx::liuding::optim::SgdOptimizerAdapter adapted(
      "optimizer.bfloat16",
      {make_group("main", "weight", adapted_parameter, options)});
  torch::optim::SGD native({native_parameter},
                           torch::optim::SGDOptions(options.learning_rate)
                               .momentum(options.momentum)
                               .weight_decay(options.weight_decay));

  for (std::int64_t step = 1; step <= 32; ++step) {
    const auto gradient = torch::full_like(adapted_parameter, 0.015625 * step);
    adapted_parameter.mutable_grad() = gradient.clone();
    native_parameter.mutable_grad() = gradient.clone();
    adapted.step();
    native.step();
  }

  EXPECT_TRUE(torch::equal(adapted_parameter, native_parameter));
}

TEST(SgdOptimizerAdapterCudaTest, MatchesLibTorchForFloat32AndFloat16) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto device = torch::Device(torch::kCUDA, 0);
  for (const auto dtype : {torch::kFloat32, torch::kFloat16}) {
    auto adapted_parameter = torch::tensor(
        {1.0, -2.0, 0.5},
        torch::TensorOptions().dtype(dtype).device(device).requires_grad(true));
    auto native_parameter =
        adapted_parameter.detach().clone().set_requires_grad(true);
    const jyppx::liuding::optim::SgdAdapterOptions options{
        .learning_rate = 0.02,
        .momentum = 0.9,
        .weight_decay = 0.01,
        .nesterov = true};
    jyppx::liuding::optim::SgdOptimizerAdapter adapted(
        "optimizer.cuda",
        {make_group("main", "weight", adapted_parameter, options)});
    torch::optim::SGD native({native_parameter},
                             torch::optim::SGDOptions(options.learning_rate)
                                 .momentum(options.momentum)
                                 .weight_decay(options.weight_decay)
                                 .nesterov(options.nesterov));

    for (std::int64_t step = 1; step <= 32; ++step) {
      const auto gradient =
          torch::full_like(adapted_parameter, 0.015625 * step);
      adapted_parameter.mutable_grad() = gradient.clone();
      native_parameter.mutable_grad() = gradient.clone();
      adapted.step();
      native.step();
    }

    EXPECT_TRUE(torch::equal(adapted_parameter, native_parameter));
  }
}

TEST(SgdOptimizerAdapterCudaTest,
     MigratesSchemaV1MomentumForExactGpuContinuation) {
  if (!cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options = jyppx::liuding::optim::SgdAdapterOptions{
      .learning_rate = 0.04, .momentum = 0.8, .weight_decay = 0.02};
  const auto tensor_options = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::kCUDA, 0)
                                  .requires_grad(true);
  auto source_parameter = torch::tensor({2.0, -1.0}, tensor_options);
  auto source = make_optimizer(source_parameter, options);
  for (std::int64_t step = 1; step <= 3; ++step) {
    source_parameter.mutable_grad() =
        torch::tensor({0.25 * step, -0.125 * step}, tensor_options);
    source->step();
  }
  source->set_learning_rates({0.0125});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto schema_v1 =
      make_schema_v1_fixture(save_participant(source_participant));

  auto resumed_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(
      resumed_parameter,
      {.learning_rate = 0.9, .momentum = 0.8, .weight_decay = 0.02});
  resumed_parameter.mutable_grad() = torch::ones_like(resumed_parameter);
  resumed->step();
  resumed_parameter.detach().copy_(source_parameter.detach());
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);
  resumed_participant.load_state(schema_v1,
                                 jyppx::liuding::RestorePolicy::exact_only);

  const auto next_gradient = torch::tensor({-0.3, 0.7}, tensor_options);
  source_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  source->step();
  resumed->step();
  EXPECT_TRUE(torch::equal(source_parameter, resumed_parameter));
  EXPECT_EQ(resumed->completed_steps(), 4U);
}

TEST(SgdOptimizerAdapterTest, PreservesExplicitPerGroupOptionsAndRateUpdates) {
  auto first = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto second = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  jyppx::liuding::optim::SgdOptimizerAdapter optimizer(
      "optimizer.main", {make_group("decay", "first", first,
                                    {.learning_rate = 0.2, .momentum = 0.8}),
                         make_group("plain", "second", second,
                                    {.learning_rate = 0.01, .momentum = 0.0})});

  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{0.2, 0.01}));
  first.mutable_grad() = torch::ones_like(first);
  second.mutable_grad() = torch::ones_like(second);
  optimizer.step();
  EXPECT_NEAR(first.item<double>(), 0.8, 1e-12);
  EXPECT_NEAR(second.item<double>(), 0.99, 1e-12);

  optimizer.set_learning_rates({0.1, 0.02});
  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{0.1, 0.02}));
  EXPECT_THROW(optimizer.set_learning_rates({0.5}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(optimizer.set_learning_rates(
                   {0.5, std::numeric_limits<double>::quiet_NaN()}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer.learning_rates(), (std::vector<double>{0.1, 0.02}));
}

TEST(SgdOptimizerAdapterTest, RejectsUnstableOrInvalidDescriptors) {
  auto parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerAdapter("optimizer.main", {}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerAdapter(
                   "", {make_group("main", "weight", parameter, {})}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SgdOptimizerAdapter(
          "optimizer.main",
          {make_group("main", "same", parameter, {}),
           make_group("other", "same",
                      parameter.detach().clone().requires_grad_(true), {})}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SgdOptimizerAdapter(
          "optimizer.main", {make_group("main", "first", parameter, {}),
                             make_group("other", "second", parameter, {})}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SgdOptimizerAdapter(
          "optimizer.main",
          {make_group("main", "integer", torch::ones({2}, torch::kLong), {})}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SgdOptimizerAdapter(
          "optimizer.main",
          {make_group(
              "main", "weight", parameter,
              {.learning_rate = 0.1, .momentum = 0.0, .nesterov = true})}),
      jyppx::liuding::InvalidArgument);
}

TEST(SgdOptimizerCheckpointTest, UsesFixedRawFieldsWithoutNativeArchive) {
  auto parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto optimizer = make_optimizer(parameter);
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant participant(
      optimizer);

  const auto manifest = participant.describe_state();
  ASSERT_EQ(manifest.schema_version, 2U);
  ASSERT_EQ(manifest.restore_level, jyppx::liuding::RestoreLevel::exact);
  ASSERT_EQ(manifest.fields.size(), 4U);
  EXPECT_EQ(manifest.fields[0].name, "topology");
  EXPECT_EQ(manifest.fields[1].name, "producer");
  EXPECT_EQ(manifest.fields[2].name, "control");
  EXPECT_EQ(manifest.fields[3].name, "momentum.0000000000");

  const auto store = save_participant(participant);
  const auto topology = store.read_blob("topology", 1024 * 1024);
  const auto producer = store.read_blob("producer", 1024);
  ASSERT_GE(topology.size(), 8U);
  ASSERT_GE(producer.size(), 8U);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(topology.data()), 8),
            "JYPPXSGD");
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(producer.data()), 8),
            "JYPPXSGP");
  EXPECT_TRUE(store.read_blob("momentum.0000000000", 1024).empty());
  EXPECT_FALSE(topology[0] == std::byte{'P'} && topology[1] == std::byte{'K'});
}

TEST(SgdOptimizerCheckpointTest,
     RestoresMomentumLearningRateAndStepForExactContinuation) {
  auto source_parameter = torch::tensor(
      {2.0, -1.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto source = make_optimizer(
      source_parameter,
      {.learning_rate = 0.04, .momentum = 0.8, .weight_decay = 0.02});
  for (std::int64_t step = 1; step <= 2; ++step) {
    source_parameter.mutable_grad() =
        torch::tensor({0.5 * step, -0.25 * step}, torch::kFloat64);
    source->step();
  }
  source->set_learning_rates({0.015});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto store = save_participant(source_participant);

  auto resumed_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(
      resumed_parameter,
      {.learning_rate = 0.9, .momentum = 0.8, .weight_decay = 0.02});
  resumed_parameter.mutable_grad() = torch::ones_like(resumed_parameter);
  resumed->step();
  resumed_parameter.detach().copy_(source_parameter.detach());
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);
  resumed_participant.load_state(store,
                                 jyppx::liuding::RestorePolicy::exact_only);

  ASSERT_EQ(resumed->completed_steps(), 2U);
  ASSERT_EQ(resumed->learning_rates().size(), 1U);
  EXPECT_DOUBLE_EQ(resumed->learning_rates()[0], 0.015);
  const auto next_gradient = torch::tensor({-0.3, 0.7}, torch::kFloat64);
  source_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  source->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(source_parameter, resumed_parameter, 1e-12, 1e-12));
}

TEST(SgdOptimizerCheckpointTest,
     MigratesSchemaV1AndWritesOnlyCurrentSchemaAfterRestore) {
  const jyppx::liuding::optim::SgdAdapterOptions source_options{
      .learning_rate = 0.04, .momentum = 0.8, .weight_decay = 0.02};
  auto source_parameter = torch::tensor(
      {2.0, -1.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto source = make_optimizer(source_parameter, source_options);
  for (std::int64_t step = 1; step <= 2; ++step) {
    source_parameter.mutable_grad() =
        torch::tensor({0.5 * step, -0.25 * step}, torch::kFloat64);
    source->step();
  }
  source->set_learning_rates({0.015});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto schema_v1 =
      make_schema_v1_fixture(save_participant(source_participant));
  ASSERT_EQ(schema_v1.manifest().schema_version, 1U);
  EXPECT_FALSE(schema_v1.contains("producer"));

  auto resumed_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(
      resumed_parameter,
      {.learning_rate = 0.9, .momentum = 0.8, .weight_decay = 0.02});
  resumed_parameter.mutable_grad() = torch::ones_like(resumed_parameter);
  resumed->step();
  resumed_parameter.detach().copy_(source_parameter.detach());
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);

  resumed_participant.load_state(schema_v1,
                                 jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(resumed->completed_steps(), 2U);
  EXPECT_EQ(resumed->learning_rates(), (std::vector<double>{0.015}));
  const auto migrated = save_participant(resumed_participant);
  EXPECT_EQ(migrated.manifest().schema_version, 2U);
  EXPECT_TRUE(migrated.contains("producer"));

  const auto next_gradient = torch::tensor({-0.3, 0.7}, torch::kFloat64);
  source_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  source->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(source_parameter, resumed_parameter, 1e-12, 1e-12));
}

TEST(SgdOptimizerCheckpointTest,
     MigratesCanonicalSchemaV1FixtureFromLibtorch251) {
  if (TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR != 5 ||
      TORCH_VERSION_PATCH != 1) {
    GTEST_SKIP() << "canonical fixture requires exact LibTorch 2.5.1";
  }

  const jyppx::liuding::optim::SgdAdapterOptions options{
      .learning_rate = 0.04, .momentum = 0.8, .weight_decay = 0.02};
  auto source_parameter = torch::tensor(
      {2.0, -1.0},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto source = make_optimizer(source_parameter, options);
  for (std::int64_t step = 1; step <= 2; ++step) {
    source_parameter.mutable_grad() =
        torch::tensor({0.5 * step, -0.25 * step}, torch::kFloat64);
    source->step();
  }
  source->set_learning_rates({0.015});

  auto resumed_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(
      resumed_parameter,
      {.learning_rate = 0.9, .momentum = 0.8, .weight_decay = 0.02});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);
  const auto schema_v1 = load_canonical_schema_v1_fixture();
  resumed_participant.load_state(schema_v1,
                                 jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(resumed->completed_steps(), 2U);
  EXPECT_EQ(resumed->learning_rates(), (std::vector<double>{0.015}));
  const auto next_gradient = torch::tensor({-0.3, 0.7}, torch::kFloat64);
  source_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  source->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(source_parameter, resumed_parameter, 1e-12, 1e-12));
}

TEST(SgdOptimizerCheckpointTest, FreshStateRestoreRemovesStaleMomentum) {
  const jyppx::liuding::optim::SgdAdapterOptions options{
      .learning_rate = 0.1, .momentum = 0.9, .dampening = 0.5};
  auto fresh_parameter = torch::tensor(
      {1.0}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto fresh = make_optimizer(fresh_parameter, options);
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant fresh_participant(
      fresh);
  const auto fresh_store = save_participant(fresh_participant);

  auto resumed_parameter =
      fresh_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(resumed_parameter, options);
  resumed_parameter.mutable_grad() = torch::tensor({4.0}, torch::kFloat64);
  resumed->step();
  resumed_parameter.detach().copy_(fresh_parameter.detach());
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);
  resumed_participant.load_state(fresh_store,
                                 jyppx::liuding::RestorePolicy::exact_only);

  fresh_parameter.mutable_grad() = torch::tensor({2.0}, torch::kFloat64);
  resumed_parameter.mutable_grad() = torch::tensor({2.0}, torch::kFloat64);
  fresh->step();
  resumed->step();
  EXPECT_TRUE(
      torch::allclose(fresh_parameter, resumed_parameter, 1e-12, 1e-12));
  EXPECT_EQ(resumed->completed_steps(), 1U);
}

TEST(SgdOptimizerCheckpointTest,
     RestoresCpuBfloat16MomentumForExactContinuation) {
  const jyppx::liuding::optim::SgdAdapterOptions options{
      .learning_rate = 0.03125, .momentum = 0.875, .weight_decay = 0.015625};
  auto source_parameter = torch::tensor(
      {1.0, -2.0},
      torch::TensorOptions().dtype(torch::kBFloat16).requires_grad(true));
  auto source = make_optimizer(source_parameter, options);
  for (std::int64_t step = 1; step <= 8; ++step) {
    source_parameter.mutable_grad() =
        torch::full_like(source_parameter, 0.03125 * step);
    source->step();
  }
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto store = save_participant(source_participant);

  auto resumed_parameter =
      source_parameter.detach().clone().set_requires_grad(true);
  auto resumed = make_optimizer(resumed_parameter, options);
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant resumed_participant(
      resumed);
  resumed_participant.load_state(store,
                                 jyppx::liuding::RestorePolicy::exact_only);

  const auto next_gradient = torch::full_like(source_parameter, -0.125);
  source_parameter.mutable_grad() = next_gradient.clone();
  resumed_parameter.mutable_grad() = next_gradient.clone();
  source->step();
  resumed->step();
  EXPECT_TRUE(torch::equal(source_parameter, resumed_parameter));
  EXPECT_EQ(resumed->completed_steps(), source->completed_steps());
}

TEST(SgdOptimizerCheckpointTest,
     RejectsTopologyMismatchAndMalformedControlWithoutMutation) {
  auto source_parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto source = make_optimizer(source_parameter);
  source_parameter.mutable_grad() = torch::ones_like(source_parameter);
  source->step();
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto source_store = save_participant(source_participant);

  auto target_parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto target = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      "optimizer.main",
      std::vector{make_group("main", "other.weight", target_parameter,
                             {.learning_rate = 0.05, .momentum = 0.9})});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant target_participant(
      target);
  const auto before_mismatch = save_participant(target_participant);
  EXPECT_THROW(target_participant.load_state(
                   source_store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_participant(target_participant).blobs(),
            before_mismatch.blobs());

  auto matching_parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto matching = make_optimizer(matching_parameter);
  matching_parameter.mutable_grad() = torch::full_like(matching_parameter, 3.0);
  matching->step();
  matching->set_learning_rates({0.2});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant matching_participant(
      matching);
  const auto before_malformed = save_participant(matching_participant);
  auto malformed = source_store;
  malformed.mutable_blob("control").front() = std::byte{'X'};
  EXPECT_THROW(matching_participant.load_state(
                   malformed, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(save_participant(matching_participant).blobs(),
            before_malformed.blobs());
}

TEST(SgdOptimizerCheckpointTest,
     RejectsUnsupportedRelabeledAndMalformedStateWithoutMutation) {
  auto source_parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto source = make_optimizer(source_parameter);
  source_parameter.mutable_grad() = torch::ones_like(source_parameter);
  source->step();
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant source_participant(
      source);
  const auto current_store = save_participant(source_participant);

  auto target_parameter = torch::ones(
      {2}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto target = make_optimizer(target_parameter);
  target_parameter.mutable_grad() = torch::full_like(target_parameter, 3.0);
  target->step();
  target->set_learning_rates({0.2});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant target_participant(
      target);
  const auto before = save_participant(target_participant);

  auto future = current_store;
  future.mutable_manifest().schema_version = 3;
  EXPECT_THROW(target_participant.load_state(
                   future, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());

  auto obsolete = current_store;
  obsolete.mutable_manifest().schema_version = 0;
  EXPECT_THROW(target_participant.load_state(
                   obsolete, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());

  auto relabeled = current_store;
  relabeled.mutable_manifest().schema_version = 1;
  EXPECT_THROW(target_participant.load_state(
                   relabeled, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());

  auto malformed_producer = current_store;
  malformed_producer.mutable_blob("producer").front() = std::byte{'X'};
  EXPECT_THROW(
      target_participant.load_state(malformed_producer,
                                    jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());

  auto incompatible_producer = current_store;
  ASSERT_GT(incompatible_producer.mutable_blob("producer").size(), 24U);
  incompatible_producer.mutable_blob("producer")[24] = std::byte{0xFF};
  EXPECT_THROW(
      target_participant.load_state(incompatible_producer,
                                    jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());

  auto malformed_v1 = make_schema_v1_fixture(current_store);
  malformed_v1.mutable_blob("control").front() = std::byte{'X'};
  EXPECT_THROW(target_participant.load_state(
                   malformed_v1, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(save_participant(target_participant).blobs(), before.blobs());
}

TEST(SgdOptimizerCheckpointTest, EnforcesParticipantResourceLimits) {
  auto parameter = torch::ones(
      {4}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto optimizer = make_optimizer(parameter);
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerCheckpointParticipant(
                   optimizer, {.max_groups = 0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerCheckpointParticipant(
                   optimizer, {.max_tensor_bytes = 8}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerCheckpointParticipant(
                   optimizer, {.max_identifier_bytes = 3}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::SgdOptimizerCheckpointParticipant(
                   optimizer, {.max_producer_bytes = 1}),
               jyppx::liuding::ResourceExhausted);

  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant participant(
      optimizer);
  parameter.requires_grad_(false);
  EXPECT_THROW(participant.describe_state(), jyppx::liuding::InvalidArgument);
}

}  // namespace
