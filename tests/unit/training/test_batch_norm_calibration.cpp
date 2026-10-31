// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../tools/cli/recipe_bn_calibration.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/training/batch_norm_calibration.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;

struct Toy final : torch::nn::Module {
  Toy() {
    bn = register_module(
        "bn",
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(3).momentum(.3)));
    dropout = register_module("dropout", torch::nn::Dropout(.8));
    conv = register_module("conv", torch::nn::Conv2d(3, 2, 1));
    train();
    bn->eval();
    dropout->eval();
    for (auto parameter : parameters())
      parameter.mutable_grad() = torch::full_like(parameter, 7.);
  }
  Tensor forward(const Tensor& input) {
    if (torch::GradMode::is_enabled() || is_training() || !bn->is_training() ||
        dropout->is_training() || conv->is_training())
      throw std::runtime_error("BN-only calibration controls are not active");
    return conv->forward(dropout->forward(bn->forward(input)));
  }
  torch::nn::BatchNorm2d bn{nullptr};
  torch::nn::Dropout dropout{nullptr};
  torch::nn::Conv2d conv{nullptr};
};

class MemoryState final : public StateWriter, public StateReader {
 public:
  explicit MemoryState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> value) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [&](const auto& item) { return item.name == name; });
    if (field == manifest_.fields.end() || blobs.contains(std::string(name)))
      throw SerializationError("Undeclared or duplicate test field");
    if (value.size() > field->max_bytes)
      throw ResourceExhausted("Test field too large");
    blobs.emplace(name, std::vector<std::byte>(value.begin(), value.end()));
  }
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& value = blobs.at(std::string(name));
    if (value.size() > maximum) throw ResourceExhausted("Test read too large");
    return value;
  }
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs;
};

using Bits = std::map<std::string, std::vector<std::byte>>;
Bits bits(const nn::ModelAdapter& model) {
  Bits result;
  const auto add = [&](const std::string& name, const Tensor& tensor) {
    const auto value = tensor.detach().contiguous();
    const auto* bytes = static_cast<const std::byte*>(value.const_data_ptr());
    result[name] = {bytes, bytes + value.nbytes()};
  };
  for (const auto& entry : model.state().named_parameters()) {
    add("parameter." + entry.key(), entry.value());
    if (entry.value().grad().defined())
      add("gradient." + entry.key(), entry.value().grad());
  }
  for (const auto& entry : model.state().named_buffers())
    add("buffer." + entry.key(), entry.value());
  return result;
}

const ArtifactDigest kTraining{.hex = std::string(64, 'a')};
const ArtifactDigest kValidation{.hex = std::string(64, 'b')};
BatchNormCalibrationOptions options() {
  return {.training_source = kTraining,
          .expected_batches = 3,
          .expected_samples = 6,
          .max_input_bytes = 1U << 20,
          .max_model_snapshot_bytes = 1U << 20};
}
BatchNormCalibrationBatch batch(std::uint64_t index,
                                const ArtifactDigest& identity = kTraining) {
  return {.images = torch::arange(24, torch::kFloat32).reshape({2, 3, 2, 2}) +
                    static_cast<double>(index),
          .training_source = identity,
          .owner = {}};
}
BatchNormCalibrationSource source() {
  return
      [index = std::uint64_t{0}](
          std::stop_token) mutable -> std::optional<BatchNormCalibrationBatch> {
        if (index == 3) return std::nullopt;
        return batch(index++);
      };
}
void controls(const std::shared_ptr<Toy>& toy, const core::RngState& rng) {
  EXPECT_TRUE(toy->is_training());
  EXPECT_FALSE(toy->bn->is_training());
  EXPECT_FALSE(toy->dropout->is_training());
  EXPECT_TRUE(toy->conv->is_training());
  EXPECT_EQ(toy->bn->options.momentum(), .3);
  EXPECT_TRUE(torch::equal(core::get_rng_state(false).cpu, rng.cpu));
}

TEST(BatchNormCalibration, SuccessPreservesParametersGradientsModesAndRng) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  const auto result = calibrate_batch_norm(model, source(), options());
  EXPECT_EQ(result.batches, 3);
  EXPECT_EQ(result.samples, 6);
  EXPECT_EQ(result.batch_norm_modules, 1);
  const auto after = bits(model);
  for (const auto& [name, value] : before)
    if (!name.starts_with("buffer.bn.")) EXPECT_EQ(value, after.at(name));
  EXPECT_EQ(toy->bn->num_batches_tracked.item<std::int64_t>(), 3);
  controls(toy, rng);
}

TEST(BatchNormCalibration, SourceErrorRestoresWholeModelAndObservableRng) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  std::uint64_t index = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            if (index == 1) {
              static_cast<void>(torch::rand({8}));
              throw std::runtime_error(
                  "Source error after actual calibration forward");
            }
            return batch(index++);
          },
          options()),
      std::runtime_error);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, CancellationDiscardsPartialPassAndReplaysExactly) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  nn::ModelCheckpointParticipant participant(model);
  MemoryState initial(participant.describe_state());
  participant.save_state(initial);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  std::stop_source stop;
  std::uint64_t index = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            if (index == 2) {
              stop.request_stop();
              return std::nullopt;
            }
            return batch(index++);
          },
          options(), stop.get_token()),
      Cancelled);
  EXPECT_EQ(index, 2);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
  calibrate_batch_norm(model, source(), options());
  const auto resumed = bits(model);
  participant.load_state(initial, RestorePolicy::exact_only);
  core::set_rng_state(rng, false);
  calibrate_batch_norm(model, source(), options());
  EXPECT_EQ(bits(model), resumed);
  controls(toy, rng);
}

TEST(BatchNormCalibration, RejectsValidationIdentityWithoutPartialCommit) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  std::uint64_t index = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            if (index == 1) return batch(index, kValidation);
            return batch(index++);
          },
          options()),
      CompatibilityError);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, CommitFailureRestoresModelAndReadyParticipant) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  cli::RecipeBatchNormCalibration stage("{\"method\":\"fixture\"}");
  stage.ready(std::string(64, 'c'), 7);
  const auto ready = stage.report();
  EXPECT_THROW(
      calibrate_batch_norm(model, source(), options(), {},
                           [&](const BatchNormCalibrationResult& result) {
                             stage.completed(std::string(64, 'd'),
                                             result.batches, result.samples);
                             try {
                               throw std::runtime_error(
                                   "Atomic checkpoint publication failed");
                             } catch (...) {
                               stage.discard_completion();
                               throw;
                             }
                           }),
      std::runtime_error);
  EXPECT_EQ(stage.report(), ready);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, ProtectedMutationAndSnapshotBoundsFailAtomically) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  auto bounded = options();
  bounded.max_model_snapshot_bytes = 1;
  std::uint64_t calls = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            ++calls;
            return batch(0);
          },
          bounded),
      ResourceExhausted);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(bits(model), before);
  std::uint64_t index = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            if (index == 3) return std::nullopt;
            if (index == 1) model.state().parameters().front().add_(1);
            return batch(index++);
          },
          options()),
      CompatibilityError);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, RejectsSameHandleAliasBeforeSourceOrForward) {
  auto toy = std::make_shared<Toy>();
  toy->register_buffer("protected_statistics_alias", toy->bn->running_mean);
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  std::uint64_t calls = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            ++calls;
            return batch(0);
          },
          options()),
      InvalidArgument);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, RejectsStorageViewAliasBeforeSourceOrForward) {
  auto toy = std::make_shared<Toy>();
  toy->register_buffer("protected_statistics_view",
                       toy->bn->running_mean.view({3}));
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  std::uint64_t calls = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            ++calls;
            return batch(0);
          },
          options()),
      InvalidArgument);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibration, RejectsUnrepresentableCounterBeforeSourceOrForward) {
  auto toy = std::make_shared<Toy>();
  auto model = nn::adapt_tensor_module(toy);
  const auto before = bits(model);
  const auto rng = core::get_rng_state(false);
  auto invalid = options();
  invalid.expected_batches =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1;
  invalid.expected_samples = invalid.expected_batches;
  std::uint64_t calls = 0;
  EXPECT_THROW(
      calibrate_batch_norm(
          model,
          [&](std::stop_token) -> std::optional<BatchNormCalibrationBatch> {
            ++calls;
            return batch(0);
          },
          invalid),
      InvalidArgument);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(bits(model), before);
  controls(toy, rng);
}

TEST(BatchNormCalibrationStage, ReadyAndCompletedRoundTripRejectChangedInputs) {
  cli::RecipeBatchNormCalibration stage("{\"training_sha256\":\"a\"}");
  stage.ready(std::string(64, 'c'), 7);
  MemoryState ready(stage.describe_state());
  stage.save_state(ready);
  cli::RecipeBatchNormCalibration resumed("{\"training_sha256\":\"a\"}");
  resumed.load_state(ready, RestorePolicy::exact_only);
  EXPECT_EQ(resumed.report(), stage.report());
  EXPECT_EQ(resumed.phase(), cli::RecipeBatchNormCalibration::Phase::ready);
  resumed.completed(std::string(64, 'd'), 3, 6);
  MemoryState complete(resumed.describe_state());
  resumed.save_state(complete);
  stage.load_state(complete, RestorePolicy::exact_only);
  EXPECT_EQ(stage.phase(), cli::RecipeBatchNormCalibration::Phase::completed);
  EXPECT_EQ(stage.report(), resumed.report());
  cli::RecipeBatchNormCalibration changed("{\"training_sha256\":\"b\"}");
  const auto unchanged = changed.report();
  EXPECT_THROW(changed.load_state(complete, RestorePolicy::exact_only),
               CompatibilityError);
  EXPECT_EQ(changed.report(), unchanged);
  const auto before_corruption = stage.report();
  complete.blobs.at("control").push_back(std::byte{' '});
  EXPECT_THROW(stage.load_state(complete, RestorePolicy::exact_only),
               SerializationError);
  EXPECT_EQ(stage.report(), before_corruption);
}
}  // namespace
