// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/data/iterable_data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace {

using jyppx::liuding::Tensor;

/** @brief In-memory participant state used by checkpoint tests. */
class MemoryState final : public jyppx::liuding::StateWriter,
                          public jyppx::liuding::StateReader {
 public:
  explicit MemoryState(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one bounded participant field. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        blobs_.contains(std::string(name))) {
      throw jyppx::liuding::SerializationError(
          "invalid in-memory participant field");
    }
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  /** @brief Returns the immutable participant manifest. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether the requested field was written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns a bounded immutable field view. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end() || found->second.size() > max_bytes) {
      throw jyppx::liuding::SerializationError(
          "missing in-memory participant field");
    }
    return found->second;
  }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Declared field limits.
  std::map<std::string, std::vector<std::byte>> blobs_;  ///< Stored fields.
};

/** @brief Transactional in-memory checkpoint sink and source. */
class MemoryCheckpoint final
    : public jyppx::liuding::training::CheckpointStateSink,
      public jyppx::liuding::training::CheckpointStateSource {
 public:
  /** @brief Starts a replacement checkpoint transaction. */
  void begin(const jyppx::liuding::training::CheckpointDescriptor& descriptor)
      override {
    descriptor_ = descriptor;
    states_.clear();
    for (const auto& manifest : descriptor.participants) {
      states_.emplace(manifest.participant_id,
                      std::make_shared<MemoryState>(manifest));
    }
    open_ = true;
  }

  /** @brief Returns a writer for one declared participant. */
  jyppx::liuding::StateWriter& writer_for(
      std::string_view participant_id) override {
    if (!open_) {
      throw jyppx::liuding::NotReady("checkpoint transaction is closed");
    }
    return *states_.at(std::string(participant_id));
  }

  /** @brief Publishes the complete transaction after required-field checks. */
  void commit() override {
    for (const auto& manifest : descriptor_.participants) {
      const auto& state = states_.at(manifest.participant_id);
      for (const auto& field : manifest.fields) {
        if (field.required && !state->contains(field.name)) {
          throw jyppx::liuding::SerializationError(
              "checkpoint field was not written");
        }
      }
    }
    open_ = false;
  }

  /** @brief Rolls back an in-flight transaction without throwing. */
  void rollback() noexcept override { open_ = false; }

  /** @brief Returns the published descriptor. */
  const jyppx::liuding::training::CheckpointDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Returns one published participant reader. */
  const jyppx::liuding::StateReader& reader_for(
      std::string_view participant_id) const override {
    return *states_.at(std::string(participant_id));
  }

 private:
  jyppx::liuding::training::CheckpointDescriptor
      descriptor_;  ///< Published topology.
  std::map<std::string, std::shared_ptr<MemoryState>> states_;  ///< Fields.
  bool open_ = false;  ///< Whether a transaction is currently being written.
};

/** @brief Sequential finite iterable source for common-pipeline coverage. */
class TestIterableSource final : public jyppx::liuding::data::IterableSource {
 public:
  /** @brief Cursor yielding five deterministic integer samples. */
  class Cursor final : public jyppx::liuding::data::IterableSourceCursor {
   public:
    /** @brief Returns the next sample unless exhausted or cancelled. */
    std::optional<jyppx::liuding::data::IterableSample> next(
        std::stop_token stop) override {
      if (closed_ || stop.stop_requested() || position_ == 5U) {
        return std::nullopt;
      }
      ++position_;
      return jyppx::liuding::data::IterableSample{
          .value = c10::IValue(static_cast<std::int64_t>(position_)),
          .sample_id = position_,
          .watermark = std::to_string(position_)};
    }

    /** @brief Serializes the next source position. */
    std::string save_state() const override {
      return std::to_string(position_);
    }

    /** @brief Restores a bounded source position. */
    void load_state(std::string_view state) override {
      position_ = static_cast<std::size_t>(std::stoull(std::string(state)));
      if (position_ > 5U) {
        throw jyppx::liuding::SerializationError(
            "invalid iterable test position");
      }
    }

    /** @brief Permanently stops the cursor. */
    void close() noexcept override { closed_ = true; }

   private:
    std::size_t position_ = 0;  ///< Number of samples already emitted.
    bool closed_ = false;       ///< Permanent close state.
  };

  /** @brief Reports an exact deterministic source contract. */
  jyppx::liuding::data::IterableSourceDescriptor descriptor() const override {
    return {.source_id = "test.iterable/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::liuding::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens one fresh cursor for the requested epoch. */
  std::unique_ptr<jyppx::liuding::data::IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<Cursor>();
  }
};

/** @brief Records events and emits one optional deterministic command. */
class RecordingCallback final
    : public jyppx::liuding::training::CallbackHandler {
 public:
  using Command =
      std::function<jyppx::liuding::training::CallbackCommandRequest(
          const jyppx::liuding::training::CallbackEvent&)>;

  RecordingCallback(
      std::string callback_id,
      jyppx::liuding::training::CallbackEventKind subscription,
      std::vector<jyppx::liuding::training::CallbackCommandKind> allowed,
      Command command = {})
      : descriptor_{.type_id = "test.callback.v1",
                    .callback_id = std::move(callback_id),
                    .subscriptions = {subscription},
                    .allowed_commands = std::move(allowed)},
        command_(std::move(command)) {}

  /** @brief Returns the immutable callback descriptor. */
  const jyppx::liuding::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records an immutable event and returns its bounded command. */
  jyppx::liuding::training::CallbackCommandRequest handle(
      const jyppx::liuding::training::CallbackEvent& event) override {
    events_.push_back(event.kind);
    return command_ ? command_(event)
                    : jyppx::liuding::training::CallbackCommandRequest{};
  }

  /** @brief Returns all observed event kinds. */
  const std::vector<jyppx::liuding::training::CallbackEventKind>& events()
      const {
    return events_;
  }

 private:
  jyppx::liuding::training::CallbackDescriptor
      descriptor_;   ///< Frozen capabilities.
  Command command_;  ///< Optional command policy.
  std::vector<jyppx::liuding::training::CallbackEventKind>
      events_;  ///< Audit log.
};

struct Domain {
  std::shared_ptr<jyppx::liuding::nn::ModelAdapter>
      model;  ///< Adapted linear model.
  std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>
      optimizer;  ///< Delegated SGD optimizer.
  std::shared_ptr<jyppx::liuding::data::BatchPipeline>
      train;  ///< Training pipeline participant.
  std::shared_ptr<jyppx::liuding::data::BatchPipeline>
      validation;  ///< Independent validation pipeline participant.
};

/** @brief Creates a two-feature linear regression domain. */
Domain make_domain(std::size_t samples = 8, std::size_t batch_size = 2) {
  const auto inputs =
      torch::arange(0, static_cast<std::int64_t>(samples * 2),
                    torch::TensorOptions().dtype(torch::kFloat32));
  const auto features = inputs.view({static_cast<std::int64_t>(samples), 2});
  const auto targets =
      (2.0 * features.select(1, 0) - features.select(1, 1) + 0.5)
          .view({static_cast<std::int64_t>(samples), 1});
  auto train_factory = [features, targets](
                           std::span<const std::size_t> indices,
                           const jyppx::liuding::data::DataWorkerContext&,
                           std::stop_token) {
    std::vector<std::int64_t> rows;
    rows.reserve(indices.size());
    for (const auto index : indices) {
      rows.push_back(static_cast<std::int64_t>(index));
    }
    const auto row_tensor = torch::tensor(rows, torch::kLong);
    return c10::ivalue::Tuple::create({features.index_select(0, row_tensor),
                                       targets.index_select(0, row_tensor)});
  };
  auto make_pipeline = [&](std::string id) {
    return std::make_shared<jyppx::liuding::data::DataPipeline>(
        std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(
            samples),
        jyppx::liuding::data::DataPipelineOptions{
            .pipeline_id = std::move(id),
            .dataset_version = "test.linear/v1",
            .dataset_size = samples,
            .schema =
                jyppx::liuding::data::BatchSchema::any("test.linear.batch", 1),
            .batch_factory_id = "test.linear.factory/v1",
            .batch_size = batch_size},
        train_factory);
  };
  auto holder = torch::nn::Linear(torch::nn::LinearOptions(2, 1));
  Domain domain;
  domain.model = std::make_shared<jyppx::liuding::nn::ModelAdapter>(
      jyppx::liuding::nn::adapt_tensor_holder(holder));
  std::vector<jyppx::liuding::optim::OptimizerParameterBinding> bindings;
  for (const auto& item : domain.model->state().named_parameters()) {
    bindings.push_back({.parameter_id = item.key(), .parameter = item.value()});
  }
  domain.optimizer =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.main",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "main",
               .parameters = std::move(bindings),
               .options = {.learning_rate = 1e-4}}});
  domain.train = make_pipeline("data.train");
  domain.validation = make_pipeline("data.validation");
  return domain;
}

/** @brief Owns two optimizers over disjoint linear-model parameters. */
struct SplitOptimizers {
  std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>
      bias;  ///< Optimizer owning only the bias parameter.
  std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>
      weight;  ///< Optimizer owning only the weight parameter.
};

/** @brief Splits one linear model's trainable parameters by stable key. */
SplitOptimizers make_split_optimizers(const Domain& domain) {
  std::map<std::string, Tensor> parameters;
  for (const auto& item : domain.model->state().named_parameters()) {
    parameters.emplace(item.key(), item.value());
  }
  return {
      .bias = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.bias",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "bias",
               .parameters = {{.parameter_id = "bias",
                               .parameter = parameters.at("bias")}},
               .options = {.learning_rate = 1e-4}}}),
      .weight = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.weight",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "weight",
               .parameters = {{.parameter_id = "weight",
                               .parameter = parameters.at("weight")}},
               .options = {.learning_rate = 1e-4}}})};
}

/** @brief Creates the complete current Trainer request for a domain. */
jyppx::liuding::training::FitRequest make_request(
    const Domain& domain,
    std::shared_ptr<jyppx::liuding::training::CallbackDispatcher> callbacks =
        {},
    std::shared_ptr<jyppx::liuding::training::SchedulerAdapter> scheduler = {},
    std::function<void(const jyppx::liuding::training::CheckpointCoordinator&)>
        checkpoint_action = {},
    jyppx::liuding::Device device = jyppx::liuding::Device(torch::kCPU),
    bool include_cuda_rng = false) {
  auto model_checkpoint =
      std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(
          *domain.model);
  auto optimizer_checkpoint = std::make_shared<
      jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
      domain.optimizer);
  auto rng_checkpoint =
      std::make_shared<jyppx::liuding::core::RngCheckpointParticipant>(
          jyppx::liuding::core::RngCheckpointParticipantOptions{
              .include_cuda = include_cuda_rng});
  auto strategy =
      std::make_shared<jyppx::liuding::training::SingleDeviceStrategy>(
          jyppx::liuding::training::SingleDeviceStrategyOptions{.device =
                                                                    device});
  const auto mapper =
      [device](const jyppx::liuding::data::BatchEnvelope& batch) {
        return jyppx::liuding::nn::Invocation{
            .positional = {
                batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
      };
  const auto target = [device](const jyppx::liuding::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1).toTensor().to(device);
  };
  const auto step = [](const jyppx::liuding::training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    const auto expected = input.target->toTensor();
    return jyppx::liuding::training::StepOutput{
        .loss = (prediction - expected).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size()};
  };
  return {.model = domain.model,
          .optimizers = {{.optimizer = domain.optimizer,
                          .checkpoint = std::move(optimizer_checkpoint),
                          .scheduler = std::move(scheduler)}},
          .training_data = domain.train,
          .training_batch_mapper = mapper,
          .training_target_mapper = target,
          .training_step = step,
          .validation_data = domain.validation,
          .validation_batch_mapper = mapper,
          .validation_target_mapper = target,
          .validation_step = step,
          .strategy = std::move(strategy),
          .callbacks = std::move(callbacks),
          .model_checkpoint = std::move(model_checkpoint),
          .rng_checkpoint = std::move(rng_checkpoint),
          .checkpoint_action = std::move(checkpoint_action)};
}

jyppx::liuding::training::TrainerOptions options(std::uint64_t max_epochs,
                                                 std::uint64_t max_steps = 0) {
  return {.trainer_id = "trainer.test",
          .run_id = "run.test",
          .trace_id = "trace.test",
          .max_epochs = max_epochs,
          .max_steps = max_steps};
}

TEST(TrainerTest, RejectsZeroCheckpointLimitsAtConstruction) {
  auto limits = options(1);
  limits.checkpoint.max_participants = 0;
  EXPECT_THROW(jyppx::liuding::training::Trainer{limits},
               jyppx::liuding::InvalidArgument);
  limits.checkpoint = {};
  limits.checkpoint.max_rollback_bytes = 0;
  EXPECT_THROW(jyppx::liuding::training::Trainer{limits},
               jyppx::liuding::InvalidArgument);
}

TEST(TrainerTest, CheckpointParticipantLimitRejectsBeforeCapture) {
  auto domain = make_domain();
  auto limits = options(1);
  limits.checkpoint.max_participants = 1;
  jyppx::liuding::training::Trainer trainer(limits);
  trainer.prepare(make_request(domain));
  MemoryCheckpoint destination;
  EXPECT_THROW(trainer.save_checkpoint(destination),
               jyppx::liuding::InvalidArgument);
  EXPECT_TRUE(destination.descriptor().participants.empty());
  EXPECT_EQ(domain.optimizer->completed_steps(), 0U);
  EXPECT_EQ(trainer.global_step(), 0U);
}

TEST(TrainerTest, ExactRollbackBudgetRestoresAndOneByteShortPreservesState) {
  auto domain = make_domain();
  jyppx::liuding::training::Trainer source(options(1));
  source.prepare(make_request(domain));
  const auto expected = source.fit();
  MemoryCheckpoint saved;
  source.save_checkpoint(saved);

  // Limits account for the current domain's declared field maxima, including
  // optional fields, rather than the smaller encoded checkpoint payload.
  std::uint64_t required = 0;
  {
    auto probe_domain = make_domain();
    jyppx::liuding::training::Trainer probe(options(1));
    probe.prepare(make_request(probe_domain));
    MemoryCheckpoint current;
    const auto descriptor = probe.save_checkpoint(current);
    for (const auto& participant : descriptor.participants) {
      for (const auto& field : participant.fields) required += field.max_bytes;
    }
  }
  ASSERT_GT(required, 1U);
  for (const auto shortage : {1U, 0U}) {
    SCOPED_TRACE(shortage);
    auto rebuilt = make_domain();
    auto limits = options(1);
    limits.checkpoint.max_rollback_bytes = required - shortage;
    jyppx::liuding::training::Trainer restored(limits);
    restored.prepare(make_request(rebuilt));
    const auto initial = rebuilt.model->state().parameters().front().clone();
    if (shortage != 0U) {
      MemoryCheckpoint before;
      restored.save_checkpoint(before);
      EXPECT_THROW(restored.restore_checkpoint(saved),
                   jyppx::liuding::ResourceExhausted);
      EXPECT_TRUE(
          torch::equal(initial, rebuilt.model->state().parameters().front()));
      EXPECT_EQ(rebuilt.optimizer->completed_steps(), 0U);
      EXPECT_EQ(restored.global_step(), 0U);
      EXPECT_EQ(restored.phase(),
                jyppx::liuding::training::TrainerPhase::prepared);
      MemoryCheckpoint after;
      restored.save_checkpoint(after);
      for (const auto& participant : before.descriptor().participants) {
        const auto& left = before.reader_for(participant.participant_id);
        const auto& right = after.reader_for(participant.participant_id);
        for (const auto& field : participant.fields) {
          ASSERT_EQ(left.contains(field.name), right.contains(field.name));
          if (!left.contains(field.name)) continue;
          const auto expected_blob =
              left.read_blob(field.name, field.max_bytes);
          const auto actual_blob = right.read_blob(field.name, field.max_bytes);
          EXPECT_TRUE(std::equal(expected_blob.begin(), expected_blob.end(),
                                 actual_blob.begin(), actual_blob.end()));
        }
      }
    } else {
      restored.restore_checkpoint(saved);
      EXPECT_EQ(restored.global_step(), expected.global_step);
      EXPECT_EQ(rebuilt.optimizer->completed_steps(), expected.global_step);
      const auto actual_parameters = rebuilt.model->state().parameters();
      const auto expected_parameters = domain.model->state().parameters();
      ASSERT_EQ(actual_parameters.size(), expected_parameters.size());
      for (std::size_t i = 0; i < actual_parameters.size(); ++i) {
        EXPECT_TRUE(torch::equal(actual_parameters[i], expected_parameters[i]));
      }
      EXPECT_EQ(restored.phase(),
                jyppx::liuding::training::TrainerPhase::completed);
    }
  }
}

TEST(TrainerTest, FitMapPipelinePublishesOrderedEventsAndMetrics) {
  auto domain = make_domain();
  auto callback = std::make_shared<RecordingCallback>(
      "callback.record",
      jyppx::liuding::training::CallbackEventKind::run_finished,
      std::vector<jyppx::liuding::training::CallbackCommandKind>{});
  auto dispatcher = std::make_shared<
      jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          callback});
  jyppx::liuding::training::Trainer trainer(options(2));
  trainer.prepare(make_request(domain, dispatcher));
  const auto result = trainer.fit();

  EXPECT_EQ(result.status, jyppx::liuding::training::FitStatus::completed);
  EXPECT_EQ(result.completed_epochs, 2U);
  EXPECT_EQ(result.global_step, 8U);
  EXPECT_TRUE(result.training_metrics.contains("training.loss"));
  EXPECT_TRUE(result.validation_metrics.contains("validation.loss"));
  ASSERT_EQ(callback->events().size(), 1U);
  EXPECT_EQ(callback->events().front(),
            jyppx::liuding::training::CallbackEventKind::run_finished);
  EXPECT_EQ(dispatcher->last_event_index(), 58U);
  trainer.close();
  EXPECT_EQ(trainer.phase(), jyppx::liuding::training::TrainerPhase::closed);
}

TEST(TrainerTest, CommitsDisjointOptimizersAsOneLogicalStep) {
  auto domain = make_domain();
  auto optimizers = make_split_optimizers(domain);
  auto request = make_request(domain);
  request.optimizers = {
      {.optimizer = optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.weight)},
      {.optimizer = optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.bias)}};
  const auto strategy =
      std::dynamic_pointer_cast<jyppx::liuding::training::SingleDeviceStrategy>(
          request.strategy);
  jyppx::liuding::training::Trainer trainer(options(1));
  trainer.prepare(std::move(request));
  const auto result = trainer.fit();

  EXPECT_EQ(result.global_step, 4U);
  EXPECT_EQ(optimizers.bias->completed_steps(), result.global_step);
  EXPECT_EQ(optimizers.weight->completed_steps(), result.global_step);
  ASSERT_TRUE(strategy);
  EXPECT_EQ(strategy->completed_optimizer_steps("optimizer.bias"), 4U);
  EXPECT_EQ(strategy->completed_optimizer_steps("optimizer.weight"), 4U);
  const auto descriptor = trainer.checkpoint_coordinator().describe();
  EXPECT_TRUE(
      std::any_of(descriptor.participants.begin(),
                  descriptor.participants.end(), [](const auto& participant) {
                    return participant.participant_id == "optimizer.bias";
                  }));
  EXPECT_TRUE(
      std::any_of(descriptor.participants.begin(),
                  descriptor.participants.end(), [](const auto& participant) {
                    return participant.participant_id == "optimizer.weight";
                  }));
}

TEST(TrainerTest, TracksIndependentSchedulerEventCursors) {
  auto domain = make_domain();
  auto optimizers = make_split_optimizers(domain);
  auto epoch_scheduler =
      std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
          optimizers.weight, jyppx::liuding::optim::StepLrSchedulerOptions{
                                 .scheduler_id = "scheduler.weight",
                                 .step_size = 1,
                                 .gamma = 0.5});
  auto metric_scheduler =
      std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
          optimizers.bias,
          jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
              .scheduler_id = "scheduler.bias", .patience = 0});
  auto request = make_request(domain);
  request.optimizers = {
      {.optimizer = optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.weight),
       .scheduler =
           std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
               epoch_scheduler)},
      {.optimizer = optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.bias),
       .scheduler =
           std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
               metric_scheduler)}};
  jyppx::liuding::training::Trainer trainer(options(2));
  trainer.prepare(std::move(request));
  EXPECT_EQ(trainer.fit().completed_epochs, 2U);
  EXPECT_EQ(epoch_scheduler->completed_steps(), 2U);
  EXPECT_EQ(metric_scheduler->completed_steps(), 2U);
}

TEST(TrainerTest, IterablePipelineCommitsFiniteBatches) {
  auto pipeline = std::make_shared<jyppx::liuding::data::IterableDataPipeline>(
      std::make_shared<TestIterableSource>(),
      jyppx::liuding::data::IterableDataPipelineOptions{
          .pipeline_id = "data.iterable.test",
          .dataset_version = "test.iterable/v1",
          .batch_factory_id = "test.iterable.factory/v1",
          .batch_size = 2},
      [](std::span<const jyppx::liuding::data::IterableSample> samples,
         std::uint64_t, std::uint64_t, std::stop_token) {
        return c10::IValue(static_cast<std::int64_t>(samples.size()));
      });
  pipeline->set_epoch(0);
  const auto first = pipeline->next();
  const auto second = pipeline->next();
  const auto third = pipeline->next();
  EXPECT_TRUE(first.has_value());
  EXPECT_TRUE(second.has_value());
  EXPECT_TRUE(third.has_value());
  EXPECT_EQ(first->value.toInt(), 2);
  EXPECT_EQ(second->value.toInt(), 2);
  EXPECT_EQ(third->value.toInt(), 1);
  EXPECT_EQ(third->source_watermark, "5");
  EXPECT_TRUE(pipeline->capability().iterable_style);
  pipeline->close();
}

TEST(TrainerTest, CpuBfloat16PrecisionScopesTrainingAndValidationAutocast) {
  auto domain = make_domain();
  auto request = make_request(domain);
  auto training_step = request.training_step;
  auto validation_step = request.validation_step;
  bool training_autocast = false;
  bool validation_autocast = false;
  request.training_step =
      [&](const jyppx::liuding::training::StepInput& input) {
        training_autocast =
            at::autocast::is_autocast_enabled(c10::DeviceType::CPU) &&
            at::autocast::get_autocast_dtype(c10::DeviceType::CPU) ==
                at::kBFloat16;
        return training_step(input);
      };
  request.validation_step =
      [&](const jyppx::liuding::training::StepInput& input) {
        validation_autocast =
            at::autocast::is_autocast_enabled(c10::DeviceType::CPU) &&
            at::autocast::get_autocast_dtype(c10::DeviceType::CPU) ==
                at::kBFloat16;
        return validation_step(input);
      };
  jyppx::liuding::training::Trainer trainer(
      {.trainer_id = "trainer.bfloat16",
       .run_id = "run.bfloat16",
       .trace_id = "trace.bfloat16",
       .max_epochs = 1,
       .precision = jyppx::liuding::training::PrecisionMode::cpu_bfloat16});
  trainer.prepare(std::move(request));
  EXPECT_EQ(trainer.fit().global_step, 4U);
  EXPECT_TRUE(training_autocast);
  EXPECT_TRUE(validation_autocast);
  EXPECT_FALSE(at::autocast::is_autocast_enabled(c10::DeviceType::CPU));
}

TEST(TrainerTest, FullPrecisionRejectsGradientScalerState) {
  auto domain = make_domain();
  auto request = make_request(domain);
  request.gradient_scaler =
      std::make_shared<jyppx::liuding::training::amp::GradScaler>();
  request.gradient_scaler_checkpoint = std::make_shared<
      jyppx::liuding::training::amp::GradScalerCheckpointParticipant>(
      request.gradient_scaler);
  jyppx::liuding::training::Trainer trainer(options(1));
  EXPECT_THROW(trainer.prepare(std::move(request)),
               jyppx::liuding::InvalidArgument);
}

TEST(TrainerCudaTest, Float16PrecisionOwnsMultiOptimizerScalerCycle) {
  if (!jyppx::liuding::cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  auto domain = make_domain();
  auto optimizers = make_split_optimizers(domain);
  auto request = make_request(domain, {}, {}, {},
                              jyppx::liuding::Device(torch::kCUDA, 0), true);
  request.optimizers = {
      {.optimizer = optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.weight)},
      {.optimizer = optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.bias)}};
  auto scaler = std::make_shared<jyppx::liuding::training::amp::GradScaler>(
      jyppx::liuding::training::amp::GradScalerOptions{
          .init_scale = 8.0,
          .growth_factor = 2.0,
          .growth_interval = 1,
          .scaler_id = "grad_scaler.trainer.cuda"});
  auto scaler_checkpoint = std::make_shared<
      jyppx::liuding::training::amp::GradScalerCheckpointParticipant>(scaler);
  request.gradient_scaler = scaler;
  request.gradient_scaler_checkpoint = scaler_checkpoint;
  jyppx::liuding::training::Trainer trainer(
      {.trainer_id = "trainer.cuda.float16",
       .run_id = "run.cuda.float16",
       .trace_id = "trace.cuda.float16",
       .max_epochs = 1,
       .precision = jyppx::liuding::training::PrecisionMode::cuda_float16});
  trainer.prepare(std::move(request));
  const auto result = trainer.fit();
  EXPECT_EQ(result.global_step, 4U);
  EXPECT_EQ(optimizers.bias->completed_steps(), 4U);
  EXPECT_EQ(optimizers.weight->completed_steps(), 4U);
  EXPECT_DOUBLE_EQ(scaler->get_scale(), 128.0);
  EXPECT_TRUE(scaler->checkpoint_ready());
  const auto descriptor = trainer.checkpoint_coordinator().describe();
  EXPECT_TRUE(std::any_of(
      descriptor.participants.begin(), descriptor.participants.end(),
      [&](const jyppx::liuding::StateManifest& participant) {
        return participant.participant_id == scaler->options().scaler_id;
      }));
}

TEST(TrainerTest, AccumulationTailFlushAndDropAreExplicit) {
  auto flush_domain = make_domain(3, 2);
  jyppx::liuding::training::Trainer flush_trainer(
      {.trainer_id = "trainer.flush",
       .run_id = "run.flush",
       .trace_id = "trace.flush",
       .max_epochs = 1,
       .accumulate_steps = 3,
       .accumulation_tail =
           jyppx::liuding::training::AccumulationTailPolicy::flush});
  flush_trainer.prepare(make_request(flush_domain));
  EXPECT_EQ(flush_trainer.fit().global_step, 1U);

  auto drop_domain = make_domain(3, 2);
  jyppx::liuding::training::Trainer drop_trainer(
      {.trainer_id = "trainer.drop",
       .run_id = "run.drop",
       .trace_id = "trace.drop",
       .max_epochs = 1,
       .accumulate_steps = 3,
       .accumulation_tail =
           jyppx::liuding::training::AccumulationTailPolicy::drop});
  drop_trainer.prepare(make_request(drop_domain));
  EXPECT_EQ(drop_trainer.fit().global_step, 0U);
}

TEST(TrainerTest, EpochSchedulerAndMetricSchedulerReceiveStrictEvents) {
  auto epoch_domain = make_domain();
  auto epoch_scheduler =
      std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
          epoch_domain.optimizer,
          jyppx::liuding::optim::StepLrSchedulerOptions{
              .scheduler_id = "scheduler.epoch", .step_size = 1, .gamma = 0.5});
  auto epoch_adapter =
      std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
          epoch_scheduler);
  jyppx::liuding::training::Trainer epoch_trainer(options(2));
  epoch_trainer.prepare(make_request(epoch_domain, {}, epoch_adapter));
  EXPECT_EQ(epoch_trainer.fit().completed_epochs, 2U);
  EXPECT_EQ(epoch_scheduler->completed_steps(), 2U);

  auto metric_domain = make_domain();
  auto metric_scheduler =
      std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
          metric_domain.optimizer,
          jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
              .scheduler_id = "scheduler.metric", .patience = 0});
  auto metric_adapter =
      std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
          metric_scheduler);
  jyppx::liuding::training::Trainer metric_trainer(options(2));
  metric_trainer.prepare(make_request(metric_domain, {}, metric_adapter));
  EXPECT_EQ(metric_trainer.fit().completed_epochs, 2U);
  EXPECT_EQ(metric_scheduler->completed_steps(), 2U);
}

TEST(TrainerTest, CallbackStopRequestsSafeBoundary) {
  auto domain = make_domain();
  auto callback = std::make_shared<RecordingCallback>(
      "callback.stop",
      jyppx::liuding::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::liuding::training::CallbackCommandKind>{
          jyppx::liuding::training::CallbackCommandKind::stop},
      [](const jyppx::liuding::training::CallbackEvent&) {
        return jyppx::liuding::training::CallbackCommandRequest{
            .kind = jyppx::liuding::training::CallbackCommandKind::stop,
            .reason = "test stop"};
      });
  auto dispatcher = std::make_shared<
      jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          callback});
  jyppx::liuding::training::Trainer trainer(options(10));
  trainer.prepare(make_request(domain, dispatcher));
  const auto result = trainer.fit();
  EXPECT_EQ(result.status, jyppx::liuding::training::FitStatus::stopped);
  EXPECT_EQ(result.completed_epochs, 0U);
  EXPECT_EQ(result.global_step, 1U);
}

TEST(TrainerTest, CheckpointResumeMatchesUninterruptedMidEpoch) {
  auto source_domain = make_domain(8, 2);
  auto source_optimizers = make_split_optimizers(source_domain);
  auto checkpoint = std::make_shared<MemoryCheckpoint>();
  auto request = make_request(source_domain);
  request.optimizers = {
      {.optimizer = source_optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           source_optimizers.weight)},
      {.optimizer = source_optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           source_optimizers.bias)}};
  auto callback = std::make_shared<RecordingCallback>(
      "callback.checkpoint",
      jyppx::liuding::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::liuding::training::CallbackCommandKind>{
          jyppx::liuding::training::CallbackCommandKind::request_checkpoint},
      [triggered = false](
          const jyppx::liuding::training::CallbackEvent& event) mutable {
        if (!triggered &&
            event.phase == jyppx::liuding::training::CallbackPhase::training) {
          triggered = true;
          return jyppx::liuding::training::CallbackCommandRequest{
              .kind = jyppx::liuding::training::CallbackCommandKind::
                  request_checkpoint,
              .reason = "mid-epoch test checkpoint"};
        }
        return jyppx::liuding::training::CallbackCommandRequest{};
      });
  request.callbacks = std::make_shared<
      jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          callback});
  request.checkpoint_action =
      [checkpoint](
          const jyppx::liuding::training::CheckpointCoordinator& coordinator) {
        coordinator.save(*checkpoint);
      };
  jyppx::liuding::training::Trainer source(options(2));
  source.prepare(std::move(request));
  const auto expected = source.fit();
  ASSERT_EQ(expected.status, jyppx::liuding::training::FitStatus::completed);

  auto restored_domain = make_domain(8, 2);
  auto restored_optimizers = make_split_optimizers(restored_domain);
  jyppx::liuding::training::Trainer restored(options(2));
  auto restored_callback = std::make_shared<RecordingCallback>(
      "callback.checkpoint",
      jyppx::liuding::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::liuding::training::CallbackCommandKind>{
          jyppx::liuding::training::CallbackCommandKind::request_checkpoint});
  auto restored_dispatcher = std::make_shared<
      jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          restored_callback});
  auto restored_request = make_request(restored_domain, restored_dispatcher);
  restored_request.optimizers = {
      {.optimizer = restored_optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           restored_optimizers.weight)},
      {.optimizer = restored_optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           restored_optimizers.bias)}};
  restored.prepare(std::move(restored_request));
  restored.restore_checkpoint(*checkpoint);
  const auto actual = restored.fit();
  EXPECT_EQ(actual.global_step, expected.global_step);
  EXPECT_EQ(actual.completed_epochs, expected.completed_epochs);
  EXPECT_EQ(restored_optimizers.bias->completed_steps(), actual.global_step);
  EXPECT_EQ(restored_optimizers.weight->completed_steps(), actual.global_step);
  EXPECT_TRUE(torch::allclose(restored_domain.model->state().parameters().at(0),
                              source_domain.model->state().parameters().at(0),
                              0.0, 0.0));
  EXPECT_TRUE(torch::allclose(restored_domain.model->state().parameters().at(1),
                              source_domain.model->state().parameters().at(1),
                              0.0, 0.0));
}

TEST(TrainerTest, InvalidStepOutputFailsAndClearsPendingGradients) {
  auto domain = make_domain();
  auto request = make_request(domain);
  request.training_step = [](const jyppx::liuding::training::StepInput&) {
    return jyppx::liuding::training::StepOutput{.sample_count = 0};
  };
  jyppx::liuding::training::Trainer trainer(options(1));
  trainer.prepare(std::move(request));
  EXPECT_THROW(trainer.fit(), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(trainer.phase(), jyppx::liuding::training::TrainerPhase::failed);
  EXPECT_EQ(trainer.global_step(), 0U);
}

TEST(TrainerTest,
     PreparationRejectsNonFreshOptimizerAndSharedValidationCursor) {
  auto domain = make_domain();
  domain.optimizer->zero_grad();
  static_cast<void>(domain.optimizer->step());
  jyppx::liuding::training::Trainer trainer(options(1));
  EXPECT_THROW(trainer.prepare(make_request(domain)), jyppx::liuding::NotReady);

  auto fresh = make_domain();
  auto request = make_request(fresh);
  request.validation_data = request.training_data;
  jyppx::liuding::training::Trainer shared_cursor(options(1));
  EXPECT_THROW(shared_cursor.prepare(std::move(request)),
               jyppx::liuding::InvalidArgument);
}

TEST(TrainerTest, PreparationRejectsDuplicateOptimizerIdentities) {
  auto domain = make_domain();
  auto optimizers = make_split_optimizers(domain);
  const auto bias_bindings = optimizers.bias->parameter_bindings();
  auto duplicate_identity =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.weight",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "bias",
               .parameters = std::vector<
                   jyppx::liuding::optim::OptimizerParameterBinding>(
                   bias_bindings.begin(), bias_bindings.end()),
               .options = {.learning_rate = 1e-4}}});
  auto request = make_request(domain);
  request.optimizers = {
      {.optimizer = optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.weight)},
      {.optimizer = duplicate_identity,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           duplicate_identity)}};
  jyppx::liuding::training::Trainer trainer(options(1));

  EXPECT_THROW(trainer.prepare(std::move(request)),
               jyppx::liuding::InvalidArgument);
}

TEST(TrainerTest, PreparationRejectsSchedulerForDifferentOptimizer) {
  auto domain = make_domain();
  auto optimizers = make_split_optimizers(domain);
  auto mismatched_scheduler =
      std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
          optimizers.bias, jyppx::liuding::optim::StepLrSchedulerOptions{
                               .scheduler_id = "scheduler.mismatched",
                               .step_size = 1,
                               .gamma = 0.5});
  auto request = make_request(domain);
  request.optimizers = {
      {.optimizer = optimizers.weight,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.weight),
       .scheduler =
           std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
               mismatched_scheduler)},
      {.optimizer = optimizers.bias,
       .checkpoint = std::make_shared<
           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
           optimizers.bias)}};
  jyppx::liuding::training::Trainer trainer(options(1));

  EXPECT_THROW(trainer.prepare(std::move(request)),
               jyppx::liuding::CompatibilityError);
}

}  // namespace
