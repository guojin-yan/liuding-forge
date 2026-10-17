// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data/iterable_data_pipeline.hpp>
#include <jyppx/data/samplers/sequential_sampler.hpp>
#include <jyppx/foundation/random_checkpoint_participant.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>
#include <jyppx/optim/event_lr_scheduler.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/scheduler_adapter.hpp>
#include <jyppx/training/single_device_strategy.hpp>
#include <jyppx/training/trainer.hpp>

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

using jyppx::Tensor;

/** @brief In-memory participant state used by checkpoint tests. */
class MemoryState final : public jyppx::StateWriter, public jyppx::StateReader {
 public:
  explicit MemoryState(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one bounded participant field. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        blobs_.contains(std::string(name))) {
      throw jyppx::SerializationError("invalid in-memory participant field");
    }
    blobs_.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  /** @brief Returns the immutable participant manifest. */
  const jyppx::StateManifest& manifest() const noexcept override {
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
      throw jyppx::SerializationError("missing in-memory participant field");
    }
    return found->second;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Declared field limits.
  std::map<std::string, std::vector<std::byte>> blobs_;  ///< Stored fields.
};

/** @brief Transactional in-memory checkpoint sink and source. */
class MemoryCheckpoint final : public jyppx::training::CheckpointStateSink,
                               public jyppx::training::CheckpointStateSource {
 public:
  /** @brief Starts a replacement checkpoint transaction. */
  void begin(const jyppx::training::CheckpointDescriptor& descriptor) override {
    descriptor_ = descriptor;
    states_.clear();
    for (const auto& manifest : descriptor.participants) {
      states_.emplace(manifest.participant_id,
                      std::make_shared<MemoryState>(manifest));
    }
    open_ = true;
  }

  /** @brief Returns a writer for one declared participant. */
  jyppx::StateWriter& writer_for(std::string_view participant_id) override {
    if (!open_) {
      throw jyppx::NotReady("checkpoint transaction is closed");
    }
    return *states_.at(std::string(participant_id));
  }

  /** @brief Publishes the complete transaction after required-field checks. */
  void commit() override {
    for (const auto& manifest : descriptor_.participants) {
      const auto& state = states_.at(manifest.participant_id);
      for (const auto& field : manifest.fields) {
        if (field.required && !state->contains(field.name)) {
          throw jyppx::SerializationError("checkpoint field was not written");
        }
      }
    }
    open_ = false;
  }

  /** @brief Rolls back an in-flight transaction without throwing. */
  void rollback() noexcept override { open_ = false; }

  /** @brief Returns the published descriptor. */
  const jyppx::training::CheckpointDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Returns one published participant reader. */
  const jyppx::StateReader& reader_for(
      std::string_view participant_id) const override {
    return *states_.at(std::string(participant_id));
  }

 private:
  jyppx::training::CheckpointDescriptor descriptor_;  ///< Published topology.
  std::map<std::string, std::shared_ptr<MemoryState>> states_;  ///< Fields.
  bool open_ = false;  ///< Whether a transaction is currently being written.
};

/** @brief Sequential finite iterable source for common-pipeline coverage. */
class TestIterableSource final : public jyppx::data::IterableSource {
 public:
  /** @brief Cursor yielding five deterministic integer samples. */
  class Cursor final : public jyppx::data::IterableSourceCursor {
   public:
    /** @brief Returns the next sample unless exhausted or cancelled. */
    std::optional<jyppx::data::IterableSample> next(
        std::stop_token stop) override {
      if (closed_ || stop.stop_requested() || position_ == 5U) {
        return std::nullopt;
      }
      ++position_;
      return jyppx::data::IterableSample{
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
        throw jyppx::SerializationError("invalid iterable test position");
      }
    }

    /** @brief Permanently stops the cursor. */
    void close() noexcept override { closed_ = true; }

   private:
    std::size_t position_ = 0;  ///< Number of samples already emitted.
    bool closed_ = false;       ///< Permanent close state.
  };

  /** @brief Reports an exact deterministic source contract. */
  jyppx::data::IterableSourceDescriptor descriptor() const override {
    return {.source_id = "test.iterable/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens one fresh cursor for the requested epoch. */
  std::unique_ptr<jyppx::data::IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<Cursor>();
  }
};

/** @brief Records events and emits one optional deterministic command. */
class RecordingCallback final : public jyppx::training::CallbackHandler {
 public:
  using Command = std::function<jyppx::training::CallbackCommandRequest(
      const jyppx::training::CallbackEvent&)>;

  RecordingCallback(std::string callback_id,
                    jyppx::training::CallbackEventKind subscription,
                    std::vector<jyppx::training::CallbackCommandKind> allowed,
                    Command command = {})
      : descriptor_{.type_id = "test.callback.v1",
                    .callback_id = std::move(callback_id),
                    .subscriptions = {subscription},
                    .allowed_commands = std::move(allowed)},
        command_(std::move(command)) {}

  /** @brief Returns the immutable callback descriptor. */
  const jyppx::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  /** @brief Records an immutable event and returns its bounded command. */
  jyppx::training::CallbackCommandRequest handle(
      const jyppx::training::CallbackEvent& event) override {
    events_.push_back(event.kind);
    return command_ ? command_(event)
                    : jyppx::training::CallbackCommandRequest{};
  }

  /** @brief Returns all observed event kinds. */
  const std::vector<jyppx::training::CallbackEventKind>& events() const {
    return events_;
  }

 private:
  jyppx::training::CallbackDescriptor descriptor_;  ///< Frozen capabilities.
  Command command_;  ///< Optional command policy.
  std::vector<jyppx::training::CallbackEventKind> events_;  ///< Audit log.
};

struct Domain {
  std::shared_ptr<jyppx::nn::ModelAdapter> model;  ///< Adapted linear model.
  std::shared_ptr<jyppx::optim::SgdOptimizerAdapter>
      optimizer;  ///< Delegated SGD optimizer.
  std::shared_ptr<jyppx::data::BatchPipeline>
      train;  ///< Training pipeline participant.
  std::shared_ptr<jyppx::data::BatchPipeline>
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
                           const jyppx::data::DataWorkerContext&,
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
    return std::make_shared<jyppx::data::DataPipeline>(
        std::make_shared<jyppx::data::samplers::SequentialSampler>(samples),
        jyppx::data::DataPipelineOptions{
            .pipeline_id = std::move(id),
            .dataset_version = "test.linear/v1",
            .dataset_size = samples,
            .schema_id = "test.linear.batch/v1",
            .batch_factory_id = "test.linear.factory/v1",
            .batch_size = batch_size},
        train_factory);
  };
  auto holder = torch::nn::Linear(torch::nn::LinearOptions(2, 1));
  Domain domain;
  domain.model = std::make_shared<jyppx::nn::ModelAdapter>(
      jyppx::nn::adapt_tensor_holder(holder));
  std::vector<jyppx::optim::OptimizerParameterBinding> bindings;
  for (const auto& item : domain.model->state().named_parameters()) {
    bindings.push_back({.parameter_id = item.key(), .parameter = item.value()});
  }
  domain.optimizer = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.main", std::vector<jyppx::optim::SgdParameterGroup>{
                            {.group_id = "main",
                             .parameters = std::move(bindings),
                             .options = {.learning_rate = 1e-4}}});
  domain.train = make_pipeline("data.train");
  domain.validation = make_pipeline("data.validation");
  return domain;
}

/** @brief Creates the complete current Trainer request for a domain. */
jyppx::training::FitRequest make_request(
    const Domain& domain,
    std::shared_ptr<jyppx::training::CallbackDispatcher> callbacks = {},
    std::shared_ptr<jyppx::training::SchedulerAdapter> scheduler = {},
    std::function<void(const jyppx::training::CheckpointCoordinator&)>
        checkpoint_action = {}) {
  auto model_checkpoint =
      std::make_shared<jyppx::nn::ModelCheckpointParticipant>(*domain.model);
  auto optimizer_checkpoint =
      std::make_shared<jyppx::optim::SgdOptimizerCheckpointParticipant>(
          domain.optimizer);
  auto rng_checkpoint = std::make_shared<jyppx::core::RngCheckpointParticipant>(
      jyppx::core::RngCheckpointParticipantOptions{.include_cuda = false});
  auto strategy = std::make_shared<jyppx::training::SingleDeviceStrategy>();
  const auto mapper = [](const jyppx::data::BatchEnvelope& batch) {
    return jyppx::nn::Invocation{
        .positional = {batch.value.toTuple()->elements().at(0)}};
  };
  const auto target = [](const jyppx::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1);
  };
  const auto step = [](const jyppx::training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    const auto expected = input.target->toTensor();
    return jyppx::training::StepOutput{
        .loss = (prediction - expected).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size()};
  };
  return {.model = domain.model,
          .optimizer = domain.optimizer,
          .training_data = domain.train,
          .training_batch_mapper = mapper,
          .training_target_mapper = target,
          .training_step = step,
          .validation_data = domain.validation,
          .validation_batch_mapper = mapper,
          .validation_target_mapper = target,
          .validation_step = step,
          .strategy = std::move(strategy),
          .scheduler = std::move(scheduler),
          .callbacks = std::move(callbacks),
          .model_checkpoint = std::move(model_checkpoint),
          .optimizer_checkpoint = std::move(optimizer_checkpoint),
          .rng_checkpoint = std::move(rng_checkpoint),
          .checkpoint_action = std::move(checkpoint_action)};
}

jyppx::training::TrainerOptions options(std::uint64_t max_epochs,
                                        std::uint64_t max_steps = 0) {
  return {.trainer_id = "trainer.test",
          .run_id = "run.test",
          .trace_id = "trace.test",
          .max_epochs = max_epochs,
          .max_steps = max_steps};
}

TEST(TrainerTest, FitMapPipelinePublishesOrderedEventsAndMetrics) {
  auto domain = make_domain();
  auto callback = std::make_shared<RecordingCallback>(
      "callback.record", jyppx::training::CallbackEventKind::run_finished,
      std::vector<jyppx::training::CallbackCommandKind>{});
  auto dispatcher = std::make_shared<jyppx::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{callback});
  jyppx::training::Trainer trainer(options(2));
  trainer.prepare(make_request(domain, dispatcher));
  const auto result = trainer.fit();

  EXPECT_EQ(result.status, jyppx::training::FitStatus::completed);
  EXPECT_EQ(result.completed_epochs, 2U);
  EXPECT_EQ(result.global_step, 8U);
  EXPECT_TRUE(result.training_metrics.contains("training.loss"));
  EXPECT_TRUE(result.validation_metrics.contains("validation.loss"));
  ASSERT_EQ(callback->events().size(), 1U);
  EXPECT_EQ(callback->events().front(),
            jyppx::training::CallbackEventKind::run_finished);
  EXPECT_EQ(dispatcher->last_event_index(), 58U);
  trainer.close();
  EXPECT_EQ(trainer.phase(), jyppx::training::TrainerPhase::closed);
}

TEST(TrainerTest, IterablePipelineCommitsFiniteBatches) {
  auto pipeline = std::make_shared<jyppx::data::IterableDataPipeline>(
      std::make_shared<TestIterableSource>(),
      jyppx::data::IterableDataPipelineOptions{
          .pipeline_id = "data.iterable.test",
          .dataset_version = "test.iterable/v1",
          .batch_factory_id = "test.iterable.factory/v1",
          .batch_size = 2},
      [](std::span<const jyppx::data::IterableSample> samples, std::uint64_t,
         std::uint64_t, std::stop_token) {
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

TEST(TrainerTest, AccumulationTailFlushAndDropAreExplicit) {
  auto flush_domain = make_domain(3, 2);
  jyppx::training::Trainer flush_trainer(
      {.trainer_id = "trainer.flush",
       .run_id = "run.flush",
       .trace_id = "trace.flush",
       .max_epochs = 1,
       .accumulate_steps = 3,
       .accumulation_tail = jyppx::training::AccumulationTailPolicy::flush});
  flush_trainer.prepare(make_request(flush_domain));
  EXPECT_EQ(flush_trainer.fit().global_step, 1U);

  auto drop_domain = make_domain(3, 2);
  jyppx::training::Trainer drop_trainer(
      {.trainer_id = "trainer.drop",
       .run_id = "run.drop",
       .trace_id = "trace.drop",
       .max_epochs = 1,
       .accumulate_steps = 3,
       .accumulation_tail = jyppx::training::AccumulationTailPolicy::drop});
  drop_trainer.prepare(make_request(drop_domain));
  EXPECT_EQ(drop_trainer.fit().global_step, 0U);
}

TEST(TrainerTest, EpochSchedulerAndMetricSchedulerReceiveStrictEvents) {
  auto epoch_domain = make_domain();
  auto epoch_scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      epoch_domain.optimizer,
      jyppx::optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.epoch", .step_size = 1, .gamma = 0.5});
  auto epoch_adapter =
      std::make_shared<jyppx::training::SchedulerAdapter>(epoch_scheduler);
  jyppx::training::Trainer epoch_trainer(options(2));
  epoch_trainer.prepare(make_request(epoch_domain, {}, epoch_adapter));
  EXPECT_EQ(epoch_trainer.fit().completed_epochs, 2U);
  EXPECT_EQ(epoch_scheduler->completed_steps(), 2U);

  auto metric_domain = make_domain();
  auto metric_scheduler =
      std::make_shared<jyppx::optim::ReduceLrOnPlateauScheduler>(
          metric_domain.optimizer,
          jyppx::optim::ReduceLrOnPlateauSchedulerOptions{
              .scheduler_id = "scheduler.metric", .patience = 0});
  auto metric_adapter =
      std::make_shared<jyppx::training::SchedulerAdapter>(metric_scheduler);
  jyppx::training::Trainer metric_trainer(options(2));
  metric_trainer.prepare(make_request(metric_domain, {}, metric_adapter));
  EXPECT_EQ(metric_trainer.fit().completed_epochs, 2U);
  EXPECT_EQ(metric_scheduler->completed_steps(), 2U);
}

TEST(TrainerTest, CallbackStopRequestsSafeBoundary) {
  auto domain = make_domain();
  auto callback = std::make_shared<RecordingCallback>(
      "callback.stop", jyppx::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::training::CallbackCommandKind>{
          jyppx::training::CallbackCommandKind::stop},
      [](const jyppx::training::CallbackEvent&) {
        return jyppx::training::CallbackCommandRequest{
            .kind = jyppx::training::CallbackCommandKind::stop,
            .reason = "test stop"};
      });
  auto dispatcher = std::make_shared<jyppx::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{callback});
  jyppx::training::Trainer trainer(options(10));
  trainer.prepare(make_request(domain, dispatcher));
  const auto result = trainer.fit();
  EXPECT_EQ(result.status, jyppx::training::FitStatus::stopped);
  EXPECT_EQ(result.completed_epochs, 0U);
  EXPECT_EQ(result.global_step, 1U);
}

TEST(TrainerTest, CheckpointResumeMatchesUninterruptedMidEpoch) {
  auto source_domain = make_domain(8, 2);
  auto checkpoint = std::make_shared<MemoryCheckpoint>();
  auto request = make_request(source_domain);
  auto callback = std::make_shared<RecordingCallback>(
      "callback.checkpoint", jyppx::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::training::CallbackCommandKind>{
          jyppx::training::CallbackCommandKind::request_checkpoint},
      [triggered = false](const jyppx::training::CallbackEvent& event) mutable {
        if (!triggered &&
            event.phase == jyppx::training::CallbackPhase::training) {
          triggered = true;
          return jyppx::training::CallbackCommandRequest{
              .kind = jyppx::training::CallbackCommandKind::request_checkpoint,
              .reason = "mid-epoch test checkpoint"};
        }
        return jyppx::training::CallbackCommandRequest{};
      });
  request.callbacks = std::make_shared<jyppx::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{callback});
  request.checkpoint_action =
      [checkpoint](const jyppx::training::CheckpointCoordinator& coordinator) {
        coordinator.save(*checkpoint);
      };
  jyppx::training::Trainer source(options(2));
  source.prepare(std::move(request));
  const auto expected = source.fit();
  ASSERT_EQ(expected.status, jyppx::training::FitStatus::completed);

  auto restored_domain = make_domain(8, 2);
  jyppx::training::Trainer restored(options(2));
  auto restored_callback = std::make_shared<RecordingCallback>(
      "callback.checkpoint", jyppx::training::CallbackEventKind::batch_finished,
      std::vector<jyppx::training::CallbackCommandKind>{
          jyppx::training::CallbackCommandKind::request_checkpoint});
  auto restored_dispatcher =
      std::make_shared<jyppx::training::CallbackDispatcher>(
          std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{
              restored_callback});
  restored.prepare(make_request(restored_domain, restored_dispatcher));
  restored.restore_checkpoint(*checkpoint);
  const auto actual = restored.fit();
  EXPECT_EQ(actual.global_step, expected.global_step);
  EXPECT_EQ(actual.completed_epochs, expected.completed_epochs);
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
  request.training_step = [](const jyppx::training::StepInput&) {
    return jyppx::training::StepOutput{.sample_count = 0};
  };
  jyppx::training::Trainer trainer(options(1));
  trainer.prepare(std::move(request));
  EXPECT_THROW(trainer.fit(), jyppx::InvalidArgument);
  EXPECT_EQ(trainer.phase(), jyppx::training::TrainerPhase::failed);
  EXPECT_EQ(trainer.global_step(), 0U);
}

TEST(TrainerTest,
     PreparationRejectsNonFreshOptimizerAndSharedValidationCursor) {
  auto domain = make_domain();
  domain.optimizer->zero_grad();
  static_cast<void>(domain.optimizer->step());
  jyppx::training::Trainer trainer(options(1));
  EXPECT_THROW(trainer.prepare(make_request(domain)), jyppx::NotReady);

  auto fresh = make_domain();
  auto request = make_request(fresh);
  request.validation_data = request.training_data;
  jyppx::training::Trainer shared_cursor(options(1));
  EXPECT_THROW(shared_cursor.prepare(std::move(request)),
               jyppx::InvalidArgument);
}

}  // namespace
