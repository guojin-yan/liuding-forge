// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/data/batch_schema.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/iterable_data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/capability.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>
#include <jyppx/liuding/nn/loss/contrastive_loss.hpp>
#include <jyppx/liuding/nn/loss/dice_loss.hpp>
#include <jyppx/liuding/nn/loss/focal_loss.hpp>
#include <jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/adadelta_adapter.hpp>
#include <jyppx/liuding/optim/adagrad_adapter.hpp>
#include <jyppx/liuding/optim/adam_adapter.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/lamb_adapter.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/nadam_adapter.hpp>
#include <jyppx/liuding/optim/optimizer_adapter.hpp>
#include <jyppx/liuding/optim/rmsprop_adapter.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>
#include <jyppx/liuding/training/metrics/ranking.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class ConsumerParticipant final : public jyppx::liuding::CheckpointParticipant {
 public:
  std::string participant_id() const override { return "consumer.state"; }

  jyppx::liuding::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = jyppx::liuding::RestoreLevel::exact,
            .fields = {{.name = "value", .max_bytes = 8}}};
  }

  void save_state(jyppx::liuding::StateWriter& writer) const override {
    const std::array<std::byte, 8> value{};
    writer.write_blob("value", value);
  }

  void load_state(const jyppx::liuding::StateReader& reader,
                  jyppx::liuding::RestorePolicy policy) override {
    if (!jyppx::liuding::restore_policy_accepts(
            policy, reader.manifest().restore_level)) {
      throw jyppx::liuding::CompatibilityError("consumer policy rejected");
    }
    (void)reader.read_blob("value", 8);
  }
};

/** @brief Minimal installed-package cursor used to consume iterable APIs. */
class ConsumerIterableCursor final
    : public jyppx::liuding::data::IterableSourceCursor {
 public:
  /** @brief Returns one stable sample and then reports exhaustion. */
  std::optional<jyppx::liuding::data::IterableSample> next(
      std::stop_token stop) override {
    if (closed_.load(std::memory_order_acquire) || stop.stop_requested() ||
        position_ != 0) {
      return std::nullopt;
    }
    ++position_;
    return jyppx::liuding::data::IterableSample{
        .value = c10::IValue(std::int64_t{7}),
        .sample_id = 11,
        .watermark = "position:1"};
  }

  /** @brief Saves the next integer position as decimal text. */
  std::string save_state() const override { return std::to_string(position_); }

  /** @brief Restores the only two valid cursor positions. */
  void load_state(std::string_view state) override {
    if (state == "0") {
      position_ = 0;
    } else if (state == "1") {
      position_ = 1;
    } else {
      throw jyppx::liuding::SerializationError("invalid consumer cursor state");
    }
  }

  /** @brief Permanently disables future pulls without blocking. */
  void close() noexcept override {
    closed_.store(true, std::memory_order_release);
  }

 private:
  std::size_t position_ = 0;  ///< Next sample position owned by the producer.
  std::atomic<bool> closed_{false};  ///< Cooperative close flag.
};

/** @brief Installed-package source factory for iterable API verification. */
class ConsumerIterableSource final
    : public jyppx::liuding::data::IterableSource {
 public:
  /** @brief Reports a deterministic exact one-sample source contract. */
  jyppx::liuding::data::IterableSourceDescriptor descriptor() const override {
    return {.source_id = "consumer.iterable-source/v1",
            .finite_epoch = true,
            .deterministic_epoch = true,
            .checkpointable = true,
            .restore_level = jyppx::liuding::RestoreLevel::exact,
            .max_replayed_samples = 0,
            .max_omitted_samples = 0};
  }

  /** @brief Opens an isolated cursor for the selected logical epoch. */
  std::unique_ptr<jyppx::liuding::data::IterableSourceCursor> open_epoch(
      std::uint64_t) const override {
    return std::make_unique<ConsumerIterableCursor>();
  }
};

}  // namespace

int main() {
  jyppx::liuding::training::Trainer installed_trainer(
      {.trainer_id = "trainer.consumer",
       .run_id = "run.consumer",
       .trace_id = "trace.consumer"});
  bool rejects_unprepared_pause = false;
  try {
    (void)installed_trainer.fit_until(1);
  } catch (const jyppx::liuding::NotReady&) {
    rejects_unprepared_pause = true;
  }
  const bool valid_trainer =
      rejects_unprepared_pause &&
      installed_trainer.phase() ==
          jyppx::liuding::training::TrainerPhase::created &&
      installed_trainer.options().trainer_id == "trainer.consumer";
  const jyppx::liuding::InvalidArgument error(
      "consumer probe",
      jyppx::liuding::ErrorDetails{.component = "InstallConsumer"});
  const bool valid_version = std::string(jyppx::liuding::version()) == "0.3.0";
  const bool valid_error =
      error.code() == jyppx::liuding::ErrorCode::invalid_argument &&
      error.details().component == "InstallConsumer";
  const bool valid_device = jyppx::liuding::parse_device("cpu").is_cpu();
  const jyppx::liuding::StateManifest state_manifest{
      .participant_id = "consumer.probe",
      .schema_version = 1,
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .fields = {{.name = "state", .max_bytes = 8}},
  };
  jyppx::liuding::validate_state_manifest(state_manifest);
  const bool valid_checkpoint_contract = jyppx::liuding::restore_policy_accepts(
      jyppx::liuding::RestorePolicy::exact_only, state_manifest.restore_level);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(std::make_shared<ConsumerParticipant>());
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  const auto checkpoint_descriptor = coordinator.describe();
  const bool valid_checkpoint_coordinator =
      checkpoint_descriptor.participants.size() == 1 &&
      checkpoint_descriptor.participants.front().participant_id ==
          "consumer.state";
  auto installed_early_stopping = std::make_shared<
      jyppx::liuding::training::callbacks::EarlyStoppingCallback>(
      jyppx::liuding::training::callbacks::EarlyStoppingOptions{
          .callback_id = "callback.consumer"});
  auto installed_callback_dispatcher = std::make_shared<
      jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          installed_early_stopping},
      jyppx::liuding::training::CallbackDispatcherOptions{
          .dispatcher_id = "callbacks.consumer"});
  jyppx::liuding::training::CheckpointParticipantRegistry callback_registry;
  jyppx::liuding::training::register_callback_checkpoint_participants(
      callback_registry, installed_callback_dispatcher);
  const bool valid_callback =
      callback_registry.participant_ids() ==
      std::vector<std::string>({"callback.consumer", "callbacks.consumer"});
  std::ostringstream installed_event_output;
  auto installed_event_sink =
      std::make_shared<jyppx::liuding::training::ConsoleEventSink>(
          installed_event_output,
          jyppx::liuding::training::ConsoleEventSinkOptions{
              .sink_id = "sink.consumer",
              .destination_id = "console.consumer"});
  jyppx::liuding::training::EventOutbox installed_outbox(
      {installed_event_sink}, jyppx::liuding::training::EventOutboxOptions{
                                  .outbox_id = "events.consumer"});
  installed_callback_dispatcher->dispatch(
      {.schema_version = 1,
       .kind = jyppx::liuding::training::CallbackEventKind::run_started,
       .phase = jyppx::liuding::training::CallbackPhase::preparing,
       .event_index = 1,
       .run_id = "run.consumer",
       .trace_id = "trace.consumer"},
      installed_outbox);
  installed_outbox.close();
  const bool valid_event_sink =
      installed_outbox.describe_state().participant_id == "events.consumer" &&
      installed_outbox.last_delivered_event_index() == 1 &&
      installed_event_output.str().find("\"run_id\":\"run.consumer\"") !=
          std::string::npos;
  jyppx::liuding::core::RngCheckpointParticipant installed_rng_participant(
      {.participant_id = "rng.consumer", .include_cuda = false});
  const bool valid_rng_checkpoint =
      installed_rng_participant.describe_state().fields.size() == 2;
  const auto indices =
      jyppx::liuding::data::samplers::SequentialSampler(3).indices();
  const bool valid_data = indices == std::vector<std::size_t>({0, 1, 2});
  auto installed_pipeline = jyppx::liuding::data::DataPipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(3),
      {.pipeline_id = "data.consumer",
       .dataset_version = "dataset.consumer/v1",
       .dataset_size = 3,
       .batch_factory_id = "factory.consumer/v1",
       .batch_size = 2},
      [](std::span<const std::size_t> batch_indices,
         const jyppx::liuding::data::DataWorkerContext&, std::stop_token) {
        return c10::IValue(static_cast<std::int64_t>(batch_indices.size()));
      });
  installed_pipeline.set_epoch(0);
  const auto installed_batch = installed_pipeline.next();
  const bool valid_data_pipeline =
      installed_batch.has_value() && installed_batch->sample_ids.size() == 2 &&
      installed_pipeline.describe_state().restore_level ==
          jyppx::liuding::RestoreLevel::exact;
  const auto installed_data_capability =
      jyppx::liuding::data::describe_data_capability(
          installed_pipeline.capability());
  const auto installed_capability_manifest =
      jyppx::liuding::make_current_capability_manifest(
          {installed_data_capability});
  const bool valid_capability_manifest =
      installed_capability_manifest.schema_version == 1 &&
      installed_capability_manifest.features.size() == 1 &&
      installed_capability_manifest.features.front().feature_id ==
          "jyppx.data.pipeline.map" &&
      installed_capability_manifest.features.front().scope_id ==
          "data.consumer" &&
      jyppx::liuding::capability_manifest_json(installed_capability_manifest)
              .find("\"scope_id\":\"data.consumer\"") != std::string::npos &&
      jyppx::liuding::capability_manifest_json(installed_capability_manifest)
              .find("\"source\":\"configured-instance\"") != std::string::npos;
  auto installed_iterable_pipeline = jyppx::liuding::data::IterableDataPipeline(
      std::make_shared<ConsumerIterableSource>(),
      {.pipeline_id = "data.iterable.consumer",
       .dataset_version = "dataset.iterable.consumer/v1",
       .batch_factory_id = "factory.iterable.consumer/v1"},
      [](std::span<const jyppx::liuding::data::IterableSample> samples,
         std::uint64_t, std::uint64_t,
         std::stop_token) { return samples.front().value; });
  installed_iterable_pipeline.set_epoch(0);
  const auto installed_iterable_batch = installed_iterable_pipeline.next();
  const bool valid_iterable_pipeline =
      installed_iterable_batch.has_value() &&
      installed_iterable_batch->value.toInt() == 7 &&
      installed_iterable_batch->source_watermark == "position:1" &&
      installed_iterable_pipeline.capability().iterable_style;
  const auto logits = torch::tensor({{2.0, -1.0}}, torch::kFloat64);
  const auto target = torch::tensor({0}, torch::kLong);
  const auto loss =
      jyppx::liuding::nn::functional::label_smoothing_cross_entropy(
          logits, target, 0.1);
  const bool valid_nn = loss.numel() == 1 && std::isfinite(loss.item<double>());
  auto installed_metric =
      std::make_shared<jyppx::liuding::training::metrics::Precision>(
          jyppx::liuding::training::metrics::ClassificationMetricOptions{
              .metric_id = "metric.consumer", .num_classes = 2});
  installed_metric->update(logits, target);
  jyppx::liuding::validate_checkpoint_participant(*installed_metric);
  const bool valid_metric =
      installed_metric->descriptor().type_id == "jyppx.metric.precision/v1" &&
      installed_metric->total_observations() == 1 &&
      installed_metric->compute() == 0.5;
  jyppx::liuding::training::metrics::BinaryAUROC installed_ranking_metric(
      {.metric_id = "metric.ranking.consumer"});
  installed_ranking_metric.update(torch::tensor({0.9, 0.8, 0.2, 0.1}),
                                  torch::tensor({1, 0, 1, 0}, torch::kLong));
  jyppx::liuding::validate_checkpoint_participant(installed_ranking_metric);
  const bool valid_ranking_metric =
      installed_ranking_metric.descriptor().type_id ==
          "jyppx.metric.binary-auroc/v1" &&
      installed_ranking_metric.observations() == 4 &&
      installed_ranking_metric.compute() == 0.75;
  jyppx::liuding::training::metrics::MulticlassAveragePrecision
      installed_classwise_metric(
          {.metric_id = "metric.classwise.consumer",
           .num_classes = 2,
           .average =
               jyppx::liuding::training::metrics::RankingAverage::macro});
  installed_classwise_metric.update(torch::tensor({{0.9, 0.1}, {0.2, 0.8}}),
                                    torch::tensor({0, 1}, torch::kLong));
  jyppx::liuding::validate_checkpoint_participant(installed_classwise_metric);
  const bool valid_classwise_metric =
      installed_classwise_metric.descriptor().type_id ==
          "jyppx.metric.multiclass-average-precision/v1" &&
      installed_classwise_metric.observations() == 4 &&
      installed_classwise_metric.compute() == 1.0;
  const auto pair_a = torch::tensor({{0.0, 0.0}}, torch::kFloat64);
  const auto pair_b = torch::tensor({{0.5, 0.0}}, torch::kFloat64);
  const auto pair_target = torch::tensor({1}, torch::kLong);
  const auto contrastive = jyppx::liuding::nn::functional::contrastive_loss(
      pair_a, pair_b, pair_target, 1.0);
  const bool valid_contrastive =
      contrastive.numel() == 1 && std::isfinite(contrastive.item<double>());
  const auto focal = jyppx::liuding::nn::functional::focal_loss(
      torch::tensor({{0.0, 0.0}}, torch::kFloat64),
      torch::tensor({{1.0, 0.0}}, torch::kFloat64), 0.25, 2.0);
  const bool valid_focal =
      focal.numel() == 1 && std::isfinite(focal.item<double>());
  jyppx::liuding::nn::DiceLossOptions dice_options;
  dice_options.mode = jyppx::liuding::nn::DiceMode::multiclass;
  const auto dice = jyppx::liuding::nn::functional::dice_loss(
      torch::tensor({{2.0, -1.0}}, torch::kFloat64),
      torch::tensor({0}, torch::kLong), dice_options);
  const bool valid_dice =
      dice.numel() == 1 && std::isfinite(dice.item<double>());

  torch::nn::Linear linear(2, 1);
  auto installed_model_state =
      std::make_shared<jyppx::liuding::nn::ModelStateView>(linear.ptr());
  jyppx::liuding::nn::ModelCheckpointParticipant installed_model_participant(
      installed_model_state);
  const bool valid_model_checkpoint =
      installed_model_participant.describe_state().fields.size() == 3;
  const auto installed_batch_schema = jyppx::liuding::data::BatchSchema::tensor(
      "consumer.features",
      {.dtype = torch::kFloat32,
       .device = jyppx::liuding::Device(torch::kCPU),
       .rank = 2,
       .shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 2}},
      1, "features");
  const bool valid_batch_schema =
      jyppx::liuding::data::batch_schema_fingerprint(installed_batch_schema)
          .size() == 16;
  jyppx::liuding::TensorSchema adapter_input;
  adapter_input.dtype = torch::kFloat32;
  adapter_input.rank = 2;
  adapter_input.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, 2};
  jyppx::liuding::TensorSchema adapter_output;
  adapter_output.dtype = torch::kFloat32;
  adapter_output.rank = 2;
  adapter_output.shape =
      std::vector<std::optional<std::int64_t>>{std::nullopt, 1};
  auto adapter = jyppx::liuding::nn::adapt_tensor_holder(
      linear, std::move(adapter_input), std::move(adapter_output));
  const auto adapted =
      adapter.invoke({.positional = {torch::ones({2, 2})}}).value.toTensor();
  const bool valid_adapter =
      adapted.sizes().vec() == std::vector<std::int64_t>({2, 1}) &&
      adapter.state().parameters().size() == 2;
  std::vector<jyppx::liuding::optim::OptimizerParameterBinding>
      strategy_bindings;
  for (const auto& item : adapter.state().named_parameters()) {
    strategy_bindings.push_back(
        {.parameter_id = item.key(), .parameter = item.value()});
  }
  auto strategy_optimizer =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.strategy",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "main",
               .parameters = std::move(strategy_bindings),
               .options = {.learning_rate = 0.0}}});
  jyppx::liuding::training::SingleDeviceStrategy installed_strategy;
  jyppx::liuding::optim::OptimizerAdapter* strategy_optimizers[] = {
      strategy_optimizer.get()};
  installed_strategy.prepare(adapter, strategy_optimizers);
  installed_strategy.zero_grad(*strategy_optimizer);
  const auto strategy_loss =
      adapter.invoke({.positional = {torch::ones({1, 2})}})
          .value.toTensor()
          .pow(2)
          .mean();
  installed_strategy.backward(strategy_loss);
  const auto strategy_outcome =
      installed_strategy.optimizer_step(*strategy_optimizer);
  const bool valid_strategy =
      installed_strategy.describe_state().restore_level ==
          jyppx::liuding::RestoreLevel::exact &&
      strategy_outcome.status ==
          jyppx::liuding::training::StrategyStepStatus::applied &&
      strategy_outcome.optimizer_completed_steps == 1;
  auto installed_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_optimizer =
      std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.main",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_parameter}},
               .options = {.learning_rate = 0.01, .momentum = 0.9}}});
  jyppx::liuding::optim::SgdOptimizerCheckpointParticipant
      installed_optimizer_participant(installed_optimizer);
  const bool valid_optimizer =
      installed_optimizer_participant.describe_state().fields.size() == 4 &&
      installed_optimizer->parameter_count() == 1;
  auto installed_adam_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_adam =
      std::make_shared<jyppx::liuding::optim::AdamOptimizerAdapter>(
          "optimizer.adam",
          std::vector<jyppx::liuding::optim::AdamParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_adam_parameter}},
               .options = {.learning_rate = 0.001, .amsgrad = true}}});
  installed_adam_parameter.mutable_grad() =
      torch::ones_like(installed_adam_parameter);
  installed_adam->step();
  jyppx::liuding::optim::AdamOptimizerCheckpointParticipant
      installed_adam_participant(installed_adam);
  const bool valid_adam =
      installed_adam_participant.describe_state().fields.size() == 6 &&
      installed_adam->completed_steps() == 1;
  auto installed_adamw_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_adamw =
      std::make_shared<jyppx::liuding::optim::AdamWOptimizerAdapter>(
          "optimizer.adamw",
          std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_adamw_parameter}},
               .options = {.learning_rate = 0.001,
                           .weight_decay = 0.01,
                           .amsgrad = true}}});
  installed_adamw_parameter.mutable_grad() =
      torch::ones_like(installed_adamw_parameter);
  installed_adamw->step();
  jyppx::liuding::optim::AdamWOptimizerCheckpointParticipant
      installed_adamw_participant(installed_adamw);
  const bool valid_adamw =
      installed_adamw_participant.describe_state().fields.size() == 6 &&
      installed_adamw->completed_steps() == 1;
  auto installed_rmsprop_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_rmsprop =
      std::make_shared<jyppx::liuding::optim::RmspropOptimizerAdapter>(
          "optimizer.rmsprop",
          std::vector<jyppx::liuding::optim::RmspropParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_rmsprop_parameter}},
               .options = {
                   .learning_rate = 0.01, .momentum = 0.9, .centered = true}}});
  installed_rmsprop_parameter.mutable_grad() =
      torch::ones_like(installed_rmsprop_parameter);
  installed_rmsprop->step();
  jyppx::liuding::optim::RmspropOptimizerCheckpointParticipant
      installed_rmsprop_participant(installed_rmsprop);
  const bool valid_rmsprop =
      installed_rmsprop_participant.describe_state().fields.size() == 6 &&
      installed_rmsprop->completed_steps() == 1;
  auto installed_adagrad_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_adagrad =
      std::make_shared<jyppx::liuding::optim::AdagradOptimizerAdapter>(
          "optimizer.adagrad",
          std::vector<jyppx::liuding::optim::AdagradParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_adagrad_parameter}},
               .options = {.learning_rate = 0.01,
                           .initial_accumulator_value = 0.1}}});
  installed_adagrad_parameter.mutable_grad() =
      torch::ones_like(installed_adagrad_parameter);
  installed_adagrad->step();
  jyppx::liuding::optim::AdagradOptimizerCheckpointParticipant
      installed_adagrad_participant(installed_adagrad);
  const bool valid_adagrad =
      installed_adagrad_participant.describe_state().fields.size() == 4 &&
      installed_adagrad->completed_steps() == 1;
  auto installed_adadelta_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_adadelta =
      std::make_shared<jyppx::liuding::optim::AdadeltaOptimizerAdapter>(
          "optimizer.adadelta",
          std::vector<jyppx::liuding::optim::AdadeltaParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_adadelta_parameter}},
               .options = {
                   .learning_rate = 1.0, .rho = 0.9, .epsilon = 1e-6}}});
  installed_adadelta_parameter.mutable_grad() =
      torch::ones_like(installed_adadelta_parameter);
  installed_adadelta->step();
  jyppx::liuding::optim::AdadeltaOptimizerCheckpointParticipant
      installed_adadelta_participant(installed_adadelta);
  const bool valid_adadelta =
      installed_adadelta_participant.describe_state().fields.size() == 5 &&
      installed_adadelta->completed_steps() == 1;
  auto installed_lamb_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_lamb =
      std::make_shared<jyppx::liuding::optim::LambOptimizerAdapter>(
          "optimizer.lamb",
          std::vector<jyppx::liuding::optim::LambParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_lamb_parameter}},
               .options = {.learning_rate = 0.001, .weight_decay = 0.01}}});
  installed_lamb_parameter.mutable_grad() =
      torch::ones_like(installed_lamb_parameter);
  installed_lamb->step();
  jyppx::liuding::optim::LambOptimizerCheckpointParticipant
      installed_lamb_participant(installed_lamb);
  const bool valid_lamb =
      installed_lamb_participant.describe_state().fields.size() == 5 &&
      installed_lamb->completed_steps() == 1;
  auto installed_nadam_parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto installed_nadam =
      std::make_shared<jyppx::liuding::optim::NAdamOptimizerAdapter>(
          "optimizer.nadam",
          std::vector<jyppx::liuding::optim::NAdamParameterGroup>{
              {.group_id = "main",
               .parameters = {{.parameter_id = "weight",
                               .parameter = installed_nadam_parameter}},
               .options = {.learning_rate = 0.002,
                           .weight_decay = 0.01,
                           .decoupled_weight_decay = true}}});
  installed_nadam_parameter.mutable_grad() =
      torch::ones_like(installed_nadam_parameter);
  installed_nadam->step();
  jyppx::liuding::optim::NAdamOptimizerCheckpointParticipant
      installed_nadam_participant(installed_nadam);
  const bool valid_nadam =
      installed_nadam_participant.describe_state().fields.size() == 5 &&
      installed_nadam->completed_steps() == 1;
  auto installed_scheduler =
      std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
          installed_optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                                   .step_size = 2, .gamma = 0.5});
  installed_scheduler->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant
      installed_scheduler_participant(installed_scheduler);
  auto installed_epoch_scheduler =
      std::make_shared<jyppx::liuding::optim::ExponentialLrScheduler>(
          installed_optimizer,
          jyppx::liuding::optim::ExponentialLrSchedulerOptions{
              .scheduler_id = "scheduler.exponential", .gamma = 0.5});
  installed_epoch_scheduler->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant
      installed_epoch_scheduler_participant(installed_epoch_scheduler);
  auto installed_restart_scheduler = std::make_shared<
      jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler>(
      installed_optimizer,
      jyppx::liuding::optim::CosineAnnealingWarmRestartsLrSchedulerOptions{
          .scheduler_id = "scheduler.restart", .t_0 = 2, .t_mult = 2});
  jyppx::liuding::training::SchedulerAdapter installed_scheduler_adapter(
      installed_restart_scheduler);
  installed_scheduler_adapter.step(
      {.trigger = jyppx::liuding::optim::SchedulerStepMode::epoch,
       .event_index = 1});
  auto installed_linear_scheduler =
      std::make_shared<jyppx::liuding::optim::WarmupLrScheduler>(
          installed_optimizer, jyppx::liuding::optim::WarmupLrSchedulerOptions{
                                   .scheduler_id = "scheduler.warmup",
                                   .warmup_iters = 2,
                                   .start_factor = 0.5});
  installed_linear_scheduler->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant
      installed_linear_scheduler_participant(installed_linear_scheduler);
  auto installed_event_scheduler =
      std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
          installed_optimizer,
          jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
              .scheduler_id = "scheduler.plateau", .patience = 0});
  installed_event_scheduler->step(1.0);
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant
      installed_event_scheduler_participant(installed_event_scheduler);
  std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>>
      scheduler_children;
  scheduler_children.push_back(
      std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
          installed_optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                                   .scheduler_id = "scheduler.child.first",
                                   .step_size = 1,
                                   .gamma = 0.5}));
  scheduler_children.push_back(
      std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
          installed_optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                                   .scheduler_id = "scheduler.child.second",
                                   .step_size = 1,
                                   .gamma = 0.5}));
  auto installed_composite =
      std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
          installed_optimizer, std::move(scheduler_children),
          jyppx::liuding::optim::SequentialLrSchedulerOptions{
              .scheduler_id = "scheduler.composite", .milestones = {1}});
  installed_composite->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant
      installed_composite_participant(installed_composite);
  const bool valid_scheduler =
      installed_scheduler_participant.describe_state().fields.size() == 2 &&
      installed_scheduler->completed_steps() == 1 &&
      installed_scheduler->descriptor().optimizer_id == "optimizer.main" &&
      installed_epoch_scheduler_participant.describe_state().fields.size() ==
          2 &&
      installed_epoch_scheduler->completed_steps() == 1 &&
      installed_scheduler_adapter.participant_id() == "scheduler.restart" &&
      installed_restart_scheduler->completed_steps() == 1 &&
      installed_linear_scheduler_participant.describe_state().fields.size() ==
          2 &&
      installed_linear_scheduler->completed_steps() == 1 &&
      installed_event_scheduler_participant.describe_state().schema_version ==
          3 &&
      installed_event_scheduler->completed_steps() == 1 &&
      installed_composite_participant.describe_state().fields.size() == 2 &&
      installed_composite->descriptor().children.size() == 2;
  const auto interop_capability =
      jyppx::liuding::interop::make_checkpoint_manifest_capability();
  const bool valid_interop =
      interop_capability.feature_id == "interop.checkpoint_manifest" &&
      interop_capability.operations.size() == 3U;
  const std::pair<std::string_view, bool> checks[] = {
      {"version", valid_version},
      {"error", valid_error},
      {"device", valid_device},
      {"checkpoint_contract", valid_checkpoint_contract},
      {"checkpoint_coordinator", valid_checkpoint_coordinator},
      {"interop", valid_interop},
      {"rng_checkpoint", valid_rng_checkpoint},
      {"model_checkpoint", valid_model_checkpoint},
      {"data", valid_data},
      {"batch_schema", valid_batch_schema},
      {"data_pipeline", valid_data_pipeline},
      {"capability_manifest", valid_capability_manifest},
      {"iterable_pipeline", valid_iterable_pipeline},
      {"nn", valid_nn},
      {"contrastive", valid_contrastive},
      {"focal", valid_focal},
      {"dice", valid_dice},
      {"adapter", valid_adapter},
      {"metric", valid_metric},
      {"ranking_metric", valid_ranking_metric},
      {"classwise_metric", valid_classwise_metric},
      {"optimizer", valid_optimizer},
      {"adam", valid_adam},
      {"adamw", valid_adamw},
      {"rmsprop", valid_rmsprop},
      {"adagrad", valid_adagrad},
      {"adadelta", valid_adadelta},
      {"lamb", valid_lamb},
      {"nadam", valid_nadam},
      {"scheduler", valid_scheduler},
      {"callback", valid_callback},
      {"event_sink", valid_event_sink},
      {"strategy", valid_strategy},
      {"trainer", valid_trainer},
  };
  bool all_valid = true;
  for (const auto& [name, passed] : checks) {
    if (!passed) {
      std::cerr << "failed installed API check: " << name << '\n';
      all_valid = false;
    }
  }
  return all_valid ? 0 : 1;
}
