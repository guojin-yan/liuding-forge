// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <map>
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

constexpr std::uint64_t kInterruptStep = 6;
constexpr std::uint64_t kEpochs = 3;
constexpr std::size_t kSamples = 8;
constexpr std::size_t kBatchSize = 2;

/** @brief Models process loss immediately after durable publication. */
struct SimulatedProcessExit final {};

/** @brief Small stochastic classifier whose dropout exercises saved RNG. */
class RecoveryClassifier final : public torch::nn::Module {
 public:
  RecoveryClassifier()
      : hidden_(register_module("hidden", torch::nn::Linear(4, 5))),
        output_(register_module("output", torch::nn::Linear(5, 3))) {
    register_buffer("input_scale", torch::tensor({0.5, 0.75, 1.0, 1.25}));
  }

  jyppx::liuding::Tensor forward(const jyppx::liuding::Tensor& input) {
    const auto scaled = input * named_buffers()["input_scale"];
    const auto activation = torch::tanh(hidden_->forward(scaled));
    return output_->forward(torch::dropout(activation, 0.25, is_training()));
  }

 private:
  torch::nn::Linear hidden_;
  torch::nn::Linear output_;
};

/** @brief Requests one checkpoint from immutable committed step metadata. */
class RecoveryCheckpointCallback final
    : public jyppx::liuding::training::CallbackHandler {
 public:
  const jyppx::liuding::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  jyppx::liuding::training::CallbackCommandRequest handle(
      const jyppx::liuding::training::CallbackEvent& event) override {
    if (event.phase == jyppx::liuding::training::CallbackPhase::training &&
        event.global_step == kInterruptStep) {
      return {.kind = jyppx::liuding::training::CallbackCommandKind::request_checkpoint,
              .reason = "durable recovery boundary after one complete epoch"};
    }
    return {};
  }

 private:
  jyppx::liuding::training::CallbackDescriptor descriptor_{
      .type_id = "test.durable_checkpoint.v1",
      .callback_id = "callback.checkpoint",
      .subscriptions = {jyppx::liuding::training::CallbackEventKind::batch_finished},
      .allowed_commands = {
          jyppx::liuding::training::CallbackCommandKind::request_checkpoint}};
};

struct RecoveryCase {
  bool cuda = false;
  bool distillation = false;
};

/** @brief Owns every mutable object discarded by the interrupted process. */
struct RecoveryDomain {
  std::shared_ptr<jyppx::liuding::nn::ModelAdapter> student;
  std::shared_ptr<jyppx::liuding::nn::ModelAdapter> teacher;
  std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> optimizer;
  std::shared_ptr<jyppx::liuding::optim::StepLrScheduler> scheduler;
  std::shared_ptr<jyppx::liuding::data::DataPipeline> training_data;
  std::shared_ptr<jyppx::liuding::data::DataPipeline> validation_data;
  std::shared_ptr<jyppx::liuding::training::metrics::F1Score> training_f1;
  std::shared_ptr<jyppx::liuding::training::metrics::F1Score> validation_f1;
  std::shared_ptr<jyppx::liuding::training::callbacks::EarlyStoppingCallback>
      early_stopping;
  std::shared_ptr<jyppx::liuding::training::CallbackDispatcher> callbacks;
  std::shared_ptr<jyppx::liuding::training::SingleDeviceStrategy> strategy;
  std::shared_ptr<std::ostringstream> event_stream;
  std::shared_ptr<jyppx::liuding::training::EventOutbox> outbox;
  std::shared_ptr<std::vector<std::uint64_t>> observed_training_samples;
  std::unique_ptr<jyppx::liuding::training::Trainer> trainer;
};

/** @brief Captures real outbox blobs without normalizing producer identity. */
class RecoveryOutboxState final : public jyppx::liuding::StateWriter,
                                  public jyppx::liuding::StateReader {
 public:
  explicit RecoveryOutboxState(const jyppx::liuding::training::EventOutbox& outbox)
      : manifest_(outbox.describe_state()) {
    outbox.save_state(*this);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const auto& candidate) { return candidate.name == name; });
    if (field == manifest_.fields.end() ||
        fields_.contains(std::string(name))) {
      throw jyppx::liuding::SerializationError(
          "Unknown or duplicate recovery state field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "Recovery state exceeds its declared bound");
    }
    fields_.emplace(std::string(name),
                    std::vector<std::byte>(data.begin(), data.end()));
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return fields_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto field = fields_.find(std::string(name));
    if (field == fields_.end()) {
      throw jyppx::liuding::SerializationError("Missing recovery state field");
    }
    if (field->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("Recovery state exceeds its read bound");
    }
    return field->second;
  }

 private:
  jyppx::liuding::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> fields_;
};

std::shared_ptr<jyppx::liuding::nn::ModelAdapter> make_recovery_model() {
  auto module = std::make_shared<RecoveryClassifier>();
  module->to(torch::kFloat64);
  return std::make_shared<jyppx::liuding::nn::ModelAdapter>(
      jyppx::liuding::nn::adapt_tensor_module(module));
}

RecoveryDomain make_recovery_domain(
    RecoveryCase scenario, const std::filesystem::path& checkpoint_path,
    bool interrupt_after_publication) {
  const auto device = scenario.cuda ? jyppx::liuding::Device(torch::kCUDA, 0)
                                    : jyppx::liuding::Device(torch::kCPU);
  RecoveryDomain domain;
  domain.student = make_recovery_model();
  if (scenario.distillation) {
    domain.teacher = make_recovery_model();
    domain.teacher->state().to(device);
    domain.teacher->state().eval();
    for (auto parameter : domain.teacher->state().parameters()) {
      parameter.requires_grad_(false);
    }
  }

  std::vector<jyppx::liuding::optim::OptimizerParameterBinding> parameters;
  for (const auto& item : domain.student->state().named_parameters()) {
    parameters.push_back(
        {.parameter_id = item.key(), .parameter = item.value()});
  }
  domain.optimizer = std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      "optimizer.student",
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "student",
           .parameters = std::move(parameters),
           .options = {
               .learning_rate = 0.05, .momentum = 0.8, .weight_decay = 0.01}}});
  domain.scheduler = std::make_shared<jyppx::liuding::optim::StepLrScheduler>(
      domain.optimizer,
      jyppx::liuding::optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.student", .step_size = 1, .gamma = 0.75});

  const auto features =
      (torch::arange(0, static_cast<std::int64_t>(kSamples * 4),
                     torch::TensorOptions().dtype(torch::kFloat64)) /
           16.0 -
       1.0)
          .reshape({static_cast<std::int64_t>(kSamples), 4});
  const auto targets = torch::tensor({0, 1, 2, 0, 1, 2, 0, 2}, torch::kLong);
  const auto factory = [features, targets](
                           std::span<const std::size_t> indices,
                           const jyppx::liuding::data::DataWorkerContext&,
                           std::stop_token) {
    std::vector<std::int64_t> rows;
    for (const auto index : indices) {
      rows.push_back(static_cast<std::int64_t>(index));
    }
    const auto selected = torch::tensor(rows, torch::kLong);
    return c10::ivalue::Tuple::create({features.index_select(0, selected),
                                       targets.index_select(0, selected)});
  };
  const auto make_pipeline = [&](std::string id, std::int64_t seed) {
    return std::make_shared<jyppx::liuding::data::DataPipeline>(
        std::make_shared<jyppx::liuding::data::samplers::RandomSampler>(kSamples, seed),
        jyppx::liuding::data::DataPipelineOptions{
            .pipeline_id = std::move(id),
            .dataset_version = "test.durable-classification/v1",
            .dataset_size = kSamples,
            .schema =
                jyppx::liuding::data::BatchSchema::any("test.durable-classification", 1),
            .batch_factory_id = "test.durable-classification.factory/v1",
            .batch_size = kBatchSize,
            .num_workers = 2,
            .prefetch_batches = 2,
            .persistent_workers = true,
            .worker_seed = 1949},
        factory);
  };
  domain.training_data = make_pipeline("data.training", 104729);
  domain.validation_data = make_pipeline("data.validation", 130363);
  domain.training_f1 = std::make_shared<jyppx::liuding::training::metrics::F1Score>(
      jyppx::liuding::training::metrics::ClassificationMetricOptions{
          .metric_id = "metric.training.f1", .num_classes = 3});
  domain.validation_f1 = std::make_shared<jyppx::liuding::training::metrics::F1Score>(
      jyppx::liuding::training::metrics::ClassificationMetricOptions{
          .metric_id = "metric.validation.f1", .num_classes = 3});
  domain.early_stopping =
      std::make_shared<jyppx::liuding::training::callbacks::EarlyStoppingCallback>(
          jyppx::liuding::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early_stopping", .patience = 100});
  domain.callbacks = std::make_shared<jyppx::liuding::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{
          std::make_shared<RecoveryCheckpointCallback>(),
          domain.early_stopping});
  domain.strategy = std::make_shared<jyppx::liuding::training::SingleDeviceStrategy>(
      jyppx::liuding::training::SingleDeviceStrategyOptions{.device = device});
  domain.event_stream = std::make_shared<std::ostringstream>();
  domain.outbox = std::make_shared<jyppx::liuding::training::EventOutbox>(
      std::vector<std::shared_ptr<jyppx::liuding::training::EventSink>>{
          std::make_shared<jyppx::liuding::training::ConsoleEventSink>(
              *domain.event_stream)});
  domain.observed_training_samples =
      std::make_shared<std::vector<std::uint64_t>>();

  const auto mapper = [device](const jyppx::liuding::data::BatchEnvelope& batch) {
    return jyppx::liuding::nn::Invocation{
        .positional = {
            batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
  };
  const auto target = [device](const jyppx::liuding::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1).toTensor().to(device);
  };
  const auto step = [teacher = domain.teacher, training_f1 = domain.training_f1,
                     validation_f1 = domain.validation_f1,
                     observed_samples = domain.observed_training_samples](
                        const jyppx::liuding::training::StepInput& input) {
    const bool training =
        input.context.phase == jyppx::liuding::training::StepPhase::training;
    const auto logits = input.model.invoke(input.invocation).value.toTensor();
    const auto labels = input.target->toTensor();
    const auto supervised =
        torch::nn::functional::cross_entropy(logits, labels);
    auto loss = supervised;
    if (teacher) {
      const auto teacher_logits =
          teacher->invoke(input.invocation).value.toTensor().detach();
      constexpr double temperature = 2.0;
      const auto soft_targets = torch::softmax(teacher_logits / temperature, 1);
      const auto soft_loss =
          -(soft_targets * torch::log_softmax(logits / temperature, 1))
               .sum(1)
               .mean() *
          (temperature * temperature);
      loss = 0.5 * supervised + 0.5 * soft_loss;
    }
    const auto& metric = training ? training_f1 : validation_f1;
    metric->update(logits.detach(), labels);
    if (training) {
      observed_samples->insert(observed_samples->end(),
                               input.batch.sample_ids.begin(),
                               input.batch.sample_ids.end());
    }
    return jyppx::liuding::training::StepOutput{
        .loss = loss,
        .metrics = {{training ? "training.f1" : "validation.f1",
                     metric->compute()}},
        .sample_count = input.batch.sample_ids.size()};
  };

  std::vector<std::shared_ptr<jyppx::liuding::CheckpointParticipant>> additional{
      domain.training_f1, domain.validation_f1};
  if (domain.teacher) {
    additional.push_back(
        std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(
            *domain.teacher, jyppx::liuding::nn::ModelCheckpointParticipantOptions{
                                 .participant_id = "model.teacher"}));
  }
  domain.trainer = std::make_unique<jyppx::liuding::training::Trainer>(
      jyppx::liuding::training::TrainerOptions{
          .trainer_id = "trainer.durable",
          .run_id = "run.durable",
          .trace_id = "trace.durable",
          .max_epochs = kEpochs,
          .restore_policy = jyppx::liuding::RestorePolicy::exact_only});
  domain.trainer->prepare(
      {.model = domain.student,
       .optimizers = {{.optimizer = domain.optimizer,
                       .checkpoint = std::make_shared<
                           jyppx::liuding::optim::SgdOptimizerCheckpointParticipant>(
                           domain.optimizer),
                       .scheduler =
                           std::make_shared<jyppx::liuding::training::SchedulerAdapter>(
                               domain.scheduler)}},
       .training_data = domain.training_data,
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .validation_data = domain.validation_data,
       .validation_batch_mapper = mapper,
       .validation_target_mapper = target,
       .validation_step = step,
       .strategy = domain.strategy,
       .callbacks = domain.callbacks,
       .event_outbox = domain.outbox,
       .model_checkpoint =
           std::make_shared<jyppx::liuding::nn::ModelCheckpointParticipant>(
               *domain.student),
       .rng_checkpoint =
           std::make_shared<jyppx::liuding::core::RngCheckpointParticipant>(
               jyppx::liuding::core::RngCheckpointParticipantOptions{.include_cuda =
                                                                scenario.cuda}),
       .additional_participants = std::move(additional),
       .checkpoint_action =
           [checkpoint_path, interrupt_after_publication](
               const jyppx::liuding::training::CheckpointCoordinator& coordinator) {
             jyppx::liuding::training::FileCheckpointStore store(checkpoint_path);
             coordinator.save(store);
             if (interrupt_after_publication) {
               throw SimulatedProcessExit{};
             }
           }});
  return domain;
}

/** @brief Compares every actual state field, including all hidden counters. */
void expect_exact_checkpoint_equal(
    const jyppx::liuding::training::FileCheckpointStore& expected,
    const jyppx::liuding::training::FileCheckpointStore& actual) {
  const auto& left = expected.descriptor();
  const auto& right = actual.descriptor();
  EXPECT_EQ(left.restore_level, jyppx::liuding::RestoreLevel::exact);
  EXPECT_EQ(right.restore_level, left.restore_level);
  ASSERT_EQ(right.participants.size(), left.participants.size());
  for (std::size_t index = 0; index != left.participants.size(); ++index) {
    const auto& expected_manifest = left.participants[index];
    const auto& actual_manifest = right.participants[index];
    SCOPED_TRACE(expected_manifest.participant_id);
    EXPECT_EQ(actual_manifest.participant_id, expected_manifest.participant_id);
    EXPECT_EQ(actual_manifest.schema_version, expected_manifest.schema_version);
    EXPECT_EQ(actual_manifest.restore_level, expected_manifest.restore_level);
    EXPECT_EQ(actual_manifest.degradation_reasons,
              expected_manifest.degradation_reasons);
    ASSERT_EQ(actual_manifest.fields.size(), expected_manifest.fields.size());
    const auto& expected_reader =
        expected.reader_for(expected_manifest.participant_id);
    const auto& actual_reader =
        actual.reader_for(actual_manifest.participant_id);
    for (const auto& field : expected_manifest.fields) {
      const auto actual_field = std::find_if(
          actual_manifest.fields.begin(), actual_manifest.fields.end(),
          [&](const auto& candidate) { return candidate.name == field.name; });
      SCOPED_TRACE(field.name);
      ASSERT_NE(actual_field, actual_manifest.fields.end());
      EXPECT_EQ(actual_field->media_type, field.media_type);
      EXPECT_EQ(actual_field->max_bytes, field.max_bytes);
      EXPECT_EQ(actual_field->required, field.required);
      ASSERT_EQ(actual_reader.contains(field.name),
                expected_reader.contains(field.name));
      if (!expected_reader.contains(field.name)) {
        continue;
      }
      const auto expected_bytes =
          expected_reader.read_blob(field.name, field.max_bytes);
      const auto actual_bytes =
          actual_reader.read_blob(field.name, field.max_bytes);
      EXPECT_EQ(
          std::vector<std::byte>(actual_bytes.begin(), actual_bytes.end()),
          std::vector<std::byte>(expected_bytes.begin(), expected_bytes.end()));
    }
  }
}

class DurableTrainingRecoveryTest
    : public ::testing::TestWithParam<RecoveryCase> {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto name = std::string(info->name());
    std::replace(name.begin(), name.end(), '/', '_');
    directory_ = std::filesystem::temp_directory_path() /
                 ("jyppx-durable-training-" + name);
    std::filesystem::create_directories(directory_);
  }

  void TearDown() override {
    ASSERT_TRUE(
        directory_.filename().string().starts_with("jyppx-durable-training-"));
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  std::filesystem::path directory_;
};

TEST_P(DurableTrainingRecoveryTest,
       RecreatedDomainMatchesUninterruptedAllParticipantState) {
  const auto scenario = GetParam();
  if (scenario.cuda && !jyppx::liuding::cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  jyppx::liuding::core::RngGuard guard(20261021, scenario.cuda);
  const auto device = scenario.cuda ? jyppx::liuding::Device(torch::kCUDA, 0)
                                    : jyppx::liuding::Device(torch::kCPU);
  jyppx::liuding::training::FitResult expected_result;
  std::vector<std::uint64_t> expected_samples;
  jyppx::liuding::Tensor expected_next_random;
  std::unique_ptr<RecoveryOutboxState> initial_outbox_state;
  std::string expected_stream_id;
  const auto baseline_final = directory_ / "baseline-final.bin";
  const auto resumed_final = directory_ / "resumed-final.bin";
  const auto recovery_path = directory_ / "recovery.bin";

  {
    auto baseline = make_recovery_domain(
        scenario, directory_ / "baseline-boundary.bin", false);
    ASSERT_EQ(baseline.outbox->pending_events(), 0U);
    initial_outbox_state =
        std::make_unique<RecoveryOutboxState>(*baseline.outbox);
    expected_stream_id = baseline.outbox->telemetry_stream_id();
    ASSERT_FALSE(expected_stream_id.empty());
    expected_result = baseline.trainer->fit();
    ASSERT_EQ(expected_result.status, jyppx::liuding::training::FitStatus::completed);
    EXPECT_EQ(expected_result.global_step, 12U);
    EXPECT_EQ(expected_result.completed_epochs, kEpochs);
    expected_samples = *baseline.observed_training_samples;
    jyppx::liuding::training::FileCheckpointStore final_store(baseline_final);
    baseline.trainer->save_checkpoint(final_store);
    expected_next_random = torch::rand(
        {16}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  }

  jyppx::liuding::core::manual_seed(20261021, scenario.cuda);
  std::vector<std::uint64_t> interrupted_samples;
  std::weak_ptr<jyppx::liuding::nn::ModelAdapter> destroyed_student;
  std::weak_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> destroyed_optimizer;
  {
    auto interrupted = make_recovery_domain(scenario, recovery_path, true);
    // Both executions must start from the same persisted producer identity.
    // A new v2 outbox generates an independent stream even when the RNG seed
    // matches; comparing independently generated streams is not replay.
    EXPECT_NE(interrupted.outbox->telemetry_stream_id(), expected_stream_id);
    interrupted.outbox->load_state(*initial_outbox_state,
                                   jyppx::liuding::RestorePolicy::exact_only);
    EXPECT_EQ(interrupted.outbox->telemetry_stream_id(), expected_stream_id);
    destroyed_student = interrupted.student;
    destroyed_optimizer = interrupted.optimizer;
    EXPECT_THROW(interrupted.trainer->fit(), SimulatedProcessExit);
    EXPECT_EQ(interrupted.trainer->global_step(), kInterruptStep);
    EXPECT_EQ(interrupted.trainer->completed_epochs(), 1U);
    EXPECT_EQ(interrupted.scheduler->completed_steps(), 1U);
    EXPECT_TRUE(interrupted.early_stopping->has_best_value());
    EXPECT_EQ(interrupted.training_f1->total_observations(), 12);
    EXPECT_EQ(interrupted.validation_f1->total_observations(), 8);
    const auto cursor = interrupted.training_data->state();
    EXPECT_EQ(cursor.epoch, 1U);
    EXPECT_EQ(cursor.next_cursor, 4U);
    EXPECT_EQ(cursor.next_sequence, 2U);
    interrupted_samples = *interrupted.observed_training_samples;
  }
  EXPECT_TRUE(destroyed_student.expired());
  EXPECT_TRUE(destroyed_optimizer.expired());
  ASSERT_TRUE(std::filesystem::exists(recovery_path));
  EXPECT_FALSE(std::filesystem::exists(recovery_path.string() + ".tmp"));

  // Construction deliberately consumes a different RNG stream and creates
  // different student/teacher weights before the durable snapshot is loaded.
  jyppx::liuding::core::manual_seed(99194853094755497ULL, scenario.cuda);
  auto resumed = make_recovery_domain(scenario, recovery_path, false);
  EXPECT_NE(resumed.outbox->telemetry_stream_id(), expected_stream_id);
  jyppx::liuding::training::FileCheckpointStore reopened(recovery_path);
  const auto restored = resumed.trainer->restore_checkpoint(reopened);
  EXPECT_EQ(restored.restore_level, jyppx::liuding::RestoreLevel::exact);
  EXPECT_EQ(resumed.outbox->telemetry_stream_id(), expected_stream_id);
  // Exact restore must retain every saved outbox field, including any pending
  // canonical envelope's captured times and per-sink delivery cursor.
  const RecoveryOutboxState restored_outbox(*resumed.outbox);
  const auto& saved_outbox =
      reopened.reader_for(resumed.outbox->participant_id());
  for (const auto& field : restored_outbox.manifest().fields) {
    SCOPED_TRACE(field.name);
    const auto saved_bytes =
        saved_outbox.read_blob(field.name, field.max_bytes);
    const auto restored_bytes =
        restored_outbox.read_blob(field.name, field.max_bytes);
    EXPECT_EQ(
        std::vector<std::byte>(restored_bytes.begin(), restored_bytes.end()),
        std::vector<std::byte>(saved_bytes.begin(), saved_bytes.end()));
  }
  EXPECT_EQ(resumed.trainer->global_step(), kInterruptStep);
  EXPECT_EQ(resumed.optimizer->completed_steps(), kInterruptStep);
  EXPECT_EQ(resumed.scheduler->completed_steps(), 1U);
  EXPECT_TRUE(resumed.early_stopping->has_best_value());
  EXPECT_EQ(resumed.training_f1->total_observations(), 12);
  EXPECT_EQ(resumed.validation_f1->total_observations(), 8);

  const auto actual_result = resumed.trainer->fit();
  EXPECT_EQ(actual_result.status, expected_result.status);
  EXPECT_EQ(actual_result.global_step, expected_result.global_step);
  EXPECT_EQ(actual_result.completed_epochs, expected_result.completed_epochs);
  EXPECT_EQ(actual_result.training_metrics, expected_result.training_metrics);
  EXPECT_EQ(actual_result.validation_metrics,
            expected_result.validation_metrics);
  EXPECT_EQ(resumed.optimizer->completed_steps(), 12U);
  EXPECT_EQ(resumed.scheduler->completed_steps(), kEpochs);
  EXPECT_EQ(resumed.strategy->completed_backward_calls(), 12U);
  EXPECT_EQ(resumed.strategy->completed_optimizer_steps("optimizer.student"),
            12U);
  EXPECT_EQ(resumed.training_f1->total_observations(), 24);
  EXPECT_EQ(resumed.validation_f1->total_observations(), 24);
  EXPECT_EQ(resumed.outbox->pending_events(), 0U);
  EXPECT_EQ(resumed.outbox->last_delivered_event_index(),
            resumed.callbacks->last_event_index());
  interrupted_samples.insert(interrupted_samples.end(),
                             resumed.observed_training_samples->begin(),
                             resumed.observed_training_samples->end());
  EXPECT_EQ(interrupted_samples, expected_samples);

  jyppx::liuding::training::FileCheckpointStore actual_store(resumed_final);
  resumed.trainer->save_checkpoint(actual_store);
  jyppx::liuding::training::FileCheckpointStore expected_store(baseline_final);
  expect_exact_checkpoint_equal(expected_store, actual_store);
  const auto actual_next_random = torch::rand(
      {16}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  EXPECT_TRUE(torch::equal(actual_next_random, expected_next_random));
}

INSTANTIATE_TEST_SUITE_P(
    SingleDevice, DurableTrainingRecoveryTest,
    ::testing::Values(RecoveryCase{.cuda = false, .distillation = false},
                      RecoveryCase{.cuda = false, .distillation = true},
                      RecoveryCase{.cuda = true, .distillation = false},
                      RecoveryCase{.cuda = true, .distillation = true}),
    [](const ::testing::TestParamInfo<RecoveryCase>& info) {
      return std::string(info.param.cuda ? "Cuda" : "Cpu") +
             (info.param.distillation ? "Distillation" : "Classification");
    });

}  // namespace
