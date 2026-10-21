// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/core/device.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/foundation/random_checkpoint_participant.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/event_sink.hpp>
#include <jyppx/training/file_checkpoint_store.hpp>
#include <jyppx/training/metrics/classification.hpp>
#include <jyppx/training/scheduler_adapter.hpp>
#include <jyppx/training/single_device_strategy.hpp>
#include <jyppx/training/trainer.hpp>

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

  jyppx::Tensor forward(const jyppx::Tensor& input) {
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
    : public jyppx::training::CallbackHandler {
 public:
  const jyppx::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  jyppx::training::CallbackCommandRequest handle(
      const jyppx::training::CallbackEvent& event) override {
    if (event.phase == jyppx::training::CallbackPhase::training &&
        event.global_step == kInterruptStep) {
      return {.kind = jyppx::training::CallbackCommandKind::request_checkpoint,
              .reason = "durable recovery boundary after one complete epoch"};
    }
    return {};
  }

 private:
  jyppx::training::CallbackDescriptor descriptor_{
      .type_id = "test.durable_checkpoint.v1",
      .callback_id = "callback.checkpoint",
      .subscriptions = {jyppx::training::CallbackEventKind::batch_finished},
      .allowed_commands = {
          jyppx::training::CallbackCommandKind::request_checkpoint}};
};

struct RecoveryCase {
  bool cuda = false;
  bool distillation = false;
};

/** @brief Owns every mutable object discarded by the interrupted process. */
struct RecoveryDomain {
  std::shared_ptr<jyppx::nn::ModelAdapter> student;
  std::shared_ptr<jyppx::nn::ModelAdapter> teacher;
  std::shared_ptr<jyppx::optim::SgdOptimizerAdapter> optimizer;
  std::shared_ptr<jyppx::optim::StepLrScheduler> scheduler;
  std::shared_ptr<jyppx::data::DataPipeline> training_data;
  std::shared_ptr<jyppx::data::DataPipeline> validation_data;
  std::shared_ptr<jyppx::training::metrics::F1Score> training_f1;
  std::shared_ptr<jyppx::training::metrics::F1Score> validation_f1;
  std::shared_ptr<jyppx::training::callbacks::EarlyStoppingCallback>
      early_stopping;
  std::shared_ptr<jyppx::training::CallbackDispatcher> callbacks;
  std::shared_ptr<jyppx::training::SingleDeviceStrategy> strategy;
  std::shared_ptr<std::ostringstream> event_stream;
  std::shared_ptr<jyppx::training::EventOutbox> outbox;
  std::shared_ptr<std::vector<std::uint64_t>> observed_training_samples;
  std::unique_ptr<jyppx::training::Trainer> trainer;
};

std::shared_ptr<jyppx::nn::ModelAdapter> make_recovery_model() {
  auto module = std::make_shared<RecoveryClassifier>();
  module->to(torch::kFloat64);
  return std::make_shared<jyppx::nn::ModelAdapter>(
      jyppx::nn::adapt_tensor_module(module));
}

RecoveryDomain make_recovery_domain(
    RecoveryCase scenario, const std::filesystem::path& checkpoint_path,
    bool interrupt_after_publication) {
  const auto device = scenario.cuda ? jyppx::Device(torch::kCUDA, 0)
                                    : jyppx::Device(torch::kCPU);
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

  std::vector<jyppx::optim::OptimizerParameterBinding> parameters;
  for (const auto& item : domain.student->state().named_parameters()) {
    parameters.push_back(
        {.parameter_id = item.key(), .parameter = item.value()});
  }
  domain.optimizer = std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.student",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "student",
           .parameters = std::move(parameters),
           .options = {
               .learning_rate = 0.05, .momentum = 0.8, .weight_decay = 0.01}}});
  domain.scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      domain.optimizer,
      jyppx::optim::StepLrSchedulerOptions{
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
                           const jyppx::data::DataWorkerContext&,
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
    return std::make_shared<jyppx::data::DataPipeline>(
        std::make_shared<jyppx::data::samplers::RandomSampler>(kSamples, seed),
        jyppx::data::DataPipelineOptions{
            .pipeline_id = std::move(id),
            .dataset_version = "test.durable-classification/v1",
            .dataset_size = kSamples,
            .schema =
                jyppx::data::BatchSchema::any("test.durable-classification", 1),
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
  domain.training_f1 = std::make_shared<jyppx::training::metrics::F1Score>(
      jyppx::training::metrics::ClassificationMetricOptions{
          .metric_id = "metric.training.f1", .num_classes = 3});
  domain.validation_f1 = std::make_shared<jyppx::training::metrics::F1Score>(
      jyppx::training::metrics::ClassificationMetricOptions{
          .metric_id = "metric.validation.f1", .num_classes = 3});
  domain.early_stopping =
      std::make_shared<jyppx::training::callbacks::EarlyStoppingCallback>(
          jyppx::training::callbacks::EarlyStoppingOptions{
              .callback_id = "callback.early_stopping", .patience = 100});
  domain.callbacks = std::make_shared<jyppx::training::CallbackDispatcher>(
      std::vector<std::shared_ptr<jyppx::training::CallbackHandler>>{
          std::make_shared<RecoveryCheckpointCallback>(),
          domain.early_stopping});
  domain.strategy = std::make_shared<jyppx::training::SingleDeviceStrategy>(
      jyppx::training::SingleDeviceStrategyOptions{.device = device});
  domain.event_stream = std::make_shared<std::ostringstream>();
  domain.outbox = std::make_shared<jyppx::training::EventOutbox>(
      std::vector<std::shared_ptr<jyppx::training::EventSink>>{
          std::make_shared<jyppx::training::ConsoleEventSink>(
              *domain.event_stream)});
  domain.observed_training_samples =
      std::make_shared<std::vector<std::uint64_t>>();

  const auto mapper = [device](const jyppx::data::BatchEnvelope& batch) {
    return jyppx::nn::Invocation{
        .positional = {
            batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
  };
  const auto target = [device](const jyppx::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1).toTensor().to(device);
  };
  const auto step = [teacher = domain.teacher, training_f1 = domain.training_f1,
                     validation_f1 = domain.validation_f1,
                     observed_samples = domain.observed_training_samples](
                        const jyppx::training::StepInput& input) {
    const bool training =
        input.context.phase == jyppx::training::StepPhase::training;
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
    return jyppx::training::StepOutput{
        .loss = loss,
        .metrics = {{training ? "training.f1" : "validation.f1",
                     metric->compute()}},
        .sample_count = input.batch.sample_ids.size()};
  };

  std::vector<std::shared_ptr<jyppx::CheckpointParticipant>> additional{
      domain.training_f1, domain.validation_f1};
  if (domain.teacher) {
    additional.push_back(
        std::make_shared<jyppx::nn::ModelCheckpointParticipant>(
            *domain.teacher, jyppx::nn::ModelCheckpointParticipantOptions{
                                 .participant_id = "model.teacher"}));
  }
  domain.trainer = std::make_unique<jyppx::training::Trainer>(
      jyppx::training::TrainerOptions{
          .trainer_id = "trainer.durable",
          .run_id = "run.durable",
          .trace_id = "trace.durable",
          .max_epochs = kEpochs,
          .restore_policy = jyppx::RestorePolicy::exact_only});
  domain.trainer->prepare(
      {.model = domain.student,
       .optimizers = {{.optimizer = domain.optimizer,
                       .checkpoint = std::make_shared<
                           jyppx::optim::SgdOptimizerCheckpointParticipant>(
                           domain.optimizer),
                       .scheduler =
                           std::make_shared<jyppx::training::SchedulerAdapter>(
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
           std::make_shared<jyppx::nn::ModelCheckpointParticipant>(
               *domain.student),
       .rng_checkpoint =
           std::make_shared<jyppx::core::RngCheckpointParticipant>(
               jyppx::core::RngCheckpointParticipantOptions{.include_cuda =
                                                                scenario.cuda}),
       .additional_participants = std::move(additional),
       .checkpoint_action =
           [checkpoint_path, interrupt_after_publication](
               const jyppx::training::CheckpointCoordinator& coordinator) {
             jyppx::training::FileCheckpointStore store(checkpoint_path);
             coordinator.save(store);
             if (interrupt_after_publication) {
               throw SimulatedProcessExit{};
             }
           }});
  return domain;
}

/** @brief Compares every actual state field, including all hidden counters. */
void expect_exact_checkpoint_equal(
    const jyppx::training::FileCheckpointStore& expected,
    const jyppx::training::FileCheckpointStore& actual) {
  const auto& left = expected.descriptor();
  const auto& right = actual.descriptor();
  EXPECT_EQ(left.restore_level, jyppx::RestoreLevel::exact);
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
  if (scenario.cuda && !jyppx::cuda_is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  jyppx::core::RngGuard guard(20261021, scenario.cuda);
  const auto device = scenario.cuda ? jyppx::Device(torch::kCUDA, 0)
                                    : jyppx::Device(torch::kCPU);
  jyppx::training::FitResult expected_result;
  std::vector<std::uint64_t> expected_samples;
  jyppx::Tensor expected_next_random;
  const auto baseline_final = directory_ / "baseline-final.bin";
  const auto resumed_final = directory_ / "resumed-final.bin";
  const auto recovery_path = directory_ / "recovery.bin";

  {
    auto baseline = make_recovery_domain(
        scenario, directory_ / "baseline-boundary.bin", false);
    expected_result = baseline.trainer->fit();
    ASSERT_EQ(expected_result.status, jyppx::training::FitStatus::completed);
    EXPECT_EQ(expected_result.global_step, 12U);
    EXPECT_EQ(expected_result.completed_epochs, kEpochs);
    expected_samples = *baseline.observed_training_samples;
    jyppx::training::FileCheckpointStore final_store(baseline_final);
    baseline.trainer->save_checkpoint(final_store);
    expected_next_random = torch::rand(
        {16}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  }

  jyppx::core::manual_seed(20261021, scenario.cuda);
  std::vector<std::uint64_t> interrupted_samples;
  std::weak_ptr<jyppx::nn::ModelAdapter> destroyed_student;
  std::weak_ptr<jyppx::optim::SgdOptimizerAdapter> destroyed_optimizer;
  {
    auto interrupted = make_recovery_domain(scenario, recovery_path, true);
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
  jyppx::core::manual_seed(99194853094755497ULL, scenario.cuda);
  auto resumed = make_recovery_domain(scenario, recovery_path, false);
  jyppx::training::FileCheckpointStore reopened(recovery_path);
  const auto restored = resumed.trainer->restore_checkpoint(reopened);
  EXPECT_EQ(restored.restore_level, jyppx::RestoreLevel::exact);
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

  jyppx::training::FileCheckpointStore actual_store(resumed_final);
  resumed.trainer->save_checkpoint(actual_store);
  jyppx::training::FileCheckpointStore expected_store(baseline_final);
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
