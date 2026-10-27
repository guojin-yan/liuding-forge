// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <torch/torch.h>

#ifdef JYPPX_LIUDING_MODELS_CUDA_METRICS
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// psapi.h requires Windows declarations.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

using namespace jyppx::liuding;

constexpr std::uint64_t kInterruptStep = 4;
constexpr std::uint64_t kEpochs = 2;
constexpr std::size_t kSamples = 3;

struct InterruptedAfterSave final {};

enum class RecoveryGate { standard, extended, high_capacity, large_capacity };

const char *GateVariable(RecoveryGate gate) {
  switch (gate) {
    case RecoveryGate::extended:
      return "JYPPX_LIUDING_DINO_EXTENDED_RECOVERY_GATE";
    case RecoveryGate::high_capacity:
      return "JYPPX_LIUDING_DINO_HIGH_CAPACITY_RECOVERY_GATE";
    case RecoveryGate::large_capacity:
      return "JYPPX_LIUDING_DINO_LARGE_RECOVERY_GATE";
    case RecoveryGate::standard:
      return nullptr;
  }
  throw std::logic_error("Unknown recovery resource gate");
}

struct RecoveryCase {
  std::string model;
  std::int64_t image_size;
  bool cuda;
  RecoveryGate gate = RecoveryGate::standard;
  std::uint64_t max_rss_bytes = 4ULL << 30U;
  std::int64_t max_cuda_allocated_bytes = 2LL << 30U;
};

void PrintTo(const RecoveryCase &scenario, std::ostream *output) {
  *output << scenario.model << (scenario.cuda ? "_CudaAmp" : "_Cpu");
}

class CheckpointCallback final : public training::CallbackHandler {
 public:
  const training::CallbackDescriptor &descriptor() const noexcept override {
    return descriptor_;
  }

  training::CallbackCommandRequest handle(
      const training::CallbackEvent &event) override {
    if (event.phase == training::CallbackPhase::training &&
        event.global_step == kInterruptStep) {
      return {.kind = training::CallbackCommandKind::request_checkpoint,
              .reason = "persist a committed DINO mid-epoch boundary"};
    }
    return {};
  }

 private:
  training::CallbackDescriptor descriptor_{
      .type_id = "test.dino_checkpoint.v1",
      .callback_id = "callback.checkpoint",
      .subscriptions = {training::CallbackEventKind::batch_finished},
      .allowed_commands = {training::CallbackCommandKind::request_checkpoint}};
};

// Leave the process-wide convolution policy as it was before this test.
class DeterministicExecution final {
 public:
  DeterministicExecution()
      : deterministic_(at::globalContext().deterministicCuDNN()),
        benchmark_(at::globalContext().benchmarkCuDNN()),
        algorithms_(at::globalContext().deterministicAlgorithms()),
        warn_only_(at::globalContext().deterministicAlgorithmsWarnOnly()) {
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicAlgorithms(true, false);
  }
  ~DeterministicExecution() {
    at::globalContext().setDeterministicCuDNN(deterministic_);
    at::globalContext().setBenchmarkCuDNN(benchmark_);
    at::globalContext().setDeterministicAlgorithms(algorithms_, warn_only_);
  }

 private:
  bool deterministic_;
  bool benchmark_;
  bool algorithms_;
  bool warn_only_;
};

struct RecoveryDomain {
  std::shared_ptr<nn::ModelAdapter> model;
  std::shared_ptr<optim::SgdOptimizerAdapter> optimizer;
  std::shared_ptr<optim::StepLrScheduler> scheduler;
  std::shared_ptr<training::amp::GradScaler> scaler;
  std::shared_ptr<data::DataPipeline> pipeline;
  std::shared_ptr<std::vector<std::uint64_t>> samples;
  std::shared_ptr<std::vector<double>> losses;
  std::unique_ptr<training::Trainer> trainer;
};

RecoveryDomain MakeDomain(const RecoveryCase &scenario,
                          const std::filesystem::path &checkpoint,
                          bool interrupt) {
  RecoveryDomain domain;
  models::ModelOptions options;
  options.dinov2.num_classes = 3;
  options.dinov3.num_classes = 3;
  options.dinov3_convnext.num_classes = 3;
  domain.model = std::make_shared<nn::ModelAdapter>(
      models::create_model(scenario.model, options));
  std::vector<optim::OptimizerParameterBinding> parameters;
  for (const auto &item : domain.model->state().named_parameters()) {
    parameters.push_back(
        {.parameter_id = item.key(), .parameter = item.value()});
  }
  domain.optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "optimizer.dino",
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "all",
           .parameters = std::move(parameters),
           .options = {.learning_rate = 0.001, .momentum = 0.8}}});
  domain.scheduler = std::make_shared<optim::StepLrScheduler>(
      domain.optimizer,
      optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.dino", .step_size = 1, .gamma = 0.75});
  domain.pipeline = std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(kSamples, 104729),
      data::DataPipelineOptions{
          .pipeline_id = "data.dino",
          .dataset_version = "test.dino-recovery/v1",
          .dataset_size = kSamples,
          .schema = data::BatchSchema::any("test.dino-recovery", 1),
          .batch_factory_id = "test.dino-recovery.factory/v1",
          .batch_size = 1,
          .num_workers = 2,
          .prefetch_batches = 2,
          .persistent_workers = true,
          .worker_seed = 1949},
      [side = scenario.image_size](std::span<const std::size_t> indices,
                                   const data::DataWorkerContext &,
                                   std::stop_token) -> c10::IValue {
        const auto id = static_cast<std::int64_t>(indices.front());
        const auto image = (torch::arange(3 * side * side, torch::kFloat32) /
                                static_cast<double>(3 * side * side) +
                            static_cast<double>(id) / 10.0)
                               .reshape({1, 3, side, side});
        return c10::ivalue::Tuple::create(
            {image, torch::tensor({id}, torch::kLong)});
      });
  const torch::Device device = scenario.cuda ? torch::Device(torch::kCUDA, 0)
                                             : torch::Device(torch::kCPU);
  domain.samples = std::make_shared<std::vector<std::uint64_t>>();
  domain.losses = std::make_shared<std::vector<double>>();
  if (scenario.cuda) {
    domain.scaler = std::make_shared<training::amp::GradScaler>(
        training::amp::GradScalerOptions{.init_scale = 128.0,
                                         .growth_interval = 3,
                                         .scaler_id = "scaler.dino"});
  }
  domain.trainer = std::make_unique<training::Trainer>(training::TrainerOptions{
      .trainer_id = "trainer.dino",
      .run_id = "run.dino",
      .trace_id = "trace.dino",
      .max_epochs = kEpochs,
      .precision = scenario.cuda ? training::PrecisionMode::cuda_float16
                                 : training::PrecisionMode::full,
      .checkpoint = {.max_rollback_bytes =
                         scenario.gate == RecoveryGate::large_capacity
                             ? 3ULL << 30U
                             : 1ULL << 30U}});
  domain.trainer->prepare(
      {.model = domain.model,
       .optimizers =
           {{.optimizer = domain.optimizer,
             .checkpoint =
                 std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
                     domain.optimizer),
             .scheduler = std::make_shared<training::SchedulerAdapter>(
                 domain.scheduler)}},
       .training_data = domain.pipeline,
       .training_batch_mapper =
           [device](const data::BatchEnvelope &batch) {
             return nn::Invocation{
                 .positional = {
                     batch.value.toTuple()->elements().at(0).toTensor().to(
                         device)}};
           },
       .training_target_mapper = [device](const data::BatchEnvelope &batch)
           -> std::optional<c10::IValue> {
         return batch.value.toTuple()->elements().at(1).toTensor().to(device);
       },
       .training_step =
           [samples = domain.samples,
            losses = domain.losses](const training::StepInput &input) {
             // Consume CPU RNG as well as the selected device RNG after worker
             // delivery. Workers never use process-global random state.
             const auto images = input.invocation.positional.front().toTensor();
             const auto noisy = images + torch::rand_like(images) * 0.01 +
                                torch::rand({1}).item<float>() * 0.01;
             const auto logits =
                 input.model.invoke({.positional = {noisy}}).value.toTensor();
             const auto loss = torch::nn::functional::cross_entropy(
                 logits.to(torch::kFloat32), input.target->toTensor());
             samples->insert(samples->end(), input.batch.sample_ids.begin(),
                             input.batch.sample_ids.end());
             losses->push_back(loss.item<double>());
             return training::StepOutput{
                 .loss = loss, .sample_count = input.batch.sample_ids.size()};
           },
       .strategy = std::make_shared<training::SingleDeviceStrategy>(
           training::SingleDeviceStrategyOptions{.device = device}),
       .gradient_scaler = domain.scaler,
       .gradient_scaler_checkpoint =
           domain.scaler ? std::make_shared<
                               training::amp::GradScalerCheckpointParticipant>(
                               domain.scaler)
                         : nullptr,
       .callbacks = std::make_shared<training::CallbackDispatcher>(
           std::vector<std::shared_ptr<training::CallbackHandler>>{
               std::make_shared<CheckpointCallback>()}),
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*domain.model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>(
           core::RngCheckpointParticipantOptions{.include_cuda =
                                                     scenario.cuda}),
       .checkpoint_action =
           [checkpoint,
            interrupt](const training::CheckpointCoordinator &owner) {
             training::FileCheckpointStore store(checkpoint);
             owner.save(store);
             if (interrupt) throw InterruptedAfterSave{};
           }});
  return domain;
}

void ExpectEqualCheckpoint(const std::filesystem::path &expected_path,
                           const std::filesystem::path &actual_path) {
  const training::FileCheckpointStore expected(expected_path);
  const training::FileCheckpointStore actual(actual_path);
  EXPECT_EQ(expected.descriptor().restore_level, RestoreLevel::exact);
  EXPECT_EQ(actual.descriptor().restore_level, RestoreLevel::exact);
  const auto &left = expected.descriptor().participants;
  const auto &right = actual.descriptor().participants;
  ASSERT_EQ(left.size(), right.size());
  std::uint64_t compared_bytes = 0;
  std::size_t compared_fields = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto &manifest = left[index];
    SCOPED_TRACE(manifest.participant_id);
    ASSERT_EQ(manifest.participant_id, right[index].participant_id);
    EXPECT_EQ(manifest.schema_version, right[index].schema_version);
    EXPECT_EQ(manifest.restore_level, right[index].restore_level);
    EXPECT_EQ(manifest.degradation_reasons, right[index].degradation_reasons);
    ASSERT_EQ(manifest.fields.size(), right[index].fields.size());
    const auto &expected_reader = expected.reader_for(manifest.participant_id);
    const auto &actual_reader = actual.reader_for(manifest.participant_id);
    for (std::size_t field_index = 0; field_index < manifest.fields.size();
         ++field_index) {
      const auto &field = manifest.fields[field_index];
      const auto &actual_field = right[index].fields[field_index];
      SCOPED_TRACE(field.name);
      ASSERT_EQ(field.name, actual_field.name);
      EXPECT_EQ(field.max_bytes, actual_field.max_bytes);
      EXPECT_EQ(field.required, actual_field.required);
      EXPECT_EQ(field.media_type, actual_field.media_type);
      ASSERT_EQ(expected_reader.contains(field.name),
                actual_reader.contains(field.name));
      if (!expected_reader.contains(field.name)) continue;
      const auto expected_blob =
          expected_reader.read_blob(field.name, field.max_bytes);
      const auto actual_blob =
          actual_reader.read_blob(field.name, field.max_bytes);
      ASSERT_EQ(expected_blob.size(), actual_blob.size());
      EXPECT_TRUE(std::equal(expected_blob.begin(), expected_blob.end(),
                             actual_blob.begin()));
      compared_bytes += expected_blob.size();
      ++compared_fields;
    }
  }
  // All model tensors and nonempty SGD momentum must participate, rather than
  // only a small head, digest, or loop counter.
  EXPECT_GT(compared_bytes, 100ULL << 20U);
  EXPECT_GT(compared_fields, 100U);
  std::cout << "CHECKPOINT_COMPARE participants=" << left.size()
            << " fields=" << compared_fields << " bytes=" << compared_bytes
            << '\n';
}

std::uint64_t PeakRssBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  }
#elif defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
  }
#endif
  return 0;
}

class DinoRecoveryTest : public ::testing::TestWithParam<RecoveryCase> {
 protected:
  void SetUp() override {
    directory_ =
        std::filesystem::temp_directory_path() /
        ("liuding-dino-recovery-" + GetParam().model + "-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory_);
  }

  void TearDown() override {
    ASSERT_TRUE(
        directory_.filename().string().starts_with("liuding-dino-recovery-"));
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  std::filesystem::path directory_;
};

TEST_P(DinoRecoveryTest, RecreatedDomainMatchesEveryParticipantAndNextRandom) {
  const auto scenario = GetParam();
  if (const auto *variable = GateVariable(scenario.gate)) {
    const auto *gate = std::getenv(variable);
    if (gate == nullptr || std::string_view(gate) != "1") {
      GTEST_SKIP() << "Set " << variable
                   << "=1 for this serial durable recovery resource profile";
    }
  }
  if (scenario.cuda && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const DeterministicExecution deterministic;
  const core::RngGuard random_guard(20261027, scenario.cuda);
  const torch::Device device = scenario.cuda ? torch::Device(torch::kCUDA, 0)
                                             : torch::Device(torch::kCPU);
#ifdef JYPPX_LIUDING_MODELS_CUDA_METRICS
  if (scenario.cuda) {
    torch::cuda::synchronize(0);
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  }
#endif
  const auto start = std::chrono::steady_clock::now();
  const auto baseline_path = directory_ / "baseline-final.bin";
  const auto checkpoint_path = directory_ / "interrupted.bin";
  const auto resumed_path = directory_ / "resumed-final.bin";
  training::FitResult expected_result;
  std::vector<std::uint64_t> expected_samples;
  std::vector<double> expected_losses;
  Tensor expected_next_cpu;
  Tensor expected_next_device;
  {
    auto baseline =
        MakeDomain(scenario, directory_ / "baseline-boundary.bin", false);
    const auto initial_head = baseline.model->state()
                                  .named_parameters()["head.weight"]
                                  .detach()
                                  .clone();
    expected_result = baseline.trainer->fit();
    ASSERT_EQ(expected_result.status, training::FitStatus::completed);
    ASSERT_EQ(expected_result.global_step, 6U);
    EXPECT_EQ(baseline.optimizer->completed_steps(), 6U);
    EXPECT_EQ(baseline.scheduler->completed_steps(), 2U);
    EXPECT_FALSE(torch::equal(
        initial_head,
        baseline.model->state().named_parameters()["head.weight"]));
    if (scenario.cuda) {
      EXPECT_DOUBLE_EQ(baseline.scaler->get_scale(), 512.0);
      EXPECT_EQ(baseline.scaler->capture_state().growth_tracker, 0U);
    }
    expected_samples = *baseline.samples;
    expected_losses = *baseline.losses;
    training::FileCheckpointStore store(baseline_path);
    baseline.trainer->save_checkpoint(store);
    expected_next_cpu = torch::rand({16});
    expected_next_device =
        torch::rand({16}, torch::TensorOptions().device(device));
  }
  core::manual_seed(20261027, scenario.cuda);
  std::vector<std::uint64_t> actual_samples;
  std::vector<double> actual_losses;
  Tensor saved_head;
  std::weak_ptr<nn::ModelAdapter> destroyed_model;
  std::weak_ptr<optim::SgdOptimizerAdapter> destroyed_optimizer;
  std::weak_ptr<training::amp::GradScaler> destroyed_scaler;
  {
    auto interrupted = MakeDomain(scenario, checkpoint_path, true);
    destroyed_model = interrupted.model;
    destroyed_optimizer = interrupted.optimizer;
    destroyed_scaler = interrupted.scaler;
    EXPECT_THROW(interrupted.trainer->fit(), InterruptedAfterSave);
    EXPECT_EQ(interrupted.trainer->global_step(), kInterruptStep);
    EXPECT_EQ(interrupted.trainer->completed_epochs(), 1U);
    EXPECT_EQ(interrupted.optimizer->completed_steps(), 4U);
    EXPECT_EQ(interrupted.scheduler->completed_steps(), 1U);
    EXPECT_EQ(interrupted.pipeline->state().epoch, 1U);
    EXPECT_EQ(interrupted.pipeline->state().next_cursor, 1U);
    EXPECT_EQ(interrupted.pipeline->state().next_sequence, 1U);
    if (scenario.cuda) {
      EXPECT_DOUBLE_EQ(interrupted.scaler->get_scale(), 256.0);
      EXPECT_EQ(interrupted.scaler->capture_state().growth_tracker, 1U);
    }
    actual_samples = *interrupted.samples;
    actual_losses = *interrupted.losses;
    saved_head = interrupted.model->state()
                     .named_parameters()["head.weight"]
                     .detach()
                     .cpu()
                     .clone();
  }
  EXPECT_TRUE(destroyed_model.expired());
  EXPECT_TRUE(destroyed_optimizer.expired());
  EXPECT_TRUE(destroyed_scaler.expired());
  ASSERT_TRUE(std::filesystem::exists(checkpoint_path));
  // Compare the published safe boundary before any rebuilt model exists.
  ExpectEqualCheckpoint(directory_ / "baseline-boundary.bin", checkpoint_path);
  core::manual_seed(99194853094755497ULL, scenario.cuda);
  {
    auto resumed = MakeDomain(scenario, checkpoint_path, false);
    EXPECT_FALSE(torch::equal(
        saved_head,
        resumed.model->state().named_parameters()["head.weight"].cpu()));
    const training::FileCheckpointStore reopened(checkpoint_path);
    const auto restored = resumed.trainer->restore_checkpoint(reopened);
    EXPECT_EQ(restored.restore_level, RestoreLevel::exact);
    EXPECT_EQ(resumed.trainer->global_step(), kInterruptStep);
    EXPECT_EQ(resumed.optimizer->completed_steps(), 4U);
    EXPECT_EQ(resumed.scheduler->completed_steps(), 1U);
    EXPECT_EQ(resumed.pipeline->state().next_cursor, 1U);
    EXPECT_EQ(resumed.pipeline->state().next_sequence, 1U);
    EXPECT_TRUE(torch::equal(
        saved_head,
        resumed.model->state().named_parameters()["head.weight"].cpu()));
    if (scenario.cuda) {
      EXPECT_DOUBLE_EQ(resumed.scaler->get_scale(), 256.0);
      EXPECT_EQ(resumed.scaler->capture_state().growth_tracker, 1U);
    }
    const auto result = resumed.trainer->fit();
    EXPECT_EQ(result.status, expected_result.status);
    EXPECT_EQ(result.global_step, expected_result.global_step);
    EXPECT_EQ(result.completed_epochs, expected_result.completed_epochs);
    EXPECT_EQ(result.training_metrics, expected_result.training_metrics);
    EXPECT_EQ(resumed.optimizer->completed_steps(), 6U);
    actual_samples.insert(actual_samples.end(), resumed.samples->begin(),
                          resumed.samples->end());
    actual_losses.insert(actual_losses.end(), resumed.losses->begin(),
                         resumed.losses->end());
    EXPECT_EQ(actual_samples, expected_samples);
    EXPECT_EQ(actual_losses, expected_losses);
    training::FileCheckpointStore store(resumed_path);
    resumed.trainer->save_checkpoint(store);
    EXPECT_TRUE(torch::equal(torch::rand({16}), expected_next_cpu));
    EXPECT_TRUE(
        torch::equal(torch::rand({16}, torch::TensorOptions().device(device)),
                     expected_next_device));
  }
  ExpectEqualCheckpoint(baseline_path, resumed_path);
  const auto peak_rss = PeakRssBytes();
  ASSERT_GT(peak_rss, 0U);
  EXPECT_LE(peak_rss, scenario.max_rss_bytes);
  std::cout << "RECOVERY_METRICS model=" << scenario.model
            << " precision=" << (scenario.cuda ? "cuda-fp16" : "fp32")
            << " image_size=" << scenario.image_size
            << " compute_threads=1 deterministic_algorithms=1"
            << " process_peak_rss_bytes=" << peak_rss << " elapsed_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count();
#ifdef JYPPX_LIUDING_MODELS_CUDA_METRICS
  if (scenario.cuda) {
    torch::cuda::synchronize(0);
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    EXPECT_LE(stats.allocated_bytes[0].peak, scenario.max_cuda_allocated_bytes);
    // Deterministic cuBLAS retains its workspace independently of model state.
    // Release it explicitly after quiescing all work, before the residency
    // gate.
    at::cuda::clearCublasWorkspaces();
    const auto released = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    EXPECT_LE(released.allocated_bytes[0].current, 32LL << 20U);
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto empty = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    EXPECT_LE(empty.reserved_bytes[0].current, 64LL << 20U);
    std::cout << " peak_cuda_allocated_bytes=" << stats.allocated_bytes[0].peak
              << " cuda_allocated_after_destroy_bytes="
              << stats.allocated_bytes[0].current
              << " cuda_allocated_after_workspace_clear_bytes="
              << released.allocated_bytes[0].current
              << " cuda_reserved_after_empty_cache_bytes="
              << empty.reserved_bytes[0].current;
  }
#endif
  std::cout << '\n';
}

INSTANTIATE_TEST_SUITE_P(
    DINO, DinoRecoveryTest,
    ::testing::Values(
        RecoveryCase{"vit_small_patch14_dinov2", 28, false},
        RecoveryCase{"dinov3_vits16", 32, false},
        RecoveryCase{"dinov3_convnext_tiny", 32, false},
        // CUDA bicubic interpolation backward is nondeterministic;
        // the native 37x37 grid exercises exact CUDA continuation.
        RecoveryCase{"vit_small_patch14_dinov2", 518, true},
        RecoveryCase{"dinov3_vits16", 32, true},
        RecoveryCase{"dinov3_convnext_tiny", 32, true},
        RecoveryCase{"vit_base_patch14_dinov2", 28, false,
                     RecoveryGate::extended, 6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_vits16plus", 32, false, RecoveryGate::extended},
        RecoveryCase{"dinov3_vitb16", 32, false, RecoveryGate::extended,
                     6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_convnext_small", 32, false,
                     RecoveryGate::extended},
        RecoveryCase{"vit_base_patch14_dinov2", 518, true,
                     RecoveryGate::extended, 6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_vits16plus", 32, true, RecoveryGate::extended},
        RecoveryCase{"dinov3_vitb16", 32, true, RecoveryGate::extended,
                     6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_convnext_small", 32, true, RecoveryGate::extended},
        RecoveryCase{"dinov3_convnext_base", 32, false,
                     RecoveryGate::high_capacity, 6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_convnext_base", 32, true,
                     RecoveryGate::high_capacity, 6ULL << 30U, 4LL << 30U},
        RecoveryCase{"dinov3_convnext_large", 32, false,
                     RecoveryGate::large_capacity, 8ULL << 30U, 5LL << 30U},
        RecoveryCase{"dinov3_convnext_large", 32, true,
                     RecoveryGate::large_capacity, 8ULL << 30U, 5LL << 30U}),
    [](const ::testing::TestParamInfo<RecoveryCase> &info) {
      return info.param.model + (info.param.cuda ? "_CudaAmp" : "_Cpu");
    });

}  // namespace

int main(int argc, char **argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
