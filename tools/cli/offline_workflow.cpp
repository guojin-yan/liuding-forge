// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/core/device.hpp>
#include <jyppx/data/data_pipeline.hpp>
#include <jyppx/data/samplers/random_sampler.hpp>
#include <jyppx/experiment/evaluation.hpp>
#include <jyppx/experiment/run_store.hpp>
#include <jyppx/experiment/run_telemetry_sink.hpp>
#include <jyppx/experiment/telemetry_exporters.hpp>
#include <jyppx/foundation/random_checkpoint_participant.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/interop/safetensors.hpp>
#include <jyppx/nn/model_adapter.hpp>
#include <jyppx/nn/model_checkpoint_participant.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/training/callback_dispatcher.hpp>
#include <jyppx/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/training/event_sink.hpp>
#include <jyppx/training/file_checkpoint_store.hpp>
#include <jyppx/training/metrics/classification.hpp>
#include <jyppx/training/scheduler_adapter.hpp>
#include <jyppx/training/single_device_strategy.hpp>
#include <jyppx/training/telemetry_adapter.hpp>
#include <jyppx/training/trainer.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::cli {
namespace {
using namespace experiment;
namespace tr = training;

constexpr std::string_view kWorkflow =
    "jyppx.offline.classification-recovery/v1";
constexpr std::string_view kBaseline = "run.offline.baseline";
constexpr std::string_view kRecovery = "run.offline.recovery";
constexpr std::uint64_t kEpochs = 3U;
constexpr std::uint64_t kBoundary = 6U;
constexpr std::size_t kSamples = 8U;
constexpr std::size_t kBatch = 2U;
constexpr std::size_t kFileLimit = 64U * 1024U * 1024U;

std::uint64_t timestamp() {
  const auto value = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  if (value < 0) throw InvalidArgument("System clock precedes Unix epoch");
  return static_cast<std::uint64_t>(value);
}

std::string utf8(const std::filesystem::path& path) {
  const auto value = path.generic_u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::span<const std::byte> byte_view(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

void write_bytes(const std::filesystem::path& path,
                 std::span<const std::byte> bytes) {
  if (bytes.size() > kFileLimit)
    throw ResourceExhausted("Offline artifact exceeds 64 MiB");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.flush();
  if (!output) throw IoError("Could not write the offline workflow artifact");
  output.close();
  if (!output) throw IoError("Could not close the offline workflow artifact");
}

void write_json(const std::filesystem::path& path, const Json& value) {
  write_bytes(path, byte_view(value.dump(2) + "\n"));
}

void require_evidence(bool condition, std::string_view message) {
  if (!condition) throw CompatibilityError(std::string(message));
}

Json reference_json(const RunArtifactReference& reference) {
  return {{"reference_id", reference.reference_id},
          {"role", reference.role},
          {"kind", reference.kind == RunArtifactKind::checkpoint ? "checkpoint"
                                                                 : "artifact"},
          {"attached_at_unix_ms", reference.attached_at_unix_ms},
          {"sha256", reference.artifact.digest.hex},
          {"size_bytes", reference.artifact.size_bytes},
          {"logical_name", reference.artifact.logical_name},
          {"media_type", reference.artifact.media_type},
          {"producer", reference.artifact.producer}};
}

Json metadata_json(const RunMetadata& metadata) {
  return {
      {"run_id", metadata.run_id},
      {"project_id", metadata.project_id},
      {"experiment_id", metadata.experiment_id},
      {"display_name", metadata.display_name},
      {"status", run_status_name(metadata.status)},
      {"created_at_unix_ms", metadata.created_at_unix_ms},
      {"updated_at_unix_ms", metadata.updated_at_unix_ms},
      {"lineage",
       {{"parent_run_id", metadata.lineage.parent_run_id
                              ? Json(*metadata.lineage.parent_run_id)
                              : Json(nullptr)},
        {"source_identity", metadata.lineage.source_digest.value_or("")},
        {"build_fingerprint", metadata.lineage.build_fingerprint.value_or("")},
        {"config_digest", metadata.lineage.config_digest.value_or("")},
        {"data_manifest_digest",
         metadata.lineage.data_manifest_digest.value_or("")}}},
      {"tags", metadata.tags}};
}

RunArtifactReference publish(SQLiteRunStore& store, std::string_view run_id,
                             std::string id, std::string role,
                             const std::filesystem::path& path,
                             std::string media_type = "application/json",
                             RunArtifactKind kind = RunArtifactKind::artifact) {
  const auto content = read_text(path, kFileLimit);
  return store.put_artifact(
      run_id, std::move(id), kind, std::move(role), byte_view(content),
      {.logical_name = utf8(path.filename()),
       .media_type = std::move(media_type),
       .producer = std::string(kWorkflow),
       .expected_size_bytes = content.size()},
      timestamp(), ArtifactDigest{.hex = sha256_hex(byte_view(content))});
}

std::vector<double> doubles(const Tensor& tensor) {
  const auto cpu =
      tensor.detach().to(torch::kCPU, torch::kFloat64).contiguous().flatten();
  return {cpu.data_ptr<double>(), cpu.data_ptr<double>() + cpu.numel()};
}

std::string tensor_digest(const Tensor& tensor) {
  const auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  return sha256_hex(std::span(
      static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()));
}

struct Fixture {
  Tensor training_features;
  Tensor validation_features;
  Tensor labels;
  Json manifest;
};

Fixture make_fixture() {
  Fixture fixture;
  fixture.training_features =
      (torch::arange(32, torch::TensorOptions().dtype(torch::kFloat64)) / 16.0 -
       1.0)
          .reshape({static_cast<std::int64_t>(kSamples), 4});
  fixture.validation_features = fixture.training_features + 0.03125;
  fixture.labels = torch::tensor({0, 1, 2, 0, 1, 2, 0, 2}, torch::kLong);
  fixture.manifest = {
      {"schema_version", 1U},
      {"dataset_id", "offline.classification.fixture/v1"},
      {"shape", Json::array({kSamples, 4U})},
      {"dtype", "float64"},
      {"training_features", doubles(fixture.training_features)},
      {"validation_features", doubles(fixture.validation_features)},
      {"labels", Json::array({0, 1, 2, 0, 1, 2, 0, 2})},
      {"purpose",
       "deterministic recovery fixture; no generalization estimate"}};
  return fixture;
}

class OfflineClassifier final : public torch::nn::Module {
 public:
  OfflineClassifier()
      : hidden_(register_module("hidden", torch::nn::Linear(4, 5))),
        output_(register_module("output", torch::nn::Linear(5, 3))) {
    register_buffer("input_scale", torch::tensor({0.5, 0.75, 1.0, 1.25}));
  }

  Tensor forward(const Tensor& input) {
    const auto hidden =
        torch::tanh(hidden_->forward(input * named_buffers()["input_scale"]));
    return output_->forward(torch::dropout(hidden, 0.25, is_training()));
  }

 private:
  torch::nn::Linear hidden_;
  torch::nn::Linear output_;
};

std::vector<std::byte> model_bytes(const nn::ModelAdapter& model) {
  std::vector<Tensor> storage;
  std::vector<interop::SafeTensorSource> sources;
  const auto append = [&](const auto& values, const std::string& prefix) {
    for (const auto& entry : values) {
      storage.push_back(entry.value().detach().to(torch::kCPU).contiguous());
      const auto& tensor = storage.back();
      require_evidence(tensor.scalar_type() == torch::kFloat64,
                       "Offline reference model must contain float64 state");
      std::vector<std::uint64_t> shape;
      for (const auto dimension : tensor.sizes())
        shape.push_back(static_cast<std::uint64_t>(dimension));
      sources.push_back({.name = prefix + entry.key(),
                         .dtype = interop::SafeTensorDtype::f64,
                         .shape = std::move(shape),
                         .data = std::span(static_cast<const std::byte*>(
                                               tensor.const_data_ptr()),
                                           tensor.nbytes())});
    }
  };
  append(model.state().named_parameters(), "parameter.");
  append(model.state().named_buffers(), "buffer.");
  return interop::serialize_safetensors(sources,
                                        {{"workflow", std::string(kWorkflow)}});
}

class BoundaryCallback final : public tr::CallbackHandler {
 public:
  const tr::CallbackDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }
  tr::CallbackCommandRequest handle(const tr::CallbackEvent& event) override {
    if (event.phase == tr::CallbackPhase::training &&
        event.global_step == kBoundary) {
      return {.kind = tr::CallbackCommandKind::request_checkpoint,
              .reason =
                  "offline reference durable boundary at committed update six"};
    }
    return {};
  }

 private:
  tr::CallbackDescriptor descriptor_{
      .type_id = "jyppx.offline.checkpoint.v1",
      .callback_id = "callback.offline.checkpoint",
      .subscriptions = {tr::CallbackEventKind::batch_finished},
      .allowed_commands = {tr::CallbackCommandKind::request_checkpoint}};
};

struct SnapshotExit final {};

struct Domain {
  std::shared_ptr<nn::ModelAdapter> model;
  std::shared_ptr<optim::SgdOptimizerAdapter> optimizer;
  std::shared_ptr<optim::StepLrScheduler> scheduler;
  std::shared_ptr<tr::metrics::F1Score> training_f1;
  std::shared_ptr<tr::metrics::F1Score> validation_f1;
  std::shared_ptr<tr::EventOutbox> outbox;
  std::shared_ptr<std::vector<std::uint64_t>> samples;
  std::unique_ptr<tr::Trainer> trainer;
};

tr::TrainingTelemetryAdapterOptions telemetry_options() {
  tr::TrainingTelemetryAdapterOptions options;
  options.project_id = "project.offline";
  for (const auto* phase : {"training", "validation"}) {
    for (const auto* name : {"loss", "f1"}) {
      const std::string key = std::string(phase) + "." + name;
      TelemetryMetricDefinition definition;
      definition.metric_id = key;
      definition.unit = std::string_view(name) == "loss"
                            ? TelemetryMetricUnit::dimensionless
                            : TelemetryMetricUnit::ratio;
      definition.aggregation = TelemetryAggregation::mean;
      definition.direction = std::string_view(name) == "loss"
                                 ? TelemetryMetricDirection::minimize
                                 : TelemetryMetricDirection::maximize;
      definition.parameters = {
          {"source", std::string_view(name) == "loss" ? "cross_entropy_mean"
                                                      : "running_macro_f1"},
          {"epoch_reduction", "sample_weighted_mean_of_batch_observations"}};
      definition.context_keys = {"split"};
      options.metrics[key] = {.definition = std::move(definition),
                              .context = {{"split", phase}}};
    }
  }
  return options;
}

Domain make_domain(const Fixture& fixture, const Device& device,
                   const std::shared_ptr<SQLiteRunStore>& store,
                   const std::filesystem::path& root, std::string run_id,
                   const std::filesystem::path& checkpoint,
                   std::shared_ptr<std::stop_source> interrupt = {}) {
  Domain domain;
  auto module = std::make_shared<OfflineClassifier>();
  module->to(torch::kFloat64);
  domain.model =
      std::make_shared<nn::ModelAdapter>(nn::adapt_tensor_module(module));
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& entry : domain.model->state().named_parameters()) {
    bindings.push_back(
        {.parameter_id = entry.key(), .parameter = entry.value()});
  }
  domain.optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "optimizer.student",
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "student",
           .parameters = std::move(bindings),
           .options = {
               .learning_rate = 0.05, .momentum = 0.8, .weight_decay = 0.01}}});
  domain.scheduler = std::make_shared<optim::StepLrScheduler>(
      domain.optimizer,
      optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.student", .step_size = 1, .gamma = 0.75});
  const auto pipeline = [&](bool validation) {
    const auto features =
        validation ? fixture.validation_features : fixture.training_features;
    const auto labels = fixture.labels;
    const auto factory = [features, labels](
                             std::span<const std::size_t> indices,
                             const data::DataWorkerContext&, std::stop_token) {
      std::vector<std::int64_t> rows;
      for (const auto index : indices)
        rows.push_back(static_cast<std::int64_t>(index));
      const auto selection = torch::tensor(rows, torch::kLong);
      return c10::ivalue::Tuple::create({features.index_select(0, selection),
                                         labels.index_select(0, selection)});
    };
    return std::make_shared<data::DataPipeline>(
        std::make_shared<data::samplers::RandomSampler>(
            kSamples, validation ? 130363 : 104729),
        data::DataPipelineOptions{
            .pipeline_id = validation ? "data.validation" : "data.training",
            .dataset_version = "offline.classification.fixture/v1",
            .dataset_size = kSamples,
            .schema = data::BatchSchema::any("offline.classification", 1),
            .batch_factory_id = "offline.classification.factory/v1",
            .batch_size = kBatch,
            .num_workers = 2,
            .prefetch_batches = 2,
            .persistent_workers = true,
            .worker_seed = 1949},
        factory);
  };
  domain.training_f1 = std::make_shared<tr::metrics::F1Score>(
      tr::metrics::ClassificationMetricOptions{
          .metric_id = "metric.training.f1", .num_classes = 3});
  domain.validation_f1 = std::make_shared<tr::metrics::F1Score>(
      tr::metrics::ClassificationMetricOptions{
          .metric_id = "metric.validation.f1", .num_classes = 3});
  auto callbacks = std::make_shared<tr::CallbackDispatcher>(
      std::vector<std::shared_ptr<tr::CallbackHandler>>{
          std::make_shared<BoundaryCallback>(),
          std::make_shared<tr::callbacks::EarlyStoppingCallback>(
              tr::callbacks::EarlyStoppingOptions{.patience = 100})});
  const auto jsonl = root / (run_id == kBaseline ? "baseline-events.jsonl"
                                                 : "recovery-events.jsonl");
  domain.outbox = std::make_shared<tr::EventOutbox>(
      std::vector<std::shared_ptr<TelemetrySink>>{
          std::make_shared<SQLiteRunTelemetrySink>(
              store, run_id, "sink.evidence", "offline.sqlite"),
          std::make_shared<JsonlTelemetrySink>(
              JsonlTelemetryOptions{.path = jsonl,
                                    .sink_id = "sink.jsonl",
                                    .destination_id = "offline.jsonl"})},
      telemetry_options());
  domain.samples = std::make_shared<std::vector<std::uint64_t>>();
  const auto mapper = [device](const data::BatchEnvelope& batch) {
    return nn::Invocation{
        .positional = {
            batch.value.toTuple()->elements().at(0).toTensor().to(device)}};
  };
  const auto target =
      [device](const data::BatchEnvelope& batch) -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1).toTensor().to(device);
  };
  const auto step = [training_f1 = domain.training_f1,
                     validation_f1 = domain.validation_f1,
                     samples = domain.samples](const tr::StepInput& input) {
    const bool training = input.context.phase == tr::StepPhase::training;
    const auto logits = input.model.invoke(input.invocation).value.toTensor();
    const auto labels = input.target->toTensor();
    const auto& metric = training ? training_f1 : validation_f1;
    metric->update(logits.detach(), labels);
    if (training)
      samples->insert(samples->end(), input.batch.sample_ids.begin(),
                      input.batch.sample_ids.end());
    return tr::StepOutput{
        .loss = torch::nn::functional::cross_entropy(logits, labels),
        .metrics = {{training ? "training.f1" : "validation.f1",
                     metric->compute()}},
        .sample_count = input.batch.sample_ids.size()};
  };
  domain.trainer = std::make_unique<tr::Trainer>(
      tr::TrainerOptions{.trainer_id = "trainer.offline",
                         .run_id = run_id,
                         .trace_id = "trace." + run_id,
                         .max_epochs = kEpochs,
                         .restore_policy = RestorePolicy::exact_only});
  domain.trainer->prepare(
      {.model = domain.model,
       .optimizers =
           {{.optimizer = domain.optimizer,
             .checkpoint =
                 std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
                     domain.optimizer),
             .scheduler =
                 std::make_shared<tr::SchedulerAdapter>(domain.scheduler)}},
       .training_data = pipeline(false),
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .validation_data = pipeline(true),
       .validation_batch_mapper = mapper,
       .validation_target_mapper = target,
       .validation_step = step,
       .strategy = std::make_shared<tr::SingleDeviceStrategy>(
           tr::SingleDeviceStrategyOptions{.device = device}),
       .callbacks = std::move(callbacks),
       .event_outbox = domain.outbox,
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*domain.model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>(
           core::RngCheckpointParticipantOptions{.include_cuda =
                                                     device.is_cuda()}),
       .additional_participants = {domain.training_f1, domain.validation_f1},
       .checkpoint_action = [checkpoint, interrupt](
                                const tr::CheckpointCoordinator& coordinator) {
         tr::FileCheckpointStore destination(checkpoint);
         coordinator.save(destination);
         if (interrupt) {
           // Model abrupt process loss after durable publication. Cancellation
           // prevents the old object's exception cleanup from delivering new
           // failure events beyond the saved outbox cursor. Destructors do no
           // I/O.
           interrupt->request_stop();
           throw SnapshotExit{};
         }
       }});
  return domain;
}

Json fit_json(const tr::FitResult& result) {
  return {{"completed_epochs", result.completed_epochs},
          {"global_step", result.global_step},
          {"training_metrics", result.training_metrics},
          {"validation_metrics", result.validation_metrics}};
}

EvaluationSuite make_suite(const ArtifactDigest& dataset,
                           const std::string& environment, std::uint64_t seed) {
  EvaluationSuite suite{.suite_id = "offline.classification.recovery",
                        .version = 1U,
                        .task = "classification",
                        .dataset_digest = dataset,
                        .environment_id = environment,
                        .seed = seed};
  for (const auto* metric : {"accuracy", "cross_entropy"}) {
    TelemetryMetricDefinition definition;
    definition.metric_id = std::string("classification.") + metric;
    const bool accuracy = std::string_view(metric) == "accuracy";
    definition.unit = accuracy ? TelemetryMetricUnit::ratio
                               : TelemetryMetricUnit::dimensionless;
    definition.aggregation = TelemetryAggregation::mean;
    definition.direction = accuracy ? TelemetryMetricDirection::maximize
                                    : TelemetryMetricDirection::minimize;
    definition.parameters = {{"adapter", "offline.classifier/v1"},
                             {"reduction", "full_fixture_mean"}};
    definition.context_keys = {"split"};
    suite.rules.push_back(
        {.rule_id = metric,
         .definition = std::move(definition),
         .context = {{"split", "validation"}},
         .minimum_samples = kSamples,
         .minimum = 0.0,
         .maximum = accuracy ? std::optional(1.0) : std::nullopt,
         .max_absolute_regression = 0.0});
  }
  return suite;
}

QualityReport evaluate_model(nn::ModelAdapter& model, const Fixture& fixture,
                             const Device& device, const EvaluationSuite& suite,
                             const EvaluationIdentity& identity) {
  return run_evaluation(
      suite, identity,
      [&](const EvaluationSuite& frozen, std::stop_token stop,
          TelemetryDeadline deadline) {
        check_telemetry_operation(stop, deadline);
        const torch::NoGradGuard guard;
        model.state().eval();
        const auto logits =
            model
                .invoke(nn::Invocation{
                    .positional = {fixture.validation_features.to(device)}})
                .value.toTensor();
        const auto labels = fixture.labels.to(device);
        const double accuracy = logits.argmax(1)
                                    .eq(labels)
                                    .to(torch::kFloat64)
                                    .mean()
                                    .item<double>();
        const double loss =
            torch::nn::functional::cross_entropy(logits, labels).item<double>();
        std::vector<TelemetryMetricPoint> observations;
        for (std::size_t index = 0U; index < frozen.rules.size(); ++index) {
          observations.push_back(
              {.definition = frozen.rules[index].definition,
               .step = {.kind = TelemetryStepKind::sample, .value = kSamples},
               .observation =
                   make_telemetry_metric_value(index == 0U ? accuracy : loss),
               .context = frozen.rules[index].context,
               .sample_count = kSamples});
        }
        return observations;
      });
}

Json comparison_json(const QualityComparison& comparison) {
  auto rules = Json::array();
  for (const auto& rule : comparison.results) {
    rules.push_back(
        {{"rule_id", rule.rule_id},
         {"verdict", quality_verdict_name(rule.verdict)},
         {"candidate_minus_baseline", rule.candidate_minus_baseline
                                          ? Json(*rule.candidate_minus_baseline)
                                          : Json(nullptr)},
         {"regression",
          rule.regression ? Json(*rule.regression) : Json(nullptr)},
         {"reason", rule.reason}});
  }
  return {{"schema_version", 1U},
          {"verdict", quality_verdict_name(comparison.verdict)},
          {"baseline_report_sha256", comparison.baseline_report_digest.hex},
          {"candidate_report_sha256", comparison.candidate_report_digest.hex},
          {"results", std::move(rules)}};
}

Json stored_evidence(const SQLiteRunStore& store, std::string_view run_id) {
  auto result = Json::array();
  RunEventPageRequest request;
  request.limit = 128U;
  do {
    const auto page = store.events(run_id, request);
    request.snapshot_through = page.snapshot_through;
    for (const auto& event : page.events) {
      result.push_back(
          {{"run_sequence", event.run_sequence},
           {"sha256", telemetry_content_digest(event.envelope).hex},
           {"event", parse_json(telemetry_projection_json(event.envelope))}});
    }
    if (!page.next_after) break;
    request.after_sequence = *page.next_after;
    if (result.size() > 4096U)
      throw ResourceExhausted("Offline event evidence exceeds 4096 records");
  } while (true);
  return result;
}

Json references_json(const SQLiteRunStore& store, std::string_view run_id) {
  auto result = Json::array();
  for (const auto& reference : store.references(run_id))
    result.push_back(reference_json(reference));
  return result;
}

Json backup_json(const RunBackupReport& report) {
  return {{"database_bytes", report.database_bytes},
          {"artifact_bytes", report.artifact_bytes},
          {"artifact_count", report.artifact_count}};
}

Json execute_workflow(const std::filesystem::path& root, const Device& device,
                      std::uint64_t seed,
                      const std::shared_ptr<SQLiteRunStore>& store) {
  const auto build = current_build_fingerprint();
  const auto build_json = parse_json(build_fingerprint_json(build));
  const auto fixture = make_fixture();
  const auto rebuilt_seed = seed ^ 0x5a5a3c3c12345678ULL;
  const Json config{
      {"schema_version", 1U},
      {"workflow", kWorkflow},
      {"device", device.str()},
      {"seed", seed},
      {"rebuilt_seed", rebuilt_seed},
      {"epochs", kEpochs},
      {"batch_size", kBatch},
      {"checkpoint_global_step", kBoundary},
      {"model", "Linear(4,5), tanh, dropout(0.25), Linear(5,3); float64"},
      {"optimizer",
       {{"algorithm", "SGD"},
        {"learning_rate", 0.05},
        {"momentum", 0.8},
        {"weight_decay", 0.01}}},
      {"scheduler",
       {{"algorithm", "StepLR"}, {"step_size", 1}, {"gamma", 0.75}}},
      {"data_pipeline",
       {{"training_sampler_seed", 104729},
        {"validation_sampler_seed", 130363},
        {"worker_seed", 1949},
        {"num_workers", 2},
        {"prefetch_batches", 2},
        {"persistent_workers", true}}},
      {"restore_policy", "exact_only"},
      {"precision", "full_float64"},
      {"intraop_threads", at::get_num_threads()},
      {"source_identity", build.git_commit}};
  write_json(root / "config.json", config);
  write_json(root / "dataset.json", fixture.manifest);
  write_json(root / "build.json", build_json);
  const ArtifactDigest config_digest{
      .hex = sha256_hex(byte_view(read_text(root / "config.json")))};
  const ArtifactDigest data_digest{
      .hex = sha256_hex(byte_view(read_text(root / "dataset.json")))};
  const std::string fingerprint = build_fingerprint_json(build);
  const auto suite = make_suite(
      data_digest, "offline-f64-" + device.str() + "-" + build.os_arch, seed);
  write_bytes(root / "evaluation-suite.json",
              byte_view(evaluation_suite_json(suite)));
  for (const auto run_id : {kBaseline, kRecovery}) {
    RunMetadata metadata{.run_id = std::string(run_id),
                         .project_id = "project.offline",
                         .experiment_id = "classification.recovery",
                         .display_name = std::string(run_id),
                         .status = RunStatus::running,
                         .created_at_unix_ms = timestamp()};
    metadata.updated_at_unix_ms = metadata.created_at_unix_ms;
    metadata.lineage = {.source_digest = build.git_commit,
                        .build_fingerprint = fingerprint,
                        .config_digest = config_digest.hex,
                        .data_manifest_digest = data_digest.hex};
    metadata.tags = {{"workflow", std::string(kWorkflow)},
                     {"device", device.str()},
                     {"seed", std::to_string(seed)}};
    store->start(metadata);
    publish(*store, run_id, "config", "config", root / "config.json");
    publish(*store, run_id, "dataset", "data_manifest", root / "dataset.json");
    publish(*store, run_id, "build", "build", root / "build.json");
    publish(*store, run_id, "suite", "evaluation_suite",
            root / "evaluation-suite.json");
  }

  core::RngGuard rng_guard(seed, device.is_cuda());
  tr::FitResult baseline_result;
  std::vector<std::uint64_t> baseline_samples;
  Tensor baseline_random;
  Tensor baseline_cpu_random;
  std::vector<std::byte> baseline_weights;
  std::string initial_digest;
  QualityReport baseline_report;
  {
    auto baseline =
        make_domain(fixture, device, store, root, std::string(kBaseline),
                    root / "baseline-boundary.jyck");
    initial_digest = sha256_hex(model_bytes(*baseline.model));
    baseline_result = baseline.trainer->fit();
    require_evidence(baseline_result.status == tr::FitStatus::completed &&
                         baseline_result.global_step == 12U,
                     "Baseline Trainer did not complete the fixed workload");
    baseline_samples = *baseline.samples;
    tr::FileCheckpointStore final_store(root / "baseline-final.jyck");
    baseline.trainer->save_checkpoint(final_store);
    baseline_weights = model_bytes(*baseline.model);
    write_bytes(root / "baseline-model.safetensors", baseline_weights);
    baseline_random = torch::rand(
        {16}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    baseline_cpu_random =
        torch::rand({16}, torch::TensorOptions().dtype(torch::kFloat64));
    baseline_report =
        evaluate_model(*baseline.model, fixture, device, suite,
                       {.evaluation_id = "evaluation.offline.baseline",
                        .run_id = std::string(kBaseline),
                        .model_digest = {.hex = sha256_hex(baseline_weights)},
                        .build_fingerprint = fingerprint,
                        .adapter_id = "offline.classifier",
                        .config_digest = config_digest});
    baseline.trainer->close();
  }

  core::manual_seed(seed, device.is_cuda());
  std::vector<std::uint64_t> recovered_samples;
  std::weak_ptr<nn::ModelAdapter> discarded_model;
  std::weak_ptr<optim::SgdOptimizerAdapter> discarded_optimizer;
  std::string saved_stream;
  auto interrupt = std::make_shared<std::stop_source>();
  {
    auto interrupted =
        make_domain(fixture, device, store, root, std::string(kRecovery),
                    root / "recovery-boundary.jyck", interrupt);
    require_evidence(
        sha256_hex(model_bytes(*interrupted.model)) == initial_digest,
        "Baseline and interrupted initial model states differ");
    discarded_model = interrupted.model;
    discarded_optimizer = interrupted.optimizer;
    bool exited = false;
    try {
      static_cast<void>(interrupted.trainer->fit(interrupt->get_token()));
    } catch (const SnapshotExit&) {
      exited = true;
    }
    require_evidence(
        exited && interrupted.trainer->global_step() == kBoundary,
        "Offline interruption did not occur at the saved safe boundary");
    recovered_samples = *interrupted.samples;
    saved_stream = interrupted.outbox->telemetry_stream_id();
    // Do not close/drain this discarded attempt with a live token: its
    // post-snapshot failure cleanup belongs only to the lost in-memory domain.
  }
  const bool destroyed =
      discarded_model.expired() && discarded_optimizer.expired();
  require_evidence(destroyed,
                   "Interrupted model and optimizer were not destroyed");
  core::manual_seed(rebuilt_seed, device.is_cuda());
  auto resumed =
      make_domain(fixture, device, store, root, std::string(kRecovery),
                  root / "recovery-boundary.jyck");
  const auto rebuilt_digest = sha256_hex(model_bytes(*resumed.model));
  require_evidence(
      rebuilt_digest != initial_digest,
      "Different reconstruction seed did not change initial weights");
  tr::FileCheckpointStore boundary(root / "recovery-boundary.jyck");
  const auto restored = resumed.trainer->restore_checkpoint(boundary);
  require_evidence(
      restored.restore_level == RestoreLevel::exact &&
          resumed.trainer->global_step() == kBoundary &&
          resumed.optimizer->completed_steps() == kBoundary &&
          resumed.scheduler->completed_steps() == 1U,
      "Checkpoint did not restore the complete exact training boundary");
  const bool stream_restored =
      resumed.outbox->telemetry_stream_id() == saved_stream;
  require_evidence(
      stream_restored,
      "Checkpoint did not restore the durable telemetry producer stream");
  const auto recovered_result = resumed.trainer->fit();
  recovered_samples.insert(recovered_samples.end(), resumed.samples->begin(),
                           resumed.samples->end());
  tr::FileCheckpointStore recovered_final(root / "recovery-final.jyck");
  resumed.trainer->save_checkpoint(recovered_final);
  const auto recovered_weights = model_bytes(*resumed.model);
  write_bytes(root / "recovery-model.safetensors", recovered_weights);
  const auto recovered_random = torch::rand(
      {16}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
  const auto recovered_cpu_random =
      torch::rand({16}, torch::TensorOptions().dtype(torch::kFloat64));
  const bool weights_equal = baseline_weights == recovered_weights;
  const bool metrics_equal =
      baseline_result.training_metrics == recovered_result.training_metrics &&
      baseline_result.validation_metrics == recovered_result.validation_metrics;
  const bool rng_equal =
      torch::equal(baseline_random, recovered_random) &&
      torch::equal(baseline_cpu_random, recovered_cpu_random);
  const bool cursor_equal = baseline_samples == recovered_samples;
  const bool counters_equal =
      recovered_result.status == baseline_result.status &&
      recovered_result.global_step == baseline_result.global_step &&
      recovered_result.completed_epochs == baseline_result.completed_epochs &&
      resumed.optimizer->completed_steps() == 12U &&
      resumed.scheduler->completed_steps() == kEpochs &&
      resumed.training_f1->total_observations() == 24 &&
      resumed.validation_f1->total_observations() == 24;
  const Json recovery{
      {"schema_version", 1U},
      {"boundary_global_step", kBoundary},
      {"destroyed_before_rebuild", destroyed},
      {"initial_seed", seed},
      {"rebuilt_seed", rebuilt_seed},
      {"initial_model_sha256", initial_digest},
      {"rebuilt_model_sha256", rebuilt_digest},
      {"parameters_and_buffers_exact", weights_equal},
      {"metrics_exact", metrics_equal},
      {"sample_order_exact", cursor_equal},
      {"counters_exact", counters_equal},
      {"next_rng_exact", rng_equal},
      {"telemetry_stream_restored", stream_restored},
      {"baseline_model_sha256", sha256_hex(baseline_weights)},
      {"recovery_model_sha256", sha256_hex(recovered_weights)},
      {"baseline_next_rng_sha256", tensor_digest(baseline_random)},
      {"recovery_next_rng_sha256", tensor_digest(recovered_random)},
      {"baseline_next_cpu_rng_sha256", tensor_digest(baseline_cpu_random)},
      {"recovery_next_cpu_rng_sha256", tensor_digest(recovered_cpu_random)},
      {"baseline", fit_json(baseline_result)},
      {"recovery", fit_json(recovered_result)},
      {"baseline_sample_order", baseline_samples},
      {"recovery_sample_order", recovered_samples},
      {"pending_events", resumed.outbox->pending_events()}};
  write_json(root / "recovery-check.json", recovery);
  require_evidence(weights_equal && metrics_equal && rng_equal &&
                       cursor_equal && counters_equal &&
                       resumed.outbox->pending_events() == 0U,
                   "Recovered training differs from the uninterrupted "
                   "baseline; inspect recovery-check.json");
  const auto recovery_report =
      evaluate_model(*resumed.model, fixture, device, suite,
                     {.evaluation_id = "evaluation.offline.recovery",
                      .run_id = std::string(kRecovery),
                      .model_digest = {.hex = sha256_hex(recovered_weights)},
                      .build_fingerprint = fingerprint,
                      .adapter_id = "offline.classifier",
                      .config_digest = config_digest});
  resumed.trainer->close();
  const auto comparison =
      compare_quality_reports(baseline_report, recovery_report);
  write_bytes(root / "baseline-quality.json",
              byte_view(quality_report_json(baseline_report)));
  write_bytes(root / "recovery-quality.json",
              byte_view(quality_report_json(recovery_report)));
  write_bytes(root / "recovery-quality.html",
              byte_view(quality_report_html(recovery_report)));
  write_json(root / "quality-comparison.json", comparison_json(comparison));
  require_evidence(baseline_report.verdict == QualityVerdict::pass &&
                       recovery_report.verdict == QualityVerdict::pass &&
                       comparison.verdict == QualityVerdict::pass,
                   "Offline measured evaluation did not pass the frozen suite");

  for (const auto run_id : {kBaseline, kRecovery}) {
    const std::string prefix = run_id == kBaseline ? "baseline" : "recovery";
    publish(*store, run_id, "checkpoint.boundary", "checkpoint.boundary",
            root / (prefix + "-boundary.jyck"),
            "application/x-jyppx-checkpoint", RunArtifactKind::checkpoint);
    publish(*store, run_id, "checkpoint.final", "checkpoint.final",
            root / (prefix + "-final.jyck"), "application/x-jyppx-checkpoint",
            RunArtifactKind::checkpoint);
    publish(*store, run_id, "model", "model.weights",
            root / (prefix + "-model.safetensors"),
            "application/x-safetensors");
    publish(*store, run_id, "quality", "evaluation.report",
            root / (prefix + "-quality.json"));
    publish(*store, run_id, "recovery.check", "recovery.evidence",
            root / "recovery-check.json");
    publish(*store, run_id, "comparison", "evaluation.comparison",
            root / "quality-comparison.json");
    publish(*store, run_id, "events.jsonl", "telemetry.projection",
            root / (prefix + "-events.jsonl"), "application/x-ndjson");
    const auto completed =
        store->transition(run_id, RunStatus::completed, timestamp());
    write_json(root / (prefix + "-run.json"), metadata_json(completed));
    write_json(root / (prefix + "-references.json"),
               references_json(*store, run_id));
    write_json(root / (prefix + "-stored-events.json"),
               stored_evidence(*store, run_id));
  }
  const auto integrity = store->integrity_check();
  require_evidence(integrity.ok(),
                   "Offline store failed the full integrity audit");
  const auto backup = store->backup(root / "backup");
  const auto restored_backup =
      SQLiteRunStore::restore_backup(root / "backup", root / "restored-store");
  SQLiteRunStore restored_store(root / "restored-store");
  const auto restored_integrity = restored_store.integrity_check();
  bool backup_equal = restored_integrity.ok();
  auto runs = Json::array();
  for (const auto run_id : {kBaseline, kRecovery}) {
    const auto metadata = store->find(run_id);
    const auto restored_metadata = restored_store.find(run_id);
    const auto events = stored_evidence(*store, run_id);
    const bool equal =
        !events.empty() && metadata && restored_metadata &&
        metadata_json(*metadata) == metadata_json(*restored_metadata) &&
        references_json(*store, run_id) ==
            references_json(restored_store, run_id) &&
        events == stored_evidence(restored_store, run_id);
    backup_equal = backup_equal && equal;
    runs.push_back({{"run_id", run_id},
                    {"event_count", events.size()},
                    {"backup_restored_evidence_equal", equal}});
  }
  const Json storage{{"schema_version", 1U},
                     {"integrity_ok", integrity.ok()},
                     {"checked_artifacts", integrity.checked_artifacts},
                     {"backup", backup_json(backup)},
                     {"restore", backup_json(restored_backup)},
                     {"restored_integrity_ok", restored_integrity.ok()},
                     {"restored_evidence_exact", backup_equal},
                     {"runs", std::move(runs)}};
  write_json(root / "storage-check.json", storage);
  require_evidence(backup_equal,
                   "Backup/restore did not retain exact metadata, references, "
                   "and neutral events");
  return {{"schema_version", 1U},
          {"workflow", kWorkflow},
          {"root", utf8(root)},
          {"device", device.str()},
          {"seed", seed},
          {"valid", true},
          {"recovery", recovery},
          {"quality", comparison_json(comparison)},
          {"storage", storage},
          {"build", build_json},
          {"summary_path", utf8(root / "summary.json")}};
}

}  // namespace

Result offline_command(const Arguments& arguments) {
  if (arguments.words().size() != 2U || arguments.words()[0] != "offline" ||
      arguments.words()[1] != "run") {
    usage(
        "offline",
        "Use offline run --root NEW_DIRECTORY [--device cpu|cuda] [--seed N].");
  }
  arguments.allow({"root", "device", "seed"});
  const auto root_name = arguments.require("root");
  const auto device_name = arguments.optional("device").value_or("cpu");
  const auto seed = arguments.number(
      "seed", 20261021U,
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()));
  if (device_name != "cpu" && device_name != "cuda")
    usage("offline.device", "Choose cpu or cuda (device index zero).");
  const auto root =
      std::filesystem::absolute(std::filesystem::u8path(root_name))
          .lexically_normal();
  std::error_code error;
  const auto status = std::filesystem::symlink_status(root, error);
  if (error && error != std::errc::no_such_file_or_directory)
    throw IoError("Could not inspect the offline output directory");
  if (std::filesystem::exists(status))
    usage("offline.root",
          "Supply a new directory; existing paths are never overwritten.");
  const bool cuda = device_name == "cuda";
  if (cuda && !cuda_is_available())
    throw DeviceError(
        "Offline workflow requires an available CUDA device zero");
  if (!std::filesystem::create_directory(root, error) || error)
    throw IoError("Could not create the offline output directory",
                  "Choose a new directory under an existing writable parent.");
  std::shared_ptr<SQLiteRunStore> store;
  try {
    store = std::make_shared<SQLiteRunStore>(root / "store");
    const auto device = cuda ? Device(torch::kCUDA, 0) : Device(torch::kCPU);
    auto summary = execute_workflow(root, device, seed, store);
    write_json(root / "summary.json", summary);
    return {std::move(summary), 0};
  } catch (...) {
    // Keep partial evidence for diagnosis. Never remove the user's requested
    // root, and never mask the original failure with secondary audit I/O.
    try {
      write_json(root / "failure.json", {{"workflow", kWorkflow},
                                         {"completed", false},
                                         {"hint",
                                          "Inspect preserved artifacts and run "
                                          "doctor for the selected device."}});
      if (store) {
        for (const auto id : {kBaseline, kRecovery}) {
          const auto metadata = store->find(id);
          if (metadata && metadata->status == RunStatus::running) {
            store->transition(id, RunStatus::failed, timestamp());
          }
        }
      }
    } catch (...) {
    }
    throw;
  }
}

}  // namespace jyppx::cli
