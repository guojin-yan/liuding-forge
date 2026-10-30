// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file search_trial.cpp @brief Native training for bounded local search and
 * explicit checkpoint resume. */
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace experiment = forge::experiment;
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr std::string_view kProject = "search-training-example";
constexpr std::string_view kExperiment = "linear-regression";

std::span<const std::byte> Bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

std::string Utf8(const fs::path& path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::string FileDigest(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open file for hashing");
  forge::Sha256 hash;
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    hash.update(Bytes(std::string_view(
        buffer.data(), static_cast<std::size_t>(input.gcount()))));
  }
  if (!input.eof()) throw std::runtime_error("Cannot read file for hashing");
  return forge::sha256_digest_hex(hash.finish());
}

std::uint64_t Now() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

forge::TelemetryMetricDefinition Definition() {
  forge::TelemetryMetricDefinition value;
  value.metric_id = "validation.mean_squared_error";
  value.unit = forge::TelemetryMetricUnit::dimensionless;
  value.aggregation = forge::TelemetryAggregation::mean;
  value.direction = forge::TelemetryMetricDirection::minimize;
  value.parameters = {{"target", "3*x+0.5"}};
  value.context_keys = {"dataset", "split"};
  return value;
}

Json SearchSpecification(const fs::path& executable,
                         std::string_view strategy) {
  if (strategy != "grid" && strategy != "random")
    throw std::runtime_error("Strategy must be grid or random");
  const auto worker = fs::canonical(executable);
  Json execution{
      {"format", "jyppx.liuding.local_run_spec"},
      {"schema_version", 1},
      {"project_id", kProject},
      {"experiment_id", kExperiment},
      {"executable", Utf8(worker)},
      {"executable_sha256", FileDigest(worker)},
      {"arguments",
       {"--store", "${store}", "--trial", "${trial_id}", "--run",
        "${training_run_id}", "--config", "${config}", "--config-sha256",
        "${config_sha256}", "--seed", "${trial_seed}"}},
      {"working_directory", Utf8(fs::current_path())},
      {"environment", {{"OMP_NUM_THREADS", "2"}, {"MKL_NUM_THREADS", "2"}}},
      {"inherit_environment", true},
      {"timeout_ms", 30000},
      {"max_attempts", 1},
      {"retry_exit_codes", Json::array()},
      {"max_stdout_bytes", 65536},
      {"max_stderr_bytes", 65536}};
  const auto definition = Definition();
  Json spec{{"format", "jyppx.liuding.local_search_spec"},
            {"schema_version", 1},
            {"project_id", kProject},
            {"experiment_id", kExperiment},
            {"strategy", strategy},
            {"seed", 71},
            {"max_trials", 3},
            {"max_duration_ms", 120000},
            {"max_failures", 1},
            {"space",
             {{{"name", "learning_rate"},
               {"config_pointer", "/learning_rate"},
               {"values", {0.005, 0.05, 0.2}}}}},
            {"configuration", {{"learning_rate", 0.05}, {"steps", 60}}},
            {"execution", std::move(execution)},
            {"objective",
             {{"definition",
               {{"schema_version", 1},
                {"metric_id", definition.metric_id},
                {"version", 1},
                {"unit", 1},
                {"aggregation", 3},
                {"direction", 2},
                {"parameters", definition.parameters},
                {"context_keys", {"dataset", "split"}}}},
              {"context",
               {{"dataset", "affine-validation-v1"}, {"split", "validation"}}},
              {"step_kind", "global_step"},
              {"selection", "last"}}},
            {"outputs",
             {{{"reference_id", "checkpoint.best"},
               {"kind", "checkpoint"},
               {"role", "best"}}}},
            {"inputs", Json::array()}};
  return spec;
}

void WriteSpecification(const fs::path& output, const Json& spec) {
  if (!output.is_absolute() || fs::exists(output))
    throw std::runtime_error(
        "Spec output must be an unused absolute file path");
  std::ofstream stream(output, std::ios::binary);
  stream << spec.dump(2) << '\n';
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write search specification");
}

void WriteSpec(const fs::path& executable, const fs::path& output,
               std::string_view strategy) {
  WriteSpecification(output, SearchSpecification(executable, strategy));
  std::cout << "Search specification written; executable SHA-256 pinned.\n";
}

// This local mode uses the same executable as search, with a separate frozen
// training-input contract. It has no random batches, scheduler or AMP state.
constexpr std::string_view kLocalProject = "local-training-example";
constexpr std::string_view kLocalModel = "example.affine.v1";

Json LocalConfiguration() {
  return {{"learning_rate", 0.05}, {"momentum", 0.9}, {"total_steps", 60}};
}

Json LocalData() {
  return {{"format", "affine-generated/v1"},
          {"train_count", 32},
          {"validation_count", 10},
          {"slope", 3.0},
          {"intercept", 0.5}};
}

std::string LocalEnvironment(std::string_view name) {
#if defined(_WIN32)
  const std::wstring key(name.begin(), name.end());
  const auto* value = _wgetenv(key.c_str());
  if (!value || !*value)
    throw std::runtime_error("Missing local worker identity");
  return Utf8(fs::path(value));
#else
  const auto* value = std::getenv(std::string(name).c_str());
  if (!value || !*value)
    throw std::runtime_error("Missing local worker identity");
  return value;
#endif
}

std::uint64_t LocalStep(std::string_view text) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value == 0 || value > 60)
    throw std::runtime_error("The local example requires 1-60 updates");
  return value;
}

Json LocalBinding(std::string_view run,
                  const experiment::RunArtifactReference& reference) {
  return {{"run_id", run},
          {"reference_id", reference.reference_id},
          {"sha256", reference.artifact.digest.hex},
          {"size_bytes", reference.artifact.size_bytes}};
}

void WriteLocalSpec(const fs::path& executable, const fs::path& output,
                    const fs::path& root, std::uint64_t until,
                    const std::string& resume_run,
                    std::uint64_t checkpoint_retry_after = 0) {
  if (!output.is_absolute() || fs::exists(output) || !root.is_absolute())
    throw std::runtime_error(
        "Use an unused absolute spec file and absolute store");
  if (checkpoint_retry_after >= until)
    throw std::runtime_error(
        "The retry demonstration must stop before its total update budget");
  experiment::SQLiteRunStore store(root);
  const auto inputs =
      "local-example-inputs." + forge::sha256_hex(Bytes(Utf8(output)));
  if (store.find(inputs))
    throw std::runtime_error(
        "Example input identity already exists; select a new spec file");
  const auto stamp = Now();
  store.start({.run_id = inputs,
               .project_id = std::string(kLocalProject),
               .experiment_id = "linear-regression",
               .display_name = "Frozen local example inputs",
               .status = experiment::RunStatus::running,
               .created_at_unix_ms = stamp,
               .updated_at_unix_ms = stamp});
  const auto publish = [&](std::string name, const Json& content) {
    const auto encoded = content.dump();
    return store.put_artifact(inputs, name,
                              experiment::RunArtifactKind::artifact, name,
                              Bytes(encoded),
                              {.logical_name = name + ".json",
                               .media_type = "application/json",
                               .producer = "liuding-forge-example-search-trial",
                               .expected_size_bytes = encoded.size()},
                              stamp);
  };
  const auto configuration = publish("configuration", LocalConfiguration());
  const auto data = publish("data_manifest", LocalData());
  store.transition(inputs, experiment::RunStatus::completed, Now());
  Json resume = nullptr;
  if (!resume_run.empty()) {
    const auto source = store.find(resume_run);
    const auto checkpoint = store.find_reference(resume_run, "checkpoint.last");
    if (!source || source->project_id != kLocalProject ||
        !experiment::is_terminal_run_status(source->status) ||
        source->status == experiment::RunStatus::interrupted || !checkpoint ||
        checkpoint->kind != experiment::RunArtifactKind::checkpoint)
      throw std::runtime_error(
          "Select a terminal example attempt with checkpoint.last");
    resume = {{"checkpoint", LocalBinding(resume_run, *checkpoint)},
              {"restore_policy", "exact_only"}};
  }
  const auto worker = fs::canonical(executable);
  Json spec{
      {"format", "jyppx.liuding.local_run_spec"},
      {"schema_version", 1},
      {"project_id", kLocalProject},
      {"experiment_id", "linear-regression"},
      {"executable", Utf8(worker)},
      {"executable_sha256", FileDigest(worker)},
      {"arguments", {"--local-training", std::to_string(until)}},
      {"working_directory", Utf8(fs::current_path())},
      {"environment", {{"OMP_NUM_THREADS", "2"}, {"MKL_NUM_THREADS", "2"}}},
      {"inherit_environment", true},
      {"timeout_ms", 30000},
      {"max_attempts", 1},
      {"retry_exit_codes", Json::array()},
      {"max_stdout_bytes", 65536},
      {"max_stderr_bytes", 65536},
      {"training",
       {{"model_id", kLocalModel},
        {"configuration", LocalBinding(inputs, configuration)},
        {"data_manifest", LocalBinding(inputs, data)},
        {"resume", std::move(resume)}}}};
  if (checkpoint_retry_after != 0) {
    spec["arguments"].push_back("--checkpoint-retry-after");
    spec["arguments"].push_back(std::to_string(checkpoint_retry_after));
    spec["max_attempts"] = 2;
    spec["retry_exit_codes"] = {23};
    spec["training"]["retry_resume"] = {
        {"checkpoint_reference_id", "checkpoint.last"},
        {"restore_policy", "exact_only"}};
  }
  std::ofstream stream(output, std::ios::binary);
  stream << spec.dump(2) << '\n';
  stream.close();
  if (!stream)
    throw std::runtime_error("Cannot write local training specification");
  std::cout << "Local training specification written; inputs and executable "
               "pinned.\n";
}

void WriteTypedSearchSpec(const fs::path& executable, const fs::path& output,
                          const fs::path& root, std::string_view strategy,
                          const std::string& resume_run) {
  if (!output.is_absolute() || fs::exists(output) || !root.is_absolute())
    throw std::runtime_error(
        "Use an unused absolute spec file and absolute store");
  auto spec = SearchSpecification(executable, strategy);
  experiment::SQLiteRunStore store(root);
  Json resume = nullptr;
  if (!resume_run.empty()) {
    const auto source = store.find(resume_run);
    const auto checkpoint = store.find_reference(resume_run, "checkpoint.last");
    if (!source || source->project_id != kLocalProject ||
        !experiment::is_terminal_run_status(source->status) ||
        source->status == experiment::RunStatus::interrupted ||
        !source->tags.contains("liuding.local.kind") ||
        source->tags.at("liuding.local.kind") != "attempt" || !checkpoint ||
        checkpoint->kind != experiment::RunArtifactKind::checkpoint)
      throw std::runtime_error(
          "Select an explicit terminal local attempt with checkpoint.last");
    // Admission rechecks confirmed native ownership. Merely generating this
    // request does not prove that a source process tree has been cleaned up.
    resume = {{"checkpoint", LocalBinding(resume_run, *checkpoint)},
              {"restore_policy", "exact_only"}};
    spec["space"][0]["values"] = {0.05};
    spec["max_trials"] = 1;
  }
  const auto inputs =
      "typed-search-inputs." + forge::sha256_hex(Bytes(Utf8(output)));
  if (store.find(inputs))
    throw std::runtime_error("Select a new typed search specification file");
  const auto stamp = Now();
  store.start({.run_id = inputs,
               .project_id = std::string(kLocalProject),
               .experiment_id = std::string(kExperiment),
               .display_name = "Frozen affine data for typed search",
               .status = experiment::RunStatus::running,
               .created_at_unix_ms = stamp,
               .updated_at_unix_ms = stamp});
  const auto encoded = LocalData().dump();
  const auto data = store.put_artifact(
      inputs, "data_manifest", experiment::RunArtifactKind::artifact,
      "data_manifest", Bytes(encoded),
      {.logical_name = "data-manifest.json",
       .media_type = "application/json",
       .producer = "liuding-forge-example-search-trial",
       .expected_size_bytes = encoded.size()},
      stamp);
  store.transition(inputs, experiment::RunStatus::completed, Now());
  spec["project_id"] = kLocalProject;
  spec["configuration"] = LocalConfiguration();
  auto& execution = spec["execution"];
  execution["project_id"] = kLocalProject;
  execution["arguments"] = {"--typed-search-training",
                            "60",
                            "--trial",
                            "${trial_id}",
                            "--run",
                            "${training_run_id}",
                            "--config-sha256",
                            "${config_sha256}",
                            "--seed",
                            "${trial_seed}"};
  execution["working_directory"] = "${trial_directory}";
  execution["training"] = {{"model_id", kLocalModel},
                           {"configuration", "${trial_configuration}"},
                           {"data_manifest", LocalBinding(inputs, data)},
                           {"resume", std::move(resume)}};
  spec["outputs"] = {{{"reference_id", "checkpoint.last"},
                      {"kind", "checkpoint"},
                      {"role", "last"}},
                     {{"reference_id", "training.result"},
                      {"kind", "artifact"},
                      {"role", "training.result"}}};
  WriteSpecification(output, spec);
  std::cout << "Typed search written; every trial owns its configuration and "
               "shares explicit model/data/checkpoint inputs.\n";
}

// Immutable participant refuses a checkpoint captured with another model,
// configuration or dataset. Update count and momentum belong to native SGD.
class LocalInputIdentity final : public forge::CheckpointParticipant {
 public:
  explicit LocalInputIdentity(std::string identity)
      : identity_(std::move(identity)) {}
  std::string participant_id() const override { return "run.inputs"; }
  forge::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = forge::RestoreLevel::exact,
            .fields = {{.name = "identity",
                        .media_type = "application/json",
                        .max_bytes = 1024,
                        .required = true}}};
  }
  void save_state(forge::StateWriter& writer) const override {
    writer.write_blob("identity", Bytes(identity_));
  }
  void load_state(const forge::StateReader& reader,
                  forge::RestorePolicy policy) override {
    const auto expected = describe_state();
    const auto& actual = reader.manifest();
    forge::validate_state_manifest(actual);
    if (!forge::restore_policy_accepts(policy, actual.restore_level) ||
        actual.participant_id != expected.participant_id ||
        actual.schema_version != expected.schema_version ||
        actual.restore_level != forge::RestoreLevel::exact ||
        !forge::state_fields_equal(actual.fields, expected.fields))
      throw forge::CompatibilityError("Local training input manifest differs");
    const auto value = reader.read_blob("identity", 1024);
    if (!std::equal(value.begin(), value.end(), Bytes(identity_).begin(),
                    Bytes(identity_).end()))
      throw forge::CompatibilityError(
          "Checkpoint model/configuration/data identity differs");
  }

 private:
  std::string identity_;
};

std::string ReadLocalArtifact(experiment::SQLiteRunStore& store,
                              std::string_view run, std::string_view id,
                              std::uint64_t limit,
                              std::string_view expected_digest) {
  const auto reference = store.find_reference(run, id);
  if (!reference || reference->artifact.size_bytes > limit ||
      reference->artifact.digest.hex != expected_digest)
    throw std::runtime_error("Pinned local input reference differs");
  experiment::FileArtifactStore cas(store.artifact_root());
  auto reader = cas.open(reference->artifact.digest);
  if (reader->size_bytes() != reference->artifact.size_bytes)
    throw std::runtime_error("Pinned local input size differs");
  std::string content(static_cast<std::size_t>(reader->size_bytes()), '\0');
  std::size_t offset = 0;
  while (offset < content.size()) {
    const auto count = reader->read(std::as_writable_bytes(
        std::span(content.data() + offset, content.size() - offset)));
    if (!count) throw std::runtime_error("Truncated local input");
    offset += count;
  }
  if (forge::sha256_hex(Bytes(content)) != expected_digest)
    throw std::runtime_error("Pinned local input bytes differ");
  return content;
}

void PublishTypedSearchTraining(
    experiment::SQLiteRunStore& store, const std::string& job,
    const std::string& attempt, const Json& training, const Json& result,
    const std::map<std::string, std::string>& arguments) {
  const auto& run_id = arguments.at("--run");
  const auto& seed = arguments.at("--seed");
  const auto config_sha =
      training.at("configuration").at("sha256").get<std::string>();
  if (arguments.at("--trial") != job || run_id != job + ".training" ||
      arguments.at("--config-sha256") != config_sha)
    throw std::runtime_error("Typed search worker identity differs");
  std::uint64_t parsed_seed = 0;
  const auto parsed =
      std::from_chars(seed.data(), seed.data() + seed.size(), parsed_seed);
  if (parsed.ec != std::errc{} || parsed.ptr != seed.data() + seed.size())
    throw std::runtime_error("Seed must be an unsigned 64-bit integer");
  static_cast<void>(parsed_seed);  // This full-batch recipe consumes no RNG.
  const auto fingerprint = forge::current_build_fingerprint();
  const auto stamp = Now();
  experiment::RunMetadata run;
  run.run_id = run_id;
  run.project_id = kLocalProject;
  run.experiment_id = kExperiment;
  run.display_name = "Native typed search training";
  run.status = experiment::RunStatus::running;
  run.created_at_unix_ms = stamp;
  run.updated_at_unix_ms = stamp;
  run.lineage.parent_run_id = job;
  run.lineage.config_digest = config_sha;
  run.lineage.source_digest = fingerprint.git_commit;
  run.lineage.build_fingerprint = forge::build_fingerprint_json(fingerprint);
  run.lineage.data_manifest_digest =
      training.at("data_manifest").at("sha256").get<std::string>();
  run.tags = {{"seed", seed},
              {"example", "typed-search"},
              {"liuding.training.model_id", std::string(kLocalModel)},
              {"liuding.training.owner_attempt", attempt}};
  store.start(run);
  try {
    std::vector<experiment::RunArtifactReference> references;
    const auto pin = [&](std::string_view owner, std::string_view from,
                         std::string to) {
      const auto reference = store.find_reference(owner, from);
      if (!reference)
        throw std::runtime_error(
            "Typed search publication lacks an input/output");
      auto copy = *reference;
      copy.reference_id = to;
      copy.role = to == "checkpoint.last" ? "last" : to;
      copy.attached_at_unix_ms = stamp;
      references.push_back(std::move(copy));
    };
    pin(job, "local.training.configuration", "search.training.configuration");
    pin(job, "local.training.data_manifest", "search.training.data_manifest");
    if (!training.at("resume").is_null())
      pin(job, "local.training.resume", "search.training.resume");
    pin(attempt, "checkpoint.last", "checkpoint.last");
    pin(attempt, "training.result", "training.result");
    store.attach(run_id, references);
    forge::TelemetryMetricPoint metric;
    metric.definition = Definition();
    metric.context = {{"dataset", "affine-validation-v1"},
                      {"split", "validation"}};
    metric.step = {.kind = forge::TelemetryStepKind::global_step,
                   .value = result.at("completed_steps").get<std::uint64_t>()};
    metric.observation = forge::make_telemetry_metric_value(
        result.at("validation_mse").get<double>());
    metric.sample_count = 10U;
    forge::TelemetryEnvelope event;
    event.time.wall_time_unix_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    event.identity = {.stream_id = run_id + ".training",
                      .event_id = run_id + ".validation-final",
                      .idempotency_key = "validation-final",
                      .sequence = 1};
    event.correlation.project_id = kLocalProject;
    event.correlation.run_id = run_id;
    event.correlation.attempt_id = job;
    event.payload = forge::TelemetryMetricBatch{{std::move(metric)}};
    store.append(run_id, "validation-final", std::span(&event, 1U));
    store.transition(run_id, experiment::RunStatus::completed, Now());
  } catch (...) {
    store.transition(run_id, experiment::RunStatus::failed, Now());
    throw;
  }
}

int LocalTrain(std::uint64_t until,
               const std::map<std::string, std::string>* search = nullptr,
               std::uint64_t checkpoint_retry_after = 0) {
  const auto root =
      fs::u8path(LocalEnvironment("JYPPX_LIUDING_LOCAL_STORE_ROOT"));
  const auto job = LocalEnvironment("JYPPX_LIUDING_LOCAL_JOB_ID");
  const auto attempt = LocalEnvironment("JYPPX_LIUDING_LOCAL_ATTEMPT_ID");
  const auto spec_sha = LocalEnvironment("JYPPX_LIUDING_LOCAL_SPEC_SHA256");
  experiment::SQLiteRunStore store(root);
  const auto metadata = store.find(attempt);
  if (!metadata || metadata->status != experiment::RunStatus::running ||
      metadata->lineage.parent_run_id != job ||
      metadata->lineage.config_digest != spec_sha)
    throw std::runtime_error(
        "Local example requires its owned running attempt");
  const auto spec =
      Json::parse(ReadLocalArtifact(store, job, "local.spec", 65536, spec_sha));
  auto training = spec.at("training");
  std::string input_suffix;
  std::string training_sha;
  if (training.contains("retry_resume")) {
    training_sha = LocalEnvironment("JYPPX_LIUDING_LOCAL_TRAINING_SHA256");
    const auto index = LocalEnvironment("JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX");
    const auto frozen = Json::parse(ReadLocalArtifact(
        store, attempt, "local.training.inputs", 65536, training_sha));
    if (frozen.at("format") != "jyppx.liuding.local_training_inputs" ||
        frozen.at("schema_version") != 1 || frozen.at("job_id") != job ||
        frozen.at("attempt_id") != attempt ||
        frozen.at("spec_sha256") != spec_sha ||
        std::to_string(frozen.at("index").get<std::uint64_t>()) != index)
      throw std::runtime_error("Frozen attempt training identity differs");
    for (const auto name : {"model_id", "configuration", "data_manifest"})
      if (frozen.at("training").at(name) != training.at(name))
        throw std::runtime_error("Retry changed an immutable training input");
    training = frozen.at("training");
    input_suffix = "." + index;
  }
  const bool first_retry_segment =
      checkpoint_retry_after != 0 &&
      LocalEnvironment("JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX") == "1";
  if (checkpoint_retry_after != 0 &&
      (search || input_suffix.empty() || checkpoint_retry_after >= until))
    throw std::runtime_error("Invalid owned checkpoint retry demonstration");
  const auto target = first_retry_segment ? checkpoint_retry_after : until;
  if (training.at("model_id").get<std::string>() != kLocalModel)
    throw std::runtime_error("This example implements example.affine.v1 only");
  const auto config_sha =
      training.at("configuration").at("sha256").get<std::string>();
  const auto data_sha =
      training.at("data_manifest").at("sha256").get<std::string>();
  const auto config = Json::parse(ReadLocalArtifact(
      store, job, "local.training.configuration" + input_suffix, 4096,
      config_sha));
  const auto data = Json::parse(ReadLocalArtifact(
      store, job, "local.training.data_manifest" + input_suffix, 4096,
      data_sha));
  auto expected_config = LocalConfiguration();
  if (search && config.is_object() && config.contains("learning_rate"))
    expected_config["learning_rate"] = config.at("learning_rate");
  if (config != expected_config || data != LocalData())
    throw std::runtime_error(
        "The learning example requires its frozen affine recipe");
  const auto rate = config.at("learning_rate").get<double>();
  if (!std::isfinite(rate) || rate <= 0.0 || rate > 1.0)
    throw std::runtime_error("Learning rate exceeds the example bounds");
  const auto work_parent = root / "local-training-example";
  fs::create_directories(work_parent);
  const auto work = work_parent / attempt;
  if (!fs::create_directory(work))
    throw std::runtime_error(
        "Attempt workspace already exists; do not relaunch it");
  at::set_num_threads(2);
  auto model = std::make_shared<torch::nn::LinearImpl>(1, 1);
  {
    const torch::NoGradGuard guard;
    model->weight.zero_();
    model->bias.zero_();
  }
  auto optimizer = std::make_shared<forge::optim::SgdOptimizerAdapter>(
      "optimizer.main",
      std::vector<forge::optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = {{.parameter_id = "weight",
                           .parameter = model->weight},
                          {.parameter_id = "bias", .parameter = model->bias}},
           .options = {.learning_rate = rate, .momentum = 0.9}}});
  forge::training::CheckpointParticipantRegistry registry;
  registry.register_participant(
      std::make_shared<forge::nn::ModelCheckpointParticipant>(
          std::make_shared<forge::nn::ModelStateView>(model),
          forge::nn::ModelCheckpointParticipantOptions{
              .max_tensors = 2,
              .max_topology_bytes = 4096,
              .max_tensor_bytes = 64,
              .max_total_bytes = 8192}));
  registry.register_participant(
      std::make_shared<forge::optim::SgdOptimizerCheckpointParticipant>(
          optimizer, forge::optim::SgdOptimizerCheckpointOptions{
                         .max_groups = 1,
                         .max_parameters = 2,
                         .max_topology_bytes = 4096,
                         .max_producer_bytes = 4096,
                         .max_control_bytes = 4096,
                         .max_tensor_bytes = 64,
                         .max_total_bytes = 16384}));
  registry.register_participant(std::make_shared<LocalInputIdentity>(Json{
      {"model_id", kLocalModel},
      {"configuration_sha256", config_sha},
      {"data_manifest_sha256", data_sha}}.dump()));
  forge::training::CheckpointCoordinator coordinator(
      registry, {.max_participants = 3, .max_rollback_bytes = 65536});
  const forge::training::FileCheckpointStoreOptions limits{
      .max_artifact_bytes = 65536, .max_field_bytes = 16384};
  const auto& resume = training.at("resume");
  if (!resume.is_null()) {
    if (resume.at("restore_policy") != "exact_only")
      throw std::runtime_error("The example requires exact_only restore");
    const auto checkpoint = ReadLocalArtifact(
        store, job, "local.training.resume" + input_suffix, 65536,
        resume.at("checkpoint").at("sha256").get<std::string>());
    const auto path = work / "resume.jyck";
    std::ofstream stream(path, std::ios::binary);
    stream.write(checkpoint.data(),
                 static_cast<std::streamsize>(checkpoint.size()));
    stream.close();
    if (!stream)
      throw std::runtime_error("Cannot stage the bounded example checkpoint");
    forge::training::FileCheckpointStore source(path, limits);
    coordinator.restore(source, forge::RestorePolicy::exact_only);
  }
  const auto initial_step = optimizer->completed_steps();
  if (initial_step >= target)
    throw std::runtime_error(
        "Select a new budget beyond the restored update count");
  const auto train_x = torch::linspace(-1.0, 1.0, 32).reshape({32, 1});
  const auto train_y = train_x * 3.0 + 0.5;
  for (auto step = initial_step; step < target; ++step) {
    optimizer->zero_grad();
    torch::mse_loss(model->forward(train_x), train_y).backward();
    optimizer->step();
  }
  const auto checkpoint = work / "checkpoint-last.jyck";
  {
    forge::training::FileCheckpointStore sink(checkpoint, limits);
    coordinator.save(sink);
  }
  const auto stamp = Now();
  const auto published = store.put_artifact_file(
      attempt, "checkpoint.last", experiment::RunArtifactKind::checkpoint,
      "last", checkpoint,
      {.logical_name = "checkpoint-last.jyck",
       .media_type = "application/octet-stream",
       .producer = "liuding-forge-example-search-trial",
       .expected_size_bytes = fs::file_size(checkpoint)},
      stamp, {.hex = FileDigest(checkpoint)});
  const torch::NoGradGuard guard;
  const auto x = torch::linspace(-0.9, 0.9, 10).reshape({10, 1});
  const auto mse =
      torch::mse_loss(model->forward(x), x * 3.0 + 0.5).item<double>();
  if (!std::isfinite(mse))
    throw std::runtime_error("Nonfinite local training metric");
  Json result{{"model_id", kLocalModel},
              {"configuration_sha256", config_sha},
              {"data_manifest_sha256", data_sha},
              {"initial_step", initial_step},
              {"completed_steps", optimizer->completed_steps()},
              {"restore_policy", "exact_only"},
              {"resumed", !resume.is_null()},
              {"validation_mse", mse},
              {"weight", model->weight.item<double>()},
              {"bias", model->bias.item<double>()},
              {"checkpoint_sha256", published.artifact.digest.hex}};
  if (!training_sha.empty()) result["training_sha256"] = training_sha;
  const auto content = result.dump();
  store.put_artifact(attempt, "training.result",
                     experiment::RunArtifactKind::artifact, "training.result",
                     Bytes(content),
                     {.logical_name = "training-result.json",
                      .media_type = "application/json",
                      .producer = "liuding-forge-example-search-trial",
                      .expected_size_bytes = content.size()},
                     stamp);
  if (search)
    PublishTypedSearchTraining(store, job, attempt, training, result, *search);
  if (first_retry_segment) {
    std::cout << "Checkpoint committed at the learning boundary; exiting 23 "
                 "for the declared exact continuation.\n";
    return 23;
  }
  std::cout << "Native model, SGD momentum and committed update count saved.\n";
  return 0;
}

void Train(const std::map<std::string, std::string>& arguments) {
  const auto& run_id = arguments.at("--run");
  const auto& trial_id = arguments.at("--trial");
  const auto& config_digest = arguments.at("--config-sha256");
  const auto config_path = fs::u8path(arguments.at("--config"));
  if (fs::file_size(config_path) > 4096U ||
      FileDigest(config_path) != config_digest)
    throw std::runtime_error("Frozen configuration size or digest differs");
  std::ifstream input(config_path, std::ios::binary);
  const auto config = Json::parse(input);
  if (!config.is_object() || config.size() != 2 || !config.contains("steps") ||
      !config.contains("learning_rate") ||
      !config.at("steps").is_number_integer() ||
      !config.at("learning_rate").is_number())
    throw std::runtime_error("Expected learning_rate and integer steps");
  const auto steps = config.at("steps").get<std::int64_t>();
  const auto rate = config.at("learning_rate").get<double>();
  if (!std::isfinite(rate) || rate <= 0.0 || rate > 1.0 || steps < 1 ||
      steps > 1024)
    throw std::runtime_error("Training configuration exceeds example bounds");
  std::uint64_t seed = 0;
  const auto& text = arguments.at("--seed");
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), seed);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::runtime_error("Seed must be an unsigned 64-bit integer");

  experiment::SQLiteRunStore store(fs::u8path(arguments.at("--store")));
  const auto fingerprint = forge::current_build_fingerprint();
  experiment::RunMetadata run;
  run.run_id = run_id;
  run.project_id = kProject;
  run.experiment_id = kExperiment;
  run.display_name = "Offline linear-regression search trial";
  run.status = experiment::RunStatus::running;
  run.created_at_unix_ms = Now();
  run.updated_at_unix_ms = run.created_at_unix_ms;
  run.lineage.parent_run_id = trial_id;
  run.lineage.config_digest = config_digest;
  run.lineage.source_digest = fingerprint.git_commit;
  run.lineage.build_fingerprint = forge::build_fingerprint_json(fingerprint);
  run.tags = {{"example", "local-search"}, {"seed", text}};
  store.start(run);
  try {
    at::set_num_threads(2);
    torch::manual_seed(seed);
    auto model = std::make_shared<torch::nn::LinearImpl>(1, 1);
    // A common explicit start isolates the learning-rate effect across trials.
    {
      const torch::NoGradGuard guard;
      model->weight.zero_();
      model->bias.zero_();
    }
    const auto train_x = torch::linspace(-1.0, 1.0, 32).reshape({32, 1});
    const auto train_y = train_x * 3.0 + 0.5;
    const auto validation_x = torch::linspace(-0.9, 0.9, 10).reshape({10, 1});
    const auto validation_y = validation_x * 3.0 + 0.5;
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(rate));
    for (std::int64_t step = 0; step < steps; ++step) {
      optimizer.zero_grad();
      const auto loss = torch::mse_loss(model->forward(train_x), train_y);
      loss.backward();
      optimizer.step();
    }
    const torch::NoGradGuard guard;
    model->eval();
    const auto loss =
        torch::mse_loss(model->forward(validation_x), validation_y)
            .item<double>();
    if (!std::isfinite(loss))
      throw std::runtime_error("Validation loss is nonfinite");
    forge::TelemetryMetricPoint metric;
    metric.definition = Definition();
    metric.context = {{"dataset", "affine-validation-v1"},
                      {"split", "validation"}};
    metric.step = {.kind = forge::TelemetryStepKind::global_step,
                   .value = static_cast<std::uint64_t>(steps)};
    metric.observation = forge::make_telemetry_metric_value(loss);
    metric.sample_count = 10U;
    forge::TelemetryEnvelope event;
    event.time.wall_time_unix_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    event.identity = {.stream_id = run_id + ".training",
                      .event_id = run_id + ".validation-final",
                      .idempotency_key = "validation-final",
                      .sequence = 1};
    event.correlation.project_id = kProject;
    event.correlation.run_id = run_id;
    event.correlation.attempt_id = trial_id;
    event.payload = forge::TelemetryMetricBatch{{std::move(metric)}};
    store.append(run_id, "validation-final", std::span(&event, 1U));

    // Publish a non-executable exact model snapshot, then verify
    // reconstruction.
    const auto checkpoint = config_path.parent_path() / "model-best.jyck";
    auto state = std::make_shared<forge::nn::ModelStateView>(model);
    forge::nn::ModelCheckpointParticipantOptions participant_options;
    participant_options.max_tensors = 2;
    participant_options.max_topology_bytes = 4096;
    participant_options.max_tensor_bytes = 64;
    participant_options.max_total_bytes = 8192;
    forge::nn::ModelCheckpointParticipant participant(state,
                                                      participant_options);
    forge::interop::FileStateArchiveOptions archive_options;
    archive_options.max_artifact_bytes = 65536;
    archive_options.max_field_bytes = 8192;
    {
      forge::interop::FileStateArchive archive(checkpoint, archive_options);
      archive.begin({participant.describe_state()});
      participant.save_state(archive.writer_for(participant.participant_id()));
      archive.commit();
    }
    auto restored = std::make_shared<torch::nn::LinearImpl>(1, 1);
    forge::nn::ModelCheckpointParticipant reader(
        std::make_shared<forge::nn::ModelStateView>(restored),
        participant_options);
    {
      archive_options.read_only = true;
      forge::interop::FileStateArchive archive(checkpoint, archive_options);
      reader.load_state(archive.reader_for(reader.participant_id()),
                        forge::RestorePolicy::exact_only);
    }
    if (!torch::equal(model->forward(validation_x),
                      restored->forward(validation_x)))
      throw std::runtime_error("Model snapshot did not restore exactly");
    store.put_artifact_file(run_id, "checkpoint.best",
                            experiment::RunArtifactKind::checkpoint, "best",
                            checkpoint,
                            {.logical_name = "model-best.jyck",
                             .media_type = "application/octet-stream",
                             .producer = "liuding-forge-example-search-trial",
                             .expected_size_bytes = fs::file_size(checkpoint)},
                            Now(), {.hex = FileDigest(checkpoint)});
    store.transition(run_id, experiment::RunStatus::completed, Now());
    std::cout << "Training completed; validation metric and exact model "
                 "snapshot persisted.\n";
  } catch (...) {
    store.transition(run_id, experiment::RunStatus::failed, Now());
    throw;
  }
}
}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    if (argc == 2 && Utf8(fs::path(argv[1])) == "--help") {
      std::cout
          << "Usage: liuding-forge-example-search-trial --write-spec "
             "ABSOLUTE_FILE [--strategy grid|random]\n"
             "Worker: --store ROOT --trial ID --run ID --config FILE "
             "--config-sha256 SHA256 --seed UINT64\n"
             "Local: --write-local-spec FILE --store ROOT --until-step 20 "
             "[--resume-run SOURCE_ATTEMPT] [--checkpoint-retry-after STEP]\n"
             "Local worker: --local-training UNTIL_STEP "
             "[--checkpoint-retry-after STEP] (owned runner only)\n";
      std::cout << "Typed search: --write-typed-search-spec FILE --store ROOT "
                   "[--strategy grid|random] [--resume-run SOURCE_ATTEMPT]\n";
      return 0;
    }
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc ||
          !arguments
               .emplace(Utf8(fs::path(argv[i])), Utf8(fs::path(argv[i + 1])))
               .second)
        throw std::runtime_error(
            "Expected distinct option/value pairs; see --help");
    }
    if (arguments.contains("--write-typed-search-spec")) {
      const auto expected_size = 2U +
                                 (arguments.contains("--strategy") ? 1U : 0U) +
                                 (arguments.contains("--resume-run") ? 1U : 0U);
      if (arguments.size() != expected_size)
        throw std::runtime_error("Unknown typed search option; see --help");
      WriteTypedSearchSpec(
          fs::path(argv[0]),
          fs::u8path(arguments.at("--write-typed-search-spec")),
          fs::u8path(arguments.at("--store")),
          arguments.contains("--strategy") ? arguments.at("--strategy")
                                           : "grid",
          arguments.contains("--resume-run") ? arguments.at("--resume-run")
                                             : "");
    } else if (arguments.contains("--typed-search-training")) {
      if (arguments.size() != 5U)
        throw std::runtime_error("Expected the typed search worker options");
      return LocalTrain(LocalStep(arguments.at("--typed-search-training")),
                        &arguments);
    } else if (arguments.contains("--write-local-spec")) {
      const auto expected_size =
          3U + (arguments.contains("--resume-run") ? 1U : 0U) +
          (arguments.contains("--checkpoint-retry-after") ? 1U : 0U);
      if (arguments.size() != expected_size)
        throw std::runtime_error(
            "Unknown local specification option; see --help");
      WriteLocalSpec(
          fs::path(argv[0]), fs::u8path(arguments.at("--write-local-spec")),
          fs::u8path(arguments.at("--store")),
          LocalStep(arguments.at("--until-step")),
          arguments.contains("--resume-run") ? arguments.at("--resume-run")
                                             : "",
          arguments.contains("--checkpoint-retry-after")
              ? LocalStep(arguments.at("--checkpoint-retry-after"))
              : 0);
    } else if (arguments.contains("--local-training")) {
      if (arguments.size() !=
          (arguments.contains("--checkpoint-retry-after") ? 2U : 1U))
        throw std::runtime_error("Unknown local worker option; see --help");
      return LocalTrain(
          LocalStep(arguments.at("--local-training")), nullptr,
          arguments.contains("--checkpoint-retry-after")
              ? LocalStep(arguments.at("--checkpoint-retry-after"))
              : 0);
    } else if (arguments.contains("--write-spec")) {
      if (arguments.size() != (arguments.contains("--strategy") ? 2U : 1U))
        throw std::runtime_error("Unknown specification option");
      WriteSpec(fs::path(argv[0]), fs::u8path(arguments.at("--write-spec")),
                arguments.contains("--strategy") ? arguments.at("--strategy")
                                                 : "grid");
    } else {
      const std::array<std::string, 6> required{"--store",         "--trial",
                                                "--run",           "--config",
                                                "--config-sha256", "--seed"};
      if (arguments.size() != required.size())
        throw std::runtime_error("Expected the six worker options; see --help");
      for (const auto& key : required) (void)arguments.at(key);
      Train(arguments);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
