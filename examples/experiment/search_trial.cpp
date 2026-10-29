// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/** @file search_trial.cpp @brief Offline native training worker for local
 * search. */
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

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

void WriteSpec(const fs::path& executable, const fs::path& output,
               std::string_view strategy) {
  if (strategy != "grid" && strategy != "random")
    throw std::runtime_error("Strategy must be grid or random");
  if (!output.is_absolute() || fs::exists(output))
    throw std::runtime_error(
        "Spec output must be an unused absolute file path");
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
  std::ofstream stream(output, std::ios::binary);
  stream << spec.dump(2) << '\n';
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write search specification");
  std::cout << "Search specification written; executable SHA-256 pinned.\n";
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
      std::cout << "Usage: liuding-forge-example-search-trial --write-spec "
                   "ABSOLUTE_FILE [--strategy grid|random]\n"
                   "Worker: --store ROOT --trial ID --run ID --config FILE "
                   "--config-sha256 SHA256 --seed UINT64\n";
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
    if (arguments.contains("--write-spec")) {
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
