// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace liuding = jyppx::liuding;
namespace training = liuding::training;
namespace experiment = liuding::experiment;
namespace nn = liuding::nn;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

liuding::ArtifactDigest digest(std::string_view text) {
  return {.hex = liuding::sha256_hex(std::as_bytes(std::span(text)))};
}

void hash_text(liuding::Sha256& hash, std::string_view text) {
  hash.update(std::as_bytes(std::span(text)));
}

void hash_tensor(liuding::Sha256& hash, const torch::Tensor& tensor) {
  require(tensor.device().is_cpu() && tensor.is_contiguous(),
          "Example identity requires contiguous CPU tensors");
  hash.update(std::span(static_cast<const std::byte*>(tensor.const_data_ptr()),
                        tensor.nbytes()));
}

// Fixed, deterministic inference keeps the checkpoint focused on evaluation
// counts/cursor. A stochastic or mutable model also needs its own participants.
struct Task {
  torch::Tensor inputs = torch::tensor({{3.F, 1.F},
                                        {0.F, 2.F},
                                        {-1.F, -1.F},
                                        {2.F, 2.F},
                                        {9.F, 0.F},
                                        {0.F, 0.F}});
  torch::Tensor targets = torch::tensor({0, 1, 1, 1, -1, 2}, torch::kInt64);
  std::array<std::size_t, 6> sources{0, 1, 0, 1, 0, 0};
  std::shared_ptr<torch::nn::LinearImpl> module =
      std::make_shared<torch::nn::LinearImpl>(2, 3);
  std::uint64_t forwards = 0;
  std::uint64_t source_calls = 0;

  Task() {
    torch::NoGradGuard guard;
    module->weight.copy_(torch::tensor({{1.F, 0.F}, {0.F, 1.F}, {-1.F, -1.F}}));
    module->bias.zero_();
  }

  liuding::ArtifactDigest model_digest() const {
    liuding::Sha256 hash;
    hash_text(hash, "Linear(2,3)/FP32/weight[3,2]/bias[3]/v1\n");
    hash_tensor(hash, module->weight);
    hash_tensor(hash, module->bias);
    return {.hex = liuding::sha256_digest_hex(hash.finish())};
  }

  training::ClassificationEvaluationIdentity identity() const {
    liuding::Sha256 hash;
    hash_text(hash, "features[6,2]/FP32/targets[6]/Int64/sources[6]/u8/v1\n");
    hash_tensor(hash, inputs);
    hash_tensor(hash, targets);
    for (const auto source : sources) {
      const auto byte = static_cast<std::byte>(source);
      hash.update(std::span(&byte, 1));
    }
    return {
        .evaluation_id = "classification.example.v1",
        .dataset_digest = {.hex = liuding::sha256_digest_hex(hash.finish())},
        .model_digest = model_digest(),
        .config_digest =
            digest("cpu/fp32/batch2/classes3/ignore-1/zero0/lowest-index-tie/"
                   "sources=a,b,empty/expected6/seed0/v1"),
        .build_fingerprint = liuding::build_fingerprint_json()};
  }

  nn::ModelAdapter adapter() {
    return nn::adapt_module(
        module,
        [this](torch::nn::LinearImpl& model,
               const nn::Invocation& call) -> c10::IValue {
          require(!model.is_training() && !torch::GradMode::is_enabled(),
                  "Evaluation must enable eval/no-grad guards");
          ++forwards;
          return model.forward(call.positional.front().toTensor());
        },
        {.positional = {liuding::ValueSchema::tensor()}},
        liuding::ValueSchema::tensor());
  }

  std::optional<training::ClassificationEvaluationBatch> next(
      std::uint64_t index, std::stop_token stop) {
    if (stop.stop_requested()) throw liuding::Cancelled("Source cancelled");
    ++source_calls;
    if (index >= 3) return std::nullopt;
    const auto begin = static_cast<std::int64_t>(index * 2);
    const auto end = begin + 2;
    return training::ClassificationEvaluationBatch{
        .invocation = {.positional = {inputs.slice(0, begin, end)}},
        .targets = targets.slice(0, begin, end),
        .source_indices = {sources.begin() + begin, sources.begin() + end}};
  }
};

training::ClassificationEvaluationOptions options() {
  return {.num_classes = 3,
          .ignore_index = -1,
          .sources = {"a", "b", "empty"},
          .max_batches = 3,
          .max_samples = 6,
          .max_batch_samples = 2,
          .max_classes = 3,
          .max_sources = 3,
          .max_confusion_bytes = 4096};
}

void write_same(const std::filesystem::path& path, const std::string& bytes) {
  if (std::filesystem::exists(path)) {
    require(std::filesystem::file_size(path) <= (1U << 20U),
            "Existing example output is unexpectedly large");
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Cannot read existing report");
    const std::string previous((std::istreambuf_iterator<char>(input)), {});
    require(previous == bytes, "Existing report differs; use a new directory");
    return;
  }
  std::ofstream output(path, std::ios::binary);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  require(static_cast<bool>(output), "Cannot write report");
}

experiment::QualityReport report(
    const training::ClassificationEvaluationResult& result,
    const training::ClassificationEvaluationIdentity& id, bool empty,
    bool strict = false) {
  experiment::EvaluationSuite suite{
      .suite_id = empty    ? "classification.example.empty"
                  : strict ? "classification.example.strict"
                           : "classification.example",
      .task = "classification",
      .dataset_digest = id.dataset_digest,
      .environment_id = "cpu/fp32/batch2",
      .seed = 0};
  std::vector<liuding::TelemetryMetricPoint> selected;
  for (const auto& point : result.observations) {
    const bool matches = empty ? point.context.at("source") == "empty"
                               : point.context.at("slice") == "overall";
    if (!matches || point.definition.metric_id != "classification.accuracy")
      continue;
    selected.push_back(point);
    suite.rules.push_back({.rule_id = "accuracy",
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1,
                           .minimum = strict ? 0.9 : 0.4});
  }
  require(selected.size() == 1, "Expected one frozen report rule");
  return experiment::evaluate_observations(
      suite,
      {.evaluation_id = id.evaluation_id,
       .run_id = "example.fixed-model",
       .model_digest = id.model_digest,
       .build_fingerprint = id.build_fingerprint,
       .adapter_id = "training.classification.example",
       .config_digest = id.config_digest},
      selected);
}

void finish(const std::filesystem::path& root,
            const training::ClassificationEvaluationResult& result,
            const training::ClassificationEvaluationIdentity& id) {
  const std::vector<std::vector<std::int64_t>> expected{
      {1, 0, 0}, {1, 1, 1}, {1, 0, 0}};
  require(result.batches == 3 && result.rows_seen == 6 &&
              result.overall.samples == 5 &&
              result.overall.confusion == expected,
          "Unexpected complete classification counts");
  require(std::abs(*result.overall.accuracy - 2.0 / 5) < 1e-14 &&
              std::abs(*result.overall.macro_precision - 4.0 / 9) < 1e-14 &&
              std::abs(*result.overall.macro_recall - 4.0 / 9) < 1e-14 &&
              std::abs(*result.overall.macro_f1 - 1.0 / 3) < 1e-14,
          "Classification scores differ from the independent count formula");
  require(result.by_source.size() == 3 && result.by_source[0].samples == 3 &&
              result.by_source[1].samples == 2 &&
              !result.by_source[2].accuracy.has_value(),
          "Source slices must preserve missing values");
  const auto accepted = report(result, id, false);
  const auto failed = report(result, id, false, true);
  const auto missing = report(result, id, true);
  require(accepted.verdict == experiment::QualityVerdict::pass &&
              failed.verdict == experiment::QualityVerdict::fail &&
              missing.verdict == experiment::QualityVerdict::incomplete,
          "Report pass/fail/missing contracts differ");
  write_same(root / "quality.json", experiment::quality_report_json(accepted));
  write_same(root / "quality.html", experiment::quality_report_html(accepted));
  write_same(root / "strict.json", experiment::quality_report_json(failed));
  write_same(root / "empty-source.json",
             experiment::quality_report_json(missing));
  // Save every score/context, including absent source values, for inspection.
  write_same(
      root / "measured.json",
      experiment::evaluation_input_json({.identity = accepted.identity,
                                         .observations = result.observations}));
  std::cout << " rows=6 samples=5 accuracy=" << *result.overall.accuracy
            << " macro_precision=" << *result.overall.macro_precision
            << " macro_recall=" << *result.overall.macro_recall
            << " macro_f1=" << *result.overall.macro_f1;
}

struct Pause {};
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3,
            "Usage: program full|pause|resume|cancel|reject-identity DIR");
    const std::string_view mode = argv[1];
    require(mode == "full" || mode == "pause" || mode == "resume" ||
                mode == "cancel" || mode == "reject-identity",
            "Unknown example mode");
    torch::set_num_threads(1);
    torch::manual_seed(0);
    std::cout << std::setprecision(17);
    const std::filesystem::path root = argv[2];
    std::filesystem::create_directories(root);
    Task task;
    auto id = task.identity();
    if (mode == "reject-identity")
      id.config_digest = digest("changed batching");
    auto session = std::make_shared<training::ClassificationEvaluationSession>(
        id, options(), 6);
    training::CheckpointParticipantRegistry registry;
    registry.register_participant(session);
    training::CheckpointCoordinator coordinator(
        registry, {.max_participants = 1, .max_rollback_bytes = 1U << 20U});
    training::FileCheckpointStore file(
        root / "evaluation.state",
        {.max_artifact_bytes = 1U << 20U, .max_field_bytes = 1U << 20U});
    if (mode == "reject-identity") {
      bool refused = false;
      try {
        coordinator.restore(file, liuding::RestorePolicy::exact_only);
      } catch (const liuding::CompatibilityError&) {
        refused = true;
      }
      require(refused && session->progress().batches == 0,
              "Changed identity must be refused without changing progress");
      std::cout
          << "mode=reject-identity refused=true forwards=0 source_calls=0\n";
      return 0;
    }
    if (mode == "resume")
      coordinator.restore(file, liuding::RestorePolicy::exact_only);
    const auto start = session->progress();
    auto model = task.adapter();
    std::stop_source stop;
    bool paused = false;
    try {
      const auto result = session->run(
          model,
          [&](auto index, auto token) { return task.next(index, token); },
          [&](const auto& committed) {
            coordinator.save(file);
            if (committed.progress().batches == 1 &&
                !committed.progress().complete) {
              if (mode == "pause") throw Pause{};
              if (mode == "cancel") stop.request_stop();
            }
          },
          stop.get_token());
      require(mode != "pause" && mode != "cancel", "Expected interruption");
      finish(root, result, id);
    } catch (const Pause&) {
      require(mode == "pause", "Unexpected pause");
      paused = true;
    } catch (const liuding::Cancelled&) {
      require(mode == "cancel" && stop.stop_requested(),
              "Unexpected cancellation");
      paused = true;
    }
    require(
        task.module->is_training() && task.model_digest() == id.model_digest,
        "Evaluation must restore mode and preserve parameters");
    const auto expected_forwards = paused ? 1 : 3 - start.batches;
    const auto expected_calls = paused           ? 1
                                : start.complete ? 0
                                                 : expected_forwards + 1;
    require(task.forwards == expected_forwards &&
                task.source_calls == expected_calls,
            "Resume/replay performed unexpected inference or source work");
    if (paused)
      require(session->progress().batches == 1 && !session->progress().complete,
              "Pause/cancel must retain one committed batch");
    std::cout << " mode=" << mode << " forwards=" << task.forwards
              << " source_calls=" << task.source_calls
              << " committed_batches=" << session->progress().batches
              << " complete="
              << (session->progress().complete ? "true" : "false")
              << " checked=true\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
