// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Test-only native training/publication boundary. The Python checker may
// corrupt the disposable STORE after .ready and before releasing --gate.
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <chrono>
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
#include <thread>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace experiment = forge::experiment;
namespace fs = std::filesystem;
using Json = nlohmann::json;

std::string Utf8(const fs::path& path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::span<const std::byte> Bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

std::string Environment(const char* name) {
#if defined(_WIN32)
  const std::string key(name);
  const std::wstring wide_key(key.begin(), key.end());
  const auto* value = _wgetenv(wide_key.c_str());
  if (!value || !*value)
    throw std::runtime_error("Missing native fixture environment");
  return Utf8(fs::path(value));
#else
  const auto* value = std::getenv(name);
  if (!value) throw std::runtime_error("Missing native fixture environment");
  return value;
#endif
}

Json ReadJson(experiment::SQLiteRunStore& store, std::string_view run,
              std::string_view reference) {
  const auto ref = store.find_reference(run, reference);
  if (!ref || ref->artifact.size_bytes > 4096)
    throw std::runtime_error("Missing bounded native fixture input");
  experiment::FileArtifactStore cas(store.artifact_root(),
                                    {.max_artifact_bytes = 4096});
  auto reader = cas.open(ref->artifact.digest);
  if (reader->size_bytes() != ref->artifact.size_bytes)
    throw std::runtime_error("Native fixture input size mismatch");
  std::string content(static_cast<std::size_t>(reader->size_bytes()), '\0');
  std::size_t offset = 0;
  while (offset < content.size()) {
    const auto got = reader->read(
        std::as_writable_bytes(std::span(content.data(), content.size()))
            .subspan(offset));
    if (!got) throw std::runtime_error("Truncated native fixture input");
    offset += got;
  }
  return Json::parse(content);
}

std::uint64_t Now() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

int Train(const fs::path& launch_log, const fs::path& gate, int exit_code) {
  const auto root = fs::u8path(Environment("JYPPX_LIUDING_LOCAL_STORE_ROOT"));
  const auto job = Environment("JYPPX_LIUDING_LOCAL_JOB_ID");
  const auto attempt = Environment("JYPPX_LIUDING_LOCAL_ATTEMPT_ID");
  const auto spec_sha = Environment("JYPPX_LIUDING_LOCAL_SPEC_SHA256");
  if (!launch_log.is_absolute() || (!gate.empty() && !gate.is_absolute()))
    throw std::runtime_error("Native fixture paths must be absolute");
  std::ofstream launches(launch_log, std::ios::binary | std::ios::app);
  launches << attempt << '\n';
  launches.close();
  if (!launches)
    throw std::runtime_error("Cannot record native fixture launch");

  experiment::SQLiteRunStore store(root);
  const auto metadata = store.find(attempt);
  if (!metadata || metadata->status != experiment::RunStatus::running ||
      metadata->lineage.parent_run_id != job ||
      metadata->lineage.config_digest != spec_sha)
    throw std::runtime_error(
        "Native fixture requires its running frozen attempt");
  const auto config = ReadJson(store, job, "local.training.configuration");
  const auto data = ReadJson(store, job, "local.training.data_manifest");
  const auto rate = config.at("learning_rate").get<double>();
  const auto steps = config.at("total_steps").get<std::uint64_t>();
  if (rate != 0.05 || steps != 60 || data.at("slope") != 3.0 ||
      data.at("intercept") != 0.5)
    throw std::runtime_error(
        "Native fixture requires the affine training recipe");
  const auto workspace = root / "local-output-fixture" / attempt;
  fs::create_directories(workspace.parent_path());
  if (!fs::create_directory(workspace))
    throw std::runtime_error("Native fixture must not launch an attempt twice");
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
  const auto x = torch::linspace(-1.0, 1.0, 32).reshape({32, 1});
  const auto y = x * 3.0 + 0.5;
  for (std::uint64_t i = 0; i < steps; ++i) {
    optimizer->zero_grad();
    torch::mse_loss(model->forward(x), y).backward();
    optimizer->step();
  }
  forge::training::CheckpointParticipantRegistry participants;
  participants.register_participant(
      std::make_shared<forge::nn::ModelCheckpointParticipant>(
          std::make_shared<forge::nn::ModelStateView>(model),
          forge::nn::ModelCheckpointParticipantOptions{
              .max_tensors = 2,
              .max_topology_bytes = 4096,
              .max_tensor_bytes = 64,
              .max_total_bytes = 8192}));
  participants.register_participant(
      std::make_shared<forge::optim::SgdOptimizerCheckpointParticipant>(
          optimizer, forge::optim::SgdOptimizerCheckpointOptions{
                         .max_groups = 1,
                         .max_parameters = 2,
                         .max_topology_bytes = 4096,
                         .max_producer_bytes = 4096,
                         .max_control_bytes = 4096,
                         .max_tensor_bytes = 64,
                         .max_total_bytes = 16384}));
  forge::training::CheckpointCoordinator coordinator(
      participants, {.max_participants = 2, .max_rollback_bytes = 65536});
  const auto checkpoint = workspace / "checkpoint-last.jyck";
  {
    forge::training::FileCheckpointStore sink(
        checkpoint, {.max_artifact_bytes = 65536, .max_field_bytes = 16384});
    coordinator.save(sink);
  }
  const auto stamp = Now();
  const auto checkpoint_size = fs::file_size(checkpoint);
  if (checkpoint_size > 65536)
    throw std::runtime_error("Native fixture checkpoint exceeds its bound");
  std::ifstream checkpoint_stream(checkpoint, std::ios::binary);
  std::string checkpoint_bytes(static_cast<std::size_t>(checkpoint_size), '\0');
  checkpoint_stream.read(checkpoint_bytes.data(),
                         static_cast<std::streamsize>(checkpoint_bytes.size()));
  if (!checkpoint_stream)
    throw std::runtime_error("Cannot read native fixture checkpoint");
  const auto published = store.put_artifact(
      attempt, "checkpoint.last", experiment::RunArtifactKind::checkpoint,
      "last", Bytes(checkpoint_bytes),
      {.logical_name = "checkpoint-last.jyck",
       .media_type = "application/octet-stream",
       .producer = "liuding-forge-test-local-output",
       .expected_size_bytes = checkpoint_size},
      stamp);
  const torch::NoGradGuard guard;
  const auto validation_x = torch::linspace(-0.9, 0.9, 10).reshape({10, 1});
  const auto mse =
      torch::mse_loss(model->forward(validation_x), validation_x * 3.0 + 0.5)
          .item<double>();
  const Json result{{"completed_steps", optimizer->completed_steps()},
                    {"validation_mse", mse},
                    {"weight", model->weight.item<double>()},
                    {"bias", model->bias.item<double>()},
                    {"checkpoint_sha256", published.artifact.digest.hex}};
  const auto encoded = result.dump();
  store.put_artifact(attempt, "training.result",
                     experiment::RunArtifactKind::artifact, "training.result",
                     Bytes(encoded),
                     {.logical_name = "training-result.json",
                      .media_type = "application/json",
                      .producer = "liuding-forge-test-local-output",
                      .expected_size_bytes = encoded.size()},
                     stamp);
  if (!gate.empty()) {
    auto ready = gate;
    ready += ".ready";
    std::ofstream signal(ready, std::ios::binary);
    signal << attempt << '\n';
    signal.close();
    if (!signal)
      throw std::runtime_error("Cannot publish native fixture readiness");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!fs::is_regular_file(gate)) {
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("Native output publication gate timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  std::cout << "Native SGD completed and checkpoint/summary published.\n";
  return exit_code;
}
}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    std::map<std::string, std::string> arguments;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc ||
          !arguments
               .emplace(Utf8(fs::path(argv[i])), Utf8(fs::path(argv[i + 1])))
               .second)
        throw std::runtime_error(
            "Expected distinct fixture option/value pairs");
    }
    if (arguments.size() != 1U + (arguments.contains("--gate") ? 1U : 0U) +
                                (arguments.contains("--exit-code") ? 1U : 0U) ||
        (arguments.contains("--exit-code") &&
         arguments.at("--exit-code") != "23"))
      throw std::runtime_error(
          "Use --launch-log FILE [--gate FILE] [--exit-code 23]");
    return Train(fs::u8path(arguments.at("--launch-log")),
                 arguments.contains("--gate")
                     ? fs::u8path(arguments.at("--gate"))
                     : fs::path{},
                 arguments.contains("--exit-code") ? 23 : 0);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
