// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#if defined(JYPPX_LIUDING_TEST_ONNX_MODELS)
#include <jyppx/liuding/models/registry.hpp>
#endif
#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {
using namespace jyppx::liuding;
using Json = nlohmann::json;

TEST(NativeOnnxSource, ProducesCheckpointAndDefinitionReferences) {
  const auto* directory = std::getenv("JYPPX_LIUDING_ONNX_SOURCE_FIXTURE");
  const auto root = directory
                        ? std::filesystem::u8path(directory)
                        : std::filesystem::temp_directory_path() /
                              ("liuding-source-fixture-" +
                               std::to_string(std::chrono::steady_clock::now()
                                                  .time_since_epoch()
                                                  .count()));
  std::filesystem::create_directories(root);
  {
    experiment::SQLiteRunStore store(root / "run-store");
    store.start({.run_id = "training-source",
                 .project_id = "native",
                 .experiment_id = "export-source",
                 .display_name = "Native source fixture",
                 .status = experiment::RunStatus::running,
                 .created_at_unix_ms = 100,
                 .updated_at_unix_ms = 100,
                 .lineage = {.parent_run_id = {},
                             .source_digest = std::string(64, 'a'),
                             .build_fingerprint = {},
                             .config_digest = std::string(64, 'b'),
                             .data_manifest_digest = std::string(64, 'c')},
                 .tags = {}});
    auto attach = [&](std::string id, std::string role,
                      std::span<const std::byte> bytes,
                      experiment::RunArtifactKind kind =
                          experiment::RunArtifactKind::artifact) {
      return store.put_artifact("training-source", std::move(id), kind,
                                std::move(role), bytes,
                                {.logical_name = "source",
                                 .media_type = "application/octet-stream",
                                 .producer = "independent.native.fixture",
                                 .expected_size_bytes = bytes.size()},
                                100);
    };
    Json summary;
    auto save = [&](const std::string& name, nn::ModelAdapter model,
                    Json factory, const Tensor& input) {
      model.state().eval();
      nn::ModelCheckpointParticipant participant(model);
      const auto path = root / (name + ".jyck");
      {
        interop::FileStateArchive archive(path);
        archive.begin({participant.describe_state()});
        participant.save_state(
            archive.writer_for(participant.participant_id()));
        archive.commit();
      }
      Sha256 hash;
      std::ifstream checkpoint_file(path, std::ios::binary);
      std::array<char, 65536> block{};
      while (checkpoint_file) {
        checkpoint_file.read(block.data(),
                             static_cast<std::streamsize>(block.size()));
        hash.update(std::as_bytes(std::span(
            block.data(), static_cast<std::size_t>(checkpoint_file.gcount()))));
      }
      const auto ref = store.put_artifact_file(
          "training-source", name + ".checkpoint",
          experiment::RunArtifactKind::checkpoint, "final", path,
          {.logical_name = name + ".jyck",
           .media_type = "application/x-jyppx-checkpoint",
           .producer = "independent.native.fixture",
           .expected_size_bytes = {}},
          100, {.hex = sha256_digest_hex(hash.finish())});
      Json definition{{"format", "jyppx.liuding.onnx_model_definition"},
                      {"schema_version", 1},
                      {"participant_id", "model.main"},
                      {"factory", std::move(factory)}};
      const auto text = definition.dump();
      attach(name + ".definition", "model.definition",
             std::as_bytes(std::span(text.data(), text.size())));
      Json invalid = definition;
      invalid["participant_id"] = "missing.model";
      const auto invalid_text = invalid.dump();
      attach(
          name + ".missing-participant", "model.definition",
          std::as_bytes(std::span(invalid_text.data(), invalid_text.size())));
      invalid = definition;
      invalid["factory"]["future"] = 1;
      const auto unknown = invalid.dump();
      attach(name + ".unknown-field", "model.definition",
             std::as_bytes(std::span(unknown.data(), unknown.size())));
      attach(name + ".wrong-role", "generic",
             std::as_bytes(std::span(text.data(), text.size())));
      const auto output = model.invoke({.positional = {input}, .keyword = {}})
                              .value.toTensor()
                              .contiguous();
      std::ofstream oracle(root / (name + "-output.bin"), std::ios::binary);
      oracle.write(static_cast<const char*>(output.const_data_ptr()),
                   static_cast<std::streamsize>(output.nbytes()));
      summary[name] = {{"checkpoint", ref.artifact.digest.hex},
                       {"size_bytes", ref.artifact.size_bytes},
                       {"output_shape", output.sizes().vec()}};
    };
    torch::nn::Linear linear(3, 2);
    const torch::NoGradGuard guard;
    linear->weight.copy_(at::arange(6, at::kFloat).reshape({2, 3}) / 7.0F);
    linear->bias.copy_(at::tensor({0.25F, -0.5F}));
    save("linear", nn::adapt_tensor_holder(linear),
         {{"id", "torch.nn.Linear"},
          {"in_features", 3},
          {"out_features", 2},
          {"bias", true}},
         at::ones({2, 3}));
    torch::nn::Linear wide(3, 131072);
    wide->weight.zero_();
    wide->bias.zero_();
    save("wide", nn::adapt_tensor_holder(wide),
         {{"id", "torch.nn.Linear"},
          {"in_features", 3},
          {"out_features", 131072},
          {"bias", true}},
         at::ones({2, 3}));
#if defined(JYPPX_LIUDING_TEST_ONNX_MODELS)
    models::ModelOptions settings;
    settings.resnet.num_classes = 2;
    torch::manual_seed(12345);
    save("resnet", models::create_model("resnet18", settings),
         {{"id", "jyppx.liuding.models"},
          {"name", "resnet18"},
          {"num_classes", 2},
          {"input_channels", 3}},
         at::ones({2, 3, 32, 32}));
#endif
    const auto bad = std::string("not a native checkpoint");
    attach("invalid.checkpoint", "final",
           std::as_bytes(std::span(bad.data(), bad.size())),
           experiment::RunArtifactKind::checkpoint);
    (void)store.transition("training-source", experiment::RunStatus::completed,
                           200);
    std::ofstream summary_file(root / "source-summary.json");
    summary_file << summary.dump();
    ASSERT_TRUE(summary_file);
    ASSERT_TRUE(store.integrity_check().ok());
  }
  if (!directory) std::filesystem::remove_all(root);
}
}  // namespace
