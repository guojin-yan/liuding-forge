// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace exporter = jyppx::liuding::onnx_export;
using Json = nlohmann::json;

exporter::ExportResult Artifact() {
  torch::nn::Linear model(3, 2);
  model->eval();
  const torch::NoGradGuard guard;
  model->weight.copy_(at::arange(6, at::kFloat).reshape({2, 3}) / 7.0F);
  model->bias.copy_(at::tensor({0.25F, -0.5F}));
  exporter::ExportOptions options;
  options.maximum_batch = 8;
  return exporter::export_model(model, at::ones({2, 3}), options);
}

TEST(NativeOnnxReport,
     CapturesExactBytesStateAndResolvedAxesWithoutRuntimeClaims) {
  const auto artifact = Artifact();
  ASSERT_TRUE(artifact.report);
  const auto& report = *artifact.report;
  EXPECT_EQ(report.model_digest, artifact.digest);
  EXPECT_EQ(report.model_size_bytes, artifact.onnx.size());
  ASSERT_EQ(report.inputs.size(), 1U);
  ASSERT_EQ(report.outputs.size(), 1U);
  ASSERT_EQ(report.state.size(), 2U);
  EXPECT_EQ(report.inputs[0].shape, (std::vector<std::int64_t>{2, 3}));
  EXPECT_EQ(report.inputs[0].size_bytes, 24U);
  const auto input = at::ones({2, 3});
  EXPECT_EQ(
      report.inputs[0].digest.hex,
      sha256_hex(std::as_bytes(std::span(input.const_data_ptr<float>(), 6U))));
  EXPECT_EQ(report.dynamic_axes.at("input").at(0), "batch");
  EXPECT_EQ(report.dynamic_axes.at("output").at(0), "batch");
  EXPECT_EQ(report.state[0].name, "state.bias");
  EXPECT_EQ(report.state[0].role, "parameter");
  EXPECT_EQ(report.state[1].name, "state.weight");
  const auto encoded = exporter::serialize_export_report(report);
  const auto document = Json::parse(encoded);
  EXPECT_EQ(document["verification"]["official_checker"], "passed");
  EXPECT_EQ(document["verification"]["runtime"], "not_checked");
  EXPECT_EQ(document["verification"]["numerical"], "not_checked");
  EXPECT_FALSE(document["verification"]["source_authenticated"].get<bool>());
  EXPECT_EQ(exporter::serialize_export_report(
                exporter::deserialize_export_report(encoded)),
            encoded);
  EXPECT_NO_THROW(exporter::verify_export_report_model(report, artifact.onnx));
  if (const auto* directory = std::getenv("JYPPX_LIUDING_ONNX_REPORT_FIXTURE");
      directory != nullptr && *directory != '\0') {
    auto fixture = report;
    fixture.source = {.run_id = "training-source",
                      .source_digest = std::string(64, 'a'),
                      .config_digest = std::string(64, 'b'),
                      .data_manifest_digest = std::string(64, 'c')};
    const auto root = std::filesystem::u8path(directory);
    std::filesystem::create_directories(root);
    std::ofstream model_file(root / "model.onnx", std::ios::binary);
    model_file.write(reinterpret_cast<const char*>(artifact.onnx.data()),
                     static_cast<std::streamsize>(artifact.onnx.size()));
    std::ofstream report_file(root / "report.json", std::ios::binary);
    report_file << exporter::serialize_export_report(fixture);
    ASSERT_TRUE(model_file && report_file);
  }
}

TEST(NativeOnnxReport, RejectsTamperingUnknownFieldsFalseClaimsAndCoercions) {
  const auto artifact = Artifact();
  auto corrupted = artifact.onnx;
  corrupted.back() ^= std::byte{1};
  EXPECT_THROW(
      exporter::verify_export_report_model(*artifact.report, corrupted),
      SerializationError);
  auto forged = *artifact.report;
  const std::vector<std::byte> opaque{std::byte{'b'}, std::byte{'a'},
                                      std::byte{'d'}};
  forged.model_digest.hex = sha256_hex(opaque);
  forged.model_size_bytes = opaque.size();
  EXPECT_THROW(exporter::verify_export_report_model(forged, opaque),
               CompatibilityError);
  const auto document =
      Json::parse(exporter::serialize_export_report(*artifact.report));
  for (const auto& field : {"unknown", "false_runtime", "dtype", "size",
                            "shape", "digest", "axis", "coercion"}) {
    auto changed = document;
    if (std::string(field) == "unknown") changed["build"]["secret"] = "bad";
    if (std::string(field) == "false_runtime")
      changed["verification"]["runtime"] = "passed";
    if (std::string(field) == "dtype")
      changed["inputs"][0]["dtype"] = "float128";
    if (std::string(field) == "size") changed["inputs"][0]["size_bytes"] = 1;
    if (std::string(field) == "shape") changed["inputs"][0]["shape"][0] = 1.5;
    if (std::string(field) == "digest")
      changed["model"]["digest"]["hex"] = "ab";
    if (std::string(field) == "axis") changed["dynamic_axes"][0]["axis"] = 1;
    if (std::string(field) == "coercion") changed["opset"] = 17.0;
    EXPECT_THROW(exporter::deserialize_export_report(changed.dump()), Error)
        << field;
  }
  auto version = document;
  version["schema_version"] = 2;
  EXPECT_THROW(exporter::deserialize_export_report(version.dump()),
               CompatibilityError);
  const auto duplicate =
      std::string("{\"format\":\"duplicate\",") + document.dump().substr(1);
  EXPECT_THROW(exporter::deserialize_export_report(duplicate),
               SerializationError);
  EXPECT_THROW(exporter::serialize_export_report(*artifact.report, 16U),
               ResourceExhausted);
  EXPECT_THROW(exporter::deserialize_export_report(document.dump(), 16U),
               ResourceExhausted);
}

TEST(NativeOnnxReport, RecordsCallerLineageAndRejectsInvalidSourceDigests) {
  auto artifact = Artifact();
  artifact.report->source = {.run_id = "trained-run",
                             .source_digest = std::string(64, 'a'),
                             .config_digest = std::string(64, 'b'),
                             .data_manifest_digest = std::string(64, 'c')};
  const auto restored = exporter::deserialize_export_report(
      exporter::serialize_export_report(*artifact.report));
  EXPECT_EQ(restored.source.run_id, "trained-run");
  EXPECT_EQ(restored.source.source_digest, std::string(64, 'a'));
  artifact.report->source.config_digest = "short";
  EXPECT_THROW(exporter::serialize_export_report(*artifact.report),
               InvalidArgument);
}

TEST(NativeOnnxReport, MultiInputSnapshotsIncludeBuffersAndResolvedSymbols) {
  torch::nn::Module module;
  const auto gain =
      module.register_buffer("gain", at::tensor({0.5F, 1.0F, 2.0F}));
  const std::vector<at::Tensor> inputs{at::ones({2, 3}), at::ones({2, 3})};
  exporter::ExportOptions options;
  options.dynamic_axes = {{"input_0", {{0, "left"}}},
                          {"input_1", {{0, "right"}}},
                          {"output_0", {{0, "left"}}},
                          {"output_1", {{0, "right"}}}};
  const auto artifact = exporter::export_module(
      module,
      [&](auto values) {
        return std::vector<at::Tensor>{values[0] * gain, values[1] + gain};
      },
      inputs, options);
  const auto report = exporter::deserialize_export_report(
      exporter::serialize_export_report(*artifact.report));
  EXPECT_EQ(report.inputs.size(), 2U);
  EXPECT_EQ(report.outputs.size(), 2U);
  ASSERT_EQ(report.state.size(), 1U);
  EXPECT_EQ(report.state[0].role, "buffer");
  EXPECT_EQ(report.dynamic_axes.at("output_1").at(0), "right");
}
}  // namespace
