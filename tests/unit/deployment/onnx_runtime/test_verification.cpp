// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>

#include <torch/torch.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

namespace {
namespace exporter = jyppx::liuding::onnx_export;
namespace runtime = jyppx::liuding::onnx_runtime;

struct LinearCase {
  exporter::ExportResult artifact;
  std::vector<float> input_values;
  std::vector<std::int64_t> input_shape;
  std::vector<float> expected_values;
  std::vector<std::int64_t> expected_shape;
};

LinearCase MakeCase() {
  torch::nn::Linear model(3, 2);
  model->eval();
  const torch::NoGradGuard guard;
  model->weight.copy_(
      at::tensor(std::vector<float>{1.0F, 0.0F, -1.0F, 0.5F, 2.0F, 1.0F})
          .reshape({2, 3}));
  model->bias.copy_(at::tensor({0.25F, -0.5F}));
  auto input =
      at::tensor(std::vector<float>{1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F})
          .reshape({2, 3});
  auto expected = model->forward(input).contiguous();
  exporter::ExportOptions options;
  options.input_names = {"features"};
  options.output_names = {"logits"};
  auto artifact = exporter::export_model(model, input, options);
  return {.artifact = std::move(artifact),
          .input_values = std::vector<float>(
              input.data_ptr<float>(), input.data_ptr<float>() + input.numel()),
          .input_shape = {input.sizes().begin(), input.sizes().end()},
          .expected_values =
              std::vector<float>(expected.data_ptr<float>(),
                                 expected.data_ptr<float>() + expected.numel()),
          .expected_shape = {expected.sizes().begin(), expected.sizes().end()}};
}

TEST(OnnxRuntimeVerification,
     ComparesAllOutputsAndSerializesActualProviderEvidence) {
  auto fixture = MakeCase();
  runtime::SessionOptions options;
  options.provider.name = "CPUExecutionProvider";
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest,
                           options);
  runtime::TensorInput input{.name = "features",
                             .values = fixture.input_values,
                             .shape = fixture.input_shape};
  runtime::TensorInput expected{.name = "logits",
                                .values = fixture.expected_values,
                                .shape = fixture.expected_shape};
  const std::array<runtime::TensorInput, 1> inputs{input};
  const std::array<runtime::TensorInput, 1> outputs{expected};
  const std::array<runtime::VerificationCase, 1> cases{
      {{.id = "linear-batch-2", .inputs = inputs, .expected = outputs}}};
  runtime::VerificationOptions policy;
  policy.reference = "native-model-forward-v1";
  policy.absolute_tolerance = 1e-6;
  policy.relative_tolerance = 1e-5;
  const auto report = runtime::verify_session(session, cases, policy);
  EXPECT_TRUE(runtime::verification_passed(report));
  ASSERT_EQ(report.cases.size(), 1U);
  ASSERT_EQ(report.cases.front().outputs.size(), 1U);
  EXPECT_EQ(report.cases.front().outputs.front().mismatched_elements, 0U);
  const auto json =
      nlohmann::json::parse(runtime::serialize_verification_report(report));
  EXPECT_EQ(json.at("verification").at("runtime_executed"), true);
  EXPECT_EQ(json.at("verification").at("numerical"), "passed");
  EXPECT_EQ(json.at("runtime").at("requested_provider"),
            "CPUExecutionProvider");
  EXPECT_FALSE(json.at("verification").at("requested_provider_observed"));
  EXPECT_EQ(json.at("verification").at("task_quality"), "not_checked");
}

TEST(OnnxRuntimeVerification,
     ReportsNumericalMismatchWithoutClaimingTaskQuality) {
  auto fixture = MakeCase();
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest);
  fixture.expected_values.front() += 1.0F;
  runtime::TensorInput input{.name = "features",
                             .values = fixture.input_values,
                             .shape = fixture.input_shape};
  runtime::TensorInput expected{.name = "logits",
                                .values = fixture.expected_values,
                                .shape = fixture.expected_shape};
  const std::array<runtime::TensorInput, 1> inputs{input};
  const std::array<runtime::TensorInput, 1> outputs{expected};
  const std::array<runtime::VerificationCase, 1> cases{
      {{.id = "tampered-reference", .inputs = inputs, .expected = outputs}}};
  const auto report = runtime::verify_session(
      session, cases, {.reference = "independent-oracle"});
  EXPECT_FALSE(runtime::verification_passed(report));
  EXPECT_GT(report.cases.front().outputs.front().mismatched_elements, 0U);
  const auto json =
      nlohmann::json::parse(runtime::serialize_verification_report(report));
  EXPECT_EQ(json.at("verification").at("numerical"), "failed");
  EXPECT_EQ(json.at("verification").at("task_quality"), "not_checked");
  EXPECT_FALSE(json.at("verification").at("reference_authenticated"));
}

TEST(OnnxRuntimeVerification, ParsesTraceProducedByTheSameSessionAndProvider) {
  auto fixture = MakeCase();
  runtime::SessionOptions options;
  options.profile_prefix =
      std::filesystem::temp_directory_path() / "liuding-onnx-verification";
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest,
                           options);
  runtime::TensorInput input{.name = "features",
                             .values = fixture.input_values,
                             .shape = fixture.input_shape};
  runtime::TensorInput expected{.name = "logits",
                                .values = fixture.expected_values,
                                .shape = fixture.expected_shape};
  const std::array<runtime::TensorInput, 1> inputs{input};
  const std::array<runtime::TensorInput, 1> outputs{expected};
  const std::array<runtime::VerificationCase, 1> cases{
      {{.id = "profiled-linear", .inputs = inputs, .expected = outputs}}};
  const auto report = runtime::verify_session(
      session, cases, {.reference = "native-model-forward-v1"});
  ASSERT_TRUE(runtime::verification_passed(report));
  const auto profile_path = session.end_profiling();
  std::ifstream stream(profile_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(stream);
  const auto size = stream.tellg();
  ASSERT_GT(size, 0);
  std::string profile(static_cast<std::size_t>(size), '\0');
  stream.seekg(0);
  stream.read(profile.data(), size);
  ASSERT_TRUE(stream);
  const auto observation = runtime::inspect_profile(profile);
  EXPECT_EQ(observation.kernels.at("CPUExecutionProvider").at("Gemm"), 1U);
  auto profiled = report;
  profiled.profile = observation;
  const auto encoded =
      nlohmann::json::parse(runtime::serialize_verification_report(profiled));
  EXPECT_EQ(encoded.at("profile")
                .at("kernel_events")
                .at("CPUExecutionProvider")
                .at("Gemm"),
            1U);
  EXPECT_TRUE(encoded.at("verification").at("requested_provider_observed"));
  std::error_code error;
  std::filesystem::remove(profile_path, error);
}

TEST(OnnxRuntimeVerification, RejectsInvalidOrUnboundedProfileData) {
  EXPECT_THROW(runtime::inspect_profile("[]"), jyppx::liuding::Error);
  const auto profile =
      R"([{"cat":"Node","ph":"X","name":"node_Gemm_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"Gemm"}}])";
  const auto observation = runtime::inspect_profile(profile);
  EXPECT_EQ(observation.kernels.at("CPUExecutionProvider").at("Gemm"), 1U);
  EXPECT_THROW(runtime::inspect_profile(profile, 4U),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(
      runtime::inspect_profile(
          R"([{"cat":"Node","ph":"X","name":"kernel_time","args":{"provider":"CPUExecutionProvider","provider":"CUDAExecutionProvider","op_name":"Gemm"}}])"),
      jyppx::liuding::SerializationError);
}

TEST(OnnxRuntimeVerification,
     PreflightsCompleteCasesBoundsNamesAndCancellation) {
  auto fixture = MakeCase();
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest);
  std::array inputs{runtime::TensorInput{.name = "features",
                                         .values = fixture.input_values,
                                         .shape = fixture.input_shape}};
  std::array expected{runtime::TensorInput{.name = "logits",
                                           .values = fixture.expected_values,
                                           .shape = fixture.expected_shape}};
  std::array cases{runtime::VerificationCase{
      .id = "bounded", .inputs = inputs, .expected = expected}};
  runtime::VerificationOptions policy;
  policy.reference = "reference";
  policy.max_case_bytes = 4U;
  EXPECT_THROW(runtime::verify_session(session, cases, policy),
               jyppx::liuding::ResourceExhausted);
  policy.max_case_bytes = 64U << 20U;
  policy.absolute_tolerance = -1.0;
  EXPECT_THROW(runtime::verify_session(session, cases, policy),
               jyppx::liuding::InvalidArgument);
  policy.absolute_tolerance = 1e-6;
  expected[0].name = "unknown";
  EXPECT_THROW(runtime::verify_session(session, cases, policy),
               jyppx::liuding::InvalidArgument);
  expected[0].name = "logits";
  fixture.expected_values[0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(runtime::verify_session(session, cases, policy),
               jyppx::liuding::InvalidArgument);
  fixture.expected_values[0] = 0.0F;
  std::stop_source cancellation;
  cancellation.request_stop();
  EXPECT_THROW(
      runtime::verify_session(session, cases, policy, cancellation.get_token()),
      jyppx::liuding::Cancelled);
  EXPECT_THROW(runtime::verify_session(session, cases, policy, {},
                                       std::chrono::steady_clock::now()),
               jyppx::liuding::DeadlineExceeded);
}
}  // namespace
