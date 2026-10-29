// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/vision.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace jyppx::liuding;
using Json = nlohmann::json;
namespace runtime = jyppx::liuding::onnx_runtime;

std::filesystem::path Root() {
  const auto* root = std::getenv("JYPPX_LIUDING_ONNX_FIXTURE_ROOT");
  if (root == nullptr || *root == '\0')
    throw std::runtime_error(
        "Run this gate through verify_onnx.py; fixtures are required");
  return std::filesystem::u8path(root);
}

std::vector<std::byte> Read(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0U || size > (256U << 20U))
    throw std::runtime_error("ONNX fixture byte count exceeds its bound");
  std::vector<std::byte> data(static_cast<std::size_t>(size));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(size));
  if (!stream) throw std::runtime_error("ONNX fixture read failed");
  return data;
}

std::string Text(const std::filesystem::path& path) {
  auto data = Read(path);
  return {reinterpret_cast<const char*>(data.data()), data.size()};
}

Json Manifest() { return Json::parse(Text(Root() / "manifest.json")); }

ArtifactDigest Digest(const Json& value) {
  if (value.at("algorithm") != "sha256")
    throw std::runtime_error("ONNX fixture digest algorithm mismatch");
  ArtifactDigest digest{.hex = value.at("hex").get<std::string>()};
  validate_artifact_digest(digest);
  return digest;
}

std::vector<std::byte> Artifact(const Json& reference) {
  const auto identity = Digest(reference.at("digest"));
  auto bytes = Read(Root() / "artifacts" / identity.hex);
  if (sha256_hex(bytes) != identity.hex ||
      bytes.size() != reference.at("size_bytes").get<std::size_t>())
    throw std::runtime_error("ONNX fixture immutable identity mismatch");
  return bytes;
}

torch::Tensor Tensor(std::span<const std::byte> bytes, const char* name) {
  const interop::SafeTensorsView view(bytes);
  const auto& entries = view.directory().tensors;
  const auto found =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto& entry) { return entry.name == name; });
  if (found == entries.end() || found->dtype != interop::SafeTensorDtype::f32)
    throw std::runtime_error("ONNX case tensor is not FP32");
  std::vector<std::int64_t> shape(found->shape.begin(), found->shape.end());
  auto result = torch::empty(shape, torch::kFloat32);
  const auto source = view.tensor_bytes(name);
  std::memcpy(result.data_ptr<float>(), source.data(), source.size());
  return result;
}

runtime::TensorOutput RunSession(runtime::Session& session,
                                 const torch::Tensor& input) {
  return session.run(std::span(input.data_ptr<float>(),
                               static_cast<std::size_t>(input.numel())),
                     std::span(input.sizes().data(), input.sizes().size()));
}

void Near(std::span<const float> actual, const torch::Tensor& expected) {
  ASSERT_EQ(actual.size(), static_cast<std::size_t>(expected.numel()));
  const auto* oracle = expected.data_ptr<float>();
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    ASSERT_TRUE(std::isfinite(actual[index]));
    ASSERT_LE(std::abs(actual[index] - oracle[index]),
              1e-5F + 1e-4F * std::abs(oracle[index]))
        << "element=" << index;
  }
}

template <typename Holder>
void CheckSameState(const char* name, Holder holder) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  holder->eval();
  auto model = nn::adapt_tensor_holder(holder);
  auto& state = model.state();
  const auto specification = Manifest().at("models").at(name);
  const auto source = Artifact(specification.at("state"));
  const auto imported = interop::import_model_weights(
      state, source, Digest(specification.at("state").at("digest")));
  ASSERT_TRUE(imported.applied);
  ASSERT_TRUE(imported.plan.ready);
  ASSERT_DOUBLE_EQ(imported.plan.tensor_coverage, 1.0);
  const auto encoded = Artifact(specification.at("artifact"));
  runtime::Session session(encoded,
                           Digest(specification.at("artifact").at("digest")));
  ASSERT_EQ(session.runtime_version(), "1.23.2");
  ASSERT_EQ(session.input().batch_symbol, session.output().batch_symbol);
  ASSERT_FALSE(session.input().batch_symbol.empty());
  std::vector<std::int64_t> batches;
  for (const auto& ref : specification.at("cases")) {
    const auto bytes = Artifact(ref);
    auto input = Tensor(bytes, "input");
    auto expected = Tensor(bytes, "output");
    auto native = holder->forward(input).contiguous();
    Near(std::span(native.template data_ptr<float>(),
                   static_cast<std::size_t>(native.numel())),
         expected);
    auto actual = RunSession(session, input);
    ASSERT_EQ(actual.shape, std::vector<std::int64_t>(expected.sizes().begin(),
                                                      expected.sizes().end()));
    Near(actual.values, expected);
    Near(actual.values, native);
    batches.push_back(input.size(0));
  }
  ASSERT_EQ(batches, (std::vector<std::int64_t>{2, 5}));
}

struct LinearFixture {
  Json specification = Manifest().at("models").at("linear");
  std::vector<std::byte> bytes = Artifact(specification.at("artifact"));
  ArtifactDigest identity = Digest(specification.at("artifact").at("digest"));
  torch::Tensor input =
      Tensor(Artifact(specification.at("cases").at(0)), "input");
};

TEST(OnnxRuntimeInterchange, LinearSameStateNativeAndOfficialOutputs) {
  CheckSameState("linear", torch::nn::Linear(8, 3));
}

TEST(OnnxRuntimeInterchange, ResNet18SameStateNativeAndOfficialOutputs) {
  CheckSameState("resnet18", models::resnet18({.num_classes = 7}));
}

TEST(OnnxRuntimeSession, RejectsUntrustedDigestAndOversizedModelBeforeParsing) {
  LinearFixture fixture;
  auto wrong = fixture.identity;
  wrong.hex.front() = wrong.hex.front() == 'a' ? 'b' : 'a';
  EXPECT_THROW(runtime::Session(fixture.bytes, wrong), CompatibilityError);
  runtime::SessionOptions options;
  options.max_model_bytes = fixture.bytes.size() - 1U;
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               ResourceExhausted);
  EXPECT_THROW(runtime::Session({}, fixture.identity), InvalidArgument);
}

TEST(OnnxRuntimeSession, RejectsMalformedBytesAndMissingOperators) {
  const auto manifest = Manifest();
  for (const auto name : {"malformed", "missing_operator", "external_data"}) {
    const auto ref = manifest.at("negative").at(name);
    const auto bytes = Artifact(ref);
    EXPECT_THROW(runtime::Session(bytes, Digest(ref.at("digest"))),
                 CompatibilityError)
        << name;
  }
}

TEST(OnnxRuntimeSession, RejectsUnsupportedDtypeArityAndNonbatchDynamicAxes) {
  const auto manifest = Manifest();
  for (const auto name : {"float64", "two_inputs", "dynamic_width"}) {
    const auto ref = manifest.at("negative").at(name);
    const auto bytes = Artifact(ref);
    EXPECT_THROW(runtime::Session(bytes, Digest(ref.at("digest"))),
                 CompatibilityError)
        << name;
  }
}

TEST(OnnxRuntimeSession, RejectsInvalidOptions) {
  LinearFixture fixture;
  for (const auto threads : {0U, 65U}) {
    runtime::SessionOptions options;
    options.cpu_threads = threads;
    EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
                 InvalidArgument);
  }
  runtime::SessionOptions options;
  options.max_output_bytes = 0U;
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               InvalidArgument);
}

TEST(OnnxRuntimeSession,
     ReportsActualProviderInventoryWithoutClaimingExecution) {
  const auto providers = runtime::available_providers();
  EXPECT_NE(
      std::find(providers.begin(), providers.end(), "CPUExecutionProvider"),
      providers.end());
  EXPECT_TRUE(runtime::provider_append_supported("CUDAExecutionProvider"));
  EXPECT_TRUE(runtime::provider_append_supported("TensorrtExecutionProvider"));
  EXPECT_TRUE(runtime::provider_append_supported("OpenVINOExecutionProvider"));
  EXPECT_TRUE(runtime::provider_append_supported("CoreMLExecutionProvider"));
  EXPECT_FALSE(runtime::provider_append_supported("DmlExecutionProvider"));
  EXPECT_FALSE(runtime::provider_append_supported("UnknownExecutionProvider"));
  EXPECT_FALSE(runtime::provider_append_supported("cpu"));
  LinearFixture fixture;
  runtime::Session session(fixture.bytes, fixture.identity);
  EXPECT_EQ(session.execution_provider().name, "CPUExecutionProvider");
  EXPECT_FALSE(session.cpu_fallback_allowed());
}

TEST(OnnxRuntimeSession, RejectsMissingOrUnwiredProvidersWithoutCpuFallback) {
  LinearFixture fixture;
  runtime::SessionOptions options;
  const auto providers = runtime::available_providers();
  bool observed_missing = false;
  for (const auto* name :
       {"CUDAExecutionProvider", "TensorrtExecutionProvider",
        "OpenVINOExecutionProvider", "CoreMLExecutionProvider"}) {
    if (std::find(providers.begin(), providers.end(), name) != providers.end())
      continue;
    observed_missing = true;
    options.provider.name = name;
    EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
                 CompatibilityError);
    options.allow_cpu_fallback = true;
    EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
                 CompatibilityError);
    options.allow_cpu_fallback = false;
  }
  EXPECT_TRUE(observed_missing);
  options.provider.name = "DmlExecutionProvider";
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               CompatibilityError);
  options.provider.name = "UnknownExecutionProvider";
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               CompatibilityError);
  options.provider.name = "CPUExecutionProvider";
  options.provider.options = {{"unapplied_option", "1"}};
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               InvalidArgument);
  options.provider.options = {{"bad", std::string("value\0tail", 10U)}};
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               InvalidArgument);
}

TEST(OnnxRuntimeSession, RejectsRankShapeElementCountAndNonfiniteInput) {
  LinearFixture fixture;
  runtime::Session session(fixture.bytes, fixture.identity);
  const std::array<std::int64_t, 2> correct{2, 8};
  const std::array<float, 16> values{};
  const std::array<std::int64_t, 2> wrong_width{2, 9};
  const std::array<std::int64_t, 2> zero_batch{0, 8};
  const std::array<std::int64_t, 1> wrong_rank{16};
  EXPECT_THROW(session.run(values, wrong_width), InvalidArgument);
  EXPECT_THROW(session.run(values, zero_batch), InvalidArgument);
  EXPECT_THROW(session.run(values, wrong_rank), InvalidArgument);
  EXPECT_THROW(session.run(std::span(values).first(15U), correct),
               InvalidArgument);
  auto bad = values;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(session.run(bad, correct), InvalidArgument);
}

TEST(OnnxRuntimeSession, EnforcesBatchAndInputOutputAllocationBounds) {
  LinearFixture fixture;
  runtime::SessionOptions options;
  options.max_batch_size = 1U;
  runtime::Session bounded_batch(fixture.bytes, fixture.identity, options);
  EXPECT_THROW(RunSession(bounded_batch, fixture.input), ResourceExhausted);
  options = {};
  options.max_input_bytes = 8U * sizeof(float);
  runtime::Session bounded_input(fixture.bytes, fixture.identity, options);
  EXPECT_THROW(RunSession(bounded_input, fixture.input), ResourceExhausted);
  options = {};
  options.max_output_bytes = 3U * sizeof(float);
  runtime::Session bounded_output(fixture.bytes, fixture.identity, options);
  EXPECT_THROW(RunSession(bounded_output, fixture.input), ResourceExhausted);
  options.max_output_bytes = sizeof(float);
  EXPECT_THROW(runtime::Session(fixture.bytes, fixture.identity, options),
               ResourceExhausted);
}

TEST(OnnxRuntimeSession, RejectsNonfiniteGraphOutput) {
  const auto ref = Manifest().at("negative").at("nonfinite_output");
  const auto bytes = Artifact(ref);
  runtime::Session session(bytes, Digest(ref.at("digest")));
  const std::array<float, 6> values{1, 2, 3, 4, 5, 6};
  const std::array<std::int64_t, 2> shape{2, 3};
  EXPECT_THROW(session.run(values, shape), NumericalError);
}

TEST(OnnxRuntimeSession, HonorsPrecancelledAndExpiredCallsThenRemainsUsable) {
  LinearFixture fixture;
  runtime::Session session(fixture.bytes, fixture.identity);
  const std::array<float, 16> values{};
  const std::array<std::int64_t, 2> shape{2, 8};
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(session.run(values, shape, stop.get_token()), Cancelled);
  EXPECT_THROW(session.run(values, shape, {}, std::chrono::steady_clock::now()),
               DeadlineExceeded);
  EXPECT_EQ(session.run(values, shape).values.size(), 6U);
}

TEST(OnnxRuntimeSession, InterruptsRealExecutionWithoutPoisoningTheSession) {
  const auto ref = Manifest().at("slow");
  const auto bytes = Artifact(ref);
  runtime::SessionOptions options;
  options.disable_graph_optimizations = true;
  runtime::Session session(bytes, Digest(ref.at("digest")), options);
  const std::array<std::int64_t, 2> shape{128, 512};
  std::vector<float> values(128U * 512U, 1.0F);
  std::stop_source stop;
  std::jthread request([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    stop.request_stop();
  });
  EXPECT_THROW(session.run(values, shape, stop.get_token()), Cancelled);
  EXPECT_THROW(session.run(values, shape, {},
                           std::chrono::steady_clock::now() +
                               std::chrono::milliseconds(2)),
               DeadlineExceeded);
  const std::array<std::int64_t, 2> short_shape{1, 512};
  EXPECT_EQ(
      session.run(std::span(values).first(512U), short_shape).values.size(),
      512U);
}

TEST(OnnxRuntimeSession, ConcurrentCallsHaveIndependentStorageAndRunOptions) {
  LinearFixture fixture;
  runtime::Session session(fixture.bytes, fixture.identity);
  auto first = std::async(std::launch::async,
                          [&] { return RunSession(session, fixture.input); });
  auto second = std::async(std::launch::async,
                           [&] { return RunSession(session, fixture.input); });
  auto a = first.get();
  auto b = second.get();
  EXPECT_EQ(a.values, b.values);
  ASSERT_FALSE(a.values.empty());
  const auto saved = b.values.front();
  a.values.front() += 1.0F;
  EXPECT_EQ(b.values.front(), saved);
}

TEST(OnnxRuntimeSession, ProfilingRecordsActualCpuNodesAndClosesIdempotently) {
  LinearFixture fixture;
  runtime::SessionOptions options;
  options.profile_prefix = Root() / "native-linear-profile";
  runtime::Session session(fixture.bytes, fixture.identity, options);
  ASSERT_FALSE(RunSession(session, fixture.input).values.empty());
  const auto profile = session.end_profiling();
  EXPECT_EQ(session.end_profiling(), profile);
  auto events = Json::parse(Text(profile));
  bool cpu_kernel = false;
  for (const auto& event : events) {
    if (event.value("cat", "") == "Node" && event.contains("args") &&
        event.at("args").value("provider", "") == "CPUExecutionProvider")
      cpu_kernel = true;
  }
  EXPECT_TRUE(cpu_kernel);
  EXPECT_FALSE(RunSession(session, fixture.input).values.empty());
  runtime::Session disabled(fixture.bytes, fixture.identity);
  EXPECT_THROW(disabled.end_profiling(), InvalidArgument);
}

}  // namespace
