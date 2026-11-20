// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/models/vision.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <stop_token>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace runtime = jyppx::liuding::onnx_runtime;
namespace exporter = jyppx::liuding::onnx_export;

struct ProfileDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("liuding-dml-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  bool keep = false;
  ProfileDirectory() {
    const auto* evidence = std::getenv("JYPPX_LIUDING_DIRECTML_EVIDENCE_DIR");
    if (evidence != nullptr && *evidence != '\0') {
      keep = true;
      path = std::filesystem::u8path(evidence) / path.filename();
    }
    std::filesystem::create_directories(path);
  }
  ~ProfileDirectory() {
    if (keep) return;
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

runtime::SessionOptions Options() {
  runtime::SessionOptions options;
  options.provider = runtime::directml_execution_provider({.device_id = 0});
  // Qualification is never allowed to succeed via CPU partition fallback.
  options.allow_cpu_fallback = false;
  return options;
}

runtime::TensorOutput RunSession(
    runtime::Session& session, const torch::Tensor& input,
    std::stop_token stop = {},
    std::optional<std::chrono::steady_clock::time_point> deadline = {}) {
  return session.run(std::span(input.const_data_ptr<float>(),
                               static_cast<std::size_t>(input.numel())),
                     std::span(input.sizes().data(), input.sizes().size()),
                     stop, deadline);
}

void Near(const runtime::TensorOutput& output, const torch::Tensor& expected) {
  ASSERT_EQ(output.shape, std::vector<std::int64_t>(expected.sizes().begin(),
                                                    expected.sizes().end()));
  ASSERT_EQ(output.values.size(), static_cast<std::size_t>(expected.numel()));
  const auto flat = expected.contiguous();
  const auto* oracle = flat.const_data_ptr<float>();
  for (std::size_t i = 0; i < output.values.size(); ++i) {
    ASSERT_TRUE(std::isfinite(output.values[i]));
    ASSERT_LE(std::abs(output.values[i] - oracle[i]),
              1e-5F + 1e-4F * std::abs(oracle[i]))
        << "element=" << i;
  }
}

void CheckProfile(runtime::Session& session) {
  const auto path = session.end_profiling();
  EXPECT_EQ(session.end_profiling(), path);
  std::ifstream stream(path);
  ASSERT_TRUE(stream.good());
  const std::string text{std::istreambuf_iterator<char>(stream), {}};
  const auto observed = runtime::inspect_profile(text);
  EXPECT_EQ(observed.kernels.size(), 1U);
  EXPECT_TRUE(observed.kernels.contains("DmlExecutionProvider"));
  const auto events = nlohmann::json::parse(text);
  std::size_t nodes = 0;
  for (const auto& event : events) {
    if (event.value("cat", "") != "Node" || !event.contains("args") ||
        !event.at("args").contains("provider"))
      continue;
    const auto& args = event.at("args");
    EXPECT_EQ(args.at("provider"), "DmlExecutionProvider") << event.dump();
    EXPECT_TRUE(args.contains("op_name"));
    ++nodes;
  }
  EXPECT_GT(nodes, 0U);
}

struct Linear {
  torch::nn::Linear model{4, 3};
  exporter::ExportResult artifact;
  Linear() {
    at::set_num_threads(1);
    torch::NoGradGuard guard;
    model->eval();
    model->weight.copy_(
        torch::arange(12, torch::kFloat32).reshape({3, 4}) * 0.125F - 0.5F);
    model->bias.copy_(torch::tensor({0.25F, -0.125F, 0.5F}));
    artifact = exporter::export_model(model, Input(2));
  }
  static torch::Tensor Input(std::int64_t batch, float offset = 0.0F) {
    return torch::arange(batch * 4, torch::kFloat32).reshape({batch, 4}) *
               0.0625F -
           0.25F + offset;
  }
  static void Scalar(const runtime::TensorOutput& actual,
                     const torch::Tensor& input) {
    ASSERT_EQ(actual.values.size(),
              static_cast<std::size_t>(input.size(0) * 3));
    constexpr float bias[]{0.25F, -0.125F, 0.5F};
    const auto* x = input.const_data_ptr<float>();
    for (std::int64_t row = 0; row < input.size(0); ++row) {
      for (std::int64_t out = 0; out < 3; ++out) {
        float expected = bias[out];
        for (std::int64_t col = 0; col < 4; ++col)
          expected += x[row * 4 + col] *
                      (static_cast<float>(out * 4 + col) * 0.125F - 0.5F);
        EXPECT_FLOAT_EQ(actual.values[row * 3 + out], expected);
      }
    }
  }
};

TEST(OnnxRuntimeDirectMlGpu, LinearDynamicBatchScalarAndActualProvider) {
  Linear fixture;
  ProfileDirectory directory;
  auto options = Options();
  options.profile_prefix = directory.path / "linear";
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest,
                           options);
  EXPECT_EQ(session.runtime_version(), JYPPX_TEST_ORT_VERSION);
  EXPECT_EQ(session.execution_provider().name, "DmlExecutionProvider");
  EXPECT_FALSE(session.cpu_fallback_allowed());
  EXPECT_EQ(session.input().shape.front(), -1);
  for (const auto batch : {1, 2, 5}) {
    const auto input = Linear::Input(batch);
    Linear::Scalar(RunSession(session, input), input);
  }
  CheckProfile(session);
  const auto input = Linear::Input(2);
  Linear::Scalar(RunSession(session, input),
                 input);  // Run remains usable after flush.
}

TEST(OnnxRuntimeDirectMlGpu, ResNet18DynamicBatchNativeAndActualProvider) {
  at::set_num_threads(1);
  torch::NoGradGuard guard;
  torch::manual_seed(1701);
  auto model = models::resnet18({.num_classes = 7});
  model->eval();
  const auto input_for = [](std::int64_t batch) {
    return (torch::arange(batch * 3 * 32 * 32, torch::kFloat32).remainder(97) /
                97.0F -
            0.5F)
        .reshape({batch, 3, 32, 32});
  };
  const auto artifact = exporter::export_model(model, input_for(2));
  ProfileDirectory directory;
  auto options = Options();
  options.profile_prefix = directory.path / "resnet18";
  runtime::Session session(artifact.onnx, artifact.digest, options);
  for (const auto batch : {1, 2, 5}) {
    const auto input = input_for(batch);
    Near(RunSession(session, input), model->forward(input));
  }
  CheckProfile(session);
}

TEST(OnnxRuntimeDirectMlGpu,
     ConcurrentCallsAndIndependentSessionsRetainOwnedResults) {
  Linear fixture;
  runtime::Session first(fixture.artifact.onnx, fixture.artifact.digest,
                         Options());
  runtime::Session second(fixture.artifact.onnx, fixture.artifact.digest,
                          Options());
  std::barrier start(8);
  std::vector<std::future<void>> callers;
  for (unsigned i = 0; i < 8; ++i) {
    callers.push_back(std::async(std::launch::async, [&, i] {
      const auto input =
          Linear::Input(1 + i % 5, static_cast<float>(i) * 0.25F);
      start.arrive_and_wait();
      // Six callers contend on first; two independently use second.
      auto& session = i < 6 ? first : second;
      for (unsigned run = 0; run < 16; ++run)
        Linear::Scalar(RunSession(session, input), input);
    }));
  }
  for (auto& caller : callers) caller.get();
  auto input = Linear::Input(5);
  runtime::TensorOutput retained;
  {
    runtime::Session temporary(fixture.artifact.onnx, fixture.artifact.digest,
                               Options());
    retained = RunSession(temporary, input);
    const auto snapshot = retained.values;
    auto next = RunSession(temporary, Linear::Input(2));
    next.values.assign(next.values.size(), 999.0F);
    EXPECT_EQ(retained.values, snapshot);
  }
  Linear::Scalar(retained, input);
  input.fill_(999.0F);
  Linear::Scalar(retained, Linear::Input(5));
}

TEST(OnnxRuntimeDirectMlGpu, CancellationDeadlineAndInvalidDeviceRecover) {
  Linear fixture;
  runtime::Session session(fixture.artifact.onnx, fixture.artifact.digest,
                           Options());
  const auto input = Linear::Input(2);
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(RunSession(session, input, stopped.get_token()), Cancelled);
  EXPECT_THROW(RunSession(session, input, {}, std::chrono::steady_clock::now()),
               DeadlineExceeded);
  Linear::Scalar(RunSession(session, input), input);
  auto invalid = Options();
  invalid.provider = runtime::directml_execution_provider(
      {.device_id = std::numeric_limits<std::int32_t>::max()});
  EXPECT_THROW(
      runtime::Session(fixture.artifact.onnx, fixture.artifact.digest, invalid),
      DeviceError);
  Linear::Scalar(RunSession(session, input), input);
}
}  // namespace
