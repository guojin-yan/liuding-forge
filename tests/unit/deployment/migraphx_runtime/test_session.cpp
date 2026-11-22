// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../../../integration/deployment/native_modules.hpp"

#include <jyppx/liuding/deployment/migraphx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <vector>

namespace {
using namespace jyppx::liuding;
namespace runtime = jyppx::liuding::migraphx_runtime;

TEST(MIGraphXSession, RejectsInvalidOptionsBeforeDeviceDiscovery) {
  const std::array model{std::byte{0}};
  const ArtifactDigest digest{.hex = sha256_hex(model)};
  const runtime::SessionOptions valid{.input_shape = {2, 8}};
  EXPECT_THROW(runtime::Session(model, ArtifactDigest{.hex = "bad"}, valid),
               InvalidArgument);
  EXPECT_THROW(runtime::Session({}, digest, valid), InvalidArgument);
  auto options = valid;
  options.device_id = -1;
  EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  options = valid;
  options.input_name = std::string("in\0put", 6);
  EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  options = valid;
  options.input_shape = {};
  EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  options = valid;
  options.input_shape = {0, 8};
  EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  options = valid;
  options.input_shape.assign(9, 1);
  EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  for (auto member : {&runtime::SessionOptions::max_model_bytes,
                      &runtime::SessionOptions::max_input_bytes,
                      &runtime::SessionOptions::max_output_bytes,
                      &runtime::SessionOptions::max_batch_size}) {
    options = valid;
    options.*member = 0;
    EXPECT_THROW(runtime::Session(model, digest, options), InvalidArgument);
  }
  options = valid;
  options.max_input_bytes = 63;
  EXPECT_THROW(runtime::Session(model, digest, options), ResourceExhausted);
  options = valid;
  options.max_batch_size = 1;
  EXPECT_THROW(runtime::Session(model, digest, options), ResourceExhausted);
  options = valid;
  options.input_shape = {std::numeric_limits<std::size_t>::max(), 2};
  EXPECT_THROW(runtime::Session(model, digest, options), ResourceExhausted);
  const std::array larger{std::byte{0}, std::byte{0}};
  options = valid;
  options.max_model_bytes = 1;
  EXPECT_THROW(runtime::Session(
                   larger, ArtifactDigest{.hex = sha256_hex(larger)}, options),
               ResourceExhausted);
  EXPECT_THROW(runtime::Session(
                   model, ArtifactDigest{.hex = std::string(64, '0')}, valid),
               CompatibilityError);
}

#ifdef JYPPX_MIGRAPHX_RUNTIME_FIXTURES
std::vector<std::byte> Read(const std::filesystem::path& path,
                            std::size_t bound, const char* sha) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0 || size > bound) throw std::runtime_error("fixed fixture size");
  std::vector<std::byte> result(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(size));
  if (!input || sha256_hex(result) != sha)
    throw std::runtime_error("fixed fixture SHA");
  return result;
}
struct Fixture {
  const char* graph;
  const char* input;
  const char* output;
};
Fixture Identities(bool resnet, int index) {
  if (resnet)
    return {
        "95fbd4284bb53e1559c3448d49200836f50731bc89ac9694ed1f478c1581a399",
        index
            ? "697c148d7f14443e742ca11f62dd028a582fb3a216647a6cd0fd99b56f6f538e"
            : "e72a6421bd642adf1c13e70636c800abdbbda4fecfbce5413b21ad42a186351"
              "e",
        index
            ? "a80ff115fbbddc54314cdebace0f302765ba4fc311e2f52328952ec23c653165"
            : "4c8b23c87357417e3c647d9bc7d8a3db1cc8f1a8897bda5088865910b56b489"
              "f"};
  return {
      "7b4f9bf96cd72143e22c906b0063958ef05b10534785eac3e9fc1b07afcb990a",
      index
          ? "bafca57808f22a3d4d4d70225cabecb085132ff4c9de22fd4692abd7abee94b7"
          : "d55f7abaa5bb2c75cc098aa3dc97579967a294f802b308803833fef7964a94e9",
      index
          ? "509b6a760633711b6655feca75f3d873d5184fc1076cc55213773a8b91be5529"
          : "3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709a"};
}
std::vector<float> Floats(const std::filesystem::path& path, const char* sha) {
  const auto bytes = Read(path, 8U << 20U, sha);
  if (bytes.size() % sizeof(float))
    throw std::runtime_error("fixed FP32 byte count");
  std::vector<float> result(bytes.size() / sizeof(float));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
void Compare(const runtime::RunResult& result,
             std::span<const float> expected) {
  ASSERT_EQ(result.values.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ASSERT_TRUE(std::isfinite(result.values[i]));
    ASSERT_LE(std::abs(static_cast<double>(result.values[i]) - expected[i]),
              1e-5 + 1e-4 * std::abs(static_cast<double>(expected[i])))
        << i;
  }
  const auto any = [&](const char* prefix) {
    return std::any_of(
        result.instructions.begin(), result.instructions.end(),
        [&](const auto& name) { return name.starts_with(prefix); });
  };
  EXPECT_TRUE(any("gpu::hip_gemm") || any("gpu::convolution") ||
              any("gpu::code_object"));
  EXPECT_TRUE(any("hip::copy_from_gpu"));
  EXPECT_TRUE(any("hip::sync_stream"));
  EXPECT_FALSE(any("cpu::"));
  EXPECT_FALSE(any("ref::"));
}
void Capture(const runtime::RunResult& result, int index) {
  const auto parent = std::filesystem::path(JYPPX_MIGRAPHX_RUNTIME_EVIDENCE);
  std::filesystem::create_directories(parent);
  const auto tick = std::chrono::system_clock::now().time_since_epoch().count();
  const auto folder =
      parent / ("case-" + std::to_string(index) + "-" + std::to_string(tick));
  if (!std::filesystem::create_directory(folder))
    throw std::runtime_error("evidence already exists");
  const auto bytes = std::as_bytes(std::span(result.values));
  std::ofstream payload(folder / "actual.f32", std::ios::binary);
  payload.write(reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
  payload.close();
  std::ofstream trace(folder / "dispatch.txt", std::ios::binary);
  for (const auto& name : result.instructions) trace << name << '\n';
  trace.close();
  if (!payload || !trace) throw std::runtime_error("evidence write failed");
  jyppx_test::Modules(folder / "modules.txt");
  std::cout << "native_runtime_evidence=" << folder.string()
            << " output_sha256=" << sha256_hex(bytes) << '\n';
}
class MIGraphXSessionGpu : public ::testing::TestWithParam<int> {};
TEST_P(MIGraphXSessionGpu, FixedReferenceOwnershipAndSerializedCalls) {
  const bool resnet = GetParam() >= 2;
  const int index = GetParam() % 2;
  const auto ids = Identities(resnet, index);
  const auto folder = std::filesystem::path(JYPPX_MIGRAPHX_RUNTIME_FIXTURES) /
                      (resnet ? "resnet18" : "linear");
  const auto graph = Read(folder / "model.onnx", 256U << 20U, ids.graph);
  const auto prefix = "case-" + std::to_string(index);
  auto input = Floats(folder / (prefix + "-input.f32"), ids.input);
  const auto retained_input = input;
  const auto expected = Floats(folder / (prefix + "-output.f32"), ids.output);
  const std::size_t batch = index ? 5 : 2;
  runtime::SessionOptions options{
      .input_shape = resnet ? std::vector<std::size_t>{batch, 3, 224, 224}
                            : std::vector<std::size_t>{batch, 8}};
  const ArtifactDigest digest{.hex = ids.graph};
  int previous = -1;
  ASSERT_EQ(hipGetDevice(&previous), hipSuccess);
  runtime::RunResult retained;
  {
    runtime::Session session(graph, digest, options);
    int current = -1;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, previous);
    EXPECT_EQ(session.options().device_id, 0);
    EXPECT_EQ(std::vector<std::size_t>(session.output_shape().begin(),
                                       session.output_shape().end()),
              (std::vector<std::size_t>{batch, resnet ? 7U : 3U}));
    EXPECT_THROW(session.run(std::span(input).first(input.size() - 1)),
                 InvalidArgument);
    auto invalid = input;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(session.run(invalid), InvalidArgument);
    retained = session.run(input);
    Capture(retained, GetParam());
    Compare(retained, expected);
    if (!resnet) {
      auto a =
          std::async(std::launch::async, [&] { return session.run(input); });
      auto b =
          std::async(std::launch::async, [&] { return session.run(input); });
      Compare(a.get(), expected);
      Compare(b.get(), expected);
    }
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, previous);
  }
  Compare(retained, expected);
  EXPECT_EQ(input, retained_input);
  if (!resnet && index == 0) {
    auto invalid_options = options;
    invalid_options.device_id = std::numeric_limits<std::int32_t>::max();
    EXPECT_THROW(runtime::Session(graph, digest, invalid_options), DeviceError);
    invalid_options = options;
    invalid_options.input_name = "missing";
    EXPECT_THROW(runtime::Session(graph, digest, invalid_options),
                 CompatibilityError);
    invalid_options = options;
    invalid_options.max_output_bytes = 1;
    EXPECT_THROW(runtime::Session(graph, digest, invalid_options),
                 ResourceExhausted);
    const std::array malformed{std::byte{0}};
    EXPECT_THROW(
        runtime::Session(malformed,
                         ArtifactDigest{.hex = sha256_hex(malformed)}, options),
        CompatibilityError);
  }
}
INSTANTIATE_TEST_SUITE_P(FixedArtifacts, MIGraphXSessionGpu,
                         ::testing::Values(0, 1, 2, 3));
#endif
}  // namespace
