// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_TEST_MIGRAPHX_SERVING_FIXTURE_HPP
#define JYPPX_TEST_MIGRAPHX_SERVING_FIXTURE_HPP
// Shared test-only owner codec, using SDK public headers and immutable
// fixtures.
#include <jyppx/liuding/deployment/migraphx_serving/backend.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <torch/torch.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace jyppx_test::migraphx_serving {
using namespace jyppx::liuding;
using namespace jyppx::liuding::serving;
namespace mgx = jyppx::liuding::migraphx_runtime;
inline std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
inline ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.hex = sha256_hex(bytes)};
}
inline std::vector<std::byte> Read(const std::filesystem::path& file,
                                   const char* digest) {
  const auto size = std::filesystem::file_size(file);
  if (!size || size > (1U << 20U))
    throw std::runtime_error("immutable fixture size");
  std::vector<std::byte> result(static_cast<std::size_t>(size));
  std::ifstream stream(file, std::ios::binary);
  stream.read(reinterpret_cast<char*>(result.data()),
              static_cast<std::streamsize>(size));
  if (!stream || stream.peek() != std::char_traits<char>::eof() ||
      sha256_hex(result) != digest)
    throw std::runtime_error("immutable fixture SHA");
  return result;
}
inline Tensor Values(std::span<const std::byte> bytes, std::int64_t batch,
                     std::int64_t width) {
  if (bytes.size() != static_cast<std::size_t>(batch * width) * sizeof(float))
    throw std::runtime_error("immutable fixture FP32 count");
  auto result = torch::empty({batch, width}, torch::kFloat32);
  std::memcpy(result.data_ptr<float>(), bytes.data(), bytes.size());
  return result;
}
struct Fixture {
  std::vector<std::byte> graph;
  Tensor input, expected;
  mgx::SessionOptions options;
  mgx::EnvironmentLock environment;
  explicit Fixture(const std::filesystem::path& fixtures, int index = 0) {
    const auto root = fixtures / "linear";
    graph = Read(
        root / "model.onnx",
        "7b4f9bf96cd72143e22c906b0063958ef05b10534785eac3e9fc1b07afcb990a");
    const auto prefix = "case-" + std::to_string(index);
    const auto batch = index ? 5 : 2;
    input = Values(Read(root / (prefix + "-input.f32"),
                        index ? "bafca57808f22a3d4d4d70225cabecb085132ff4c9de22"
                                "fd4692abd7abee94b7"
                              : "d55f7abaa5bb2c75cc098aa3dc97579967a294f802b308"
                                "803833fef7964a94e9"),
                   batch, 8);
    expected = Values(Read(root / (prefix + "-output.f32"),
                           index ? "509b6a760633711b6655feca75f3d873d5184fc1076"
                                   "cc55213773a8b91be5529"
                                 : "3b106db38a6dca7aafe95d1af546237ca38ac4d74a0"
                                   "84a8f1b0cea5a361c709a"),
                      batch, 3);
    options.input_shape = {static_cast<std::size_t>(batch), 8};
    const mgx::Session observation(graph, Digest(graph), options);
    environment = mgx::observe_environment(observation);
  }
  nn::Invocation Input() const {
    return {.positional = {}, .keyword = {{"input", input}}};
  }
  WarmupCase Known() const {
    return {.input = Input(),
            .expected = expected,
            .relative_tolerance = 1e-4,
            .absolute_tolerance = 1e-5};
  }
  PredictRequest Request(std::string version = "migraphx-v1") const {
    PredictRequest request;
    request.input = Input();
    request.context.request_id = "migraphx-request";
    request.context.model_version = std::move(version);
    return request;
  }
};
inline std::string InputText(const mgx::EnvironmentLock& environment) {
  const auto schema = migraphx_invocation_schema(environment.input_name,
                                                 environment.input_shape);
  return "fixed-migraphx-input-v1;" + environment.input_name + ";" +
         canonical_value_schema(schema.keyword.begin()->second);
}
struct Bundle {
  std::filesystem::path root;
  RuntimeDescriptor descriptor;
  interop::ModelBundleManifest manifest;
  std::vector<interop::BundlePublishSource> sources;
  std::size_t publication = 0;
  explicit Bundle(const std::filesystem::path& directory,
                  const Fixture& fixture,
                  const mgx::EnvironmentLock* override_environment = nullptr)
      : root(directory) {
    if (!std::filesystem::create_directories(root / "sources"))
      throw std::runtime_error("fresh bundle evidence required");
    const auto& environment =
        override_environment ? *override_environment : fixture.environment;
    const auto lock = mgx::serialize_environment_lock(environment);
    descriptor = migraphx_runtime_descriptor(environment);
    manifest.bundle_id = "migraphx-v1";
    manifest.build = current_build_fingerprint();
    manifest.runtime = {.backend_id = descriptor.backend_id,
                        .backend_version = descriptor.backend_version,
                        .environment_lock = descriptor.environment_lock,
                        .dependencies = descriptor.dependencies,
                        .devices = {"rocm"},
                        .dtypes = {"float32"},
                        .os_arch = {environment.os_arch},
                        .dynamic_shapes = false};
    Add("graph", interop::BundleFileRole::model, fixture.graph);
    manifest.models.push_back({.id = "primary",
                               .format = interop::BundleModelFormat::onnx,
                               .file_ids = {"graph"},
                               .tensors = {}});
    manifest.input_schema = Description(
        "input", interop::BundleFileRole::schema, InputText(environment));
    manifest.output_schema =
        Description("output", interop::BundleFileRole::schema,
                    canonical_value_schema(
                        migraphx_result_schema(environment.output_shape)));
    manifest.preprocessing =
        Description("pre", interop::BundleFileRole::prepost, "identity-v1");
    manifest.postprocessing =
        Description("post", interop::BundleFileRole::prepost, "identity-v1");
    manifest.evaluation =
        Description("evaluation", interop::BundleFileRole::evaluation,
                    "original-fixed-reference");
    manifest.sbom = Description("sbom", interop::BundleFileRole::sbom,
                                "observed-native-runtime");
    manifest.provenance =
        Description("provenance", interop::BundleFileRole::provenance,
                    "native-cpp-offline-test");
    auto ref =
        Description("runtime-lock", interop::BundleFileRole::evidence, lock);
    ref.schema_id = std::string(mgx::kEnvironmentLockSchemaId);
    manifest.evidence.push_back(std::move(ref));
  }
  void Add(std::string id, interop::BundleFileRole role,
           std::span<const std::byte> bytes) {
    const auto path = root / "sources" / (id + ".data");
    if (std::filesystem::exists(path))
      throw std::runtime_error("source overwrite refused");
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) throw std::runtime_error("bundle source write");
    sources.push_back({.file_id = id, .source_path = path});
    manifest.files.push_back(
        {.id = id,
         .relative_path =
             "artifacts/" + id +
             (role == interop::BundleFileRole::model ? ".onnx" : ".data"),
         .role = role,
         .artifact = {.digest = Digest(bytes),
                      .size_bytes = bytes.size(),
                      .logical_name = id,
                      .media_type = role == interop::BundleFileRole::model
                                        ? "application/onnx"
                                        : "application/octet-stream",
                      .producer = "jyppx.test.migraphx-serving"}});
  }
  interop::BundleDescriptorRef Description(std::string id,
                                           interop::BundleFileRole role,
                                           std::string_view text) {
    Add(id, role, Bytes(text));
    return {.schema_id = "jyppx.test.migraphx." + id,
            .schema_version = 1U,
            .file_id = std::move(id)};
  }
  std::filesystem::path Publish(std::string version = "migraphx-v1") {
    manifest.bundle_id = std::move(version);
    const auto directory = root / ("bundle-" + std::to_string(publication++));
    static_cast<void>(
        interop::publish_model_bundle(directory, manifest, sources));
    return directory;
  }
  static MIGraphXPipelineFactory Factory() {
    return [](const VerifiedBundle& bundle,
              const mgx::EnvironmentLock& environment) {
      const auto bind = [&](const interop::BundleDescriptorRef& ref,
                            std::string_view expected) {
        const auto& bytes = bundle.files.at(ref.file_id);
        if (ref.schema_id != "jyppx.test.migraphx." + ref.file_id ||
            ref.schema_version != 1U || bytes.size() != expected.size() ||
            !std::equal(bytes.begin(), bytes.end(), Bytes(expected).begin()))
          throw CompatibilityError("Owner descriptor contents rejected");
        return DescriptorBinding{.schema_id = ref.schema_id,
                                 .schema_version = ref.schema_version,
                                 .digest = Digest(bytes)};
      };
      const auto& m = bundle.verification.manifest;
      return Pipeline{
          .version = "fixed-migraphx-identity-v1",
          .input = bind(m.input_schema, InputText(environment)),
          .output = bind(m.output_schema,
                         canonical_value_schema(
                             migraphx_result_schema(environment.output_shape))),
          .preprocessing = bind(m.preprocessing, "identity-v1"),
          .postprocessing = bind(m.postprocessing, "identity-v1"),
          .request_schema = migraphx_invocation_schema(environment.input_name,
                                                       environment.input_shape),
          .response_schema = migraphx_result_schema(environment.output_shape),
          .preprocess = {},
          .postprocess = {}};
    };
  }
  MIGraphXRuntimeBackend Backend(const Fixture& fixture,
                                 MIGraphXPipelineFactory factory = {}) const {
    return MIGraphXRuntimeBackend(descriptor, fixture.options,
                                  factory ? std::move(factory) : Factory());
  }
};
inline void Compare(const c10::IValue& output, const Fixture& fixture) {
  if (!output.isTensor()) throw std::runtime_error("response tensor required");
  const auto actual = output.toTensor();
  if (!actual.device().is_cpu() || actual.scalar_type() != torch::kFloat32 ||
      !actual.is_contiguous() || actual.sizes() != fixture.expected.sizes())
    throw std::runtime_error("response signature");
  const auto* a = actual.const_data_ptr<float>();
  const auto* e = fixture.expected.const_data_ptr<float>();
  for (std::int64_t i = 0; i < actual.numel(); ++i)
    if (!std::isfinite(a[i]) ||
        std::abs(static_cast<double>(a[i]) - e[i]) >
            1e-5 + 1e-4 * std::abs(static_cast<double>(e[i])))
      throw std::runtime_error("original FP32 reference tolerance");
}
inline void WriteOutput(const std::filesystem::path& path,
                        const c10::IValue& value) {
  const auto tensor = value.toTensor();
  const auto bytes =
      std::as_bytes(std::span(tensor.const_data_ptr<float>(),
                              static_cast<std::size_t>(tensor.numel())));
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) throw std::runtime_error("output evidence write");
}
}  // namespace jyppx_test::migraphx_serving
#endif
