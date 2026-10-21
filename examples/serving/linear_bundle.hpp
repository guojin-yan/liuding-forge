// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_EXAMPLE_SERVING_LINEAR_BUNDLE_HPP
#define JYPPX_EXAMPLE_SERVING_LINEAR_BUNDLE_HPP
// Fixed, generated data-only example: y[n,c] = x[n,c] * [2,-3][c].
// This is local application code; a bundle never selects or loads a C++ class.
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/version.hpp>
#include <jyppx/serving/serving.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::examples::serving {
using namespace jyppx::serving;
inline std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
inline ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.hex = sha256_hex(bytes)};
}
inline RuntimeDescriptor Descriptor(Device device = Device(c10::kCPU)) {
  return {.backend_id = "jyppx.example.native-linear",
          .backend_version = "1",
          .format = "safetensors",
          .environment_lock =
              Digest(Bytes(std::string("example-lock-v1:libtorch:") +
                           build_info().libtorch_version)),
          .dependencies = {{.name = "libtorch",
                            .version = build_info().libtorch_version}},
          .device = device,
          .dtypes = {torch::kFloat32},
          .dynamic_shapes = true,
          .tensor_batching = true,
          .max_batch_size = 8U,
          .oom_retry_safe = true};
}
inline TensorSchema TensorContract(Device device) {
  return {.dtype = torch::kFloat32,
          .device = device,
          .rank = 2,
          .shape = std::vector<std::optional<std::int64_t>>{std::nullopt, 2}};
}
inline nn::InvocationSchema Input(Device device) {
  return {.positional = {ValueSchema::tensor(TensorContract(device), "input")},
          .keyword = {},
          .allow_extra_keywords = false};
}
inline ValueSchema Output(Device device) {
  return ValueSchema::tensor(TensorContract(device), "output");
}
inline std::string InputDescriptor(Device device) {
  return canonical_value_schema(Input(device).positional.front());
}
inline std::string OutputDescriptor(Device device) {
  return canonical_value_schema(Output(device));
}
inline constexpr std::string_view kIdentity = "jyppx.example.identity.v1";
using Hook = std::function<void(const Tensor&)>;
struct Linear final : torch::nn::Module {
  Tensor weight;
  Hook hook;
  explicit Linear(Device device, Hook before = {}) : hook(std::move(before)) {
    weight = register_buffer(
        "weight",
        torch::zeros(
            {2}, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
  }
  Tensor forward(const Tensor& input) {
    if (hook) hook(input);
    return input * weight;
  }
};
inline NativeModelFactory Factory(Hook hook = {}) {
  return [hook = std::move(hook)](const VerifiedBundle& bundle,
                                  const RuntimeDescriptor& runtime) {
    const auto bind = [&](const interop::BundleDescriptorRef& ref,
                          std::string_view expected) {
      const auto& bytes = bundle.files.at(ref.file_id);
      if (bytes.size() != expected.size() ||
          !std::equal(bytes.begin(), bytes.end(), Bytes(expected).begin()))
        throw CompatibilityError("Example owner descriptor mismatch");
      const auto& files = bundle.verification.manifest.files;
      const auto file =
          std::find_if(files.begin(), files.end(),
                       [&](const auto& f) { return f.id == ref.file_id; });
      if (file == files.end())
        throw CompatibilityError("Missing example descriptor");
      return DescriptorBinding{.schema_id = ref.schema_id,
                               .schema_version = ref.schema_version,
                               .digest = file->artifact.digest};
    };
    const auto& m = bundle.verification.manifest;
    auto module = std::make_shared<Linear>(runtime.device, hook);
    auto adapter = std::make_shared<nn::ModelAdapter>(
        nn::adapt_tensor_module(module, TensorContract(runtime.device),
                                TensorContract(runtime.device)));
    Pipeline pipeline{
        .version = "example-linear-identity-v1",
        .input = bind(m.input_schema, InputDescriptor(runtime.device)),
        .output = bind(m.output_schema, OutputDescriptor(runtime.device)),
        .preprocessing = bind(m.preprocessing, kIdentity),
        .postprocessing = bind(m.postprocessing, kIdentity),
        .request_schema = Input(runtime.device),
        .response_schema = Output(runtime.device),
        .preprocess = {},
        .postprocess = {}};
    return NativeModel{.adapter = std::move(adapter),
                       .pipeline = std::move(pipeline)};
  };
}
inline WarmupCase KnownVector(Device device) {
  return {
      .input = {.positional = {torch::ones({1, 2}, torch::TensorOptions()
                                                       .dtype(torch::kFloat32)
                                                       .device(device))},
                .keyword = {}},
      .expected =
          torch::tensor({{2.0F, -3.0F}}, torch::TensorOptions().device(device)),
      .relative_tolerance = 0.0,
      .absolute_tolerance = 0.0};
}
inline PredictRequest Request(Device device, std::string id = "request-1",
                              float value = 1.0F) {
  return {
      .input =
          {.positional = {torch::full(
               {1, 2}, value,
               torch::TensorOptions().dtype(torch::kFloat32).device(device))},
           .keyword = {}},
      .context = {.request_id = std::move(id),
                  .trace_id = "trace-example",
                  .project_id = "project-example",
                  .principal_id = "local-operator",
                  .deployment_id = "local",
                  .model_version = "example-linear-v1"},
      .deadline = {}};
}
inline void Write(const std::filesystem::path& path,
                  std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream)
    throw std::runtime_error("Cannot write generated local example bundle");
}
/** Generated local fixture with an exclusively owned temporary directory. */
class BundleFixture final {
 public:
  std::filesystem::path root, directory;
  RuntimeDescriptor descriptor;
  interop::ModelBundleManifest manifest;
  std::vector<interop::BundlePublishSource> sources;
  explicit BundleFixture(Device device = Device(c10::kCPU))
      : descriptor(Descriptor(device)) {
    static std::atomic<std::uint64_t> sequence{0};
    root = std::filesystem::temp_directory_path() /
           ("jyppx-serving-example-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(++sequence));
    std::filesystem::create_directories(root / "sources");
    directory = root / "bundle";
    manifest.bundle_id = "example-linear-v1";
    manifest.build = current_build_fingerprint();
    manifest.runtime = {.backend_id = descriptor.backend_id,
                        .backend_version = descriptor.backend_version,
                        .environment_lock = descriptor.environment_lock,
                        .dependencies = descriptor.dependencies,
                        .devices = {device.is_cuda() ? "cuda" : "cpu"},
                        .dtypes = {"float32"},
                        .os_arch = {serving_platform_id()},
                        .dynamic_shapes = true};
    const std::array<float, 2> values{2.0F, -3.0F};
    const interop::SafeTensorSource state{
        .name = "weight",
        .dtype = interop::SafeTensorDtype::f32,
        .shape = {2},
        .data = std::as_bytes(std::span(values)),
        .byte_order = interop::SafeTensorByteOrder::little};
    const auto shards =
        interop::shard_safetensors_for_bundle(std::span(&state, 1), 4096);
    interop::BundleModel model{
        .id = "primary",
        .format = interop::BundleModelFormat::safetensors,
        .file_ids = {},
        .tensors = {}};
    for (const auto& shard : shards) {
      AddSource(shard.file.id, shard.bytes);
      manifest.files.push_back(shard.file);
      model.file_ids.push_back(shard.file.id);
      model.tensors.insert(model.tensors.end(), shard.tensors.begin(),
                           shard.tensors.end());
    }
    manifest.models.push_back(std::move(model));
    manifest.input_schema = AddDescriptor(
        "input", interop::BundleFileRole::schema, InputDescriptor(device));
    manifest.output_schema = AddDescriptor(
        "output", interop::BundleFileRole::schema, OutputDescriptor(device));
    manifest.preprocessing = AddDescriptor(
        "preprocess", interop::BundleFileRole::prepost, kIdentity);
    manifest.postprocessing = AddDescriptor(
        "postprocess", interop::BundleFileRole::prepost, kIdentity);
    manifest.evaluation = AddDescriptor(
        "evaluation", interop::BundleFileRole::evaluation,
        R"({"schema_version":1,"evidence":"generated exact arithmetic known vector","trained_model":false})");
    manifest.sbom = AddDescriptor(
        "sbom", interop::BundleFileRole::sbom,
        R"({"schema_version":1,"dependencies":["libtorch"],"environment":"generated local fixture"})");
    manifest.provenance = AddDescriptor(
        "provenance", interop::BundleFileRole::provenance,
        R"({"schema_version":1,"origin":"generated local arithmetic fixture","dataset":null})");
  }
  ~BundleFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  BundleFixture(const BundleFixture&) = delete;
  BundleFixture& operator=(const BundleFixture&) = delete;
  void AddSource(std::string id, std::span<const std::byte> bytes) {
    const auto path = root / "sources" / (id + ".data");
    Write(path, bytes);
    sources.push_back({.file_id = std::move(id), .source_path = path});
  }
  interop::BundleDescriptorRef AddDescriptor(std::string id,
                                             interop::BundleFileRole role,
                                             std::string_view content) {
    AddSource(id, Bytes(content));
    manifest.files.push_back(
        {.id = id,
         .relative_path = "metadata/" + id + ".data",
         .role = role,
         .artifact = {.digest = Digest(Bytes(content)),
                      .size_bytes = content.size(),
                      .logical_name = id,
                      .media_type = "application/octet-stream",
                      .producer = "jyppx.example.serving"}});
    return {.schema_id = "jyppx.example." + id,
            .schema_version = 1,
            .file_id = std::move(id)};
  }
  void Publish(const interop::ModelBundleVerifyOptions& verify = {}) {
    static_cast<void>(
        interop::publish_model_bundle(directory, manifest, sources, verify));
  }
  std::shared_ptr<Predictor> Load(Hook hook = {},
                                  const LoadOptions& options = {}) {
    NativeStateBackend backend(descriptor, Factory(std::move(hook)));
    const std::array<WarmupCase, 1> cases{KnownVector(descriptor.device)};
    return backend.load(directory, {}, cases, options);
  }
};
}  // namespace jyppx::examples::serving
#endif
