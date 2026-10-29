// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file onnx.cpp @brief Builds real multi-I/O ONNX bundles in C++, verifies
 * independent scalar references and publishes an ORT serving version safely. */
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/deployment/onnx_serving/backend.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/serving/reload.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace exporter = forge::onnx_export;
namespace ort = forge::onnx_runtime;
namespace serving = forge::serving;
namespace interop = forge::interop;
namespace fs = std::filesystem;
using Tensors = std::vector<at::Tensor>;

constexpr std::string_view kInput =
    R"({"schema_version":1,"names":["left","right"],"dtype":"float32","device":"cpu","shape":[null,2],"batch_symbol":"batch","maximum_batch":8})";
constexpr std::string_view kOutput =
    R"({"schema_version":1,"names":["sum","difference"],"dtype":"float32","device":"cpu","shape":[null,2],"batch_symbol":"batch"})";
constexpr std::string_view kIdentity =
    R"({"schema_version":1,"stage":"identity"})";

std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
forge::ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.hex = forge::sha256_hex(bytes)};
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    static std::atomic<std::uint64_t> sequence{0};
    path = fs::temp_directory_path() /
           ("liuding-serving-onnx-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(++sequence));
    if (!fs::create_directory(path))
      throw std::runtime_error("Cannot create an exclusive example directory");
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  fs::path path;
};

// This normal application-owned C++ model uses the same generic exporter as
// trained models. Changing gain changes the actual graph weights, not a label.
class PairModel final : public torch::nn::Module {
 public:
  explicit PairModel(float gain)
      : gain_(register_buffer("gain", torch::tensor({gain}))) {}
  Tensors forward(const at::Tensor& left, const at::Tensor& right) const {
    return {left * gain_ + right, left - right};
  }

 private:
  at::Tensor gain_;
};

struct Sample {
  forge::nn::Invocation input;
  c10::IValue expected;
};

Sample MakeSample(std::int64_t batch, float gain) {
  auto left = torch::empty({batch, 2}, torch::kFloat32);
  auto right = torch::empty({batch, 2}, torch::kFloat32);
  auto sum = torch::empty({batch, 2}, torch::kFloat32);
  auto difference = torch::empty({batch, 2}, torch::kFloat32);
  for (std::int64_t i = 0; i < batch * 2; ++i) {
    const float a = static_cast<float>(i) / 4.0F - 1.0F;
    const float b = static_cast<float>(i) / 8.0F + 0.125F;
    left.data_ptr<float>()[i] = a;
    right.data_ptr<float>()[i] = b;
    // Independent scalar oracle; neither native forward nor ORT computes it.
    sum.data_ptr<float>()[i] = a * gain + b;
    difference.data_ptr<float>()[i] = a - b;
  }
  c10::Dict<std::string, at::Tensor> expected;
  expected.insert("sum", std::move(sum));
  expected.insert("difference", std::move(difference));
  return {.input = {.positional = {},
                    .keyword = {{"right", std::move(right)},
                                {"left", std::move(left)}}},
          .expected = std::move(expected)};
}

std::array<serving::WarmupCase, 2> KnownVectors(float gain) {
  auto first = MakeSample(1, gain);
  auto larger = MakeSample(5, gain);
  return {
      {{.input = std::move(first.input), .expected = std::move(first.expected)},
       {.input = std::move(larger.input),
        .expected = std::move(larger.expected)}}};
}

void Check(const serving::PredictResponse& reply, const Sample& sample,
           std::string_view version) {
  if (reply.runtime.model_version != version ||
      !reply.output.value.isGenericDict())
    throw std::runtime_error("Wrong serving version or output structure");
  const auto actual = reply.output.value.toGenericDict();
  const auto expected = sample.expected.toGenericDict();
  if (actual.size() != expected.size())
    throw std::runtime_error("Output names differ from the scalar oracle");
  for (const auto& field : expected) {
    const auto found = actual.find(field.key());
    if (found == actual.end() || !found->value().isTensor())
      throw std::runtime_error("Missing named output");
    const auto tensor = found->value().toTensor();
    const auto reference = field.value().toTensor();
    if (tensor.device() != at::Device(at::kCPU) ||
        tensor.scalar_type() != at::kFloat ||
        tensor.sizes() != reference.sizes() || !tensor.is_contiguous())
      throw std::runtime_error(
          "Output metadata differs from the scalar oracle");
    for (std::int64_t i = 0; i < tensor.numel(); ++i)
      if (!std::isfinite(tensor.const_data_ptr<float>()[i]) ||
          std::abs(tensor.const_data_ptr<float>()[i] -
                   reference.const_data_ptr<float>()[i]) > 1e-6F)
        throw std::runtime_error(
            "ORT serving differs from the independent scalar oracle");
  }
}

// The owner decoder accepts these exact known descriptor payloads. It also
// checks actual graph names, fixed extents and shared batch symbols, rather
// than treating a matching descriptor hash as a decoded application schema.
serving::OnnxPipelineFactory Owner() {
  return [](const serving::VerifiedBundle& bundle,
            std::span<const ort::TensorSignature> inputs,
            std::span<const ort::TensorSignature> outputs) {
    const auto signatures = [](auto values,
                               std::array<std::string_view, 2> names) {
      if (values.size() != names.size())
        throw std::runtime_error("Example graph I/O count differs");
      for (std::size_t i = 0; i < names.size(); ++i)
        if (values[i].name != names[i] ||
            values[i].shape != std::vector<std::int64_t>{-1, 2} ||
            values[i].batch_symbol != "batch")
          throw std::runtime_error(
              "Example graph signature differs from the owner schema");
    };
    signatures(inputs, {"left", "right"});
    signatures(outputs, {"sum", "difference"});
    const auto bind = [&](const interop::BundleDescriptorRef& ref,
                          std::string_view expected, std::string_view schema) {
      const auto& bytes = bundle.files.at(ref.file_id);
      if (ref.schema_id != schema || ref.schema_version != 1U ||
          bytes.size() != expected.size() ||
          !std::equal(bytes.begin(), bytes.end(), Bytes(expected).begin()))
        throw std::runtime_error(
            "Example owner descriptor does not match its codec");
      return serving::DescriptorBinding{.schema_id = ref.schema_id,
                                        .schema_version = ref.schema_version,
                                        .digest = Digest(bytes)};
    };
    const auto& manifest = bundle.verification.manifest;
    return serving::Pipeline{
        .version = "example-onnx-pair-identity-v1",
        .input =
            bind(manifest.input_schema, kInput, "jyppx.example.onnx_input"),
        .output =
            bind(manifest.output_schema, kOutput, "jyppx.example.onnx_output"),
        .preprocessing =
            bind(manifest.preprocessing, kIdentity, "jyppx.example.identity"),
        .postprocessing =
            bind(manifest.postprocessing, kIdentity, "jyppx.example.identity"),
        .request_schema = serving::onnx_invocation_schema(inputs),
        .response_schema = serving::onnx_result_schema(outputs),
        .preprocess = {},
        .postprocess = {}};
  };
}

ort::SessionOptions SessionOptions() {
  ort::SessionOptions options;
  options.provider.name = "CPUExecutionProvider";
  options.max_model_bytes = 1U << 20U;
  options.max_input_bytes = 64U << 10U;
  options.max_output_bytes = 64U << 10U;
  options.max_batch_size = 8;
  return options;
}

class Bundle final {
 public:
  Bundle(std::string version, float gain) {
    fs::create_directory(temporary.path / "sources");
    directory = temporary.path / "bundle";
    PairModel model(gain);
    model.eval();
    const std::array<at::Tensor, 2> examples{torch::zeros({2, 2}),
                                             torch::ones({2, 2})};
    exporter::ExportOptions export_options;
    export_options.minimum_batch = 1;
    export_options.maximum_batch = 8;
    export_options.max_output_bytes = 1U << 20U;
    export_options.input_names = {"left", "right"};
    export_options.output_names = {"sum", "difference"};
    export_options.dynamic_axes = {{"left", {{0, "batch"}}},
                                   {"right", {{0, "batch"}}},
                                   {"sum", {{0, "batch"}}},
                                   {"difference", {{0, "batch"}}}};
    const auto artifact = exporter::export_module(
        model,
        [&model](std::span<const at::Tensor> values) {
          return model.forward(values[0], values[1]);
        },
        examples, export_options);
    std::string lock_bytes;
    ort::EnvironmentLock environment;
    {
      ort::Session observed(artifact.onnx, artifact.digest, SessionOptions());
      environment = ort::observe_environment(observed);
      lock_bytes = ort::serialize_environment_lock(environment);
    }  // Do not retain an observation Session while loading the serving
       // instance.
    descriptor.backend_id = environment.runtime.backend_id;
    descriptor.backend_version = environment.runtime.backend_version;
    descriptor.format = "onnx";
    descriptor.environment_lock = Digest(Bytes(lock_bytes));
    for (const auto& entry : environment.runtime.dependencies)
      descriptor.dependencies.push_back(
          {.name = entry.name, .version = entry.version});
    descriptor.device = at::Device(at::kCPU);
    descriptor.dynamic_shapes = environment.runtime.dynamic_shapes;
    descriptor.execution_provider = environment.provider.name;
    descriptor.provider_options = environment.provider.options;
    descriptor.allow_cpu_fallback = environment.allow_cpu_fallback;
    // Dynamic graph requests are allowed by SessionOptions. This example does
    // not opt in to DynamicBatcher or native OOM splitting/retry.
    descriptor.tensor_batching = false;
    descriptor.max_batch_size = 1;
    descriptor.oom_retry_safe = false;
    manifest.bundle_id = std::move(version);
    manifest.build = forge::current_build_fingerprint();
    manifest.runtime = {.backend_id = descriptor.backend_id,
                        .backend_version = descriptor.backend_version,
                        .environment_lock = descriptor.environment_lock,
                        .dependencies = descriptor.dependencies,
                        .devices = environment.runtime.devices,
                        .dtypes = environment.runtime.dtypes,
                        .os_arch = environment.runtime.os_arch,
                        .dynamic_shapes = environment.runtime.dynamic_shapes};
    Add("model", interop::BundleFileRole::model, "model.onnx", artifact.onnx);
    manifest.models.push_back({.id = "primary",
                               .format = interop::BundleModelFormat::onnx,
                               .file_ids = {"model"},
                               .tensors = {}});
    manifest.input_schema = Metadata("input", interop::BundleFileRole::schema,
                                     "jyppx.example.onnx_input", kInput);
    manifest.output_schema = Metadata("output", interop::BundleFileRole::schema,
                                      "jyppx.example.onnx_output", kOutput);
    manifest.preprocessing =
        Metadata("preprocess", interop::BundleFileRole::prepost,
                 "jyppx.example.identity", kIdentity);
    manifest.postprocessing =
        Metadata("postprocess", interop::BundleFileRole::prepost,
                 "jyppx.example.identity", kIdentity);
    manifest.evaluation = Metadata(
        "evaluation", interop::BundleFileRole::evaluation,
        "jyppx.example.evaluation",
        R"({"schema_version":1,"reference":"independent scalar arithmetic","trained_model":false})");
    manifest.sbom = Metadata(
        "sbom", interop::BundleFileRole::sbom, "jyppx.example.sbom",
        R"({"schema_version":1,"kind":"demo dependency inventory","versions":"runtime-lock"})");
    manifest.provenance = Metadata(
        "provenance", interop::BundleFileRole::provenance,
        "jyppx.example.provenance",
        R"({"schema_version":1,"origin":"generated local C++ arithmetic model","training_dataset":null})");
    manifest.evidence.push_back(
        Metadata("runtime-lock", interop::BundleFileRole::evidence,
                 std::string(ort::kEnvironmentLockSchemaId), lock_bytes));
    static_cast<void>(
        interop::publish_model_bundle(directory, manifest, sources));
  }
  Bundle(const Bundle&) = delete;
  Bundle& operator=(const Bundle&) = delete;
  fs::path directory;
  serving::RuntimeDescriptor descriptor;

 private:
  TemporaryDirectory temporary;
  interop::ModelBundleManifest manifest;
  std::vector<interop::BundlePublishSource> sources;

  void Add(const std::string& id, interop::BundleFileRole role,
           std::string relative, std::span<const std::byte> bytes) {
    const bool model = role == interop::BundleFileRole::model;
    const auto path = temporary.path / "sources" / id;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) throw std::runtime_error("Cannot stage generated example data");
    manifest.files.push_back(
        {.id = id,
         .relative_path = std::move(relative),
         .role = role,
         .artifact = {.digest = Digest(bytes),
                      .size_bytes = bytes.size(),
                      .logical_name = id,
                      .media_type = model ? "application/onnx"
                                          : "application/octet-stream",
                      .producer = "jyppx.example.onnx_serving"}});
    sources.push_back({.file_id = id, .source_path = path});
  }
  interop::BundleDescriptorRef Metadata(const std::string& id,
                                        interop::BundleFileRole role,
                                        std::string schema,
                                        std::string_view content) {
    Add(id, role, "metadata/" + id + ".json", Bytes(content));
    return {.schema_id = std::move(schema), .schema_version = 1, .file_id = id};
  }
};
}  // namespace

int main() {
  try {
    using namespace std::chrono_literals;
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    Bundle first("example-onnx-pair-v1", 2.0F);
    Bundle next("example-onnx-pair-v2", 5.0F);
    serving::OnnxRuntimeBackend initial_backend(first.descriptor,
                                                SessionOptions(), Owner());
    serving::OnnxRuntimeBackend candidate_backend(next.descriptor,
                                                  SessionOptions(), Owner());
    serving::LoadOptions load;
    load.max_snapshot_bytes = 4U << 20U;
    load.request_limits.max_tensor_bytes = 64U << 10U;
    load.output_limits.max_tensor_bytes = 64U << 10U;
    const auto known_first = KnownVectors(2.0F);
    const auto known_next = KnownVectors(5.0F);
    const auto reservation = initial_backend.load_reservation_bytes(load);
    serving::ReloadablePredictor predictor(
        initial_backend.load(first.directory, {}, known_first, load),
        {.initial_reservation_bytes = reservation,
         .max_resident_bytes = 2 * reservation});
    auto old_lease =
        std::make_unique<serving::PredictionLease>(predictor.acquire());
    const auto publication =
        predictor.reload(candidate_backend, next.directory, {}, known_next,
                         load, {.drain_timeout = 0ms});
    if (publication.previous_drained)
      throw std::runtime_error(
          "An active old-version lease was drained prematurely");
    for (const std::int64_t batch : {1, 2, 5}) {
      const auto old_sample = MakeSample(batch, 2.0F);
      const auto new_sample = MakeSample(batch, 5.0F);
      serving::PredictRequest request;
      request.input = old_sample.input;
      request.context.request_id = "batch-" + std::to_string(batch);
      request.context.project_id = "local-demo";
      request.context.principal_id = "local-operator";
      const auto old_reply = old_lease->predict(request);
      const auto new_reply = predictor.predict(request);
      Check(old_reply, old_sample, "example-onnx-pair-v1");
      Check(new_reply, new_sample, "example-onnx-pair-v2");
      std::cout << "batch=" << batch
                << ": pinned v1 gain=2, current v2 gain=5; "
                << "both named outputs match the independent scalar oracle\n";
    }
    old_lease.reset();
    if (!predictor.drain_previous(std::chrono::steady_clock::now() + 1s))
      throw std::runtime_error("Old ONNX session did not drain");
    const auto state = predictor.statistics();
    if (state.resident_instances != 1U)
      throw std::runtime_error("Old ONNX session remains resident");
    std::cout << "provider=CPUExecutionProvider, resident_instances="
              << state.resident_instances
              << ", peak_reserved_bytes=" << state.peak_reserved_bytes << '\n';
    if (!predictor.close(std::chrono::steady_clock::now() + 1s))
      throw std::runtime_error("Current ONNX serving session did not close");
    std::cout << "C++ ONNX bundle serving, dynamic batches and version drain "
                 "passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
