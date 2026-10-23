// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INSTALL_INTEROP_CONSUMER_CHECKS_HPP
#define JYPPX_LIUDING_INSTALL_INTEROP_CONSUMER_CHECKS_HPP

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>
#include <jyppx/liuding/interop/export_backend.hpp>
#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/interop/weight_import_report.hpp>

#if JYPPX_LIUDING_COMPONENT_NN_ENABLED
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/interop/torchscript_compatibility.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#else
#if __has_include(<jyppx/liuding/interop/model_weight_import.hpp>)
#error "NN OFF package must not install model_weight_import.hpp"
#endif
#if __has_include(<jyppx/liuding/interop/torchscript_compatibility.hpp>)
#error "NN OFF package must not install torchscript_compatibility.hpp"
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace consumer {
using namespace jyppx::liuding;
namespace interop = jyppx::liuding::interop;

inline void Require(bool condition, const char* operation) {
  if (!condition)
    throw InternalError(
        "Installed interop consumer contract failed",
        ErrorDetails{.component = "install.interop_consumer",
                     .operation = operation,
                     .expected = "installed public API contract",
                     .actual = "contract mismatch",
                     .hint = "Inspect the installed package."});
}

inline std::span<const std::byte> Bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

inline ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.algorithm = ArtifactDigestAlgorithm::sha256,
          .hex = sha256_hex(bytes)};
}

inline ArtifactRef Reference(std::string_view content, std::string name,
                             std::string media = "application/json") {
  return {.digest = Digest(Bytes(content)),
          .size_bytes = content.size(),
          .logical_name = std::move(name),
          .media_type = std::move(media),
          .producer = "install.consumer.fixture"};
}

struct TempRoot {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("jyppx-interop-consumer-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  TempRoot() {
    Require(std::filesystem::create_directory(path), "unique-temp-directory");
  }
  ~TempRoot() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

inline void Write(const std::filesystem::path& path,
                  std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  Require(static_cast<bool>(output), "write-fixture");
}

struct TensorFixture {
  // Canonical little-endian bits for float32 [1,2,3] and int64 scalar 7.
  const std::array<std::byte, 12> weight{
      std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x3f},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x40},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x40}, std::byte{0x40}};
  const std::array<std::byte, 8> count{
      std::byte{0x07}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};

  std::vector<interop::SafeTensorSource> Sources() const {
    return {{.name = "weight",
             .dtype = interop::SafeTensorDtype::f32,
             .shape = {3U},
             .data = weight,
             .byte_order = interop::SafeTensorByteOrder::little},
            {.name = "count",
             .dtype = interop::SafeTensorDtype::i64,
             .shape = {},
             .data = count,
             .byte_order = interop::SafeTensorByteOrder::little}};
  }
};

inline std::vector<std::byte> SafeTensors(const TensorFixture& fixture) {
  const auto sources = fixture.Sources();
  auto encoded = interop::serialize_safetensors(
      sources, {{"producer", "install-consumer"}});
  const interop::SafeTensorsView view(encoded);
  Require(view.directory().tensors.size() == 2U &&
              view.directory().payload_size == 20U &&
              view.directory().metadata.at("producer") == "install-consumer",
          "safetensors-directory");
  Require(view.digest() == Digest(encoded), "safetensors-artifact-digest");
  const auto weight = view.tensor_bytes("weight");
  Require(std::equal(weight.begin(), weight.end(), fixture.weight.begin(),
                     fixture.weight.end()),
          "safetensors-weight-bytes");
  Require(
      interop::safetensors_dtype_from_name("F32") ==
              interop::SafeTensorDtype::f32 &&
          interop::safetensors_dtype_name(interop::SafeTensorDtype::i64) ==
              "I64" &&
          interop::safetensors_dtype_size(interop::SafeTensorDtype::i64) == 8U,
      "safetensors-dtype-helpers");
  return encoded;
}

inline std::string WeightPlan(std::span<const std::byte> encoded) {
  const interop::SafeTensorsView view(encoded);
  std::vector<interop::WeightTensorDescriptor> inventory;
  for (const auto& tensor : view.directory().tensors)
    inventory.push_back(
        {.name = tensor.name, .dtype = tensor.dtype, .shape = tensor.shape});
  const auto plan =
      interop::plan_weight_import(view.digest(), inventory, inventory);
  Require(plan.report().ready && plan.report().tensor_coverage == 1.0 &&
              plan.report().element_coverage == 1.0 &&
              plan.assignments().size() == 2U,
          "strict-weight-plan");
  return plan.report_json();
}

inline void Archive(const std::filesystem::path& root) {
  const std::array<std::byte, 8> content{};
  StateManifest state{.participant_id = "consumer.state",
                      .schema_version = 1U,
                      .restore_level = RestoreLevel::exact,
                      .fields = {{.name = "value",
                                  .media_type = "application/octet-stream",
                                  .max_bytes = 8U,
                                  .required = true}}};
  interop::CheckpointArtifactManifest manifest{
      .schema_version = interop::CheckpointArtifactManifest::format_version,
      .build = current_build_fingerprint(),
      .participants = {{.state = state,
                        .blobs = {{.field_name = "value",
                                   .relative_path = "blobs/value.bin",
                                   .size_bytes = content.size(),
                                   .digest = Digest(content)}}}}};
  const auto json = interop::checkpoint_artifact_manifest_json(manifest);
  const auto decoded = interop::parse_checkpoint_artifact_manifest_json(json);
  Require(decoded.participants.size() == 1U &&
              json == interop::checkpoint_artifact_manifest_json(decoded) &&
              interop::make_checkpoint_manifest_capability().feature_id ==
                  "interop.checkpoint_manifest",
          "checkpoint-manifest");
  const auto path = root / "last.bin";
  {
    interop::FileStateArchive archive(path);
    archive.begin({state});
    archive.writer_for(state.participant_id).write_blob("value", content);
    archive.rollback();
    archive.begin({state});
    archive.writer_for(state.participant_id).write_blob("value", content);
    archive.commit();
  }
  interop::FileStateArchive archive(path, {.read_only = true});
  const auto restored =
      archive.reader_for(state.participant_id).read_blob("value", 8U);
  Require(archive.path() == path &&
              archive.manifest().participants.size() == 1U &&
              std::equal(restored.begin(), restored.end(), content.begin(),
                         content.end()),
          "file-state-archive-roundtrip");
}

inline void Bundle(const std::filesystem::path& root,
                   const TensorFixture& fixture,
                   std::string_view weight_report) {
  interop::ModelBundleManifest manifest;
  manifest.bundle_id = "installed-consumer-v1";
  manifest.build = current_build_fingerprint();
  auto runtime_platform = manifest.build.os_arch;
  for (auto& value : runtime_platform)
    if (value >= 'A' && value <= 'Z')
      value = static_cast<char>(value - 'A' + 'a');
  manifest.runtime = {
      .backend_id = "consumer.safetensors",
      .backend_version = "1.0.0",
      .environment_lock = Digest(Bytes("install-consumer-lock-v1")),
      .dependencies = {{.name = "safetensors", .version = "0.6.2"}},
      .devices = {"cpu"},
      .dtypes = {"float32", "int64"},
      .os_arch = {runtime_platform},
      .dynamic_shapes = false};
  std::vector<interop::BundlePublishSource> sources;
  interop::BundleModel model{.id = "primary",
                             .format = interop::BundleModelFormat::safetensors,
                             .file_ids = {},
                             .tensors = {}};
  const auto tensors = fixture.Sources();
  auto shards = interop::shard_safetensors_for_bundle(tensors, 128U);
  Require(shards.size() == 2U, "bounded-whole-tensor-shards");
  for (auto& shard : shards) {
    const auto path = root / "sources" / (shard.file.id + ".data");
    Write(path, shard.bytes);
    sources.push_back({.file_id = shard.file.id, .source_path = path});
    model.file_ids.push_back(shard.file.id);
    model.tensors.insert(model.tensors.end(), shard.tensors.begin(),
                         shard.tensors.end());
    manifest.files.push_back(std::move(shard.file));
  }
  manifest.models.push_back(std::move(model));
  const auto descriptor = [&](std::string id, interop::BundleFileRole role,
                              std::string_view content) {
    const auto source = root / "sources" / (id + ".data");
    Write(source, Bytes(content));
    sources.push_back({.file_id = id, .source_path = source});
    manifest.files.push_back({.id = id,
                              .relative_path = "metadata/" + id + ".json",
                              .role = role,
                              .artifact = Reference(content, id)});
    return interop::BundleDescriptorRef{
        .schema_id = "consumer." + id, .schema_version = 1U, .file_id = id};
  };
  manifest.input_schema = descriptor(
      "input", interop::BundleFileRole::schema,
      R"({"schema_id":"consumer.input","schema_version":1,"tensor":"float32[3]"})");
  manifest.output_schema = descriptor(
      "output", interop::BundleFileRole::schema,
      R"({"schema_id":"consumer.output","schema_version":1,"tensor":"float32[3]"})");
  manifest.preprocessing = descriptor(
      "preprocess", interop::BundleFileRole::prepost,
      R"({"schema_id":"consumer.preprocess","schema_version":1,"operation":"identity"})");
  manifest.postprocessing = descriptor(
      "postprocess", interop::BundleFileRole::prepost,
      R"({"schema_id":"consumer.postprocess","schema_version":1,"operation":"identity"})");
  manifest.labels = descriptor(
      "labels", interop::BundleFileRole::labels,
      R"({"schema_id":"consumer.labels","schema_version":1,"labels":["sample"]})");
  manifest.evaluation = descriptor(
      "evaluation", interop::BundleFileRole::evaluation,
      R"({"schema_id":"consumer.evaluation","schema_version":1,"status":"fixture-only"})");
  manifest.sbom = descriptor(
      "sbom", interop::BundleFileRole::sbom,
      R"({"schema_id":"consumer.sbom","schema_version":1,"status":"fixture-only"})");
  manifest.provenance = descriptor(
      "provenance", interop::BundleFileRole::provenance,
      R"({"schema_id":"consumer.provenance","schema_version":1,"producer":"install-consumer"})");
  manifest.weight_import_reports = {
      descriptor("weight-import", interop::BundleFileRole::weight_import_report,
                 weight_report)};
  interop::validate_model_bundle_manifest(manifest);
  const auto json = interop::serialize_model_bundle_manifest(manifest);
  const auto restored = interop::deserialize_model_bundle_manifest(json);
  Require(interop::serialize_model_bundle_manifest(restored) == json,
          "complete-bundle-manifest-roundtrip");
  const auto payload = interop::model_bundle_signing_payload(restored);
  Require(payload.starts_with("JYPPX-MODEL-BUNDLE-SIGNATURE-V1\n") &&
              payload.find("\"signatures\"") == std::string::npos,
          "domain-separated-signing-payload");
  const auto published =
      interop::publish_model_bundle(root / "bundle", restored, sources);
  const auto verified = interop::verify_model_bundle_directory(root / "bundle");
  std::uint64_t declared_bytes = 0U;
  for (const auto& file : restored.files)
    declared_bytes += file.artifact.size_bytes;
  Require(
      published.manifest_digest == verified.manifest_digest &&
          verified.verified_tensors == 2U &&
          verified.verified_file_bytes == declared_bytes &&
          verified.integrity_only_models == 0U &&
          verified.signature_status ==
              interop::BundleSignatureStatus::unsigned_bundle &&
          verified.manifest.files.size() == 11U &&
          interop::serialize_model_bundle_manifest(verified.manifest) == json,
      "published-exhaustive-bundle-verification");
  interop::ModelBundleVerifyOptions options;
  options.require_trusted_signature = true;
  bool rejected = false;
  try {
    static_cast<void>(
        interop::verify_model_bundle_directory(root / "bundle", options));
  } catch (const CompatibilityError&) {
    rejected = true;
  }
  Require(rejected, "unsigned-bundle-cannot-become-trusted");
}

inline void BackendProtocol() {
  const std::array<interop::BackendDependency, 6> declared{
      {{.name = "python", .version = "3.12.8"},
       {.name = "torch", .version = "2.5.1+cpu"},
       {.name = "onnx", .version = "1.17.0"},
       {.name = "onnxscript", .version = "0.1.0"},
       {.name = "onnxruntime", .version = "1.20.1"},
       {.name = "safetensors", .version = "0.6.2"}}};
  const auto lock = Digest(Bytes("declared-protocol-fixture-lock"));
  const auto descriptor =
      interop::make_onnx_sidecar_descriptor("1.0.0", lock, declared);
  interop::validate_backend_descriptor(descriptor);
  interop::BackendRegistry exporters(interop::BackendRole::exporter);
  exporters.add(descriptor);
  Require(exporters.find(descriptor.backend_id) != nullptr &&
              exporters.find("missing") == nullptr &&
              interop::serialize_backend_descriptor(descriptor)
                      .find("declared-unprobed") != std::string::npos,
          "explicit-backend-catalog");
  const auto compat = interop::make_torchscript_compatibility_descriptor(
      "1.0.0", lock, current_build_fingerprint().libtorch_version);
  interop::BackendRegistry runtimes(interop::BackendRole::runtime);
  runtimes.add(compat);
  Require(runtimes.find(compat.backend_id) != nullptr &&
              compat.compatibility_only && compat.deprecated,
          "deprecated-runtime-catalog");
  // Unexecuted protocol fixtures: no imported factory, official checker or ORT.
  interop::ExportRequest request;
  request.request_id = "installed-codec-fixture";
  request.backend_id = descriptor.backend_id;
  request.source_kind = interop::ExportSourceKind::python_factory;
  request.source = Reference("fixture-code-reference", "fixture-code");
  request.state = Reference("fixture-state-reference", "fixture-state");
  request.factory_entry_point = "consumer_fixture:factory";
  request.invocation_schema =
      Reference("fixture-input-schema", "fixture-schema");
  request.verification_cases = {Reference("fixture-case", "fixed-case")};
  request.trust_source = true;
  request.environment_lock = lock;
  request.options = {{"opset", "18"}};
  const auto encoded = interop::serialize_sidecar_request(request);
  request = interop::deserialize_sidecar_request(encoded);
  Require(
      interop::serialize_sidecar_request(request) == encoded &&
          interop::backend_request_digest(request) == Digest(Bytes(encoded)) &&
          interop::preflight_backend(descriptor, request).decision ==
              interop::SupportDecision::conditional,
      "request-codec-unprobed");
  interop::BackendProbe probe;
  probe.backend_id = descriptor.backend_id;
  probe.backend_version = descriptor.backend_version;
  probe.protocol_version = descriptor.protocol_version;
  probe.environment_lock = lock;
  probe.dependencies = descriptor.dependencies;
  probe.operations = descriptor.operations;
  probe.devices = {"cpu"};
  probe.dtypes = {"float32"};
  probe.evidence = Reference("unexecuted codec fixture", "unexecuted-probe");
  const auto probe_json = interop::serialize_sidecar_probe(probe);
  probe = interop::deserialize_sidecar_probe(probe_json);
  Require(
      interop::serialize_sidecar_probe(probe) == probe_json &&
          interop::preflight_backend(descriptor, request, &probe).decision !=
              interop::SupportDecision::supported,
      "probe-codec-is-not-proof");
  interop::ExportCandidate candidate;
  candidate.request_id = request.request_id;
  candidate.backend_id = descriptor.backend_id;
  candidate.backend_version = descriptor.backend_version;
  candidate.artifact = Reference("not-an-onnx-model", "unexecuted-output");
  candidate.verification.request_digest =
      interop::backend_request_digest(request);
  candidate.verification.artifact_digest = candidate.artifact.digest;
  candidate.verification.environment_lock = lock;
  candidate.verification.evidence =
      Reference("no verification executed", "unexecuted-validation");
  candidate.verification.absolute_tolerance = request.absolute_tolerance;
  candidate.verification.relative_tolerance = request.relative_tolerance;
  const auto candidate_json = interop::serialize_sidecar_candidate(candidate);
  candidate = interop::deserialize_sidecar_candidate(candidate_json);
  Require(interop::serialize_sidecar_candidate(candidate) == candidate_json,
          "candidate-codec");
  bool rejected = false;
  try {
    interop::validate_export_candidate_metadata(descriptor, request, probe,
                                                candidate);
  } catch (const CompatibilityError&) {
    rejected = true;
  }
  Require(rejected, "unexecuted-onnx-candidate-cannot-publish");
}

#if JYPPX_LIUDING_COMPONENT_NN_ENABLED
inline void OptionalNN(std::span<const std::byte> weights) {
  struct Model final : torch::nn::Module {
    torch::Tensor weight = register_parameter("weight", torch::zeros({3}));
    torch::Tensor count =
        register_buffer("count", torch::tensor(0, torch::kInt64));
  };
  auto model = std::make_shared<Model>();
  nn::ModelStateView state(model);
  const auto imported =
      interop::import_model_weights(state, weights, Digest(weights));
  Require(imported.applied && imported.applied_tensors == 2U &&
              torch::equal(model->weight, torch::tensor({1.0F, 2.0F, 3.0F})) &&
              model->count.item<std::int64_t>() == 7,
          "installed-nn-staged-weight-import");
  torch::jit::Module scripted("InstalledCompatibilityFixture");
  scripted.register_buffer("bias", model->weight.detach().clone());
  scripted.define(R"JIT(
def forward(self, x: Tensor, scale: float = 2.0) -> Tensor:
    return (x + self.bias) * scale
)JIT");
  std::ostringstream stream(std::ios::out | std::ios::binary);
  scripted.save(stream);
  const auto archive = stream.str();
  auto scale = ValueSchema{};
  scale.kind = ValueKind::floating;
  scale.required = false;
  const nn::InvocationSchema schema{.positional = {ValueSchema::tensor()},
                                    .keyword = {{"scale", scale}},
                                    .allow_extra_keywords = false};
  const auto digest = Digest(Bytes(archive));
  bool rejected = false;
  try {
    static_cast<void>(interop::TorchScriptCompatibilityRuntime::load(
        Bytes(archive), digest, schema, ValueSchema::tensor()));
  } catch (const CompatibilityError&) {
    rejected = true;
  }
  Require(rejected, "installed-runtime-requires-explicit-trust");
  interop::TorchScriptLoadOptions options;
  options.trust_source = true;
  options.device = Device(c10::kCPU);
  auto runtime = interop::TorchScriptCompatibilityRuntime::load(
      Bytes(archive), digest, schema, ValueSchema::tensor(), options);
  const auto output =
      runtime.invoke({.positional = {torch::ones({3})}, .keyword = {}});
  Require(torch::equal(output.value.toTensor(),
                       torch::tensor({4.0F, 6.0F, 8.0F})) &&
              runtime.inspect().artifact_digest == digest &&
              runtime.inspect().open && runtime.inspect().device == "cpu",
          "installed-real-jit-cpu-forward");
  runtime.close();
  Require(!runtime.inspect().open &&
              torch::equal(output.value.toTensor(),
                           torch::tensor({4.0F, 6.0F, 8.0F})),
          "installed-runtime-close-retains-output-snapshot");
}
#endif

inline void Run() {
  TempRoot root;
  TensorFixture fixture;
  const auto weights = SafeTensors(fixture);
  const auto report = WeightPlan(weights);
  Archive(root.path);
  Bundle(root.path, fixture, report);
  BackendProtocol();
#if JYPPX_LIUDING_COMPONENT_NN_ENABLED
  OptionalNN(weights);
#endif
}

}  // namespace consumer
#endif  // JYPPX_LIUDING_INSTALL_INTEROP_CONSUMER_CHECKS_HPP
