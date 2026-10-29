// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../serving/detail.hpp"

#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/deployment/onnx_serving/backend.hpp>

#include <ATen/core/Dict.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <utility>

namespace jyppx::liuding::serving {
namespace {
namespace ort = onnx_runtime;

[[noreturn]] void Mismatch(std::string operation) {
  throw CompatibilityError(
      "ONNX Serving declaration does not match the verified execution point",
      detail::Details(std::move(operation),
                      "exact owner schema, runtime and provider"));
}

ValueSchema TensorContract(const ort::TensorSignature& signature) {
  if (signature.name.empty() || signature.name.size() > 4096U ||
      signature.shape.empty() || signature.shape.size() > 8U)
    detail::Invalid("onnx.signature", "bounded named rank 1..8 graph tensor");
  std::vector<std::optional<std::int64_t>> shape;
  for (std::size_t i = 0; i < signature.shape.size(); ++i) {
    const auto dimension = signature.shape[i];
    if (dimension == -1 && i == 0 && !signature.batch_symbol.empty())
      shape.push_back(std::nullopt);
    else if (dimension > 0)
      shape.push_back(dimension);
    else
      detail::Invalid("onnx.signature",
                      "positive extents or a named dynamic leading batch");
  }
  if ((signature.shape.front() == -1) != !signature.batch_symbol.empty())
    detail::Invalid("onnx.signature",
                    "one exact leading dynamic batch declaration");
  return ValueSchema::tensor(
      {.dtype = torch::kFloat32,
       .device = Device(c10::kCPU),
       .rank = static_cast<std::int64_t>(signature.shape.size()),
       .shape = std::move(shape)},
      signature.name);
}

bool SameTensor(const ValueSchema& owner, const ValueSchema& graph) {
  if (owner.kind != ValueKind::tensor || !owner.required || owner.nullable ||
      !owner.tensor_constraints || !owner.elements.empty() ||
      !owner.fields.empty())
    return false;
  const auto& a = *owner.tensor_constraints;
  const auto& b = *graph.tensor_constraints;
  return a.dtype == b.dtype && a.device == b.device && a.rank == b.rank &&
         a.shape == b.shape;
}

void GraphBoundary(const Pipeline& pipeline, const ort::Session& session) {
  nn::validate_invocation_schema(pipeline.request_schema);
  validate_value_schema(pipeline.response_schema);
  const auto input = onnx_invocation_schema(session.inputs());
  const auto output = onnx_result_schema(session.outputs());
  if (!pipeline.request_schema.positional.empty() ||
      pipeline.request_schema.allow_extra_keywords ||
      pipeline.request_schema.keyword.size() != input.keyword.size())
    Mismatch("onnx.owner.input");
  for (const auto& [name, schema] : input.keyword) {
    const auto found = pipeline.request_schema.keyword.find(name);
    if (found == pipeline.request_schema.keyword.end() ||
        !SameTensor(found->second, schema))
      Mismatch("onnx.owner.input");
  }
  const auto& owner = pipeline.response_schema;
  if (owner.kind != ValueKind::dictionary || !owner.required ||
      owner.nullable || owner.allow_extra_fields ||
      owner.fields.size() != output.fields.size())
    Mismatch("onnx.owner.output");
  for (const auto& [name, schema] : output.fields) {
    const auto found = owner.fields.find(name);
    if (found == owner.fields.end() || !SameTensor(found->second, schema))
      Mismatch("onnx.owner.output");
  }
}

ort::EnvironmentRequirements Requirements(
    const interop::BundleRuntimeRequirements& runtime) {
  ort::EnvironmentRequirements result;
  result.backend_id = runtime.backend_id;
  result.backend_version = runtime.backend_version;
  for (const auto& dependency : runtime.dependencies)
    result.dependencies.push_back({dependency.name, dependency.version});
  result.devices = runtime.devices;
  result.dtypes = runtime.dtypes;
  result.os_arch = runtime.os_arch;
  result.dynamic_shapes = runtime.dynamic_shapes;
  return result;
}

void Declaration(const RuntimeDescriptor& descriptor,
                 const interop::BundleRuntimeRequirements& runtime) {
  if (runtime.backend_id != descriptor.backend_id ||
      runtime.backend_version != descriptor.backend_version ||
      runtime.environment_lock != descriptor.environment_lock ||
      runtime.dynamic_shapes != descriptor.dynamic_shapes)
    Mismatch("onnx.runtime");
  auto expected = descriptor.dependencies, actual = runtime.dependencies;
  const auto order = [](const auto& a, const auto& b) {
    return a.name < b.name;
  };
  std::sort(expected.begin(), expected.end(), order);
  std::sort(actual.begin(), actual.end(), order);
  if (expected.size() != actual.size()) Mismatch("onnx.dependencies");
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (expected[i].name != actual[i].name ||
        expected[i].version != actual[i].version)
      Mismatch("onnx.dependencies");
}

const interop::BundleFile& File(const interop::ModelBundleManifest& manifest,
                                std::string_view id) {
  const auto found =
      std::find_if(manifest.files.begin(), manifest.files.end(),
                   [&](const auto& file) { return file.id == id; });
  if (found == manifest.files.end()) Mismatch("onnx.file");
  return *found;
}

const interop::BundleDescriptorRef& Lock(
    const interop::ModelBundleManifest& manifest) {
  const interop::BundleDescriptorRef* result = nullptr;
  for (const auto& ref : manifest.evidence) {
    if (ref.schema_id != ort::kEnvironmentLockSchemaId) continue;
    if (result || ref.schema_version != 1U) Mismatch("onnx.environment_lock");
    result = &ref;
  }
  if (!result) Mismatch("onnx.environment_lock");
  return *result;
}

std::string GraphContract(const ort::Session& session) {
  std::string result = "onnx-named-fp32-v1;";
  const auto append = [&](std::string_view text) {
    result += std::to_string(text.size()) + ":";
    result.append(text);
    result += ';';
  };
  for (const auto signatures : {session.inputs(), session.outputs()}) {
    append(std::to_string(signatures.size()));
    std::vector<const ort::TensorSignature*> ordered;
    for (const auto& signature : signatures) ordered.push_back(&signature);
    std::sort(ordered.begin(), ordered.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });
    for (const auto* signature : ordered) {
      append(signature->name);
      append(signature->batch_symbol);
      append(std::to_string(signature->shape.size()));
      for (const auto dimension : signature->shape)
        append(std::to_string(dimension));
    }
  }
  return result;
}

nn::InvocationResult Invoke(const ort::Session& session,
                            const nn::Invocation& input, std::stop_token stop,
                            const Deadline& deadline) {
  detail::Boundary(stop, deadline);
  const auto signature = onnx_invocation_schema(session.inputs());
  nn::validate_invocation(input, signature);
  std::vector<ort::TensorInput> tensors;
  tensors.reserve(session.inputs().size());
  for (const auto& graph : session.inputs()) {
    const auto& tensor = input.keyword.at(graph.name).toTensor();
    if (!tensor.is_contiguous())
      detail::Invalid("onnx.input", "owned contiguous CPU FP32 input snapshot");
    tensors.push_back(
        {.name = graph.name,
         .values = std::span(tensor.const_data_ptr<float>(),
                             static_cast<std::size_t>(tensor.numel())),
         .shape = std::span(tensor.sizes().data(), tensor.sizes().size())});
  }
  auto output = session.run(tensors, stop, deadline);
  c10::Dict<std::string, Tensor> values;
  for (auto& result : output) {
    detail::Boundary(stop, deadline);
    // The vector is already an independent, pre-bounded Session output.
    // Transfer its ownership into Tensor storage instead of copying it into
    // another buffer.
    auto storage =
        std::make_shared<std::vector<float>>(std::move(result.values));
    auto tensor = torch::from_blob(
        storage->data(), result.shape,
        [storage](void*) { static_cast<void>(storage); },
        torch::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU));
    values.insert(result.name, std::move(tensor));
  }
  return {.value = std::move(values),
          .schema = onnx_result_schema(session.outputs())};
}

void InputBoundary(const ort::Session& session, const nn::Invocation& input,
                   std::uint64_t max_batch_size) {
  nn::validate_invocation(input, onnx_invocation_schema(session.inputs()));
  std::map<std::string, std::int64_t> batches;
  for (const auto& graph : session.inputs()) {
    const auto& tensor = input.keyword.at(graph.name).toTensor();
    const auto batch = tensor.size(0);
    if (batch <= 0 || static_cast<std::uint64_t>(batch) > max_batch_size)
      detail::Resource("onnx.input.batch");
    if (!tensor.is_contiguous())
      detail::Invalid("onnx.input",
                      "contiguous graph inputs after trusted preprocessing");
    if (!graph.batch_symbol.empty()) {
      const auto [position, inserted] =
          batches.emplace(graph.batch_symbol, batch);
      if (!inserted && position->second != batch)
        detail::Invalid("onnx.input.batch",
                        "equal sizes for one declared batch symbol");
    }
  }
}
}  // namespace

nn::InvocationSchema onnx_invocation_schema(
    std::span<const ort::TensorSignature> signatures) {
  if (signatures.empty() || signatures.size() > 4096U)
    detail::Invalid("onnx.signature", "nonempty bounded named graph inputs");
  nn::InvocationSchema result;
  for (const auto& signature : signatures)
    if (!result.keyword.emplace(signature.name, TensorContract(signature))
             .second)
      detail::Invalid("onnx.signature", "unique graph input names");
  return result;
}

ValueSchema onnx_result_schema(
    std::span<const ort::TensorSignature> signatures) {
  if (signatures.empty() || signatures.size() > 4096U)
    detail::Invalid("onnx.signature", "nonempty bounded named graph outputs");
  std::map<std::string, ValueSchema> fields;
  for (const auto& signature : signatures)
    if (!fields.emplace(signature.name, TensorContract(signature)).second)
      detail::Invalid("onnx.signature", "unique graph output names");
  return ValueSchema::dictionary(std::move(fields), false, "onnx.outputs");
}

OnnxRuntimeBackend::OnnxRuntimeBackend(RuntimeDescriptor descriptor,
                                       ort::SessionOptions session,
                                       OnnxPipelineFactory factory,
                                       std::string model_id)
    : descriptor_(std::move(descriptor)),
      session_(std::move(session)),
      factory_(std::move(factory)),
      model_id_(std::move(model_id)) {
  detail::ValidateRuntime(descriptor_);
  detail::Text(model_id_, true);
  if (!factory_ || descriptor_.backend_id != ort::kEnvironmentBackendId ||
      descriptor_.format != "onnx" || descriptor_.device != Device(c10::kCPU) ||
      descriptor_.dtypes != std::vector<Dtype>{torch::kFloat32} ||
      descriptor_.tensor_batching || descriptor_.oom_retry_safe ||
      descriptor_.execution_provider != session_.provider.name ||
      descriptor_.provider_options != session_.provider.options ||
      descriptor_.allow_cpu_fallback != session_.allow_cpu_fallback ||
      !session_.profile_prefix.empty())
    detail::Invalid("onnx.backend",
                    "explicit named FP32 host I/O, exact EP/options/fallback, "
                    "no native retry/batching or implicit profile files");
}

RuntimeDescriptor OnnxRuntimeBackend::descriptor() const { return descriptor_; }

std::uint64_t OnnxRuntimeBackend::load_reservation_bytes(
    const LoadOptions& options) const {
  if (!options.max_snapshot_bytes) detail::Resource("reload.onnx.snapshot");
  return options.max_snapshot_bytes;
}

std::shared_ptr<Predictor> OnnxRuntimeBackend::load(
    const std::filesystem::path& directory,
    const interop::ModelBundleVerifyOptions& verify,
    std::span<const WarmupCase> warmup, const LoadOptions& options) const {
  return detail::Backend("onnx.load", [&] {
    detail::ValidateLoad(options);
    static_cast<void>(load_reservation_bytes(options));
    if (warmup.empty() || warmup.size() > options.max_warmup_cases)
      detail::Invalid("warmup", "at least one bounded known-vector case");
    detail::Boundary(options.stop, options.deadline);
    const auto metadata =
        interop::verify_model_bundle_directory(directory, verify);
    Declaration(descriptor_, metadata.manifest.runtime);
    const auto& models = metadata.manifest.models;
    const auto model =
        std::find_if(models.begin(), models.end(),
                     [&](const auto& item) { return item.id == model_id_; });
    if (model == models.end() ||
        model->format != interop::BundleModelFormat::onnx ||
        model->file_ids.size() != 1U)
      Mismatch("onnx.model");
    const auto model_file = File(metadata.manifest, model->file_ids.front());
    if (model_file.artifact.size_bytes > session_.max_model_bytes)
      detail::Resource("onnx.model.bytes");
    std::vector<std::string_view> ids;
    ids.reserve(metadata.manifest.files.size());
    for (const auto& file : metadata.manifest.files) ids.push_back(file.id);
    auto snapshot = interop::read_model_bundle_snapshot(
        directory, metadata.manifest_digest, ids, verify,
        options.max_snapshot_bytes);
    VerifiedBundle bundle{.verification = std::move(snapshot.verification),
                          .files = std::move(snapshot.files)};
    // No path is reopened for Session, descriptors or environment checks.
    detail::Boundary(options.stop, options.deadline);
    auto session_options = session_;
    session_options.max_input_bytes =
        std::min(session_options.max_input_bytes,
                 options.request_limits.max_tensor_bytes);
    session_options.max_output_bytes =
        std::min(session_options.max_output_bytes,
                 options.output_limits.max_tensor_bytes);
    auto session = std::make_shared<ort::Session>(
        bundle.files.at(model_file.id), model_file.artifact.digest,
        session_options);
    detail::Boundary(options.stop, options.deadline);
    const auto& manifest = bundle.verification.manifest;
    const auto& lock = Lock(manifest);
    const auto& encoded = bundle.files.at(lock.file_id);
    static_cast<void>(ort::verify_environment_lock(
        *session,
        {reinterpret_cast<const char*>(encoded.data()), encoded.size()},
        manifest.runtime.environment_lock, Requirements(manifest.runtime)));
    if (session->execution_provider().name != descriptor_.execution_provider ||
        session->execution_provider().options != descriptor_.provider_options ||
        session->cpu_fallback_allowed() != descriptor_.allow_cpu_fallback)
      Mismatch("onnx.provider");
    auto pipeline = factory_(bundle, session->inputs(), session->outputs());
    detail::Boundary(options.stop, options.deadline);
    GraphBoundary(pipeline, *session);
    detail::ValidateBinding(pipeline.input, manifest.input_schema, bundle);
    detail::ValidateBinding(pipeline.output, manifest.output_schema, bundle);
    detail::ValidateBinding(pipeline.preprocessing, manifest.preprocessing,
                            bundle);
    detail::ValidateBinding(pipeline.postprocessing, manifest.postprocessing,
                            bundle);
    auto preprocess = std::move(pipeline.preprocess);
    const auto maximum_batch = session_options.max_batch_size;
    pipeline.preprocess = [session, preprocess = std::move(preprocess),
                           maximum_batch](const nn::Invocation& input) {
      auto processed = preprocess ? preprocess(input) : input;
      InputBoundary(*session, processed, maximum_batch);
      return processed;
    };
    auto bounded = options;
    bounded.request_limits.max_tensor_bytes = session_options.max_input_bytes;
    bounded.output_limits.max_tensor_bytes = session_options.max_output_bytes;
    return detail::PredictorFactory::Create(
        descriptor_, manifest.bundle_id, std::move(pipeline),
        [session](const nn::Invocation& input, std::stop_token stop,
                  const Deadline& deadline) {
          return Invoke(*session, input, stop, deadline);
        },
        [session] { static_cast<void>(session); }, warmup, bounded, true,
        GraphContract(*session));
  });
}
}  // namespace jyppx::liuding::serving
