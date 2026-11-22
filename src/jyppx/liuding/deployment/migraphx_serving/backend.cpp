// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../serving/detail.hpp"

#include <jyppx/liuding/deployment/migraphx_serving/backend.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <limits>

namespace jyppx::liuding::serving {
namespace {
namespace mgx = migraphx_runtime;
[[noreturn]] void Mismatch(std::string operation) {
  throw CompatibilityError(
      "MIGraphX Serving execution contract mismatch",
      detail::Details(std::move(operation),
                      "exact verified fixed GPU environment and owner schema"));
}
std::vector<std::int64_t> Dimensions(std::span<const std::size_t> shape) {
  if (shape.empty() || shape.size() > 8U)
    detail::Invalid("migraphx.shape", "positive rank 1..8");
  std::vector<std::int64_t> result;
  std::uint64_t count = 1;
  for (const auto dimension : shape) {
    if (!dimension ||
        dimension > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()) /
                        sizeof(float) / count)
      detail::Invalid("migraphx.shape",
                      "positive representable FP32 dimensions");
    count *= dimension;
    result.push_back(static_cast<std::int64_t>(dimension));
  }
  return result;
}
ValueSchema Tensor(std::span<const std::size_t> shape) {
  const auto dimensions = Dimensions(shape);
  std::vector<std::optional<std::int64_t>> fixed;
  for (const auto dimension : dimensions) fixed.push_back(dimension);
  return ValueSchema::tensor({.dtype = torch::kFloat32,
                              .device = Device(c10::kCPU),
                              .rank = static_cast<std::int64_t>(shape.size()),
                              .shape = std::move(fixed)});
}
std::vector<interop::BundleDependency> Dependencies(
    const mgx::EnvironmentLock& environment) {
  return {{.name = "hip-headers", .version = environment.hip_header_version},
          {.name = "hip-runtime",
           .version = std::to_string(environment.hip_runtime_version)},
          {.name = "libtorch", .version = environment.libtorch_version},
          {.name = "liuding-forge", .version = environment.adapter_version},
          {.name = "migraphx-headers",
           .version = environment.migraphx_header_version}};
}
void SameDependencies(std::vector<interop::BundleDependency> a,
                      std::vector<interop::BundleDependency> b) {
  const auto order = [](const auto& x, const auto& y) {
    return x.name < y.name;
  };
  std::sort(a.begin(), a.end(), order);
  std::sort(b.begin(), b.end(), order);
  if (a.size() != b.size()) Mismatch("migraphx.dependencies");
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].name != b[i].name || a[i].version != b[i].version ||
        (i && a[i].name == a[i - 1].name))
      Mismatch("migraphx.dependencies");
}
const interop::BundleDescriptorRef& Lock(
    const interop::ModelBundleManifest& manifest) {
  const interop::BundleDescriptorRef* selected = nullptr;
  for (const auto& ref : manifest.evidence) {
    if (ref.schema_id != mgx::kEnvironmentLockSchemaId) continue;
    if (selected || ref.schema_version != 1U)
      Mismatch("migraphx.environment_lock");
    selected = &ref;
  }
  if (!selected) Mismatch("migraphx.environment_lock");
  return *selected;
}
void GraphBoundary(const Pipeline& pipeline, const mgx::Session& session) {
  nn::validate_invocation_schema(pipeline.request_schema);
  validate_value_schema(pipeline.response_schema);
  const auto input = migraphx_invocation_schema(session.options().input_name,
                                                session.options().input_shape);
  if (!pipeline.request_schema.positional.empty() ||
      pipeline.request_schema.allow_extra_keywords ||
      pipeline.request_schema.keyword.size() != 1U)
    Mismatch("migraphx.owner.input");
  const auto found =
      pipeline.request_schema.keyword.find(session.options().input_name);
  if (found == pipeline.request_schema.keyword.end() ||
      canonical_value_schema(found->second) !=
          canonical_value_schema(input.keyword.begin()->second) ||
      canonical_value_schema(pipeline.response_schema) !=
          canonical_value_schema(
              migraphx_result_schema(session.output_shape())))
    Mismatch("migraphx.owner.schema");
}
void InputBoundary(const mgx::Session& session, const nn::Invocation& input) {
  nn::validate_invocation(
      input, migraphx_invocation_schema(session.options().input_name,
                                        session.options().input_shape));
  if (!input.keyword.at(session.options().input_name)
           .toTensor()
           .is_contiguous())
    detail::Invalid("migraphx.input",
                    "contiguous CPU FP32 after trusted preprocessing");
}
nn::InvocationResult Invoke(const mgx::Session& session,
                            const nn::Invocation& input, std::stop_token stop,
                            const Deadline& deadline) {
  detail::Boundary(stop, deadline);
  InputBoundary(session, input);
  const auto& tensor =
      input.keyword.at(session.options().input_name).toTensor();
  auto output =
      session.run(std::span(tensor.const_data_ptr<float>(),
                            static_cast<std::size_t>(tensor.numel())));
  // Native run completes host-copy/synchronization and owns its result before
  // a cooperative interruption can discard it. No GPU work is detached.
  detail::Boundary(stop, deadline);
  const auto dimensions = Dimensions(output.shape);
  auto storage = std::make_shared<std::vector<float>>(std::move(output.values));
  auto value = torch::from_blob(
      storage->data(), dimensions,
      [storage](void*) { static_cast<void>(storage); },
      torch::TensorOptions().dtype(torch::kFloat32).device(c10::kCPU));
  return {.value = std::move(value),
          .schema = migraphx_result_schema(output.shape)};
}
std::string Contract(const mgx::Session& session) {
  const auto input = migraphx_invocation_schema(session.options().input_name,
                                                session.options().input_shape);
  const auto& name = session.options().input_name;
  return "migraphx-fixed-fp32-v1;" + std::to_string(name.size()) + ":" + name +
         ";" + canonical_value_schema(input.keyword.begin()->second) + ";" +
         canonical_value_schema(migraphx_result_schema(session.output_shape()));
}
}  // namespace

nn::InvocationSchema migraphx_invocation_schema(
    std::string_view name, std::span<const std::size_t> shape) {
  if (name.empty() || name.size() > 255U ||
      std::any_of(name.begin(), name.end(),
                  [](unsigned char c) { return c < 32 || c > 126; }))
    detail::Invalid("migraphx.input.name",
                    "bounded nonempty printable ASCII graph name");
  nn::InvocationSchema result;
  result.keyword.emplace(std::string(name), Tensor(shape));
  return result;
}
ValueSchema migraphx_result_schema(std::span<const std::size_t> shape) {
  return Tensor(shape);
}
RuntimeDescriptor migraphx_runtime_descriptor(
    const mgx::EnvironmentLock& environment) {
  const auto lock = mgx::serialize_environment_lock(environment);
  RuntimeDescriptor result;
  result.backend_id = kMIGraphXBackendId;
  result.backend_version = environment.adapter_version;
  result.format = "onnx";
  result.environment_lock = {
      .hex = sha256_hex(std::as_bytes(std::span(lock.data(), lock.size())))};
  result.dependencies = Dependencies(environment);
  result.dynamic_shapes = false;
  result.execution_provider = "migraphx.gpu";
  result.provider_options = {
      {"device_id", std::to_string(environment.device_id)}};
  detail::ValidateRuntime(result);
  return result;
}
MIGraphXRuntimeBackend::MIGraphXRuntimeBackend(RuntimeDescriptor descriptor,
                                               mgx::SessionOptions session,
                                               MIGraphXPipelineFactory factory,
                                               std::string model_id)
    : descriptor_(std::move(descriptor)),
      session_(std::move(session)),
      factory_(std::move(factory)),
      model_id_(std::move(model_id)) {
  detail::ValidateRuntime(descriptor_);
  detail::Text(model_id_, true);
  if (!factory_ || descriptor_.backend_id != kMIGraphXBackendId ||
      descriptor_.format != "onnx" || descriptor_.device != Device(c10::kCPU) ||
      descriptor_.dtypes != std::vector<Dtype>{torch::kFloat32} ||
      descriptor_.dynamic_shapes || descriptor_.tensor_batching ||
      descriptor_.oom_retry_safe || descriptor_.allow_cpu_fallback ||
      descriptor_.execution_provider != "migraphx.gpu" ||
      descriptor_.provider_options !=
          std::map<std::string, std::string>{
              {"device_id", std::to_string(session_.device_id)}} ||
      session_.device_id < 0 || !session_.max_model_bytes ||
      !session_.max_input_bytes || !session_.max_output_bytes ||
      !session_.max_batch_size)
    detail::Invalid(
        "migraphx.backend",
        "explicit fixed GPU device, CPU FP32 I/O, no batching/retry/fallback");
  static_cast<void>(
      migraphx_invocation_schema(session_.input_name, session_.input_shape));
}
RuntimeDescriptor MIGraphXRuntimeBackend::descriptor() const {
  return descriptor_;
}
std::uint64_t MIGraphXRuntimeBackend::load_reservation_bytes(
    const LoadOptions& options) const {
  if (!options.max_snapshot_bytes) detail::Resource("reload.migraphx.snapshot");
  return options.max_snapshot_bytes;
}
std::shared_ptr<Predictor> MIGraphXRuntimeBackend::load(
    const std::filesystem::path& directory,
    const interop::ModelBundleVerifyOptions& verify,
    std::span<const WarmupCase> warmup, const LoadOptions& options) const {
  return detail::Backend("migraphx.load", [&] {
    detail::ValidateLoad(options);
    static_cast<void>(load_reservation_bytes(options));
    if (warmup.empty() || warmup.size() > options.max_warmup_cases)
      detail::Invalid("warmup", "nonempty bounded independent known vectors");
    detail::Boundary(options.stop, options.deadline);
    const auto metadata =
        interop::verify_model_bundle_directory(directory, verify);
    const auto& declared = metadata.manifest.runtime;
    if (declared.backend_id != descriptor_.backend_id ||
        declared.backend_version != descriptor_.backend_version ||
        declared.environment_lock != descriptor_.environment_lock ||
        declared.dynamic_shapes ||
        declared.devices != std::vector<std::string>{"rocm"} ||
        declared.dtypes != std::vector<std::string>{"float32"})
      Mismatch("migraphx.runtime");
    SameDependencies(declared.dependencies, descriptor_.dependencies);
    const auto& models = metadata.manifest.models;
    const auto model =
        std::find_if(models.begin(), models.end(),
                     [&](const auto& item) { return item.id == model_id_; });
    if (model == models.end() ||
        model->format != interop::BundleModelFormat::onnx ||
        model->file_ids.size() != 1U)
      Mismatch("migraphx.model");
    const auto file = std::find_if(
        metadata.manifest.files.begin(), metadata.manifest.files.end(),
        [&](const auto& item) { return item.id == model->file_ids.front(); });
    if (file == metadata.manifest.files.end()) Mismatch("migraphx.file");
    if (file->artifact.size_bytes > session_.max_model_bytes)
      detail::Resource("migraphx.model.bytes");
    const auto model_file = *file;
    std::vector<std::string_view> ids;
    for (const auto& item : metadata.manifest.files) ids.push_back(item.id);
    auto snapshot = interop::read_model_bundle_snapshot(
        directory, metadata.manifest_digest, ids, verify,
        options.max_snapshot_bytes);
    VerifiedBundle bundle{.verification = std::move(snapshot.verification),
                          .files = std::move(snapshot.files)};
    const auto& manifest = bundle.verification.manifest;
    const auto& encoded = bundle.files.at(Lock(manifest).file_id);
    const std::string_view text(reinterpret_cast<const char*>(encoded.data()),
                                encoded.size());
    if (text.size() > mgx::kEnvironmentLockMaxBytes)
      detail::Resource("migraphx.environment_lock.bytes");
    if (sha256_hex(encoded) != descriptor_.environment_lock.hex)
      Mismatch("migraphx.environment_lock.digest");
    const auto lock = mgx::parse_environment_lock(text);
    if (lock.adapter_version != descriptor_.backend_version ||
        declared.os_arch != std::vector<std::string>{lock.os_arch} ||
        lock.device_id != session_.device_id ||
        lock.input_name != session_.input_name ||
        lock.input_shape != session_.input_shape)
      Mismatch("migraphx.specialization");
    SameDependencies(Dependencies(lock), declared.dependencies);
    detail::Boundary(options.stop, options.deadline);
    auto bounded = options;
    auto session_options = session_;
    session_options.max_input_bytes = std::min(
        session_.max_input_bytes, options.request_limits.max_tensor_bytes);
    session_options.max_output_bytes = std::min(
        session_.max_output_bytes, options.output_limits.max_tensor_bytes);
    auto session = std::make_shared<mgx::Session>(
        bundle.files.at(model_file.id), model_file.artifact.digest,
        session_options);
    detail::Boundary(options.stop, options.deadline);
    const auto actual = mgx::verify_environment_lock(
        *session, text, descriptor_.environment_lock);
    auto pipeline = factory_(bundle, actual);
    detail::Boundary(options.stop, options.deadline);
    GraphBoundary(pipeline, *session);
    detail::ValidateBinding(pipeline.input, manifest.input_schema, bundle);
    detail::ValidateBinding(pipeline.output, manifest.output_schema, bundle);
    detail::ValidateBinding(pipeline.preprocessing, manifest.preprocessing,
                            bundle);
    detail::ValidateBinding(pipeline.postprocessing, manifest.postprocessing,
                            bundle);
    auto preprocess = std::move(pipeline.preprocess);
    pipeline.preprocess = [session, preprocess = std::move(preprocess)](
                              const nn::Invocation& input) {
      auto processed = preprocess ? preprocess(input) : input;
      InputBoundary(*session, processed);
      return processed;
    };
    bounded.request_limits.max_tensor_bytes = session_options.max_input_bytes;
    bounded.output_limits.max_tensor_bytes = session_options.max_output_bytes;
    return detail::PredictorFactory::Create(
        descriptor_, manifest.bundle_id, std::move(pipeline),
        [session](const nn::Invocation& input, std::stop_token stop,
                  const Deadline& deadline) {
          return Invoke(*session, input, stop, deadline);
        },
        [session] { static_cast<void>(session); }, warmup, bounded, true,
        Contract(*session));
  });
}
}  // namespace jyppx::liuding::serving
