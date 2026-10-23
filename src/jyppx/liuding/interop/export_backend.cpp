// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "metadata_validation.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/export_backend.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <utility>

namespace jyppx::liuding::interop {
namespace {

namespace md = metadata_detail;
using md::Json;
constexpr std::string_view kComponent = "interop.export_backend";
constexpr std::string_view kProtocol = "jyppx.export.sidecar";
constexpr std::uint32_t kProtocolVersion = 1U;
constexpr std::size_t kMaxValues = 256U;
constexpr std::uint64_t kMaxArtifactBytes = 16ULL << 30U;

constexpr std::array<std::string_view, 6> kRequiredOnnxDependencies{
    "python", "torch", "onnx", "onnxscript", "onnxruntime", "safetensors"};

std::string_view Name(ArtifactFormat value) noexcept {
  switch (value) {
    case ArtifactFormat::jyppx_checkpoint:
      return "jyppx_checkpoint";
    case ArtifactFormat::jyppx_model_bundle:
      return "jyppx_model_bundle";
    case ArtifactFormat::safetensors:
      return "safetensors";
    case ArtifactFormat::libtorch_archive:
      return "libtorch_archive";
    case ArtifactFormat::torchscript:
      return "torchscript";
    case ArtifactFormat::onnx:
      return "onnx";
  }
  return "unknown";
}

std::string_view Name(ExportSourceKind value) noexcept {
  switch (value) {
    case ExportSourceKind::native_state:
      return "native_state";
    case ExportSourceKind::python_factory:
      return "python_factory";
    case ExportSourceKind::exported_program:
      return "exported_program";
    case ExportSourceKind::torchscript:
      return "torchscript";
    case ExportSourceKind::native_graph_adapter:
      return "native_graph_adapter";
  }
  return "unknown";
}

std::string_view Name(BackendOperation value) noexcept {
  switch (value) {
    case BackendOperation::export_model:
      return "export_model";
    case BackendOperation::load:
      return "load";
    case BackendOperation::run:
      return "run";
    case BackendOperation::inspect:
      return "inspect";
    case BackendOperation::trace:
      return "trace";
  }
  return "unknown";
}

std::string_view Name(ShapeInferenceStatus value) noexcept {
  switch (value) {
    case ShapeInferenceStatus::not_checked:
      return "not_checked";
    case ShapeInferenceStatus::passed:
      return "passed";
    case ShapeInferenceStatus::not_applicable:
      return "not_applicable";
    case ShapeInferenceStatus::failed:
      return "failed";
  }
  return "unknown";
}

template <class Enum>
void EnumValue(Enum value) {
  if (Name(value) == "unknown")
    md::Invalid(kComponent, "enum", "a known stable enumeration identity");
}

template <class Enum>
Enum ReadEnum(const Json& value, std::initializer_list<Enum> values) {
  const auto token = md::String(value, kComponent);
  for (const auto candidate : values) {
    if (Name(candidate) == token) return candidate;
  }
  throw CompatibilityError(
      "Unsupported export backend enum token",
      md::Details(kComponent, "enum", "a supported versioned machine token"));
}

ArtifactFormat ReadFormat(const Json& value) {
  return ReadEnum(
      value,
      {ArtifactFormat::jyppx_checkpoint, ArtifactFormat::jyppx_model_bundle,
       ArtifactFormat::safetensors, ArtifactFormat::libtorch_archive,
       ArtifactFormat::torchscript, ArtifactFormat::onnx});
}

ExportSourceKind ReadSource(const Json& value) {
  return ReadEnum(
      value, {ExportSourceKind::native_state, ExportSourceKind::python_factory,
              ExportSourceKind::exported_program, ExportSourceKind::torchscript,
              ExportSourceKind::native_graph_adapter});
}

BackendOperation ReadOperation(const Json& value) {
  return ReadEnum(value, {BackendOperation::export_model,
                          BackendOperation::load, BackendOperation::run,
                          BackendOperation::inspect, BackendOperation::trace});
}

ShapeInferenceStatus ReadShapeStatus(const Json& value) {
  return ReadEnum(
      value,
      {ShapeInferenceStatus::not_checked, ShapeInferenceStatus::passed,
       ShapeInferenceStatus::not_applicable, ShapeInferenceStatus::failed});
}

template <class Collection, class Value>
bool Contains(const Collection& collection, const Value& value) {
  return std::find(collection.begin(), collection.end(), value) !=
         collection.end();
}

void Count(std::size_t count, bool allow_empty = false) {
  if (count > kMaxValues) md::Resource(kComponent, "collection", kMaxValues);
  if (!allow_empty && count == 0U)
    md::Invalid(kComponent, "collection", "a nonempty explicit finite matrix");
}

template <class Enum>
void Enums(const std::vector<Enum>& values, bool allow_empty = false) {
  Count(values.size(), allow_empty);
  std::set<Enum> unique;
  for (const auto value : values) {
    EnumValue(value);
    if (!unique.insert(value).second)
      md::Invalid(kComponent, "collection", "unique explicit enum values");
  }
}

void Tokens(const std::vector<std::string>& values, bool allow_empty = false) {
  Count(values.size(), allow_empty);
  std::set<std::string_view> unique;
  for (const auto& value : values) {
    md::Token(value, kComponent, "matrix.token");
    if (!unique.insert(value).second)
      md::Invalid(kComponent, "matrix", "unique explicit matrix entries");
  }
}

void Version(std::string_view value) {
  md::Text(value, 256U, kComponent, "version");
  if (value.front() < '0' || value.front() > '9' ||
      !std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
               (byte >= '0' && byte <= '9') || byte == '.' || byte == '+' ||
               byte == '-' || byte == '_';
      }))
    md::Invalid(kComponent, "version", "an exact pinned version, not a range");
}

void Dependencies(const std::vector<BackendDependency>& dependencies,
                  bool allow_empty = false) {
  Count(dependencies.size(), allow_empty);
  std::set<std::string_view> names;
  for (const auto& dependency : dependencies) {
    md::Token(dependency.name, kComponent, "dependency.name");
    Version(dependency.version);
    if (!names.insert(dependency.name).second)
      md::Invalid(kComponent, "dependencies", "unique package identities");
  }
}

void Schema(std::uint32_t actual, std::uint32_t expected) {
  if (actual != expected)
    throw CompatibilityError(
        "Unsupported export metadata schema",
        md::Details(kComponent, "schema_version", "supported exact version"));
}

void FactoryEntryPoint(std::string_view value) {
  md::Text(value, 4096U, kComponent, "factory_entry_point");
  const auto colon = value.find(':');
  if (colon == std::string_view::npos || colon == 0U ||
      colon + 1U == value.size() ||
      value.find(':', colon + 1U) != std::string_view::npos)
    md::Invalid(kComponent, "factory_entry_point", "module.path:factory_name");
  // Entry points are import identifiers, never file paths or shell commands.
  for (std::size_t begin = 0U; begin < value.size();) {
    const auto end = value.find_first_of(".:", begin);
    const auto part =
        value.substr(begin, end == std::string_view::npos ? value.size() - begin
                                                          : end - begin);
    const auto initial = [](unsigned char byte) {
      return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
             byte == '_';
    };
    if (part.empty() || !initial(static_cast<unsigned char>(part.front())) ||
        !std::all_of(part.begin(), part.end(), [&](unsigned char byte) {
          return initial(byte) || (byte >= '0' && byte <= '9');
        }))
      md::Invalid(kComponent, "factory_entry_point",
                  "Python import identifiers");
    if (end == std::string_view::npos) break;
    begin = end + 1U;
    if (begin == value.size())
      md::Invalid(kComponent, "factory_entry_point",
                  "nonempty import identifiers");
  }
}

void Resources(const SidecarResources& resources) {
  if (resources.cpu_threads == 0U || resources.cpu_threads > 1024U ||
      resources.memory_bytes == 0U || resources.memory_bytes > (1ULL << 40U) ||
      resources.wall_time_ms == 0U || resources.wall_time_ms > 86400000U ||
      resources.max_output_bytes == 0U ||
      resources.max_output_bytes > kMaxArtifactBytes ||
      resources.max_control_bytes == 0U ||
      resources.max_control_bytes > 64U * 1024U * 1024U)
    md::Invalid(kComponent, "resources",
                "positive finite sidecar resource bounds");
}

void Tolerance(double absolute, double relative) {
  if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0.0 ||
      relative < 0.0)
    md::Invalid(kComponent, "tolerance",
                "finite nonnegative numeric tolerances");
}

void NonemptyArtifact(const ArtifactRef& artifact, std::uint64_t maximum) {
  md::Artifact(artifact, kComponent, maximum);
  if (artifact.size_bytes == 0U)
    md::Invalid(kComponent, "artifact.bytes",
                "nonempty independently verified bytes");
}

void ValidateRequest(const ExportRequest& request) {
  Schema(request.schema_version, ExportRequest::format_version);
  md::Token(request.request_id, kComponent, "request.id");
  md::Token(request.backend_id, kComponent, "request.backend");
  EnumValue(request.operation);
  EnumValue(request.source_kind);
  EnumValue(request.target);
  NonemptyArtifact(request.source, kMaxArtifactBytes);
  std::uint64_t total_input_bytes = 0U;
  const auto count_input = [&](const ArtifactRef& artifact) {
    md::Add(total_input_bytes, artifact.size_bytes, kMaxArtifactBytes,
            kComponent, "request.total_input_bytes");
  };
  count_input(request.source);
  if (request.state) NonemptyArtifact(*request.state, kMaxArtifactBytes);
  if (request.state) count_input(*request.state);
  if (request.source_kind == ExportSourceKind::python_factory) {
    FactoryEntryPoint(request.factory_entry_point);
    if (!request.state)
      md::Invalid(kComponent, "source.state", "explicit factory tensor state");
  } else if (!request.factory_entry_point.empty() || request.state) {
    md::Invalid(kComponent, "source.factory",
                "factory metadata only for PythonFactory");
  }
  md::Token(request.device, kComponent, "request.device");
  md::Token(request.dtype, kComponent, "request.dtype");
  md::Token(request.quantization_mode, kComponent, "request.quantization");
  md::Token(request.prepost_mode, kComponent, "request.prepost");
  if (request.invocation_schema)
    NonemptyArtifact(*request.invocation_schema, kMaxArtifactBytes);
  if (request.invocation_schema) count_input(*request.invocation_schema);
  Count(request.verification_cases.size(), true);
  for (const auto& artifact : request.verification_cases) {
    NonemptyArtifact(artifact, kMaxArtifactBytes);
    count_input(artifact);
  }
  Count(request.options.size(), true);
  for (const auto& [key, value] : request.options) {
    md::Token(key, kComponent, "request.option");
    md::Text(value, 4096U, kComponent, "request.option.value");
  }
  validate_artifact_digest(request.environment_lock);
  Resources(request.resources);
  Tolerance(request.absolute_tolerance, request.relative_tolerance);
}

void ValidateProbe(const BackendProbe& probe) {
  Schema(probe.schema_version, BackendProbe::format_version);
  md::Token(probe.backend_id, kComponent, "probe.backend");
  Version(probe.backend_version);
  validate_artifact_digest(probe.environment_lock);
  Dependencies(probe.dependencies, true);
  Enums(probe.operations, true);
  Tokens(probe.devices, true);
  Tokens(probe.dtypes, true);
  if (probe.request_digest) validate_artifact_digest(*probe.request_digest);
  if (probe.model_preflight_passed && !probe.request_digest)
    md::Invalid(kComponent, "probe.request_digest",
                "an exact model preflight request binding");
  Count(probe.unsupported_operators.size(), true);
  std::set<std::string_view> operators;
  for (const auto& operation : probe.unsupported_operators) {
    md::Text(operation, 4096U, kComponent, "probe.operator");
    if (!operators.insert(operation).second)
      md::Invalid(kComponent, "probe.operators", "unique operator identities");
  }
  NonemptyArtifact(probe.evidence, kMaxArtifactBytes);
}

void ValidateCandidate(const ExportCandidate& candidate) {
  Schema(candidate.schema_version, ExportCandidate::format_version);
  md::Token(candidate.request_id, kComponent, "candidate.request_id");
  md::Token(candidate.backend_id, kComponent, "candidate.backend_id");
  Version(candidate.backend_version);
  NonemptyArtifact(candidate.artifact, kMaxArtifactBytes);
  const auto& verification = candidate.verification;
  validate_artifact_digest(verification.request_digest);
  validate_artifact_digest(verification.artifact_digest);
  validate_artifact_digest(verification.environment_lock);
  EnumValue(verification.shape_inference);
  md::Text(verification.shape_inference_reason, 4096U, kComponent,
           "verification.shape_reason", true);
  Tolerance(verification.absolute_tolerance, verification.relative_tolerance);
  NonemptyArtifact(verification.evidence, kMaxArtifactBytes);
  if (verification.cases_run > kMaxValues ||
      verification.cases_passed > verification.cases_run ||
      verification.dynamic_cases_passed > verification.cases_passed)
    md::Invalid(kComponent, "verification.cases",
                "consistent bounded case counts");
}

template <class Operation>
auto Boundary(Operation&& operation, bool encoded_input = false)
    -> decltype(operation()) {
  try {
    return operation();
  } catch (const nlohmann::json::exception&) {
    if (encoded_input)
      md::Malformed(kComponent, "json", "the exact supported control schema");
    md::Invalid(kComponent, "json", "bounded encodable metadata");
  } catch (const std::bad_alloc&) {
    md::Resource(kComponent, "memory", 64U * 1024U * 1024U);
  } catch (const std::length_error&) {
    md::Resource(kComponent, "memory", 64U * 1024U * 1024U);
  }
}

}  // namespace

void validate_backend_descriptor(const BackendDescriptor& descriptor) {
  Boundary([&] {
    Schema(descriptor.schema_version, BackendDescriptor::format_version);
    md::Token(descriptor.backend_id, kComponent, "descriptor.id");
    Version(descriptor.backend_version);
    if ((descriptor.role != BackendRole::exporter &&
         descriptor.role != BackendRole::runtime) ||
        (descriptor.execution != BackendExecution::in_process &&
         descriptor.execution != BackendExecution::out_of_process))
      md::Invalid(kComponent, "descriptor.role",
                  "a known role/execution identity");
    if ((descriptor.execution == BackendExecution::out_of_process &&
         descriptor.protocol_version != kProtocolVersion) ||
        (descriptor.execution == BackendExecution::in_process &&
         descriptor.protocol_version != 0U))
      md::Invalid(kComponent, "descriptor.protocol",
                  "sidecar v1 or in-process zero");
    validate_artifact_digest(descriptor.environment_lock);
    Enums(descriptor.formats);
    Enums(descriptor.sources);
    Enums(descriptor.operations);
    Tokens(descriptor.devices);
    Tokens(descriptor.dtypes);
    Tokens(descriptor.quantization_modes);
    Tokens(descriptor.prepost_modes);
    Dependencies(descriptor.dependencies);
    Count(descriptor.allowed_options.size(), true);
    for (const auto& [key, values] : descriptor.allowed_options) {
      md::Token(key, kComponent, "descriptor.option");
      Count(values.size());
      std::set<std::string_view> unique;
      for (const auto& value : values) {
        md::Text(value, 4096U, kComponent, "descriptor.option.value");
        if (!unique.insert(value).second)
          md::Invalid(kComponent, "descriptor.option",
                      "unique exact option values");
      }
    }
    for (const auto operation : descriptor.operations) {
      const bool exporting = operation == BackendOperation::export_model ||
                             operation == BackendOperation::trace;
      if (exporting != (descriptor.role == BackendRole::exporter))
        md::Invalid(kComponent, "descriptor.role",
                    "separate export/runtime operations");
      if (operation == BackendOperation::trace &&
          (!descriptor.compatibility_only || !descriptor.deprecated))
        md::Invalid(kComponent, "descriptor.trace",
                    "deprecated compatibility scope");
    }
    if (descriptor.max_artifact_bytes == 0U ||
        descriptor.max_artifact_bytes > kMaxArtifactBytes ||
        descriptor.max_verification_cases == 0U ||
        descriptor.max_verification_cases > kMaxValues)
      md::Invalid(kComponent, "descriptor.limits",
                  "positive bounded artifact/case limits");
  });
}

BackendDescriptor make_onnx_sidecar_descriptor(
    std::string backend_version, ArtifactDigest environment_lock,
    std::span<const BackendDependency> dependencies) {
  return Boundary([&] {
    BackendDescriptor descriptor;
    descriptor.backend_id = "onnx.sidecar";
    descriptor.backend_version = std::move(backend_version);
    descriptor.role = BackendRole::exporter;
    descriptor.execution = BackendExecution::out_of_process;
    descriptor.protocol_version = kProtocolVersion;
    descriptor.environment_lock = std::move(environment_lock);
    descriptor.formats = {ArtifactFormat::onnx};
    descriptor.sources = {ExportSourceKind::python_factory,
                          ExportSourceKind::exported_program};
    descriptor.operations = {BackendOperation::export_model};
    descriptor.devices = {"cpu", "cuda"};
    descriptor.dtypes = {"float32"};
    descriptor.dynamic_shapes = true;
    descriptor.quantization_modes = {"none"};
    descriptor.prepost_modes = {"external"};
    descriptor.allowed_options = {{"opset", {"18"}}};
    if (dependencies.size() > kMaxValues)
      md::Resource(kComponent, "dependencies.count", kMaxValues);
    descriptor.dependencies.assign(dependencies.begin(), dependencies.end());
    validate_backend_descriptor(descriptor);
    for (const auto name : kRequiredOnnxDependencies) {
      if (std::none_of(descriptor.dependencies.begin(),
                       descriptor.dependencies.end(),
                       [&](const auto& item) { return item.name == name; }))
        md::Invalid(kComponent, "dependencies",
                    "all six official sidecar package pins");
    }
    return descriptor;
  });
}

BackendDescriptor make_torchscript_compatibility_descriptor(
    std::string backend_version, ArtifactDigest environment_lock,
    std::string libtorch_version) {
  return Boundary([&] {
    BackendDescriptor descriptor;
    descriptor.backend_id = "torchscript.compatibility";
    descriptor.backend_version = std::move(backend_version);
    descriptor.role = BackendRole::runtime;
    descriptor.execution = BackendExecution::in_process;
    descriptor.protocol_version = 0U;
    descriptor.environment_lock = std::move(environment_lock);
    descriptor.formats = {ArtifactFormat::torchscript};
    descriptor.sources = {ExportSourceKind::torchscript};
    descriptor.operations = {BackendOperation::load, BackendOperation::run,
                             BackendOperation::inspect};
    descriptor.devices = {"cpu", "cuda"};
    descriptor.dtypes = {"float32"};
    descriptor.quantization_modes = {"none"};
    descriptor.prepost_modes = {"external"};
    descriptor.dependencies = {
        {.name = "libtorch", .version = std::move(libtorch_version)}};
    descriptor.compatibility_only = true;
    descriptor.deprecated = true;
    validate_backend_descriptor(descriptor);
    return descriptor;
  });
}

BackendRegistry::BackendRegistry(BackendRole role) : role_(role) {
  if (role != BackendRole::exporter && role != BackendRole::runtime)
    md::Invalid(kComponent, "registry.role", "exporter or runtime");
}

void BackendRegistry::add(BackendDescriptor descriptor) {
  Boundary([&] {
    validate_backend_descriptor(descriptor);
    if (descriptor.role != role_ ||
        descriptors_.contains(descriptor.backend_id))
      md::Invalid(kComponent, "registry.add",
                  "a unique descriptor with matching role");
    if (descriptors_.size() >= 64U)
      md::Resource(kComponent, "registry.backends", 64U);
    auto id = descriptor.backend_id;
    descriptors_.emplace(std::move(id), std::move(descriptor));
  });
}

const BackendDescriptor* BackendRegistry::find(
    std::string_view backend_id) const {
  const auto found = descriptors_.find(backend_id);
  return found == descriptors_.end() ? nullptr : &found->second;
}

SupportReport preflight_backend(const BackendDescriptor& descriptor,
                                const ExportRequest& request,
                                const BackendProbe* probe) {
  return Boundary([&] {
    validate_backend_descriptor(descriptor);
    ValidateRequest(request);
    if (probe != nullptr) ValidateProbe(*probe);
    SupportReport report{.decision = SupportDecision::supported,
                         .backend_id = descriptor.backend_id,
                         .backend_version = descriptor.backend_version,
                         .issues = {}};
    const auto issue = [&](BackendIssueKind kind, std::string code,
                           std::string subject, std::string expected,
                           std::string actual, bool conditional = false) {
      if (!conditional)
        report.decision = SupportDecision::unsupported;
      else if (report.decision == SupportDecision::supported)
        report.decision = SupportDecision::conditional;
      report.issues.push_back({.kind = kind,
                               .code = std::move(code),
                               .subject = std::move(subject),
                               .expected = std::move(expected),
                               .actual = std::move(actual)});
    };
    const auto matrix = [&](const auto& values, const auto& value,
                            BackendIssueKind kind, std::string_view subject,
                            std::string actual) {
      if (!Contains(values, value))
        issue(kind, std::string(subject) + ".unsupported", std::string(subject),
              "an explicitly declared matrix entry", std::move(actual));
    };
    if (request.backend_id != descriptor.backend_id)
      issue(BackendIssueKind::environment, "backend.mismatch", "backend_id",
            descriptor.backend_id, request.backend_id);
    matrix(descriptor.operations, request.operation,
           BackendIssueKind::operation, "operation",
           std::string(Name(request.operation)));
    matrix(descriptor.sources, request.source_kind, BackendIssueKind::source,
           "source", std::string(Name(request.source_kind)));
    matrix(descriptor.formats, request.target, BackendIssueKind::source,
           "format", std::string(Name(request.target)));
    matrix(descriptor.devices, request.device, BackendIssueKind::device,
           "device", request.device);
    matrix(descriptor.dtypes, request.dtype, BackendIssueKind::dtype, "dtype",
           request.dtype);
    matrix(descriptor.quantization_modes, request.quantization_mode,
           BackendIssueKind::option, "quantization", request.quantization_mode);
    matrix(descriptor.prepost_modes, request.prepost_mode,
           BackendIssueKind::option, "prepost", request.prepost_mode);
    if (request.dynamic_shapes && !descriptor.dynamic_shapes)
      issue(BackendIssueKind::dynamic_shape, "dynamic_shape.unsupported",
            "dynamic_shapes", "declared dynamic implementation", "absent");
    if (!(request.environment_lock == descriptor.environment_lock))
      issue(BackendIssueKind::environment, "environment.lock_mismatch",
            "environment_lock", "selected descriptor lock", "different lock");
    if (request.source_kind != ExportSourceKind::native_state &&
        !request.trust_source)
      issue(BackendIssueKind::trust, "source.trust_required", "trust_source",
            "explicit executable-source authorization", "false");
    if (request.operation == BackendOperation::trace && !request.trace_opt_in)
      issue(BackendIssueKind::trust, "trace.opt_in_required", "trace_opt_in",
            "explicit deprecated trace opt-in", "false");
    for (const auto& [key, value] : request.options) {
      const auto allowed = descriptor.allowed_options.find(key);
      if (allowed == descriptor.allowed_options.end() ||
          !Contains(allowed->second, value))
        issue(BackendIssueKind::option, "option.unsupported", key,
              "an exact allow-listed option/value", "not allowed");
    }
    if (descriptor.backend_id == "onnx.sidecar" &&
        !request.options.contains("opset"))
      issue(BackendIssueKind::option, "option.required", "opset",
            "explicit pinned opset", "absent");
    if (request.source.size_bytes > descriptor.max_artifact_bytes ||
        (request.state &&
         request.state->size_bytes > descriptor.max_artifact_bytes) ||
        request.resources.max_output_bytes > descriptor.max_artifact_bytes)
      issue(BackendIssueKind::option, "artifact.limit", "artifact_bytes",
            "within declared backend byte limit", "exceeded");
    if ((request.invocation_schema && request.invocation_schema->size_bytes >
                                          descriptor.max_artifact_bytes) ||
        std::any_of(
            request.verification_cases.begin(),
            request.verification_cases.end(), [&](const ArtifactRef& artifact) {
              return artifact.size_bytes > descriptor.max_artifact_bytes;
            }))
      issue(BackendIssueKind::verification, "verification_artifact.limit",
            "verification_bytes", "within declared backend byte limit",
            "exceeded");
    if (request.verification_cases.size() > descriptor.max_verification_cases)
      issue(BackendIssueKind::verification, "cases.limit", "verification_cases",
            "within declared backend case limit", "exceeded");
    const bool executing =
        request.operation == BackendOperation::export_model ||
        request.operation == BackendOperation::trace ||
        request.operation == BackendOperation::run;
    if (executing &&
        (!request.invocation_schema || request.verification_cases.empty()))
      issue(BackendIssueKind::verification, "cases.required",
            "verification_cases",
            "explicit invocation schema and nonempty cases", "absent");
    if (executing && request.dynamic_shapes &&
        request.verification_cases.size() < 2U)
      issue(BackendIssueKind::verification, "dynamic_cases.required",
            "verification_cases", "at least one fixed and one dynamic case",
            "insufficient cases");
    if (probe == nullptr) {
      issue(BackendIssueKind::environment, "environment.unprobed", "probe",
            "real current environment and independently verified evidence",
            "absent", true);
      return report;
    }
    if (probe->backend_id != descriptor.backend_id ||
        probe->backend_version != descriptor.backend_version)
      issue(BackendIssueKind::environment, "probe.backend_mismatch", "backend",
            "selected backend identity and implementation version",
            "different identity/version");
    if (probe->protocol_version != descriptor.protocol_version)
      issue(BackendIssueKind::protocol, "probe.protocol_mismatch",
            "protocol_version", std::to_string(descriptor.protocol_version),
            std::to_string(probe->protocol_version));
    if (!(probe->environment_lock == descriptor.environment_lock))
      issue(BackendIssueKind::environment, "probe.lock_mismatch",
            "environment_lock", "selected exact environment lock",
            "different lock");
    for (const auto& required : descriptor.dependencies) {
      const auto observed = std::find_if(
          probe->dependencies.begin(), probe->dependencies.end(),
          [&](const auto& item) { return item.name == required.name; });
      if (observed == probe->dependencies.end())
        issue(BackendIssueKind::dependency, "dependency.missing", required.name,
              required.version, "missing");
      else if (observed->version != required.version)
        issue(BackendIssueKind::dependency, "dependency.version_mismatch",
              required.name, required.version, observed->version);
    }
    matrix(probe->operations, request.operation, BackendIssueKind::operation,
           "probe.operation", std::string(Name(request.operation)));
    matrix(probe->devices, request.device, BackendIssueKind::device,
           "probe.device", request.device);
    matrix(probe->dtypes, request.dtype, BackendIssueKind::dtype, "probe.dtype",
           request.dtype);
    if (request.dynamic_shapes && !probe->dynamic_shapes_verified)
      issue(BackendIssueKind::dynamic_shape, "probe.dynamic_unverified",
            "dynamic_shapes", "real verified dynamic-shape validation",
            "absent");
    if (!probe->request_digest)
      issue(BackendIssueKind::operation, "model.preflight_required",
            "source_model",
            "real request-bound model/operator/schema preflight", "absent",
            true);
    else if (!(*probe->request_digest == backend_request_digest(request)))
      issue(BackendIssueKind::environment, "probe.request_mismatch",
            "request_digest", "the complete current canonical request",
            "different request");
    else if (!probe->model_preflight_passed)
      issue(BackendIssueKind::operation, "model.preflight_failed",
            "source_model", "successful real model/operator/schema preflight",
            "failed");
    for (const auto& operation : probe->unsupported_operators)
      issue(BackendIssueKind::operation, "operator.unsupported", operation,
            "a supported official exporter/runtime operator", "unsupported");
    if (request.target == ArtifactFormat::onnx &&
        request.operation == BackendOperation::export_model &&
        (!probe->official_onnx_exporter || !probe->onnx_checker ||
         !probe->onnx_runtime))
      issue(BackendIssueKind::verification,
            "onnx.official_validation_unavailable", "onnx",
            "official exporter, onnx.checker, and real ORT", "incomplete");
    return report;
  });
}

namespace {

Json DependenciesJson(const std::vector<BackendDependency>& dependencies) {
  Json result = Json::array();
  std::vector<const BackendDependency*> sorted;
  sorted.reserve(dependencies.size());
  for (const auto& dependency : dependencies) sorted.push_back(&dependency);
  std::sort(sorted.begin(), sorted.end(),
            [](const auto* left, const auto* right) {
              return left->name < right->name;
            });
  for (const auto* dependency : sorted)
    result.push_back(
        Json{{"name", dependency->name}, {"version", dependency->version}});
  return result;
}

template <class Enum>
Json EnumsJson(const std::vector<Enum>& values) {
  std::vector<std::string_view> names;
  names.reserve(values.size());
  for (const auto value : values) names.push_back(Name(value));
  std::sort(names.begin(), names.end());
  return names;
}

Json StringsJson(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  return values;
}

Json OptionalArtifactJson(const std::optional<ArtifactRef>& value) {
  return value ? md::ArtifactJson(*value) : Json(nullptr);
}

Json RequestJson(const ExportRequest& request) {
  Json cases = Json::array();
  // Case order is part of the request identity and must not be canonicalized
  // away; it binds each runtime/numeric verification to the supplied examples.
  for (const auto& artifact : request.verification_cases)
    cases.push_back(md::ArtifactJson(artifact));
  const auto& resources = request.resources;
  return Json{
      {"schema_version", request.schema_version},
      {"request_id", request.request_id},
      {"backend_id", request.backend_id},
      {"operation", Name(request.operation)},
      {"source_kind", Name(request.source_kind)},
      {"source", md::ArtifactJson(request.source)},
      {"state", OptionalArtifactJson(request.state)},
      {"factory_entry_point", request.factory_entry_point},
      {"target", Name(request.target)},
      {"device", request.device},
      {"dtype", request.dtype},
      {"dynamic_shapes", request.dynamic_shapes},
      {"quantization_mode", request.quantization_mode},
      {"prepost_mode", request.prepost_mode},
      {"invocation_schema", OptionalArtifactJson(request.invocation_schema)},
      {"verification_cases", std::move(cases)},
      {"options", request.options},
      {"trust_source", request.trust_source},
      {"trace_opt_in", request.trace_opt_in},
      {"environment_lock", md::DigestJson(request.environment_lock)},
      {"resources", Json{{"cpu_threads", resources.cpu_threads},
                         {"memory_bytes", resources.memory_bytes},
                         {"wall_time_ms", resources.wall_time_ms},
                         {"max_output_bytes", resources.max_output_bytes},
                         {"max_control_bytes", resources.max_control_bytes}}},
      {"absolute_tolerance", request.absolute_tolerance},
      {"relative_tolerance", request.relative_tolerance}};
}

Json ProbeJson(const BackendProbe& probe) {
  return Json{
      {"schema_version", probe.schema_version},
      {"backend_id", probe.backend_id},
      {"backend_version", probe.backend_version},
      {"protocol_version", probe.protocol_version},
      {"environment_lock", md::DigestJson(probe.environment_lock)},
      {"dependencies", DependenciesJson(probe.dependencies)},
      {"operations", EnumsJson(probe.operations)},
      {"devices", StringsJson(probe.devices)},
      {"dtypes", StringsJson(probe.dtypes)},
      {"dynamic_shapes_verified", probe.dynamic_shapes_verified},
      {"official_onnx_exporter", probe.official_onnx_exporter},
      {"onnx_checker", probe.onnx_checker},
      {"onnx_runtime", probe.onnx_runtime},
      {"request_digest", probe.request_digest
                             ? md::DigestJson(*probe.request_digest)
                             : Json(nullptr)},
      {"model_preflight_passed", probe.model_preflight_passed},
      {"unsupported_operators", StringsJson(probe.unsupported_operators)},
      {"evidence", md::ArtifactJson(probe.evidence)}};
}

Json CandidateJson(const ExportCandidate& candidate) {
  const auto& verification = candidate.verification;
  return Json{
      {"schema_version", candidate.schema_version},
      {"request_id", candidate.request_id},
      {"backend_id", candidate.backend_id},
      {"backend_version", candidate.backend_version},
      {"artifact", md::ArtifactJson(candidate.artifact)},
      {"verification",
       Json{{"request_digest", md::DigestJson(verification.request_digest)},
            {"artifact_digest", md::DigestJson(verification.artifact_digest)},
            {"environment_lock", md::DigestJson(verification.environment_lock)},
            {"schema_passed", verification.schema_passed},
            {"checker_passed", verification.checker_passed},
            {"shape_inference", Name(verification.shape_inference)},
            {"shape_inference_reason", verification.shape_inference_reason},
            {"runtime_passed", verification.runtime_passed},
            {"numerical_passed", verification.numerical_passed},
            {"cases_run", verification.cases_run},
            {"cases_passed", verification.cases_passed},
            {"dynamic_cases_passed", verification.dynamic_cases_passed},
            {"absolute_tolerance", verification.absolute_tolerance},
            {"relative_tolerance", verification.relative_tolerance},
            {"evidence", md::ArtifactJson(verification.evidence)}}}};
}

Json Envelope(std::string_view kind, Json payload) {
  return Json{{"protocol", kProtocol},
              {"protocol_version", kProtocolVersion},
              {"kind", kind},
              {"payload", std::move(payload)}};
}

void ControlLimit(std::size_t maximum) {
  if (maximum == 0U || maximum > 64U * 1024U * 1024U)
    md::Invalid(kComponent, "control.limit", "a positive limit at most 64 MiB");
}

std::string Dump(const Json& json, std::size_t maximum) {
  ControlLimit(maximum);
  auto bytes = json.dump();
  if (bytes.size() > maximum)
    md::Resource(kComponent, "control.bytes", maximum);
  return bytes;
}

Json ReadEnvelope(std::string_view bytes, std::string_view kind,
                  std::size_t maximum) {
  auto envelope = md::Parse(bytes, maximum, kComponent);
  md::Fields(envelope, {"protocol", "protocol_version", "kind", "payload"},
             kComponent);
  if (md::String(envelope.at("protocol"), kComponent) != kProtocol ||
      md::Uint(envelope.at("protocol_version"), kComponent) != kProtocolVersion)
    throw CompatibilityError(
        "Unsupported export sidecar protocol",
        md::Details(kComponent, "protocol", "jyppx.export.sidecar v1"));
  if (md::String(envelope.at("kind"), kComponent) != kind)
    md::Malformed(kComponent, "control.kind",
                  "the exact expected message kind");
  return std::move(envelope.at("payload"));
}

std::uint32_t ReadU32(const Json& value) {
  const auto integer = md::Uint(value, kComponent);
  if (integer > (std::numeric_limits<std::uint32_t>::max)())
    md::Malformed(kComponent, "json.uint32", "a uint32 integer");
  return static_cast<std::uint32_t>(integer);
}

std::size_t ReadSize(const Json& value) {
  const auto integer = md::Uint(value, kComponent);
  if (integer > (std::numeric_limits<std::size_t>::max)())
    md::Malformed(kComponent, "json.size",
                  "a platform-sized nonnegative integer");
  return static_cast<std::size_t>(integer);
}

double ReadNumber(const Json& value) {
  if (!value.is_number())
    md::Malformed(kComponent, "json.number", "a finite JSON number");
  const auto number = value.get<double>();
  if (!std::isfinite(number))
    md::Malformed(kComponent, "json.number", "a finite JSON number");
  return number;
}

void ReadArray(const Json& value) {
  if (!value.is_array())
    md::Malformed(kComponent, "json.array", "a JSON array");
  Count(value.size(), true);
}

std::optional<ArtifactRef> ReadOptionalArtifact(const Json& value) {
  return value.is_null()
             ? std::nullopt
             : std::optional<ArtifactRef>(md::ReadArtifact(value, kComponent));
}

std::vector<BackendDependency> ReadDependencies(const Json& value) {
  ReadArray(value);
  std::vector<BackendDependency> result;
  result.reserve(value.size());
  for (const auto& item : value) {
    md::Fields(item, {"name", "version"}, kComponent);
    result.push_back({.name = md::String(item.at("name"), kComponent),
                      .version = md::String(item.at("version"), kComponent)});
  }
  return result;
}

std::vector<BackendOperation> ReadOperations(const Json& value) {
  ReadArray(value);
  std::vector<BackendOperation> result;
  result.reserve(value.size());
  for (const auto& item : value) result.push_back(ReadOperation(item));
  return result;
}

std::vector<std::string> ReadStrings(const Json& value) {
  ReadArray(value);
  std::vector<std::string> result;
  result.reserve(value.size());
  for (const auto& item : value) result.push_back(md::String(item, kComponent));
  return result;
}

ExportRequest ReadRequest(const Json& value) {
  md::Fields(value,
             {"schema_version",
              "request_id",
              "backend_id",
              "operation",
              "source_kind",
              "source",
              "state",
              "factory_entry_point",
              "target",
              "device",
              "dtype",
              "dynamic_shapes",
              "quantization_mode",
              "prepost_mode",
              "invocation_schema",
              "verification_cases",
              "options",
              "trust_source",
              "trace_opt_in",
              "environment_lock",
              "resources",
              "absolute_tolerance",
              "relative_tolerance"},
             kComponent);
  ExportRequest request;
  request.schema_version = ReadU32(value.at("schema_version"));
  request.request_id = md::String(value.at("request_id"), kComponent);
  request.backend_id = md::String(value.at("backend_id"), kComponent);
  request.operation = ReadOperation(value.at("operation"));
  request.source_kind = ReadSource(value.at("source_kind"));
  request.source = md::ReadArtifact(value.at("source"), kComponent);
  request.state = ReadOptionalArtifact(value.at("state"));
  request.factory_entry_point =
      md::String(value.at("factory_entry_point"), kComponent);
  request.target = ReadFormat(value.at("target"));
  request.device = md::String(value.at("device"), kComponent);
  request.dtype = md::String(value.at("dtype"), kComponent);
  request.dynamic_shapes = md::Boolean(value.at("dynamic_shapes"), kComponent);
  request.quantization_mode =
      md::String(value.at("quantization_mode"), kComponent);
  request.prepost_mode = md::String(value.at("prepost_mode"), kComponent);
  request.invocation_schema =
      ReadOptionalArtifact(value.at("invocation_schema"));
  ReadArray(value.at("verification_cases"));
  for (const auto& item : value.at("verification_cases"))
    request.verification_cases.push_back(md::ReadArtifact(item, kComponent));
  if (!value.at("options").is_object())
    md::Malformed(kComponent, "options", "a bounded string-to-string object");
  Count(value.at("options").size(), true);
  for (const auto& [key, item] : value.at("options").items())
    request.options.emplace(key, md::String(item, kComponent));
  request.trust_source = md::Boolean(value.at("trust_source"), kComponent);
  request.trace_opt_in = md::Boolean(value.at("trace_opt_in"), kComponent);
  request.environment_lock =
      md::ReadDigest(value.at("environment_lock"), kComponent);
  const auto& resources = value.at("resources");
  md::Fields(resources,
             {"cpu_threads", "memory_bytes", "wall_time_ms", "max_output_bytes",
              "max_control_bytes"},
             kComponent);
  request.resources.cpu_threads = ReadU32(resources.at("cpu_threads"));
  request.resources.memory_bytes =
      md::Uint(resources.at("memory_bytes"), kComponent);
  request.resources.wall_time_ms =
      md::Uint(resources.at("wall_time_ms"), kComponent);
  request.resources.max_output_bytes =
      md::Uint(resources.at("max_output_bytes"), kComponent);
  request.resources.max_control_bytes =
      ReadSize(resources.at("max_control_bytes"));
  request.absolute_tolerance = ReadNumber(value.at("absolute_tolerance"));
  request.relative_tolerance = ReadNumber(value.at("relative_tolerance"));
  ValidateRequest(request);
  return request;
}

BackendProbe ReadProbe(const Json& value) {
  md::Fields(
      value,
      {"schema_version", "backend_id", "backend_version", "protocol_version",
       "environment_lock", "dependencies", "operations", "devices", "dtypes",
       "dynamic_shapes_verified", "official_onnx_exporter", "onnx_checker",
       "onnx_runtime", "request_digest", "model_preflight_passed",
       "unsupported_operators", "evidence"},
      kComponent);
  BackendProbe probe;
  probe.schema_version = ReadU32(value.at("schema_version"));
  probe.backend_id = md::String(value.at("backend_id"), kComponent);
  probe.backend_version = md::String(value.at("backend_version"), kComponent);
  probe.protocol_version = ReadU32(value.at("protocol_version"));
  probe.environment_lock =
      md::ReadDigest(value.at("environment_lock"), kComponent);
  probe.dependencies = ReadDependencies(value.at("dependencies"));
  probe.operations = ReadOperations(value.at("operations"));
  probe.devices = ReadStrings(value.at("devices"));
  probe.dtypes = ReadStrings(value.at("dtypes"));
  probe.dynamic_shapes_verified =
      md::Boolean(value.at("dynamic_shapes_verified"), kComponent);
  probe.official_onnx_exporter =
      md::Boolean(value.at("official_onnx_exporter"), kComponent);
  probe.onnx_checker = md::Boolean(value.at("onnx_checker"), kComponent);
  probe.onnx_runtime = md::Boolean(value.at("onnx_runtime"), kComponent);
  if (!value.at("request_digest").is_null())
    probe.request_digest =
        md::ReadDigest(value.at("request_digest"), kComponent);
  probe.model_preflight_passed =
      md::Boolean(value.at("model_preflight_passed"), kComponent);
  probe.unsupported_operators = ReadStrings(value.at("unsupported_operators"));
  probe.evidence = md::ReadArtifact(value.at("evidence"), kComponent);
  ValidateProbe(probe);
  return probe;
}

ExportCandidate ReadCandidate(const Json& value) {
  md::Fields(value,
             {"schema_version", "request_id", "backend_id", "backend_version",
              "artifact", "verification"},
             kComponent);
  ExportCandidate candidate;
  candidate.schema_version = ReadU32(value.at("schema_version"));
  candidate.request_id = md::String(value.at("request_id"), kComponent);
  candidate.backend_id = md::String(value.at("backend_id"), kComponent);
  candidate.backend_version =
      md::String(value.at("backend_version"), kComponent);
  candidate.artifact = md::ReadArtifact(value.at("artifact"), kComponent);
  const auto& input = value.at("verification");
  md::Fields(input,
             {"request_digest", "artifact_digest", "environment_lock",
              "schema_passed", "checker_passed", "shape_inference",
              "shape_inference_reason", "runtime_passed", "numerical_passed",
              "cases_run", "cases_passed", "dynamic_cases_passed",
              "absolute_tolerance", "relative_tolerance", "evidence"},
             kComponent);
  auto& verification = candidate.verification;
  verification.request_digest =
      md::ReadDigest(input.at("request_digest"), kComponent);
  verification.artifact_digest =
      md::ReadDigest(input.at("artifact_digest"), kComponent);
  verification.environment_lock =
      md::ReadDigest(input.at("environment_lock"), kComponent);
  verification.schema_passed =
      md::Boolean(input.at("schema_passed"), kComponent);
  verification.checker_passed =
      md::Boolean(input.at("checker_passed"), kComponent);
  verification.shape_inference = ReadShapeStatus(input.at("shape_inference"));
  verification.shape_inference_reason =
      md::String(input.at("shape_inference_reason"), kComponent);
  verification.runtime_passed =
      md::Boolean(input.at("runtime_passed"), kComponent);
  verification.numerical_passed =
      md::Boolean(input.at("numerical_passed"), kComponent);
  verification.cases_run = md::Uint(input.at("cases_run"), kComponent);
  verification.cases_passed = md::Uint(input.at("cases_passed"), kComponent);
  verification.dynamic_cases_passed =
      md::Uint(input.at("dynamic_cases_passed"), kComponent);
  verification.absolute_tolerance = ReadNumber(input.at("absolute_tolerance"));
  verification.relative_tolerance = ReadNumber(input.at("relative_tolerance"));
  verification.evidence = md::ReadArtifact(input.at("evidence"), kComponent);
  ValidateCandidate(candidate);
  return candidate;
}

}  // namespace

ArtifactDigest backend_request_digest(const ExportRequest& request) {
  return Boundary([&] {
    ValidateRequest(request);
    const auto bytes = Dump(Envelope("export.request", RequestJson(request)),
                            request.resources.max_control_bytes);
    return ArtifactDigest{
        .algorithm = ArtifactDigestAlgorithm::sha256,
        .hex =
            sha256_hex(std::as_bytes(std::span(bytes.data(), bytes.size())))};
  });
}

std::string serialize_backend_descriptor(const BackendDescriptor& descriptor,
                                         std::size_t maximum_bytes) {
  return Boundary([&] {
    validate_backend_descriptor(descriptor);
    Json options = Json::object();
    for (const auto& [key, values] : descriptor.allowed_options)
      options[key] = StringsJson(values);
    return Dump(
        Json{{"schema_version", descriptor.schema_version},
             {"status", "declared-unprobed"},
             {"backend_id", descriptor.backend_id},
             {"backend_version", descriptor.backend_version},
             {"role", descriptor.role == BackendRole::exporter ? "exporter"
                                                               : "runtime"},
             {"execution",
              descriptor.execution == BackendExecution::out_of_process
                  ? "out_of_process"
                  : "in_process"},
             {"protocol_version", descriptor.protocol_version},
             {"environment_lock", md::DigestJson(descriptor.environment_lock)},
             {"formats", EnumsJson(descriptor.formats)},
             {"sources", EnumsJson(descriptor.sources)},
             {"operations", EnumsJson(descriptor.operations)},
             {"devices", StringsJson(descriptor.devices)},
             {"dtypes", StringsJson(descriptor.dtypes)},
             {"dynamic_shapes", descriptor.dynamic_shapes},
             {"quantization_modes", StringsJson(descriptor.quantization_modes)},
             {"prepost_modes", StringsJson(descriptor.prepost_modes)},
             {"allowed_options", std::move(options)},
             {"dependencies", DependenciesJson(descriptor.dependencies)},
             {"compatibility_only", descriptor.compatibility_only},
             {"deprecated", descriptor.deprecated},
             {"max_artifact_bytes", descriptor.max_artifact_bytes},
             {"max_verification_cases", descriptor.max_verification_cases}},
        maximum_bytes);
  });
}

std::string serialize_sidecar_request(const ExportRequest& request) {
  return Boundary([&] {
    ValidateRequest(request);
    return Dump(Envelope("export.request", RequestJson(request)),
                request.resources.max_control_bytes);
  });
}

ExportRequest deserialize_sidecar_request(std::string_view json,
                                          std::size_t maximum_bytes) {
  return Boundary(
      [&] {
        auto request =
            ReadRequest(ReadEnvelope(json, "export.request", maximum_bytes));
        if (json.size() > request.resources.max_control_bytes)
          md::Resource(kComponent, "request.control.bytes",
                       request.resources.max_control_bytes);
        return request;
      },
      true);
}

std::string serialize_sidecar_probe(const BackendProbe& probe,
                                    std::size_t maximum_bytes) {
  return Boundary([&] {
    ValidateProbe(probe);
    return Dump(Envelope("probe.result", ProbeJson(probe)), maximum_bytes);
  });
}

BackendProbe deserialize_sidecar_probe(std::string_view json,
                                       std::size_t maximum_bytes) {
  return Boundary(
      [&] {
        return ReadProbe(ReadEnvelope(json, "probe.result", maximum_bytes));
      },
      true);
}

std::string serialize_sidecar_candidate(const ExportCandidate& candidate,
                                        std::size_t maximum_bytes) {
  return Boundary([&] {
    ValidateCandidate(candidate);
    return Dump(Envelope("export.candidate", CandidateJson(candidate)),
                maximum_bytes);
  });
}

ExportCandidate deserialize_sidecar_candidate(std::string_view json,
                                              std::size_t maximum_bytes) {
  return Boundary(
      [&] {
        return ReadCandidate(
            ReadEnvelope(json, "export.candidate", maximum_bytes));
      },
      true);
}

void validate_export_candidate_metadata(const BackendDescriptor& descriptor,
                                        const ExportRequest& request,
                                        const BackendProbe& probe,
                                        const ExportCandidate& candidate) {
  Boundary([&] {
    const auto support = preflight_backend(descriptor, request, &probe);
    ValidateCandidate(candidate);
    const auto reject = [](std::string_view operation) {
      throw CompatibilityError(
          "Export candidate metadata is not ready for independent "
          "verification/publication",
          md::Details(kComponent, operation,
                      "complete official verification bound to exact inputs"));
    };
    if (support.decision != SupportDecision::supported) reject("preflight");
    if (request.operation != BackendOperation::export_model ||
        request.target != ArtifactFormat::onnx ||
        descriptor.role != BackendRole::exporter)
      reject("onnx.scope");
    if (candidate.request_id != request.request_id ||
        candidate.backend_id != descriptor.backend_id ||
        candidate.backend_version != descriptor.backend_version)
      reject("identity");
    if (candidate.artifact.media_type != "application/onnx" ||
        candidate.artifact.size_bytes > descriptor.max_artifact_bytes ||
        candidate.artifact.size_bytes > request.resources.max_output_bytes)
      reject("artifact");
    const auto& verification = candidate.verification;
    if (verification.evidence.size_bytes >
        request.resources.max_output_bytes - candidate.artifact.size_bytes)
      reject("total_output_bytes");
    if (!(verification.request_digest == backend_request_digest(request)) ||
        !(verification.artifact_digest == candidate.artifact.digest) ||
        !(verification.environment_lock == request.environment_lock))
      reject("digest_binding");
    if (!verification.schema_passed || !verification.checker_passed ||
        !verification.runtime_passed || !verification.numerical_passed)
      reject("official_stages");
    if (verification.shape_inference != ShapeInferenceStatus::passed &&
        !(verification.shape_inference ==
              ShapeInferenceStatus::not_applicable &&
          !verification.shape_inference_reason.empty()))
      reject("shape_inference");
    if (verification.cases_run != request.verification_cases.size() ||
        verification.cases_passed != verification.cases_run)
      reject("verification_cases");
    if ((request.dynamic_shapes &&
         (verification.dynamic_cases_passed == 0U ||
          verification.dynamic_cases_passed >= verification.cases_run)) ||
        (!request.dynamic_shapes && verification.dynamic_cases_passed != 0U))
      reject("dynamic_cases");
    if (verification.absolute_tolerance != request.absolute_tolerance ||
        verification.relative_tolerance != request.relative_tolerance)
      reject("numeric_tolerance");
  });
}

}  // namespace jyppx::liuding::interop
