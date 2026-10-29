// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#if defined(JYPPX_LIUDING_CLI_ONNX_EXPORT) && \
    defined(JYPPX_LIUDING_CLI_ONNX_RUNTIME)
#include <jyppx/liuding/deployment/onnx_export/report.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

namespace jyppx::liuding::cli {
namespace {
constexpr std::string_view kRuntimeBackend = "jyppx.liuding.onnx_runtime";
constexpr std::string_view kRuntimeLock = "jyppx.liuding.onnx_runtime_lock";

std::string Lower(std::string value) {
  for (auto& byte : value)
    if (byte >= 'A' && byte <= 'Z') byte = static_cast<char>(byte - 'A' + 'a');
  return value;
}

std::span<const std::byte> Bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

std::string CheckedFile(const Arguments& arguments, const char* path,
                        const char* digest, std::size_t maximum) {
  const ArtifactDigest expected{.hex = arguments.require(digest)};
  validate_artifact_digest(expected);
  auto data = read_text(arguments.require(path), maximum);
  if (sha256_hex(Bytes(data)) != expected.hex)
    throw SerializationError("ONNX verification input SHA-256 mismatch");
  return data;
}

std::string_view TextBytes(std::span<const std::byte> bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

struct VerificationInputs {
  std::string standalone;
  std::string external_report;
  std::string external_cases;
  std::string external_tensors;
  std::optional<interop::ModelBundleSnapshot> bundle;
  std::string model_id;
  std::string file_id;
  std::string report_id;
  std::string cases_id;
  std::string tensors_id;
  std::string runtime_id;

  std::string_view bytes() const {
    return bundle ? TextBytes(bundle->files.at(file_id)) : standalone;
  }
  std::string_view report() const {
    return report_id.empty() ? external_report
                             : TextBytes(bundle->files.at(report_id));
  }
  std::string_view cases() const {
    return cases_id.empty() ? external_cases
                            : TextBytes(bundle->files.at(cases_id));
  }
  std::string_view tensors() const {
    return tensors_id.empty() ? external_tensors
                              : TextBytes(bundle->files.at(tensors_id));
  }

  bool contained() const { return !report_id.empty(); }

  Json document(bool runtime_checked = false) const {
    if (!bundle) return nullptr;
    const auto& verified = bundle->verification;
    const auto status = verified.signature_status;
    return {
        {"manifest_sha256", verified.manifest_digest.hex},
        {"bundle_id", verified.manifest.bundle_id},
        {"model_id", model_id},
        {"file_id", file_id},
        {"signature_status",
         status == interop::BundleSignatureStatus::unsigned_bundle ? "unsigned"
         : status == interop::BundleSignatureStatus::trusted
             ? "trusted"
             : "present_unverified"},
        {"authenticated", status == interop::BundleSignatureStatus::trusted},
        {"descriptor_payloads_validated", false},
        {"runtime_requirements_checked", runtime_checked}};
  }

  Json evidence() const {
    if (!contained()) return nullptr;
    return {{"source", "verified_bundle_snapshot"},
            {"report_file_id", report_id},
            {"cases_file_id", cases_id},
            {"tensors_file_id", tensors_id},
            {"export_report_schema_validated", true},
            {"export_report_model_identity_verified", true},
            {"official_onnx_checker", "passed"},
            {"case_schema_validated", true},
            {"tensor_inventory_validated", true},
            {"case_bindings_executed", true}};
  }
};

void ReadBundle(const Arguments& arguments, VerificationInputs& result,
                std::uint64_t model_bytes) {
  const auto directory = std::filesystem::u8path(arguments.require("bundle"));
  const ArtifactDigest digest{.hex = arguments.require("bundle-sha256")};
  validate_artifact_digest(digest);
  result.model_id = arguments.require("model-id");
  interop::ModelBundleVerifyOptions options;
  options.limits.max_manifest_bytes = 4U << 20U;
  options.limits.max_file_bytes = 256U << 20U;
  options.limits.max_total_bytes = 512U << 20U;
  std::uint64_t retained_limit = model_bytes + (4U << 20U);
  // Discovery validates the whole bundle. Final retained bytes are read and
  // hashed together, again bound to the caller's original manifest digest.
  {
    const auto discovered = interop::read_model_bundle_snapshot(
        directory, digest, {}, options, 4U << 20U);
    const auto& manifest = discovered.verification.manifest;
    const auto found = std::find_if(
        manifest.models.begin(), manifest.models.end(),
        [&](const auto& model) { return model.id == result.model_id; });
    if (found == manifest.models.end() ||
        found->format != interop::BundleModelFormat::onnx)
      throw CompatibilityError("Select an existing single-file ONNX variant");
    result.file_id = found->file_ids.front();
    const auto size = [&](const std::string& id, std::uint64_t maximum) {
      const auto file =
          std::find_if(manifest.files.begin(), manifest.files.end(),
                       [&](const auto& entry) { return entry.id == id; });
      if (file == manifest.files.end())
        throw InvalidArgument("Unknown ONNX bundle file id");
      if (file->artifact.size_bytes > maximum)
        throw ResourceExhausted(
            "ONNX bundle verification input exceeds its byte limit");
      return file->artifact.size_bytes;
    };
    static_cast<void>(size(result.file_id, model_bytes));
    if (result.contained()) {
      if (manifest.schema_version != 2U)
        throw CompatibilityError(
            "Bundle-contained ONNX evidence requires schema v2");
      const auto evidence = [&](const std::string& id, std::string_view schema,
                                std::uint64_t maximum) {
        const auto ref = std::find_if(
            manifest.evidence.begin(), manifest.evidence.end(),
            [&](const auto& entry) { return entry.file_id == id; });
        if (ref == manifest.evidence.end() || ref->schema_id != schema ||
            ref->schema_version != 1U)
          throw CompatibilityError(
              "ONNX evidence needs an explicit supported owner schema and "
              "version");
        retained_limit += size(id, maximum);
      };
      evidence(result.report_id, "jyppx.liuding.onnx_export_report", 4U << 20U);
      evidence(result.cases_id, "jyppx.liuding.onnx_verification_cases",
               4U << 20U);
      evidence(result.tensors_id, "jyppx.liuding.onnx_verification_tensors",
               64U << 20U);
    }
    const bool owns_runtime_lock = std::any_of(
        manifest.evidence.begin(), manifest.evidence.end(),
        [](const auto& ref) { return ref.schema_id == kRuntimeLock; });
    if (owns_runtime_lock && manifest.runtime.backend_id != kRuntimeBackend)
      throw CompatibilityError(
          "ONNX runtime lock requires its declared backend");
    if (manifest.runtime.backend_id == kRuntimeBackend) {
      for (const auto& ref : manifest.evidence) {
        if (ref.schema_id != kRuntimeLock) continue;
        if (!result.runtime_id.empty() || ref.schema_version != 1U)
          throw CompatibilityError(
              "Select exactly one supported ONNX runtime lock");
        result.runtime_id = ref.file_id;
      }
      if (result.runtime_id.empty())
        throw CompatibilityError(
            "ONNX runtime declaration requires its environment lock evidence");
      retained_limit += size(result.runtime_id, 64U << 10U);
    }
  }
  std::vector<std::string_view> selected{result.file_id};
  if (result.contained()) {
    selected.insert(selected.end(),
                    {result.report_id, result.cases_id, result.tensors_id});
  }
  if (!result.runtime_id.empty()) selected.push_back(result.runtime_id);
  result.bundle = interop::read_model_bundle_snapshot(
      directory, digest, selected, options, retained_limit);
}

VerificationInputs Inputs(const Arguments& arguments, bool smoke) {
  VerificationInputs result;
  const bool contained = arguments.optional("report-id").has_value() ||
                         arguments.optional("cases-id").has_value() ||
                         arguments.optional("tensors-id").has_value();
  if (smoke && !contained)
    throw InvalidArgument(
        "ONNX smoke requires bundle-contained report, cases and tensors ids");
  if (contained) {
    if (!arguments.optional("bundle"))
      throw InvalidArgument("ONNX evidence file ids require --bundle");
    for (const auto* option : {"path", "report", "report-sha256", "cases",
                               "cases-sha256", "tensors", "tensors-sha256"})
      if (arguments.optional(option))
        throw InvalidArgument(
            "Do not mix bundle evidence ids and external ONNX verification "
            "inputs");
    result.report_id = arguments.require("report-id");
    result.cases_id = arguments.require("cases-id");
    result.tensors_id = arguments.require("tensors-id");
    if (result.report_id.empty() || result.cases_id.empty() ||
        result.tensors_id.empty())
      throw InvalidArgument("ONNX evidence ids must be nonempty");
    ReadBundle(arguments, result, 256U << 20U);
  } else {
    result.external_report =
        CheckedFile(arguments, "report", "report-sha256", 4U << 20U);
    const auto report =
        onnx_export::deserialize_export_report(result.external_report);
    if (report.model_size_bytes > (256U << 20U))
      throw ResourceExhausted("ONNX verification model exceeds 256 MiB");
    if (arguments.optional("bundle")) {
      if (arguments.optional("path"))
        throw InvalidArgument("Choose exactly one ONNX --path or --bundle");
      ReadBundle(arguments, result, report.model_size_bytes);
    } else {
      if (arguments.optional("bundle-sha256") || arguments.optional("model-id"))
        throw InvalidArgument("Bundle identity options require --bundle");
      result.standalone =
          read_text(arguments.require("path"),
                    static_cast<std::size_t>(report.model_size_bytes));
    }
    result.external_cases =
        CheckedFile(arguments, "cases", "cases-sha256", 4U << 20U);
    result.external_tensors =
        CheckedFile(arguments, "tensors", "tensors-sha256", 64U << 20U);
  }
  return result;
}

struct OwnedTensor {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

struct OwnedCase {
  std::string id;
  std::vector<OwnedTensor> inputs;
  std::vector<OwnedTensor> expected;
  std::vector<onnx_runtime::TensorInput> input_views;
  std::vector<onnx_runtime::TensorInput> expected_views;
};

std::vector<OwnedTensor> Tensors(const Json& value,
                                 const interop::SafeTensorsView& tensors,
                                 std::set<std::string>& used,
                                 std::uint64_t& remaining) {
  if (!value.is_array() || value.empty() || value.size() > 64U)
    throw SerializationError("ONNX cases need bounded nonempty tensor arrays");
  std::vector<OwnedTensor> result;
  result.reserve(value.size());
  for (const auto& binding : value) {
    fields(binding, {"name", "tensor"});
    if (!binding.at("name").is_string() || !binding.at("tensor").is_string())
      throw SerializationError("ONNX case tensor names must be strings");
    const auto key = binding.at("tensor").get<std::string>();
    const auto& infos = tensors.directory().tensors;
    const auto found =
        std::find_if(infos.begin(), infos.end(),
                     [&](const auto& info) { return info.name == key; });
    if (found == infos.end() || found->dtype != interop::SafeTensorDtype::f32 ||
        found->shape.empty() || found->shape.size() > 8U)
      throw CompatibilityError(
          "ONNX verification requires named FP32 tensors of rank 1-8");
    const auto data = tensors.tensor_bytes(key);
    if (data.size() > remaining)
      throw ResourceExhausted("ONNX verification suite payload exceeds 64 MiB");
    remaining -= data.size();
    OwnedTensor tensor{.name = binding.at("name").get<std::string>(),
                       .shape = {},
                       .values = {}};
    for (const auto extent : found->shape) {
      if (extent == 0U || extent > (1ULL << 32U))
        throw InvalidArgument(
            "ONNX verification tensor dimensions are invalid");
      tensor.shape.push_back(static_cast<std::int64_t>(extent));
    }
    static_assert(std::endian::native == std::endian::little);
    tensor.values.resize(data.size() / sizeof(float));
    std::memcpy(tensor.values.data(), data.data(), data.size());
    used.insert(key);
    result.push_back(std::move(tensor));
  }
  return result;
}

struct TraceCleanup {
  std::filesystem::path directory;
  ~TraceCleanup() {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
  }
};

std::filesystem::path TraceDirectory() {
  const auto root = std::filesystem::temp_directory_path();
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  for (int attempt = 0; attempt < 128; ++attempt) {
    const auto directory =
        root / ("liuding-onnx-verify-" + std::to_string(tick) + "-" +
                std::to_string(attempt));
    std::error_code error;
    if (std::filesystem::create_directory(directory, error)) {
#if !defined(_WIN32)
      std::filesystem::permissions(directory, std::filesystem::perms::owner_all,
                                   std::filesystem::perm_options::replace,
                                   error);
      if (error) {
        std::error_code ignored;
        std::filesystem::remove(directory, ignored);
        throw IoError("Cannot restrict ONNX profile directory permissions");
      }
#endif
      return directory;
    }
    if (error)
      throw IoError("Cannot create ONNX verification profile directory");
  }
  throw IoError("Cannot allocate unique ONNX verification profile directory");
}
struct PreparedVerification {
  VerificationInputs inputs;
  onnx_export::ExportReport export_report;
  onnx_runtime::VerificationOptions policy;
  std::vector<OwnedCase> owned;
  std::vector<onnx_runtime::VerificationCase> cases;
};

PreparedVerification Prepare(const Arguments& arguments, bool smoke) {
  PreparedVerification result;
  result.inputs = Inputs(arguments, smoke);
  result.export_report =
      onnx_export::deserialize_export_report(result.inputs.report());
  onnx_export::verify_export_report_model(result.export_report,
                                          Bytes(result.inputs.bytes()));
  const auto case_json = result.inputs.cases();
  const auto tensor_bytes = result.inputs.tensors();
  interop::SafeTensorsLimits tensor_limits;
  tensor_limits.max_file_bytes = 64U << 20U;
  tensor_limits.max_payload_bytes = 64U << 20U;
  tensor_limits.max_tensor_bytes = 64U << 20U;
  tensor_limits.max_tensors = 8192U;
  const interop::SafeTensorsView tensors(Bytes(tensor_bytes), tensor_limits);
  const auto suite = parse_json(case_json);
  fields(suite, {"format", "schema_version", "reference", "absolute_tolerance",
                 "relative_tolerance", "cases"});
  if (suite.at("format") != "jyppx.liuding.onnx_verification_cases" ||
      unsigned_number(suite.at("schema_version"), 1U) != 1U ||
      !suite.at("reference").is_string() ||
      !suite.at("absolute_tolerance").is_number() ||
      !suite.at("relative_tolerance").is_number())
    throw SerializationError("Invalid ONNX verification suite schema");
  auto& policy = result.policy;
  policy.reference = suite.at("reference").get<std::string>();
  policy.absolute_tolerance = suite.at("absolute_tolerance").get<double>();
  policy.relative_tolerance = suite.at("relative_tolerance").get<double>();
  policy.max_case_bytes = 64U << 20U;
  const auto& case_array = suite.at("cases");
  if (!case_array.is_array() || case_array.empty() || case_array.size() > 64U)
    throw ResourceExhausted("ONNX verification suite requires 1-64 cases");
  auto& owned = result.owned;
  owned.reserve(case_array.size());
  std::set<std::string> used;
  std::uint64_t remaining = 64U << 20U;
  for (const auto& item : case_array) {
    fields(item, {"id", "inputs", "expected"});
    if (!item.at("id").is_string())
      throw SerializationError("Invalid ONNX case ID");
    owned.push_back(
        {.id = item.at("id").get<std::string>(),
         .inputs = Tensors(item.at("inputs"), tensors, used, remaining),
         .expected = Tensors(item.at("expected"), tensors, used, remaining),
         .input_views = {},
         .expected_views = {}});
  }
  if (used.size() != tensors.directory().tensors.size())
    throw InvalidArgument(
        "ONNX verification tensor file has undeclared tensors");
  auto& cases = result.cases;
  cases.reserve(owned.size());
  for (auto& item : owned) {
    for (const auto& tensor : item.inputs)
      item.input_views.push_back({.name = tensor.name,
                                  .values = tensor.values,
                                  .shape = tensor.shape});
    for (const auto& tensor : item.expected)
      item.expected_views.push_back({.name = tensor.name,
                                     .values = tensor.values,
                                     .shape = tensor.shape});
    cases.push_back({.id = item.id,
                     .inputs = item.input_views,
                     .expected = item.expected_views});
  }
  return result;
}

Json RuntimeDeclaration(
    const interop::BundleRuntimeRequirements& requirements) {
  Json dependencies = Json::array();
  auto sorted = requirements.dependencies;
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.name < b.name; });
  for (const auto& entry : sorted)
    dependencies.push_back({{"name", entry.name}, {"version", entry.version}});
  return {{"backend_id", requirements.backend_id},
          {"backend_version", requirements.backend_version},
          {"dependencies", std::move(dependencies)},
          {"devices", requirements.devices},
          {"dtypes", requirements.dtypes},
          {"os_arch", requirements.os_arch},
          {"dynamic_shapes", requirements.dynamic_shapes}};
}

Json TensorSchema(std::span<const onnx_runtime::TensorSignature> signatures,
                  const onnx_export::ExportReport& report) {
  Json tensors = Json::array();
  for (const auto& tensor : signatures)
    tensors.push_back({{"name", tensor.name},
                       {"shape", tensor.shape},
                       {"batch_symbol", tensor.batch_symbol},
                       {"dtype", "float32"}});
  return {{"format", "jyppx.liuding.onnx_tensor_schema"},
          {"schema_version", 1},
          {"tensors", std::move(tensors)},
          {"minimum_batch", report.minimum_batch},
          {"maximum_batch", report.maximum_batch}};
}

Json ObserveEnvironment(const onnx_runtime::Session& session,
                        const BuildFingerprint& build) {
  const auto& provider = session.execution_provider();
  const auto domain = provider.name == "CPUExecutionProvider" ? "cpu"
                      : provider.name == "CUDAExecutionProvider" ||
                              provider.name == "TensorrtExecutionProvider"
                          ? "cuda"
                          : Lower(provider.name);
  const auto dynamic = [](auto signatures) {
    return std::any_of(
        signatures.begin(), signatures.end(),
        [](const auto& tensor) { return !tensor.batch_symbol.empty(); });
  };
  return {{"format", kRuntimeLock},
          {"schema_version", 1},
          {"backend_id", kRuntimeBackend},
          {"backend_version", build.jyppx_version},
          {"dependencies",
           Json::array(
               {{{"name", "libtorch"}, {"version", build.libtorch_version}},
                {{"name", "liuding-forge"}, {"version", build.jyppx_version}},
                {{"name", "onnxruntime"},
                 {"version", session.runtime_version()}}})},
          {"devices", Json::array({domain})},
          {"dtypes", Json::array({"float32"})},
          {"os_arch", Json::array({Lower(build.os_arch)})},
          {"dynamic_shapes",
           dynamic(session.inputs()) || dynamic(session.outputs())},
          {"provider",
           {{"name", provider.name},
            {"options", provider.options},
            {"allow_cpu_fallback", session.cpu_fallback_allowed()}}}};
}

void CheckEnvironment(const interop::BundleRuntimeRequirements& requirements,
                      std::string_view encoded, const Json& observed) {
  if (sha256_hex(Bytes(encoded)) != requirements.environment_lock.hex)
    throw CompatibilityError("ONNX runtime environment lock digest mismatch");
  auto lock = parse_json(encoded);
  fields(lock, {"format", "schema_version", "backend_id", "backend_version",
                "dependencies", "devices", "dtypes", "os_arch",
                "dynamic_shapes", "provider"});
  if (lock != observed)
    throw CompatibilityError(
        "ONNX runtime environment does not match this execution point");
  lock.erase("format");
  lock.erase("schema_version");
  lock.erase("provider");
  if (lock != RuntimeDeclaration(requirements))
    throw CompatibilityError(
        "ONNX bundle runtime declaration differs from its checked lock");
}

interop::BundleRuntimeRequirements Requirements(const Json& lock,
                                                std::string_view encoded) {
  interop::BundleRuntimeRequirements result;
  result.backend_id = lock.at("backend_id").get<std::string>();
  result.backend_version = lock.at("backend_version").get<std::string>();
  result.environment_lock.hex = sha256_hex(Bytes(encoded));
  for (const auto& dependency : lock.at("dependencies"))
    result.dependencies.push_back(
        {dependency.at("name").get<std::string>(),
         dependency.at("version").get<std::string>()});
  result.devices = lock.at("devices").get<std::vector<std::string>>();
  result.dtypes = lock.at("dtypes").get<std::vector<std::string>>();
  result.os_arch = lock.at("os_arch").get<std::vector<std::string>>();
  result.dynamic_shapes = lock.at("dynamic_shapes").get<bool>();
  return result;
}

struct ExecutedVerification {
  onnx_runtime::VerificationReport report;
  std::string profile_json;
  Json document;
  std::string encoded;
  bool passed = false;
  bool runtime_checked = false;
  Json environment;
  Json input_schema;
  Json output_schema;
};

ExecutedVerification Execute(const Arguments& arguments,
                             const PreparedVerification& prepared, bool smoke,
                             bool capture_environment = false) {
  const auto& export_report = prepared.export_report;
  const auto& owner = prepared.inputs;
  const auto& cases = prepared.cases;
  onnx_runtime::SessionOptions options;
  options.provider.name = arguments.require("provider");
  options.max_batch_size =
      static_cast<std::uint64_t>(export_report.maximum_batch);
  for (const auto& setting : arguments.repeated("provider-option")) {
    const auto separator = setting.find('=');
    if (separator == 0U || separator == std::string::npos ||
        separator + 1U == setting.size() ||
        !options.provider.options
             .emplace(setting.substr(0, separator),
                      setting.substr(separator + 1U))
             .second)
      throw InvalidArgument("Use unique provider-option key=value pairs");
  }
  for (const auto& item : cases)
    for (const auto& input : item.inputs)
      if (input.shape.front() < export_report.minimum_batch)
        throw InvalidArgument(
            "ONNX verification batch is below the export policy");
  const auto timeout = arguments.number("timeout-ms", 60000U, 3600000U);
  if (timeout == 0U)
    throw InvalidArgument("ONNX verification timeout must be positive");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
  const TraceCleanup temporary{TraceDirectory()};
  options.profile_prefix = temporary.directory / "actual";
  onnx_runtime::Session session(Bytes(owner.bytes()),
                                export_report.model_digest, options);
  Json environment = nullptr;
  const bool runtime_checked = !owner.runtime_id.empty();
  if (runtime_checked || capture_environment)
    environment = ObserveEnvironment(session, current_build_fingerprint());
  if (runtime_checked)
    CheckEnvironment(owner.bundle->verification.manifest.runtime,
                     TextBytes(owner.bundle->files.at(owner.runtime_id)),
                     environment);
  const auto input_schema = TensorSchema(session.inputs(), export_report);
  const auto output_schema = TensorSchema(session.outputs(), export_report);
  auto report = onnx_runtime::verify_session(session, cases, prepared.policy,
                                             {}, deadline);
  const auto profile_path = session.end_profiling();
  auto profile_json = read_text(profile_path, 16U << 20U);
  report.profile = onnx_runtime::inspect_profile(profile_json);
  if (report.profile->kernels.size() != 1U ||
      !report.profile->kernels.contains(options.provider.name))
    throw CompatibilityError(
        "ORT trace does not exclusively observe the requested provider");
  Json document{
      {"format", "jyppx.liuding.onnx_verification_evidence"},
      {"schema_version", runtime_checked     ? 4
                         : owner.contained() ? 3
                                             : 2},
      {"input_bundle", owner.document(runtime_checked)},
      {"source", Json::parse(owner.report()).at("source")},
      {"export_report_sha256", sha256_hex(Bytes(owner.report()))},
      {"cases_sha256", sha256_hex(Bytes(owner.cases()))},
      {"tensors_sha256", sha256_hex(Bytes(owner.tensors()))},
      {"execution",
       Json::parse(onnx_runtime::serialize_verification_report(report))}};
  if (owner.contained()) {
    document["verification_inputs"] = owner.evidence();
    document["source_run_lineage_verified"] = !smoke;
  }
  if (runtime_checked) {
    document["source_run_lineage_verified"] = !smoke;
    document["runtime_requirements"] = {
        {"checked", true},
        {"file_id", owner.runtime_id},
        {"environment_lock_sha256",
         owner.bundle->verification.manifest.runtime.environment_lock.hex},
        {"hardware_resource_qualification", false}};
  }
  auto encoded = document.dump();
  if (encoded.size() > (4U << 20U))
    throw ResourceExhausted("ONNX validation evidence exceeds 4 MiB");
  const bool passed = onnx_runtime::verification_passed(report);
  return {.report = std::move(report),
          .profile_json = std::move(profile_json),
          .document = std::move(document),
          .encoded = std::move(encoded),
          .passed = passed,
          .runtime_checked = runtime_checked,
          .environment = std::move(environment),
          .input_schema = input_schema,
          .output_schema = output_schema};
}

Json DeliveryMetadata(std::string_view encoded) {
  auto metadata = parse_json(encoded);
  fields(metadata,
         {"format", "schema_version", "bundle_id", "checkpoint",
          "preprocessing", "postprocessing", "labels", "evaluation", "sbom"});
  if (metadata.at("format") != "jyppx.liuding.onnx_delivery_metadata" ||
      unsigned_number(metadata.at("schema_version"), 4294967295ULL) != 1U ||
      !metadata.at("bundle_id").is_string())
    throw SerializationError("Invalid ONNX delivery metadata schema");
  const auto& checkpoint = metadata.at("checkpoint");
  fields(checkpoint, {"sha256", "size_bytes"});
  if (!checkpoint.at("sha256").is_string() ||
      unsigned_number(checkpoint.at("size_bytes"),
                      (std::numeric_limits<std::uint64_t>::max)()) == 0U)
    throw SerializationError(
        "Declare a nonempty source checkpoint content reference");
  validate_artifact_digest({.hex = checkpoint.at("sha256").get<std::string>()});
  for (const auto key :
       {"preprocessing", "postprocessing", "labels", "evaluation", "sbom"}) {
    const auto& descriptor = metadata.at(key);
    if (std::string_view(key) == "labels" && descriptor.is_null()) continue;
    fields(descriptor, {"schema_id", "schema_version", "content"});
    if (!descriptor.at("schema_id").is_string() ||
        unsigned_number(descriptor.at("schema_version"), 4294967295ULL) == 0U ||
        !descriptor.at("content").is_object())
      throw SerializationError(
          "Use explicit owner-versioned JSON delivery descriptors");
  }
  return metadata;
}

Result Pack(const Arguments& arguments, const Json* reconstruction = nullptr,
            const std::filesystem::path& definition = {},
            const std::filesystem::path& inputs = {},
            const std::filesystem::path& tensors = {}) {
  arguments.allow({"path", "report", "report-sha256", "cases", "cases-sha256",
                   "tensors", "tensors-sha256", "provider", "provider-option",
                   "timeout-ms", "bundle", "bundle-sha256", "model-id",
                   "report-id", "cases-id", "tensors-id", "metadata",
                   "metadata-sha256", "output"});
  const auto metadata_bytes =
      CheckedFile(arguments, "metadata", "metadata-sha256", 4U << 20U);
  const auto metadata = DeliveryMetadata(metadata_bytes);
  if (reconstruction && (metadata.at("checkpoint").at("sha256") !=
                             reconstruction->at("checkpoint").at("sha256") ||
                         metadata.at("checkpoint").at("size_bytes") !=
                             reconstruction->at("checkpoint").at("size_bytes")))
    throw CompatibilityError(
        "Delivery checkpoint differs from the restored original run "
        "checkpoint");
  const auto output = std::filesystem::u8path(arguments.require("output"));
  std::error_code error;
  if (std::filesystem::symlink_status(output, error).type() !=
      std::filesystem::file_type::not_found)
    throw IoError("ONNX package output must be a new directory");
  const auto prepared = Prepare(arguments, false);
  const auto& source = prepared.export_report.source;
  if (!source.run_id || !source.source_digest || !source.config_digest ||
      !source.data_manifest_digest)
    throw InvalidArgument(
        "ONNX delivery requires explicit training run/code/config/data source "
        "references");
  const auto executed = Execute(arguments, prepared, true, true);
  if (!executed.passed)
    return {Json{{"operation", "model.onnx.pack"},
                 {"published", false},
                 {"numerical", "failed"},
                 {"runtime_executed", true},
                 {"source_authenticated", false},
                 {"report", executed.document},
                 {"runtime_profile", executed.profile_json}},
            4};
  const auto lock_bytes = executed.environment.dump();
  if (lock_bytes.size() > (64U << 10U))
    throw ResourceExhausted("ONNX runtime lock exceeds 64 KiB");
  interop::ModelBundleManifest manifest;
  manifest.bundle_id = metadata.at("bundle_id").get<std::string>();
  manifest.build = executed.report.build;
  manifest.runtime = Requirements(executed.environment, lock_bytes);
  CheckEnvironment(manifest.runtime, lock_bytes, executed.environment);
  const TraceCleanup temporary{TraceDirectory()};
  std::vector<interop::BundlePublishSource> sources;
  const auto add = [&](const std::string& id, interop::BundleFileRole role,
                       std::string path, std::string_view data,
                       std::string media) {
    const auto local = temporary.directory / id;
    std::ofstream file(local, std::ios::binary);
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    file.close();
    if (!file) throw IoError("Cannot stage ONNX bundle data");
    manifest.files.push_back(
        {.id = id,
         .relative_path = std::move(path),
         .role = role,
         .artifact = {.digest = {.hex = sha256_hex(Bytes(data))},
                      .size_bytes = data.size(),
                      .logical_name = id,
                      .media_type = std::move(media),
                      .producer = "onnx.delivery.native"}});
    sources.push_back({.file_id = id, .source_path = local});
  };
  const auto descriptor = [&](const std::string& id,
                              interop::BundleFileRole role, std::string schema,
                              std::uint32_t version, const Json& content) {
    add(id, role, "metadata/" + id + ".json", content.dump(),
        "application/json");
    return interop::BundleDescriptorRef{.schema_id = std::move(schema),
                                        .schema_version = version,
                                        .file_id = id};
  };
  const auto supplied = [&](const char* key, interop::BundleFileRole role) {
    const auto& item = metadata.at(key);
    return descriptor(key, role, item.at("schema_id").get<std::string>(),
                      item.at("schema_version").get<std::uint32_t>(),
                      item.at("content"));
  };
  add("model", interop::BundleFileRole::model, "models/model.onnx",
      prepared.inputs.bytes(), "application/onnx");
  manifest.models.push_back({.id = "primary",
                             .format = interop::BundleModelFormat::onnx,
                             .file_ids = {"model"},
                             .tensors = {}});
  manifest.input_schema =
      descriptor("input", interop::BundleFileRole::schema,
                 "jyppx.liuding.onnx_tensor_schema", 1U, executed.input_schema);
  manifest.output_schema = descriptor("output", interop::BundleFileRole::schema,
                                      "jyppx.liuding.onnx_tensor_schema", 1U,
                                      executed.output_schema);
  manifest.preprocessing =
      supplied("preprocessing", interop::BundleFileRole::prepost);
  manifest.postprocessing =
      supplied("postprocessing", interop::BundleFileRole::prepost);
  if (!metadata.at("labels").is_null())
    manifest.labels = supplied("labels", interop::BundleFileRole::labels);
  manifest.evaluation =
      supplied("evaluation", interop::BundleFileRole::evaluation);
  manifest.sbom = supplied("sbom", interop::BundleFileRole::sbom);
  manifest.provenance = descriptor(
      "provenance", interop::BundleFileRole::provenance,
      "jyppx.liuding.onnx_delivery_provenance", 1U,
      {{"format", "jyppx.liuding.onnx_delivery_provenance"},
       {"schema_version", 1},
       {"source", executed.document.at("source")},
       {"checkpoint", metadata.at("checkpoint")},
       {"checkpoint_association",
        reconstruction ? "run_reference_verified" : "caller_declared"},
       {"source_run_lineage_verified", reconstruction != nullptr},
       {"source_authenticated", false},
       {"delivery_metadata_sha256", sha256_hex(Bytes(metadata_bytes))},
       {"export_report_sha256", sha256_hex(Bytes(prepared.inputs.report()))},
       {"verification_report_sha256", sha256_hex(Bytes(executed.encoded))},
       {"runtime_profile_sha256", sha256_hex(Bytes(executed.profile_json))}});
  const auto evidence = [&](std::string id, std::string schema,
                            std::uint32_t version, std::string path,
                            std::string_view data,
                            std::string media = "application/json") {
    add(id, interop::BundleFileRole::evidence, std::move(path), data,
        std::move(media));
    manifest.evidence.push_back({.schema_id = std::move(schema),
                                 .schema_version = version,
                                 .file_id = std::move(id)});
  };
  evidence("export-report", "jyppx.liuding.onnx_export_report", 1U,
           "evidence/export-report.json", prepared.inputs.report());
  evidence("cases", "jyppx.liuding.onnx_verification_cases", 1U,
           "evidence/cases.json", prepared.inputs.cases());
  evidence("tensors", "jyppx.liuding.onnx_verification_tensors", 1U,
           "evidence/cases.safetensors", prepared.inputs.tensors(),
           "application/safetensors");
  evidence("verification", "jyppx.liuding.onnx_verification_evidence",
           executed.document.at("schema_version").get<std::uint32_t>(),
           "evidence/verification.json", executed.encoded);
  evidence("profile", "jyppx.liuding.onnx_runtime_profile", 1U,
           "evidence/profile.json", executed.profile_json);
  evidence("runtime-lock", std::string(kRuntimeLock), 1U,
           "evidence/runtime-lock.json", lock_bytes);
  evidence("delivery-metadata", "jyppx.liuding.onnx_delivery_metadata", 1U,
           "evidence/delivery-metadata.json", metadata_bytes);
  if (reconstruction) {
    const auto proof = reconstruction->dump();
    evidence("source-reconstruction",
             "jyppx.liuding.onnx_source_reconstruction", 1U,
             "evidence/source-reconstruction.json", proof);
    const auto definition_bytes = read_text(definition, 4U << 20U);
    const auto input_bytes = read_text(inputs, 4U << 20U);
    const auto tensor_bytes = read_text(tensors, 64U << 20U);
    if (reconstruction->at("source") !=
            Json::parse(prepared.inputs.report()).at("source") ||
        reconstruction->at("definition").at("sha256") !=
            sha256_hex(Bytes(definition_bytes)) ||
        reconstruction->at("definition").at("size_bytes") !=
            definition_bytes.size() ||
        reconstruction->at("input_spec_sha256") !=
            sha256_hex(Bytes(input_bytes)) ||
        reconstruction->at("input_tensors_sha256") !=
            sha256_hex(Bytes(tensor_bytes)) ||
        reconstruction->at("export_report_sha256") !=
            sha256_hex(Bytes(prepared.inputs.report())) ||
        reconstruction->at("cases_sha256") !=
            sha256_hex(Bytes(prepared.inputs.cases())) ||
        reconstruction->at("tensors_sha256") !=
            sha256_hex(Bytes(prepared.inputs.tensors())) ||
        reconstruction->at("source_case_count") != prepared.cases.size())
      throw SerializationError(
          "ONNX reconstruction proof differs from the retained source bytes");
    evidence("model-definition", "jyppx.liuding.onnx_model_definition", 1U,
             "evidence/model-definition.json", definition_bytes);
    evidence("source-inputs", "jyppx.liuding.onnx_export_inputs", 1U,
             "evidence/source-inputs.json", input_bytes);
    evidence("source-tensors", "jyppx.liuding.onnx_export_input_tensors", 1U,
             "evidence/source-inputs.safetensors", tensor_bytes,
             "application/safetensors");
  }
  if (prepared.inputs.bundle)
    evidence("input-manifest", "jyppx.model_bundle.manifest",
             prepared.inputs.bundle->verification.manifest.schema_version,
             "evidence/input-manifest.json",
             TextBytes(prepared.inputs.bundle->manifest_bytes));
  interop::ModelBundleVerifyOptions policy;
  policy.limits.max_manifest_bytes = 4U << 20U;
  policy.limits.max_file_bytes = 256U << 20U;
  policy.limits.max_total_bytes = 512U << 20U;
  const auto verified =
      interop::publish_model_bundle(output, manifest, sources, policy);
  return {Json{{"operation", "model.onnx.pack"},
               {"published", true},
               {"manifest_sha256", verified.manifest_digest.hex},
               {"bundle_id", manifest.bundle_id},
               {"model_id", "primary"},
               {"report_id", "export-report"},
               {"cases_id", "cases"},
               {"tensors_id", "tensors"},
               {"runtime_lock_id", "runtime-lock"},
               {"runtime_executed", true},
               {"numerical", "passed"},
               {"runtime_requirements_checked", true},
               {"descriptor_payloads_validated", false},
               {"source_run_lineage_verified", reconstruction != nullptr},
               {"source_authenticated", false},
               {"task_quality_qualified", false},
               {"signature_status", "unsigned"}},
          0};
}
}  // namespace

Result onnx_pack_command(const Arguments& arguments) { return Pack(arguments); }

Result onnx_pack_reconstructed(const Arguments& arguments,
                               const Json& reconstruction,
                               const std::filesystem::path& definition,
                               const std::filesystem::path& inputs,
                               const std::filesystem::path& tensors) {
  return Pack(arguments, &reconstruction, definition, inputs, tensors);
}

Result onnx_smoke_command(const Arguments& arguments) {
  arguments.allow({"bundle", "bundle-sha256", "model-id", "report-id",
                   "cases-id", "tensors-id", "provider", "provider-option",
                   "timeout-ms"});
  const auto prepared = Prepare(arguments, true);
  const auto executed = Execute(arguments, prepared, true);
  return {
      Json{{"operation", "model.onnx.smoke"},
           {"runtime_executed", true},
           {"numerical", executed.passed ? "passed" : "failed"},
           {"requested_provider_observed", true},
           {"source_authenticated", false},
           {"source_run_lineage_verified", false},
           {"input_bundle", prepared.inputs.document(executed.runtime_checked)},
           {"report", executed.document},
           {"runtime_profile", executed.profile_json}},
      executed.passed ? 0 : 4};
}

Result onnx_verify_command(const Arguments& arguments) {
  arguments.allow({"path", "report", "report-sha256", "cases", "cases-sha256",
                   "tensors", "tensors-sha256", "provider", "provider-option",
                   "timeout-ms", "root", "id", "reference", "bundle",
                   "bundle-sha256", "model-id", "report-id", "cases-id",
                   "tensors-id"});
  const auto prepared = Prepare(arguments, false);
  const auto& model_owner = prepared.inputs;
  const auto model = model_owner.bytes();
  const auto export_json = model_owner.report();
  const auto case_json = model_owner.cases();
  const auto tensor_bytes = model_owner.tensors();
  const auto& export_report = prepared.export_report;
  const auto root = std::filesystem::u8path(arguments.require("root"));
  if (!std::filesystem::is_regular_file(root / "runs.sqlite3"))
    throw IoError("ONNX verification requires an existing run store");
  experiment::SQLiteRunStore store(root);
  const auto id = arguments.require("id");
  const auto reference = arguments.require("reference");
  if (reference.empty() || reference.size() > 220U ||
      !export_report.source.run_id || id == *export_report.source.run_id)
    throw InvalidArgument(
        "Use a distinct verification run and a reference of 1-220 bytes");
  const auto source = store.find(*export_report.source.run_id);
  if (!source ||
      source->lineage.source_digest != export_report.source.source_digest ||
      source->lineage.config_digest != export_report.source.config_digest ||
      source->lineage.data_manifest_digest !=
          export_report.source.data_manifest_digest)
    throw CompatibilityError("ONNX verification source run lineage mismatch");
  if (store.find(id))
    throw InvalidArgument(
        "ONNX verification run ID is already reserved; use a new attempt ID");
  const auto executed = Execute(arguments, prepared, false);
  const auto& report = executed.report;
  const auto& document = executed.document;
  const auto& encoded = executed.encoded;
  const auto& profile_json = executed.profile_json;
  const bool passed = executed.passed;
  const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  if (time < 0) throw InvalidArgument("System clock precedes Unix epoch");
  experiment::RunMetadata metadata{
      .run_id = id,
      .project_id = source->project_id,
      .experiment_id = source->experiment_id,
      .display_name = "ONNX verification " + id,
      .status = experiment::RunStatus::running,
      .created_at_unix_ms = static_cast<std::uint64_t>(time),
      .lineage = {.parent_run_id = source->run_id,
                  .source_digest = export_report.source.source_digest,
                  .build_fingerprint = build_fingerprint_json(report.build),
                  .config_digest = export_report.source.config_digest,
                  .data_manifest_digest =
                      export_report.source.data_manifest_digest},
      .tags = {{"operation", "onnx.verify"},
               {"onnx_sha256", export_report.model_digest.hex},
               {"numerical", passed ? "passed" : "failed"}}};
  metadata.updated_at_unix_ms = metadata.created_at_unix_ms;
  if (model_owner.bundle)
    metadata.tags.emplace("input_bundle_sha256",
                          model_owner.bundle->verification.manifest_digest.hex);
  (void)store.start(metadata);
  std::vector<experiment::RunArtifactUpload> uploads;
  const auto append = [&](std::string suffix, std::string role,
                          std::string media, std::string_view data) {
    uploads.push_back(
        {.reference_id = reference + "." + suffix,
         .role = std::move(role),
         .bytes = Bytes(data),
         .descriptor = {.logical_name = std::move(suffix),
                        .media_type = std::move(media),
                        .producer = "onnx_runtime.verification",
                        .expected_size_bytes = data.size()},
         .expected_digest = ArtifactDigest{.hex = sha256_hex(Bytes(data))},
         .attached_at_unix_ms = metadata.created_at_unix_ms});
  };
  append("onnx", "onnx.model", "application/onnx", model);
  append("export", "onnx.export_report", "application/json", export_json);
  append("cases", "onnx.verification_cases", "application/json", case_json);
  append("tensors", "onnx.verification_tensors", "application/octet-stream",
         tensor_bytes);
  append("verification", "onnx.verification_report", "application/json",
         encoded);
  append("profile", "onnx.runtime_profile", "application/json", profile_json);
  if (model_owner.bundle)
    append("bundle", "onnx.input_bundle_manifest", "application/json",
           TextBytes(model_owner.bundle->manifest_bytes));
  if (!model_owner.runtime_id.empty())
    append("runtime_lock", "onnx.runtime_lock", "application/json",
           TextBytes(model_owner.bundle->files.at(model_owner.runtime_id)));
  const auto receipts = store.put_artifacts(id, uploads);
  (void)store.transition(
      id,
      passed ? experiment::RunStatus::completed : experiment::RunStatus::failed,
      metadata.created_at_unix_ms);
  auto artifacts = Json::array();
  for (const auto& receipt : receipts)
    artifacts.push_back({{"reference_id", receipt.reference_id},
                         {"role", receipt.role},
                         {"sha256", receipt.artifact.digest.hex},
                         {"size_bytes", receipt.artifact.size_bytes}});
  return {Json{{"operation", "model.onnx.verify"},
               {"run_id", id},
               {"source_run_id", source->run_id},
               {"runtime_executed", true},
               {"numerical", passed ? "passed" : "failed"},
               {"requested_provider_observed", true},
               {"source_authenticated", false},
               {"input_bundle", model_owner.document(executed.runtime_checked)},
               {"artifacts", artifacts},
               {"report", document}},
          passed ? 0 : 4};
}
}  // namespace jyppx::liuding::cli
#else
namespace jyppx::liuding::cli {
Result onnx_pack_command(const Arguments&) {
  throw CompatibilityError("ONNX packing components are disabled",
                           "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT and "
                           "JYPPX_LIUDING_BUILD_ONNX_RUNTIME.");
}
Result onnx_smoke_command(const Arguments&) {
  throw CompatibilityError("ONNX smoke components are disabled",
                           "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT and "
                           "JYPPX_LIUDING_BUILD_ONNX_RUNTIME.");
}
Result onnx_verify_command(const Arguments&) {
  throw CompatibilityError("ONNX verification components are disabled",
                           "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT and "
                           "JYPPX_LIUDING_BUILD_ONNX_RUNTIME.");
}
}  // namespace jyppx::liuding::cli
#endif
