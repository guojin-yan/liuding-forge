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
#include <jyppx/liuding/interop/safetensors.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <set>

namespace jyppx::liuding::cli {
namespace {
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
    if (std::filesystem::create_directory(directory, error)) return directory;
    if (error)
      throw IoError("Cannot create ONNX verification profile directory");
  }
  throw IoError("Cannot allocate unique ONNX verification profile directory");
}
}  // namespace

Result onnx_verify_command(const Arguments& arguments) {
  arguments.allow({"path", "report", "report-sha256", "cases", "cases-sha256",
                   "tensors", "tensors-sha256", "provider", "provider-option",
                   "timeout-ms", "root", "id", "reference"});
  const auto export_json =
      CheckedFile(arguments, "report", "report-sha256", 4U << 20U);
  const auto export_report =
      onnx_export::deserialize_export_report(export_json);
  if (export_report.model_size_bytes > (256U << 20U))
    throw ResourceExhausted("ONNX verification model exceeds 256 MiB");
  const auto model =
      read_text(arguments.require("path"),
                static_cast<std::size_t>(export_report.model_size_bytes));
  onnx_export::verify_export_report_model(export_report, Bytes(model));
  const auto case_json =
      CheckedFile(arguments, "cases", "cases-sha256", 4U << 20U);
  const auto tensor_bytes =
      CheckedFile(arguments, "tensors", "tensors-sha256", 64U << 20U);
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
  onnx_runtime::VerificationOptions policy;
  policy.reference = suite.at("reference").get<std::string>();
  policy.absolute_tolerance = suite.at("absolute_tolerance").get<double>();
  policy.relative_tolerance = suite.at("relative_tolerance").get<double>();
  policy.max_case_bytes = 64U << 20U;
  const auto& case_array = suite.at("cases");
  if (!case_array.is_array() || case_array.empty() || case_array.size() > 64U)
    throw ResourceExhausted("ONNX verification suite requires 1-64 cases");
  std::vector<OwnedCase> owned;
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
  std::vector<onnx_runtime::VerificationCase> cases;
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
  onnx_runtime::Session session(Bytes(model), export_report.model_digest,
                                options);
  auto report =
      onnx_runtime::verify_session(session, cases, policy, {}, deadline);
  const auto profile_path = session.end_profiling();
  const auto profile_json = read_text(profile_path, 16U << 20U);
  report.profile = onnx_runtime::inspect_profile(profile_json);
  if (report.profile->kernels.size() != 1U ||
      !report.profile->kernels.contains(options.provider.name))
    throw CompatibilityError(
        "ORT trace does not exclusively observe the requested provider");
  const Json document{
      {"format", "jyppx.liuding.onnx_verification_evidence"},
      {"schema_version", 1},
      {"source", Json::parse(export_json).at("source")},
      {"export_report_sha256", sha256_hex(Bytes(export_json))},
      {"cases_sha256", sha256_hex(Bytes(case_json))},
      {"tensors_sha256", sha256_hex(Bytes(tensor_bytes))},
      {"execution",
       Json::parse(onnx_runtime::serialize_verification_report(report))}};
  const auto encoded = document.dump();
  if (encoded.size() > (4U << 20U))
    throw ResourceExhausted("ONNX validation evidence exceeds 4 MiB");
  const bool passed = onnx_runtime::verification_passed(report);
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
               {"artifacts", artifacts},
               {"report", document}},
          passed ? 0 : 4};
}
}  // namespace jyppx::liuding::cli
#else
namespace jyppx::liuding::cli {
Result onnx_verify_command(const Arguments&) {
  throw CompatibilityError("ONNX verification components are disabled",
                           "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT and "
                           "JYPPX_LIUDING_BUILD_ONNX_RUNTIME.");
}
}  // namespace jyppx::liuding::cli
#endif
