// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_runtime/verification.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

namespace jyppx::liuding::onnx_runtime {
namespace {
using Json = nlohmann::json;
using Deadline = std::optional<std::chrono::steady_clock::time_point>;

void Text(std::string_view value, std::size_t maximum = 4096U) {
  if (value.empty() || value.size() > maximum ||
      std::any_of(value.begin(), value.end(),
                  [](unsigned char c) { return c < 32U || c == 127U; }))
    throw InvalidArgument("Invalid ONNX verification text");
}

void Check(std::stop_token stop, Deadline deadline) {
  if (stop.stop_requested()) throw Cancelled("ONNX verification cancelled");
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("ONNX verification deadline exceeded");
}

void Policy(const VerificationOptions& options) {
  Text(options.reference);
  if (!std::isfinite(options.absolute_tolerance) ||
      !std::isfinite(options.relative_tolerance) ||
      options.absolute_tolerance < 0.0 || options.absolute_tolerance > 1.0 ||
      options.relative_tolerance < 0.0 || options.relative_tolerance > 1.0 ||
      options.max_cases == 0U || options.max_cases > 1024U ||
      options.max_case_bytes < sizeof(float) ||
      options.max_case_bytes > (1ULL << 32U))
    throw InvalidArgument("Invalid ONNX verification policy");
}

std::uint64_t Count(std::span<const std::int64_t> shape,
                    std::uint64_t maximum_bytes) {
  if (shape.empty() || shape.size() > 8U)
    throw InvalidArgument("ONNX verification tensor rank must be 1-8");
  std::uint64_t count = 1U;
  const auto maximum = maximum_bytes / sizeof(float);
  for (const auto extent : shape) {
    if (extent <= 0)
      throw InvalidArgument("ONNX verification dimensions must be positive");
    if (static_cast<std::uint64_t>(extent) > maximum ||
        count > maximum / static_cast<std::uint64_t>(extent))
      throw ResourceExhausted(
          "ONNX verification tensor exceeds its byte bound");
    count *= static_cast<std::uint64_t>(extent);
  }
  return count;
}

std::vector<const TensorInput*> Ordered(
    std::span<const TensorInput> tensors,
    std::span<const TensorSignature> signatures,
    std::map<std::string, std::int64_t>& symbols, bool inputs,
    std::uint64_t& remaining, std::stop_token stop, Deadline deadline) {
  if (tensors.size() != signatures.size())
    throw InvalidArgument("ONNX verification requires every named tensor");
  std::vector<const TensorInput*> result(signatures.size(), nullptr);
  for (const auto& tensor : tensors) {
    Check(stop, deadline);
    const auto found =
        std::find_if(signatures.begin(), signatures.end(),
                     [&](const auto& s) { return s.name == tensor.name; });
    if (found == signatures.end())
      throw InvalidArgument("Unknown ONNX verification tensor name");
    const auto index = static_cast<std::size_t>(found - signatures.begin());
    if (result[index] || tensor.shape.size() != found->shape.size())
      throw InvalidArgument("Duplicate ONNX verification name or invalid rank");
    for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis)
      if (tensor.shape[axis] <= 0 || (found->shape[axis] != -1 &&
                                      found->shape[axis] != tensor.shape[axis]))
        throw InvalidArgument("ONNX verification tensor shape mismatch");
    if (!found->batch_symbol.empty()) {
      const auto entry = symbols.find(found->batch_symbol);
      if (entry == symbols.end()) {
        if (!inputs) throw InvalidArgument("Unbound ONNX output batch symbol");
        symbols.emplace(found->batch_symbol, tensor.shape.front());
      } else if (entry->second != tensor.shape.front()) {
        throw InvalidArgument("ONNX verification batch symbol mismatch");
      }
    }
    const auto count = Count(tensor.shape, remaining);
    if (count != tensor.values.size())
      throw InvalidArgument("ONNX verification tensor element count mismatch");
    remaining -= count * sizeof(float);
    for (std::size_t i = 0; i < tensor.values.size(); ++i) {
      if (i % 4096U == 0U) Check(stop, deadline);
      if (!std::isfinite(tensor.values[i]))
        throw InvalidArgument("ONNX verification values must be finite");
    }
    result[index] = &tensor;
  }
  return result;
}

VerificationTensor Identity(const TensorInput& tensor) {
  static_assert(std::endian::native == std::endian::little);
  return {.name = std::string(tensor.name),
          .shape = {tensor.shape.begin(), tensor.shape.end()},
          .digest = {.hex = sha256_hex(std::as_bytes(tensor.values))},
          .size_bytes = tensor.values.size_bytes()};
}

Json Tensor(const VerificationTensor& tensor) {
  Text(tensor.name);
  validate_artifact_digest(tensor.digest);
  if (Count(tensor.shape, 1ULL << 32U) * sizeof(float) != tensor.size_bytes)
    throw InvalidArgument(
        "ONNX verification tensor size does not match its shape");
  return {{"name", tensor.name},
          {"dtype", "float32"},
          {"shape", tensor.shape},
          {"sha256", tensor.digest.hex},
          {"size_bytes", tensor.size_bytes}};
}
}  // namespace

VerificationReport verify_session(const Session& session,
                                  std::span<const VerificationCase> cases,
                                  const VerificationOptions& options,
                                  std::stop_token stop, Deadline deadline) {
  Check(stop, deadline);
  Policy(options);
  if (cases.empty() || cases.size() > options.max_cases)
    throw ResourceExhausted("ONNX verification case count exceeds its bound");
  // Check the complete suite before any model execution. No sample copies.
  std::set<std::string_view> ids;
  for (const auto& item : cases) {
    Text(item.id, 128U);
    if (!ids.insert(item.id).second)
      throw InvalidArgument("ONNX verification case IDs must be unique");
    std::map<std::string, std::int64_t> symbols;
    auto remaining = options.max_case_bytes;
    (void)Ordered(item.inputs, session.inputs(), symbols, true, remaining, stop,
                  deadline);
    (void)Ordered(item.expected, session.outputs(), symbols, false, remaining,
                  stop, deadline);
  }
  VerificationReport report{
      .model_digest = session.model_digest(),
      .build = current_build_fingerprint(),
      .runtime_version = session.runtime_version(),
      .requested_provider = session.execution_provider(),
      .allow_cpu_fallback = session.cpu_fallback_allowed(),
      .policy = options,
      .cases = {},
      .profile = std::nullopt};
  report.cases.reserve(cases.size());
  for (const auto& item : cases) {
    Check(stop, deadline);
    std::map<std::string, std::int64_t> symbols;
    auto remaining = options.max_case_bytes;
    const auto inputs = Ordered(item.inputs, session.inputs(), symbols, true,
                                remaining, stop, deadline);
    const auto expected = Ordered(item.expected, session.outputs(), symbols,
                                  false, remaining, stop, deadline);
    CaseVerification observation{
        .id = std::string(item.id), .inputs = {}, .outputs = {}};
    for (const auto* input : inputs)
      observation.inputs.push_back(Identity(*input));
    const auto started = std::chrono::steady_clock::now();
    const auto actual = session.run(item.inputs, stop, deadline);
    observation.elapsed_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto& output = actual[index];
      const auto& reference = *expected[index];
      if (output.name != reference.name ||
          output.shape.size() != reference.shape.size() ||
          !std::equal(output.shape.begin(), output.shape.end(),
                      reference.shape.begin()) ||
          output.values.size() != reference.values.size())
        throw CompatibilityError(
            "ONNX verification actual output signature differs");
      OutputComparison comparison{.actual = Identity({.name = output.name,
                                                      .values = output.values,
                                                      .shape = output.shape}),
                                  .expected = Identity(reference)};
      for (std::size_t i = 0; i < output.values.size(); ++i) {
        if (i % 4096U == 0U) Check(stop, deadline);
        const double error = std::abs(static_cast<double>(output.values[i]) -
                                      reference.values[i]);
        comparison.maximum_absolute_error =
            (std::max)(comparison.maximum_absolute_error, error);
        if (error > options.absolute_tolerance +
                        options.relative_tolerance *
                            std::abs(static_cast<double>(reference.values[i])))
          ++comparison.mismatched_elements;
      }
      observation.outputs.push_back(std::move(comparison));
    }
    report.cases.push_back(std::move(observation));
  }
  Check(stop, deadline);
  return report;
}

ProfileObservation inspect_profile(std::string_view json,
                                   std::size_t maximum_bytes) {
  if (maximum_bytes == 0U || maximum_bytes > (64U << 20U))
    throw InvalidArgument("Invalid ONNX profile byte bound");
  if (json.empty() || json.size() > maximum_bytes)
    throw ResourceExhausted("ONNX profile exceeds its byte bound");
  std::vector<std::set<std::string>> keys;
  std::size_t nodes = 0U;
  try {
    const auto callback = [&](int depth, Json::parse_event_t event,
                              Json& value) {
      if (depth > 20 || ++nodes > 2097152U)
        throw ResourceExhausted("ONNX profile structure exceeds its bound");
      if (event == Json::parse_event_t::object_start) keys.emplace_back();
      if (event == Json::parse_event_t::key &&
          (keys.empty() ||
           !keys.back().insert(value.get<std::string>()).second))
        throw SerializationError("Duplicate ONNX profile field");
      if (event == Json::parse_event_t::object_end) keys.pop_back();
      if (value.is_string() &&
          value.get_ref<const std::string&>().size() > 16384U)
        throw ResourceExhausted("ONNX profile string exceeds its bound");
      return true;
    };
    const auto events = Json::parse(json.begin(), json.end(), callback);
    if (!events.is_array() || events.size() > 131072U)
      throw SerializationError("Expected bounded ORT event array");
    ProfileObservation result{
        .digest = {.hex = sha256_hex(
                       std::as_bytes(std::span(json.data(), json.size())))},
        .size_bytes = json.size(),
        .kernels = {}};
    for (const auto& event : events) {
      if (!event.is_object())
        throw SerializationError("Expected ORT event object");
      if (event.value("cat", "") != "Node" || event.value("ph", "") != "X")
        continue;
      const auto name = event.at("name").get<std::string>();
      if (!name.ends_with("_kernel_time")) continue;
      const auto provider = event.at("args").at("provider").get<std::string>();
      const auto operation = event.at("args").at("op_name").get<std::string>();
      Text(provider, 128U);
      Text(operation, 256U);
      if (result.kernels.size() >= 32U && !result.kernels.contains(provider))
        throw ResourceExhausted("Too many ONNX profile providers");
      auto& operations = result.kernels[provider];
      if (operations.size() >= 4096U && !operations.contains(operation))
        throw ResourceExhausted("Too many ONNX profile operations");
      ++operations[operation];
    }
    if (result.kernels.empty())
      throw CompatibilityError(
          "ORT profile contains no actual provider kernel events");
    return result;
  } catch (const Json::exception&) {
    std::throw_with_nested(SerializationError("Invalid ORT profiling JSON"));
  }
}

bool verification_passed(const VerificationReport& report) noexcept {
  if (report.cases.empty()) return false;
  for (const auto& item : report.cases) {
    if (item.outputs.empty()) return false;
    for (const auto& output : item.outputs)
      if (output.mismatched_elements != 0U) return false;
  }
  return true;
}

std::string serialize_verification_report(const VerificationReport& report,
                                          std::size_t maximum_bytes) {
  if (maximum_bytes == 0U || maximum_bytes > (64U << 20U))
    throw InvalidArgument("Invalid ONNX verification report byte bound");
  Policy(report.policy);
  validate_artifact_digest(report.model_digest);
  Text(report.runtime_version);
  Text(report.requested_provider.name, 128U);
  if (report.cases.empty() || report.cases.size() > report.policy.max_cases)
    throw InvalidArgument("Invalid ONNX verification report case count");
  Json cases = Json::array();
  std::set<std::string> ids;
  std::uint64_t mismatches = 0U;
  for (const auto& item : report.cases) {
    Text(item.id, 128U);
    if (!ids.insert(item.id).second || item.inputs.empty() ||
        item.outputs.empty() ||
        item.inputs.size() + item.outputs.size() > 4096U)
      throw InvalidArgument("Invalid ONNX verification report case");
    Json inputs = Json::array(), outputs = Json::array();
    std::set<std::string> input_names, output_names;
    for (const auto& input : item.inputs) {
      if (!input_names.insert(input.name).second)
        throw InvalidArgument("Duplicate verification input");
      inputs.push_back(Tensor(input));
    }
    for (const auto& output : item.outputs) {
      if (!output_names.insert(output.actual.name).second ||
          output.actual.name != output.expected.name ||
          output.actual.shape != output.expected.shape ||
          !std::isfinite(output.maximum_absolute_error) ||
          output.maximum_absolute_error < 0.0 ||
          output.mismatched_elements > output.actual.size_bytes / sizeof(float))
        throw InvalidArgument("Invalid ONNX verification comparison");
      mismatches += output.mismatched_elements;
      outputs.push_back(
          {{"actual", Tensor(output.actual)},
           {"expected", Tensor(output.expected)},
           {"mismatched_elements", output.mismatched_elements},
           {"maximum_absolute_error", output.maximum_absolute_error}});
    }
    cases.push_back({{"id", item.id},
                     {"inputs", inputs},
                     {"outputs", outputs},
                     {"elapsed_nanoseconds", item.elapsed_nanoseconds}});
  }
  Json profile = nullptr;
  bool provider_observed = false;
  if (report.profile) {
    validate_artifact_digest(report.profile->digest);
    if (report.profile->size_bytes == 0U || report.profile->kernels.empty())
      throw InvalidArgument("Invalid ONNX profile observation");
    for (const auto& [provider, operations] : report.profile->kernels) {
      Text(provider, 128U);
      if (operations.empty())
        throw InvalidArgument("Empty ONNX profile provider");
      for (const auto& [operation, count] : operations) {
        Text(operation, 256U);
        if (count == 0U) throw InvalidArgument("Empty ONNX profile operation");
      }
    }
    profile = {{"sha256", report.profile->digest.hex},
               {"size_bytes", report.profile->size_bytes},
               {"kernel_events", report.profile->kernels}};
    provider_observed =
        report.profile->kernels.contains(report.requested_provider.name);
  }
  const Json result{
      {"format", "jyppx.liuding.onnx_runtime_verification"},
      {"schema_version", 1},
      {"model_sha256", report.model_digest.hex},
      {"build", Json::parse(build_fingerprint_json(report.build))},
      {"runtime",
       {{"name", "onnxruntime"},
        {"version", report.runtime_version},
        {"requested_provider", report.requested_provider.name},
        {"provider_options", report.requested_provider.options},
        {"allow_cpu_fallback", report.allow_cpu_fallback}}},
      {"comparison",
       {{"absolute_tolerance", report.policy.absolute_tolerance},
        {"relative_tolerance", report.policy.relative_tolerance},
        {"max_cases", report.policy.max_cases},
        {"max_case_bytes", report.policy.max_case_bytes},
        {"reference", report.policy.reference}}},
      {"tensor_byte_order", "little"},
      {"cases", cases},
      {"profile", profile},
      {"verification",
       {{"runtime_executed", true},
        {"numerical", verification_passed(report) ? "passed" : "failed"},
        {"mismatched_elements", mismatches},
        {"requested_provider_observed", provider_observed},
        {"reference_authenticated", false},
        {"task_quality", "not_checked"},
        {"resource_qualification", "not_checked"},
        {"performance", "not_checked"}}}};
  const auto encoded = result.dump();
  if (encoded.size() > maximum_bytes)
    throw ResourceExhausted("ONNX verification report exceeds its byte bound");
  return encoded;
}

}  // namespace jyppx::liuding::onnx_runtime
