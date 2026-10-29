// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/report.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/csrc/jit/serialization/export.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace jyppx::liuding::onnx_export {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Invalid() {
  throw SerializationError(
      "Invalid native ONNX export report",
      ErrorDetails{.component = "onnx_export.report", .operation = "metadata"});
}

void Fields(const Json& value, std::initializer_list<std::string_view> names) {
  if (!value.is_object() || value.size() != names.size()) Invalid();
  for (const auto name : names)
    if (!value.contains(std::string(name))) Invalid();
}

std::string Text(const Json& value) {
  if (!value.is_string()) Invalid();
  auto text = value.get<std::string>();
  if (text.empty() || text.size() > 4096U ||
      std::any_of(text.begin(), text.end(),
                  [](unsigned char c) { return c < 0x20U || c == 0x7fU; }))
    Invalid();
  return text;
}

std::uint64_t Uint(const Json& value) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (!value.is_number_integer() || value.get<std::int64_t>() < 0) Invalid();
  return static_cast<std::uint64_t>(value.get<std::int64_t>());
}

Json Digest(const ArtifactDigest& value) {
  validate_artifact_digest(value);
  return {{"algorithm", "sha256"}, {"hex", value.hex}};
}

ArtifactDigest ReadDigest(const Json& value) {
  Fields(value, {"algorithm", "hex"});
  if (Text(value.at("algorithm")) != "sha256") Invalid();
  ArtifactDigest result{.hex = Text(value.at("hex"))};
  validate_artifact_digest(result);
  return result;
}

std::optional<std::string> Optional(const Json& value, bool digest) {
  if (value.is_null()) return std::nullopt;
  auto text = Text(value);
  if (digest) validate_artifact_digest({.hex = text});
  if (!digest && text.size() > 256U) Invalid();
  return text;
}

Json Source(const ExportSource& source) {
  const auto optional = [](const auto& value) {
    return value ? Json(*value) : Json(nullptr);
  };
  return {{"run_id", optional(source.run_id)},
          {"source_digest", optional(source.source_digest)},
          {"config_digest", optional(source.config_digest)},
          {"data_manifest_digest", optional(source.data_manifest_digest)}};
}

std::size_t Width(std::string_view dtype) {
  if (dtype == "bool" || dtype == "uint8" || dtype == "int8") return 1U;
  if (dtype == "int16" || dtype == "float16" || dtype == "bfloat16") return 2U;
  if (dtype == "int32" || dtype == "float32") return 4U;
  if (dtype == "int64" || dtype == "float64") return 8U;
  Invalid();
}

Json Tensors(std::span<const ExportTensorRecord> tensors, bool state) {
  if (tensors.size() > (state ? 65536U : 64U) || (!state && tensors.empty()))
    Invalid();
  auto result = Json::array();
  std::set<std::string> names;
  for (const auto& tensor : tensors) {
    (void)Text(tensor.name);
    if (!names.insert(tensor.name).second || tensor.shape.size() > 8U ||
        (state ? tensor.role != "parameter" && tensor.role != "buffer"
               : tensor.role != "input" && tensor.role != "output"))
      Invalid();
    std::uint64_t size = Width(tensor.dtype);
    for (const auto extent : tensor.shape) {
      if (extent <= 0 ||
          static_cast<std::uint64_t>(extent) > (1ULL << 32U) / size)
        Invalid();
      size *= static_cast<std::uint64_t>(extent);
    }
    if (tensor.size_bytes != size || size > (1ULL << 32U)) Invalid();
    result.push_back({{"name", tensor.name},
                      {"dtype", tensor.dtype},
                      {"shape", tensor.shape},
                      {"digest", Digest(tensor.digest)},
                      {"size_bytes", tensor.size_bytes},
                      {"role", tensor.role}});
  }
  return result;
}

std::vector<ExportTensorRecord> ReadTensors(const Json& values) {
  if (!values.is_array() || values.size() > 65536U) Invalid();
  std::vector<ExportTensorRecord> result;
  result.reserve(values.size());
  for (const auto& value : values) {
    Fields(value, {"name", "dtype", "shape", "digest", "size_bytes", "role"});
    ExportTensorRecord tensor{.name = Text(value.at("name")),
                              .dtype = Text(value.at("dtype")),
                              .digest = ReadDigest(value.at("digest")),
                              .size_bytes = Uint(value.at("size_bytes")),
                              .role = Text(value.at("role"))};
    if (!value.at("shape").is_array() || value.at("shape").size() > 8U)
      Invalid();
    for (const auto& extent : value.at("shape")) {
      const auto dimension = Uint(extent);
      if (dimension == 0U || dimension > (1ULL << 32U)) Invalid();
      tensor.shape.push_back(static_cast<std::int64_t>(dimension));
    }
    result.push_back(std::move(tensor));
  }
  return result;
}

Json Document(const ExportReport& report) {
  if (report.model_size_bytes == 0U ||
      report.model_size_bytes > (1ULL << 32U) || report.opset != 17U ||
      report.minimum_batch < 1 || report.maximum_batch < report.minimum_batch ||
      report.maximum_batch > 65536)
    Invalid();
  const auto source = Source(report.source);
  (void)Optional(source.at("run_id"), false);
  (void)Optional(source.at("source_digest"), false);
  for (const auto field : {"config_digest", "data_manifest_digest"})
    (void)Optional(source.at(field), true);
  Json axes = Json::array();
  std::map<std::string, const ExportTensorRecord*> tensors;
  for (const auto* list : {&report.inputs, &report.outputs})
    for (const auto& tensor : *list)
      if (!tensors.emplace(tensor.name, &tensor).second) Invalid();
  std::map<std::string, std::int64_t> symbols;
  for (const auto& [name, dimensions] : report.dynamic_axes) {
    const auto found = tensors.find(name);
    if (found == tensors.end() || dimensions.size() != 1U ||
        dimensions.begin()->first != 0 || found->second->shape.empty())
      Invalid();
    const auto& symbol = dimensions.begin()->second;
    (void)Text(symbol);
    if (symbol.size() > 128U) Invalid();
    const auto extent = found->second->shape.front();
    const auto [entry, inserted] = symbols.emplace(symbol, extent);
    if (!inserted && entry->second != extent) Invalid();
    axes.push_back({{"name", name}, {"axis", 0}, {"symbol", symbol}});
  }
  if (report.operators.empty() || report.operators.size() > 65536U ||
      !std::is_sorted(report.operators.begin(), report.operators.end()) ||
      std::adjacent_find(report.operators.begin(), report.operators.end()) !=
          report.operators.end())
    Invalid();
  for (const auto& op : report.operators) {
    (void)Text(op);
    if (!op.starts_with("onnx::")) Invalid();
  }
  for (const auto& tensor : report.inputs)
    if (tensor.role != "input") Invalid();
  for (const auto& tensor : report.outputs)
    if (tensor.role != "output") Invalid();
  const auto build = Json::parse(build_fingerprint_json(report.build));
  for (const auto& field : build.items()) (void)Text(field.value());
  return {{"format", "jyppx.liuding.onnx_export_report"},
          {"schema_version", ExportReport::format_version},
          {"model",
           {{"digest", Digest(report.model_digest)},
            {"size_bytes", report.model_size_bytes}}},
          {"build", build},
          {"source", source},
          {"exporter", "native-cpp-trace"},
          {"serializer", "libtorch.official-onnx"},
          {"tensor_byte_order", "little"},
          {"opset", report.opset},
          {"external_data", false},
          {"minimum_batch", report.minimum_batch},
          {"maximum_batch", report.maximum_batch},
          {"dynamic_axes", axes},
          {"inputs", Tensors(report.inputs, false)},
          {"outputs", Tensors(report.outputs, false)},
          {"state", Tensors(report.state, true)},
          {"operators", report.operators},
          {"verification",
           {{"official_checker", "passed"},
            {"runtime", "not_checked"},
            {"numerical", "not_checked"},
            {"source_authenticated", false}}}};
}

Json Parse(std::string_view text, std::size_t maximum) {
  if (maximum == 0U || maximum > (64U << 20U))
    throw InvalidArgument("Invalid ONNX report byte limit");
  if (text.empty() || text.size() > maximum)
    throw ResourceExhausted("Native ONNX export report exceeds its byte limit");
  std::vector<std::set<std::string>> keys;
  std::size_t nodes = 0U;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 12 || ++nodes > 1048576U)
      throw ResourceExhausted(
          "Native ONNX export report structure exceeds its limit");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        (keys.empty() || !keys.back().insert(value.get<std::string>()).second))
      Invalid();
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    if (value.is_string() && value.get_ref<const std::string&>().size() > 4096U)
      Invalid();
    return true;
  };
  return Json::parse(text.begin(), text.end(), callback);
}
}  // namespace

std::string serialize_export_report(const ExportReport& report,
                                    std::size_t maximum_bytes) {
  if (maximum_bytes == 0U || maximum_bytes > (64U << 20U))
    throw InvalidArgument("Invalid ONNX report byte limit");
  try {
    auto encoded = Document(report).dump();
    if (encoded.size() > maximum_bytes)
      throw ResourceExhausted(
          "Native ONNX export report exceeds its byte limit");
    return encoded;
  } catch (const Json::exception&) {
    std::throw_with_nested(SerializationError("Invalid ONNX report JSON"));
  }
}

ExportReport deserialize_export_report(std::string_view json,
                                       std::size_t maximum_bytes) {
  try {
    const auto value = Parse(json, maximum_bytes);
    if (Uint(value.at("schema_version")) != ExportReport::format_version)
      throw CompatibilityError("Unsupported native ONNX export report version");
    ExportReport result;
    const auto& model = value.at("model");
    Fields(model, {"digest", "size_bytes"});
    result.model_digest = ReadDigest(model.at("digest"));
    result.model_size_bytes = Uint(model.at("size_bytes"));
    const auto& build = value.at("build");
    result.build = {
        .jyppx_version = Text(build.at("jyppx_version")),
        .git_commit = Text(build.at("git_commit")),
        .compiler = Text(build.at("compiler")),
        .cxx_standard = Text(build.at("cxx_standard")),
        .os_arch = Text(build.at("os_arch")),
        .build_type = Text(build.at("build_type")),
        .libtorch_version = Text(build.at("libtorch_version")),
        .libtorch_build = Text(build.at("libtorch_build")),
        .accelerator_runtime = Text(build.at("accelerator_runtime")),
        .dependency_lock_digest = Text(build.at("dependency_lock_digest"))};
    const auto& source = value.at("source");
    result.source = {
        .run_id = Optional(source.at("run_id"), false),
        .source_digest = Optional(source.at("source_digest"), false),
        .config_digest = Optional(source.at("config_digest"), true),
        .data_manifest_digest =
            Optional(source.at("data_manifest_digest"), true)};
    const auto opset = Uint(value.at("opset"));
    if (opset != 17U) Invalid();
    result.opset = static_cast<std::uint32_t>(opset);
    for (const auto field : {"minimum_batch", "maximum_batch"})
      if (Uint(value.at(field)) > 65536U) Invalid();
    result.minimum_batch =
        static_cast<std::int64_t>(Uint(value.at("minimum_batch")));
    result.maximum_batch =
        static_cast<std::int64_t>(Uint(value.at("maximum_batch")));
    const auto& axes = value.at("dynamic_axes");
    if (!axes.is_array() || axes.size() > 128U) Invalid();
    for (const auto& axis : axes) {
      Fields(axis, {"name", "axis", "symbol"});
      if (Uint(axis.at("axis")) != 0U ||
          !result.dynamic_axes
               .emplace(Text(axis.at("name")),
                        std::map<std::int64_t, std::string>{
                            {0, Text(axis.at("symbol"))}})
               .second)
        Invalid();
    }
    result.inputs = ReadTensors(value.at("inputs"));
    result.outputs = ReadTensors(value.at("outputs"));
    result.state = ReadTensors(value.at("state"));
    if (!value.at("operators").is_array() ||
        value.at("operators").size() > 65536U)
      Invalid();
    for (const auto& op : value.at("operators"))
      result.operators.push_back(Text(op));
    // Compare all fields recursively, rejecting unknown fields and false
    // claims.
    const auto canonical = Document(result);
    if (value != canonical || !value.at("external_data").is_boolean() ||
        !value.at("verification").at("source_authenticated").is_boolean())
      Invalid();
    return result;
  } catch (const Json::exception&) {
    std::throw_with_nested(
        SerializationError("Malformed native ONNX export report"));
  }
}

void verify_export_report_model(const ExportReport& report,
                                std::span<const std::byte> onnx) {
  (void)serialize_export_report(report);
  if (onnx.size() != report.model_size_bytes ||
      sha256_hex(onnx) != report.model_digest.hex)
    throw SerializationError("ONNX model does not match its export report");
  try {
    torch::jit::check_onnx_proto(
        std::string(reinterpret_cast<const char*>(onnx.data()), onnx.size()));
  } catch (const std::exception&) {
    std::throw_with_nested(
        CompatibilityError("Official ONNX report/model check failed"));
  }
}

}  // namespace jyppx::liuding::onnx_export
