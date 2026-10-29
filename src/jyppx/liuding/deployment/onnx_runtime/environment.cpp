// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include <jyppx/liuding/deployment/onnx_runtime/environment.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <exception>
#include <initializer_list>
#include <new>
#include <nlohmann/json.hpp>
#include <set>
#include <span>
#include <utility>

namespace jyppx::liuding::onnx_runtime {
namespace {
using Json = nlohmann::json;

[[noreturn]] void Invalid() {
  throw InvalidArgument("Invalid ONNX Runtime environment declaration");
}
[[noreturn]] void Schema() {
  throw SerializationError("Invalid ONNX Runtime environment-lock schema");
}
void Bound(std::size_t count, std::size_t maximum) {
  if (count > maximum)
    throw ResourceExhausted(
        "ONNX Runtime environment metadata exceeds its bound");
}
void Text(std::string_view value, std::size_t maximum = 128U,
          bool empty_allowed = false) {
  Bound(value.size(), maximum);
  if ((!empty_allowed && value.empty()) ||
      std::any_of(value.begin(), value.end(),
                  [](unsigned char c) { return c < 32U || c == 127U; }))
    Invalid();
}
void UniqueStrings(const std::vector<std::string>& values) {
  Bound(values.size(), 64U);
  if (values.empty()) Invalid();
  std::set<std::string_view> seen;
  for (const auto& value : values) {
    Text(value);
    if (!seen.insert(value).second) Invalid();
  }
}
Json Declaration(const EnvironmentRequirements& value) {
  Text(value.backend_id);
  Text(value.backend_version);
  if (value.backend_id != kEnvironmentBackendId || value.dependencies.empty())
    Invalid();
  Bound(value.dependencies.size(), 128U);
  UniqueStrings(value.devices);
  UniqueStrings(value.dtypes);
  UniqueStrings(value.os_arch);
  auto dependencies = value.dependencies;
  std::sort(dependencies.begin(), dependencies.end(),
            [](const auto& a, const auto& b) { return a.name < b.name; });
  Json entries = Json::array();
  std::string_view previous;
  for (const auto& entry : dependencies) {
    Text(entry.name);
    Text(entry.version);
    if (entry.name == previous) Invalid();
    previous = entry.name;
    entries.push_back({{"name", entry.name}, {"version", entry.version}});
  }
  return {{"backend_id", value.backend_id},
          {"backend_version", value.backend_version},
          {"dependencies", std::move(entries)},
          {"devices", value.devices},
          {"dtypes", value.dtypes},
          {"os_arch", value.os_arch},
          {"dynamic_shapes", value.dynamic_shapes}};
}
Json Document(const EnvironmentLock& value) {
  auto result = Declaration(value.runtime);
  Text(value.provider.name);
  Bound(value.provider.options.size(), 128U);
  for (const auto& [key, setting] : value.provider.options) {
    Text(key, 256U);
    Text(setting, 4096U, true);
  }
  result["format"] = kEnvironmentLockSchemaId;
  result["schema_version"] = 1;
  result["provider"] = {{"name", value.provider.name},
                        {"options", value.provider.options},
                        {"allow_cpu_fallback", value.allow_cpu_fallback}};
  return result;
}
void Fields(const Json& value, std::initializer_list<std::string_view> names) {
  if (!value.is_object() || value.size() != names.size()) Schema();
  for (const auto name : names)
    if (!value.contains(std::string(name))) Schema();
}
std::string String(const Json& value) {
  if (!value.is_string()) Schema();
  return value.get<std::string>();
}
std::vector<std::string> Strings(const Json& value) {
  if (!value.is_array()) Schema();
  Bound(value.size(), 64U);
  std::vector<std::string> result;
  result.reserve(value.size());
  for (const auto& entry : value) result.push_back(String(entry));
  return result;
}
bool Boolean(const Json& value) {
  if (!value.is_boolean()) Schema();
  return value.get<bool>();
}
Json Parse(std::string_view encoded) {
  Bound(encoded.size(), kEnvironmentLockMaxBytes);
  if (encoded.empty()) Schema();
  std::vector<std::set<std::string>> keys;
  std::size_t nodes = 0U;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 8 || ++nodes > 4096U)
      throw ResourceExhausted(
          "ONNX Runtime environment JSON exceeds its bound");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        !keys.back().insert(value.get<std::string>()).second)
      Schema();
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  return Json::parse(encoded.begin(), encoded.end(), callback);
}
std::string Lower(std::string value) {
  for (auto& byte : value)
    if (byte >= 'A' && byte <= 'Z') byte = static_cast<char>(byte - 'A' + 'a');
  return value;
}
}  // namespace

EnvironmentLock observe_environment(const Session& session) {
  const auto build = current_build_fingerprint();
  const auto& provider = session.execution_provider();
  const auto domain = provider.name == "CPUExecutionProvider" ? "cpu"
                      : provider.name == "CUDAExecutionProvider" ||
                              provider.name == "TensorrtExecutionProvider"
                          ? "cuda"
                          : Lower(provider.name);
  const auto dynamic = [](auto signatures) {
    return std::any_of(
        signatures.begin(), signatures.end(),
        [](const auto& signature) { return !signature.batch_symbol.empty(); });
  };
  EnvironmentLock result{
      .runtime = {.backend_id = std::string(kEnvironmentBackendId),
                  .backend_version = build.jyppx_version,
                  .dependencies = {{.name = "libtorch",
                                    .version = build.libtorch_version},
                                   {.name = "liuding-forge",
                                    .version = build.jyppx_version},
                                   {.name = "onnxruntime",
                                    .version = session.runtime_version()}},
                  .devices = {domain},
                  .dtypes = {"float32"},
                  .os_arch = {Lower(build.os_arch)},
                  .dynamic_shapes =
                      dynamic(session.inputs()) || dynamic(session.outputs())},
      .provider = provider,
      .allow_cpu_fallback = session.cpu_fallback_allowed()};
  static_cast<void>(Document(result));
  return result;
}

EnvironmentLock parse_environment_lock(std::string_view encoded) {
  try {
    const auto json = Parse(encoded);
    Fields(json, {"format", "schema_version", "backend_id", "backend_version",
                  "dependencies", "devices", "dtypes", "os_arch",
                  "dynamic_shapes", "provider"});
    const auto& version = json.at("schema_version");
    if (String(json.at("format")) != kEnvironmentLockSchemaId ||
        !version.is_number_integer() || version != 1)
      Schema();
    EnvironmentLock result;
    result.runtime.backend_id = String(json.at("backend_id"));
    result.runtime.backend_version = String(json.at("backend_version"));
    const auto& dependencies = json.at("dependencies");
    if (!dependencies.is_array()) Schema();
    Bound(dependencies.size(), 128U);
    for (const auto& entry : dependencies) {
      Fields(entry, {"name", "version"});
      result.runtime.dependencies.push_back(
          {.name = String(entry.at("name")),
           .version = String(entry.at("version"))});
    }
    result.runtime.devices = Strings(json.at("devices"));
    result.runtime.dtypes = Strings(json.at("dtypes"));
    result.runtime.os_arch = Strings(json.at("os_arch"));
    result.runtime.dynamic_shapes = Boolean(json.at("dynamic_shapes"));
    const auto& provider = json.at("provider");
    Fields(provider, {"name", "options", "allow_cpu_fallback"});
    result.provider.name = String(provider.at("name"));
    result.allow_cpu_fallback = Boolean(provider.at("allow_cpu_fallback"));
    const auto& options = provider.at("options");
    if (!options.is_object()) Schema();
    Bound(options.size(), 128U);
    for (const auto& [key, value] : options.items())
      result.provider.options.emplace(key, String(value));
    static_cast<void>(Document(result));
    return result;
  } catch (const InvalidArgument&) {
    std::throw_with_nested(SerializationError(
        "Invalid ONNX Runtime environment-lock declaration"));
  } catch (const Json::exception&) {
    std::throw_with_nested(
        SerializationError("Cannot parse ONNX Runtime environment lock"));
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(
        ResourceExhausted("Cannot allocate ONNX Runtime environment lock"));
  }
}

std::string serialize_environment_lock(const EnvironmentLock& environment) {
  try {
    auto result = Document(environment).dump();
    Bound(result.size(), kEnvironmentLockMaxBytes);
    return result;
  } catch (const Json::exception&) {
    std::throw_with_nested(
        InvalidArgument("Cannot encode ONNX Runtime environment lock"));
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(
        ResourceExhausted("Cannot allocate ONNX Runtime environment lock"));
  }
}

EnvironmentLock verify_environment_lock(
    const Session& session, std::string_view encoded,
    const ArtifactDigest& expected, const EnvironmentRequirements& declared) {
  validate_artifact_digest(expected);
  Bound(encoded.size(), kEnvironmentLockMaxBytes);
  if (sha256_hex(std::as_bytes(std::span(encoded.data(), encoded.size()))) !=
      expected.hex)
    throw CompatibilityError("ONNX Runtime environment lock digest mismatch");
  auto parsed = parse_environment_lock(encoded);
  const auto observed = observe_environment(session);
  if (Document(parsed) != Document(observed))
    throw CompatibilityError(
        "ONNX Runtime environment does not match this execution point");
  if (Declaration(parsed.runtime) != Declaration(declared))
    throw CompatibilityError(
        "ONNX bundle runtime declaration differs from its checked lock");
  return parsed;
}
}  // namespace jyppx::liuding::onnx_runtime
