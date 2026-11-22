// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#include "../../foundation/nested_error.hpp"

#include <jyppx/liuding/deployment/migraphx_runtime/environment.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>
#include <limits>
#include <migraphx/migraphx.h>
#include <migraphx/version.h>
#include <nlohmann/json.hpp>
#include <set>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace jyppx::liuding::migraphx_runtime {
namespace {
using Json = nlohmann::json;
[[noreturn]] void Invalid() {
  throw InvalidArgument("Invalid MIGraphX environment declaration");
}
[[noreturn]] void Schema() {
  throw SerializationError("Invalid MIGraphX environment-lock schema");
}
[[noreturn]] void Mismatch() {
  throw CompatibilityError("MIGraphX environment does not match this session");
}
void Bound(std::uint64_t count, std::uint64_t maximum) {
  if (count > maximum)
    throw ResourceExhausted("MIGraphX environment metadata exceeds its bound");
}
void Text(std::string_view value, std::size_t maximum = 128U) {
  Bound(value.size(), maximum);
  if (value.empty() ||
      std::any_of(value.begin(), value.end(),
                  [](unsigned char c) { return c < 32 || c > 126; }))
    Invalid();
}
void Hex(std::string_view value, std::size_t length) {
  if (value.size() != length ||
      std::any_of(value.begin(), value.end(), [](char c) {
        return !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
      }))
    Invalid();
}
void Shape(const std::vector<std::size_t>& shape) {
  Bound(shape.size(), 8);
  if (shape.empty()) Invalid();
  std::uint64_t count = 1;
  for (const auto dimension : shape) {
    if (!dimension ||
        dimension > std::numeric_limits<std::uint64_t>::max() / 4U / count)
      Invalid();
    count *= dimension;
  }
}
Json Module(const EnvironmentModule& value) {
  Hex(value.sha256, 64);
  Bound(value.size_bytes, kEnvironmentModuleMaxBytes);
  if (!value.size_bytes) Invalid();
  return {{"sha256", value.sha256}, {"size_bytes", value.size_bytes}};
}
Json Document(const EnvironmentLock& value) {
  for (const auto* text : {&value.adapter_version, &value.libtorch_version,
                           &value.os_arch, &value.migraphx_header_version,
                           &value.hip_header_version, &value.device_pci_bus_id})
    Text(*text);
  Text(value.device_architecture, 255);
  Text(value.input_name, 255);
  Hex(value.device_uuid, 32);
  if (value.device_uuid == std::string(32, '0') || value.device_id < 0 ||
      value.hip_runtime_version <= 0 || value.hip_driver_version <= 0)
    Invalid();
  Shape(value.input_shape);
  Shape(value.output_shape);
  return {{"format", kEnvironmentLockSchemaId},
          {"schema_version", 1},
          {"build",
           {{"adapter_version", value.adapter_version},
            {"libtorch_version", value.libtorch_version},
            {"os_arch", value.os_arch}}},
          {"sdk_headers",
           {{"migraphx", value.migraphx_header_version},
            {"hip", value.hip_header_version}}},
          {"modules",
           {{"migraphx_c", Module(value.migraphx_c)},
            {"hip_runtime", Module(value.hip_runtime)}}},
          {"hip",
           {{"runtime_version", value.hip_runtime_version},
            {"driver_version", value.hip_driver_version},
            {"device_id", value.device_id},
            {"architecture", value.device_architecture},
            {"uuid", value.device_uuid},
            {"pci_bus_id", value.device_pci_bus_id}}},
          {"program",
           {{"target", "gpu"},
            {"dtype", "float32"},
            {"offload_copy", true},
            {"fast_math", false},
            {"exhaustive_tune", false},
            {"input_name", value.input_name},
            {"input_shape", value.input_shape},
            {"output_shape", value.output_shape}}}};
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
std::uint64_t Integer(const Json& value, std::uint64_t maximum) {
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
    Schema();
  const auto result = value.get<std::uint64_t>();
  if (result > maximum) Schema();
  return result;
}
EnvironmentModule ParseModule(const Json& value) {
  Fields(value, {"sha256", "size_bytes"});
  return {.sha256 = String(value.at("sha256")),
          .size_bytes = Integer(value.at("size_bytes"),
                                std::numeric_limits<std::uint64_t>::max())};
}
std::vector<std::size_t> ParseShape(const Json& value) {
  if (!value.is_array()) Schema();
  Bound(value.size(), 8);
  std::vector<std::size_t> result;
  for (const auto& dimension : value)
    result.push_back(static_cast<std::size_t>(
        Integer(dimension, std::numeric_limits<std::size_t>::max())));
  return result;
}
Json Parse(std::string_view encoded) {
  Bound(encoded.size(), kEnvironmentLockMaxBytes);
  if (encoded.empty()) Schema();
  std::vector<std::set<std::string>> keys;
  std::size_t nodes = 0;
  const auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if (depth > 6 || ++nodes > 512U)
      throw ResourceExhausted("MIGraphX environment JSON exceeds its bound");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        !keys.back().insert(value.get<std::string>()).second)
      Schema();
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    return true;
  };
  return Json::parse(encoded.begin(), encoded.end(), callback);
}
std::filesystem::path ModulePath(const void* address) {
#ifdef _WIN32
  HMODULE module{};
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(address), &module))
    Mismatch();
  std::vector<wchar_t> path(32768);
  const auto size =
      GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
  if (!size || size >= path.size()) Mismatch();
  return std::filesystem::path(path.data(), path.data() + size);
#else
  Dl_info info{};
  if (!dladdr(address, &info) || !info.dli_fname) Mismatch();
  const std::filesystem::path path(info.dli_fname);
  if (!path.is_absolute()) Mismatch();
  return path;
#endif
}
#ifdef _WIN32
const void* LoadedExport(const wchar_t* module_name, const char* symbol) {
  // HIP's Windows declarations can yield an executable import trampoline when
  // taking a linked function address. Resolve the export from the already
  // loaded DLL so its backing file, rather than the caller EXE, is identified.
  // GetModuleHandle/GetProcAddress never load another library.
  const auto module = GetModuleHandleW(module_name);
  if (!module) Mismatch();
  const auto address = GetProcAddress(module, symbol);
  if (!address) Mismatch();
  return reinterpret_cast<const void*>(address);
}
#endif
EnvironmentModule ObserveModule(const void* address) {
  const auto path = ModulePath(address);
  if (!std::filesystem::is_regular_file(path)) Mismatch();
  const auto size = std::filesystem::file_size(path);
  const auto modified = std::filesystem::last_write_time(path);
  Bound(size, kEnvironmentModuleMaxBytes);
  if (!size) Mismatch();
  std::ifstream stream(path, std::ios::binary);
  if (!stream) Mismatch();
  Sha256 digest;
  std::array<std::byte, 65536> buffer;
  std::uint64_t remaining = size;
  while (remaining) {
    const auto count = static_cast<std::streamsize>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    stream.read(reinterpret_cast<char*>(buffer.data()), count);
    if (stream.gcount() != count || !stream) Mismatch();
    digest.update(std::span(buffer.data(), static_cast<std::size_t>(count)));
    remaining -= static_cast<std::uint64_t>(count);
  }
  if (stream.peek() != std::char_traits<char>::eof() ||
      std::filesystem::file_size(path) != size ||
      std::filesystem::last_write_time(path) != modified)
    Mismatch();
  return {.sha256 = sha256_digest_hex(digest.finish()), .size_bytes = size};
}
void Hip(hipError_t result) {
  if (result != hipSuccess)
    throw DeviceError("Cannot observe MIGraphX HIP environment",
                      {.component = "migraphx_runtime.environment",
                       .actual = std::to_string(result)});
}
template <class Char, std::size_t Count>
std::string BoundedString(const Char (&data)[Count]) {
  const auto end = std::find(std::begin(data), std::end(data), '\0');
  if (end == std::end(data)) Mismatch();
  return std::string(std::begin(data), end);
}
}  // namespace

EnvironmentLock observe_environment(const Session& session) {
  try {
    const auto build = current_build_fingerprint();
    EnvironmentLock result;
    result.adapter_version = build.jyppx_version;
    result.libtorch_version = build.libtorch_version;
    result.os_arch = build.os_arch;
    for (auto& c : result.os_arch)
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    result.migraphx_header_version =
        std::to_string(MIGRAPHX_VERSION_MAJOR) + "." +
        std::to_string(MIGRAPHX_VERSION_MINOR) + "." +
        std::to_string(MIGRAPHX_VERSION_PATCH);
    result.hip_header_version = std::to_string(HIP_VERSION_MAJOR) + "." +
                                std::to_string(HIP_VERSION_MINOR) + "." +
                                std::to_string(HIP_VERSION_PATCH);
#ifdef _WIN32
    result.migraphx_c = ObserveModule(
        LoadedExport(L"migraphx_c.dll", "migraphx_program_create"));
    result.hip_runtime =
        ObserveModule(LoadedExport(L"amdhip64_7.dll", "hipRuntimeGetVersion"));
#else
    result.migraphx_c =
        ObserveModule(reinterpret_cast<const void*>(&migraphx_program_create));
    result.hip_runtime =
        ObserveModule(reinterpret_cast<const void*>(&hipRuntimeGetVersion));
#endif
    int runtime = 0, driver = 0;
    Hip(hipRuntimeGetVersion(&runtime));
    Hip(hipDriverGetVersion(&driver));
    result.hip_runtime_version = runtime;
    result.hip_driver_version = driver;
    result.device_id = session.options().device_id;
    hipDeviceProp_t properties{};
    Hip(hipGetDeviceProperties(&properties, result.device_id));
    result.device_architecture = BoundedString(properties.gcnArchName);
    hipUUID uuid{};
    Hip(hipDeviceGetUuid(&uuid, result.device_id));
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : uuid.bytes) {
      const auto value = static_cast<unsigned char>(byte);
      result.device_uuid += hex[value >> 4U];
      result.device_uuid += hex[value & 15U];
    }
    char pci[128]{};
    Hip(hipDeviceGetPCIBusId(pci, sizeof(pci), result.device_id));
    result.device_pci_bus_id = BoundedString(pci);
    result.input_name = session.options().input_name;
    result.input_shape = session.options().input_shape;
    result.output_shape.assign(session.output_shape().begin(),
                               session.output_shape().end());
    static_cast<void>(Document(result));
    return result;
  } catch (const std::filesystem::filesystem_error&) {
    ::jyppx::liuding::detail::throw_nested_error(
        CompatibilityError("Cannot read loaded MIGraphX/HIP backing file"));
  } catch (const std::bad_alloc&) {
    ::jyppx::liuding::detail::throw_nested_error(
        ResourceExhausted("Cannot allocate MIGraphX environment metadata"));
  }
}
EnvironmentLock parse_environment_lock(std::string_view encoded) {
  try {
    const auto json = Parse(encoded);
    Fields(json, {"format", "schema_version", "build", "sdk_headers", "modules",
                  "hip", "program"});
    if (String(json.at("format")) != kEnvironmentLockSchemaId ||
        Integer(json.at("schema_version"), 1) != 1)
      Schema();
    const auto& build = json.at("build");
    Fields(build, {"adapter_version", "libtorch_version", "os_arch"});
    const auto& headers = json.at("sdk_headers");
    Fields(headers, {"migraphx", "hip"});
    const auto& modules = json.at("modules");
    Fields(modules, {"migraphx_c", "hip_runtime"});
    const auto& hip = json.at("hip");
    Fields(hip, {"runtime_version", "driver_version", "device_id",
                 "architecture", "uuid", "pci_bus_id"});
    const auto& program = json.at("program");
    Fields(program,
           {"target", "dtype", "offload_copy", "fast_math", "exhaustive_tune",
            "input_name", "input_shape", "output_shape"});
    if (String(program.at("target")) != "gpu" ||
        String(program.at("dtype")) != "float32")
      Schema();
    for (const auto key : {"offload_copy", "fast_math", "exhaustive_tune"})
      if (!program.at(key).is_boolean() ||
          program.at(key).get<bool>() !=
              (std::string_view(key) == "offload_copy"))
        Schema();
    EnvironmentLock result{
        .adapter_version = String(build.at("adapter_version")),
        .libtorch_version = String(build.at("libtorch_version")),
        .os_arch = String(build.at("os_arch")),
        .migraphx_header_version = String(headers.at("migraphx")),
        .hip_header_version = String(headers.at("hip")),
        .migraphx_c = ParseModule(modules.at("migraphx_c")),
        .hip_runtime = ParseModule(modules.at("hip_runtime")),
        .hip_runtime_version = static_cast<std::int32_t>(
            Integer(hip.at("runtime_version"),
                    std::numeric_limits<std::int32_t>::max())),
        .hip_driver_version = static_cast<std::int32_t>(
            Integer(hip.at("driver_version"),
                    std::numeric_limits<std::int32_t>::max())),
        .device_id = static_cast<std::int32_t>(Integer(
            hip.at("device_id"), std::numeric_limits<std::int32_t>::max())),
        .device_architecture = String(hip.at("architecture")),
        .device_uuid = String(hip.at("uuid")),
        .device_pci_bus_id = String(hip.at("pci_bus_id")),
        .input_name = String(program.at("input_name")),
        .input_shape = ParseShape(program.at("input_shape")),
        .output_shape = ParseShape(program.at("output_shape"))};
    static_cast<void>(Document(result));
    return result;
  } catch (const InvalidArgument&) {
    ::jyppx::liuding::detail::throw_nested_error(
        SerializationError("Invalid MIGraphX environment-lock declaration"));
  } catch (const Json::exception&) {
    ::jyppx::liuding::detail::throw_nested_error(
        SerializationError("Cannot parse MIGraphX environment lock"));
  } catch (const std::bad_alloc&) {
    ::jyppx::liuding::detail::throw_nested_error(
        ResourceExhausted("Cannot allocate MIGraphX environment lock"));
  }
}
std::string serialize_environment_lock(const EnvironmentLock& environment) {
  try {
    auto result = Document(environment).dump();
    Bound(result.size(), kEnvironmentLockMaxBytes);
    return result;
  } catch (const Json::exception&) {
    ::jyppx::liuding::detail::throw_nested_error(
        InvalidArgument("Cannot encode MIGraphX environment lock"));
  } catch (const std::bad_alloc&) {
    ::jyppx::liuding::detail::throw_nested_error(
        ResourceExhausted("Cannot allocate MIGraphX environment lock"));
  }
}
EnvironmentLock verify_environment_lock(const Session& session,
                                        std::string_view encoded,
                                        const ArtifactDigest& expected) {
  validate_artifact_digest(expected);
  Bound(encoded.size(), kEnvironmentLockMaxBytes);
  if (sha256_hex(std::as_bytes(std::span(encoded.data(), encoded.size()))) !=
      expected.hex)
    throw CompatibilityError("MIGraphX environment lock SHA-256 mismatch");
  auto parsed = parse_environment_lock(encoded);
  if (Document(parsed) != Document(observe_environment(session))) Mismatch();
  return parsed;
}
}  // namespace jyppx::liuding::migraphx_runtime
