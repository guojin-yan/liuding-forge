// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "native_otlp_worker.hpp"

#include "native_telemetry_protocol.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#ifdef JYPPX_LIUDING_CLI_NATIVE_OTLP_HTTP
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
#endif

namespace jyppx::liuding::cli {
namespace {
#ifdef JYPPX_LIUDING_CLI_NATIVE_OTLP_HTTP
constexpr std::size_t kConfigBytes = 64U * 1024U;
constexpr std::size_t kRequestBytes = 16U * 1024U * 1024U;
constexpr std::size_t kReplyBytes = 128U * 1024U;

struct WorkerConfig {
  experiment::OtlpHttpOptions http;
  SidecarProcessOptions process;
};

std::uint64_t positive_number(const Json& object, const char* name,
                              std::uint64_t fallback, std::uint64_t maximum) {
  const auto value = object.contains(name)
                         ? unsigned_number(object.at(name), maximum)
                         : fallback;
  if (!value) throw InvalidArgument("Native OTLP limits must be positive");
  return value;
}

WorkerConfig configuration(const Json& value) {
  fields(value,
         {"schema_version", "endpoint", "resource_instance_id",
          "working_directory"},
         {"clock_mappings", "environment", "inherit_environment",
          "operation_timeout_ms", "startup_timeout_ms", "cleanup_timeout_ms",
          "request_timeout_ms", "max_seen", "max_seen_bytes",
          "max_batch_prepared_bytes"});
  if (unsigned_number(value.at("schema_version"), 1U) != 1U)
    throw CompatibilityError("Unsupported native OTLP configuration version");
  WorkerConfig result;
  const auto& endpoint = value.at("endpoint");
  fields(endpoint, {"host", "port", "tls"});
  result.http.endpoint.host = endpoint.at("host").get<std::string>();
  result.http.endpoint.port = static_cast<std::uint16_t>(
      positive_number(endpoint, "port", 4318U, UINT16_MAX));
  result.http.endpoint.tls = endpoint.at("tls").get<bool>();
  result.http.mapping.resource_instance_id =
      value.at("resource_instance_id").get<std::string>();
  if (value.contains("clock_mappings")) {
    const auto& mappings = value.at("clock_mappings");
    if (!mappings.is_object() || mappings.size() > 64U)
      throw ResourceExhausted("Native OTLP clock mapping bound");
    for (const auto& [name, anchor] : mappings.items()) {
      fields(anchor, {"monotonic_ns", "unix_ns"});
      result.http.mapping.clock_mappings.emplace(
          name, experiment::OtlpClockAnchor{
                    unsigned_number(anchor.at("monotonic_ns"), UINT64_MAX),
                    unsigned_number(anchor.at("unix_ns"), UINT64_MAX)});
    }
  }
  result.http.request_timeout = std::chrono::milliseconds(
      positive_number(value, "request_timeout_ms", 30000U, 30000U));
  result.http.max_seen = static_cast<std::size_t>(
      positive_number(value, "max_seen", 100000U, 100000U));
  result.http.max_seen_bytes = positive_number(
      value, "max_seen_bytes", 32U * 1024U * 1024U, 32U * 1024U * 1024U);
  result.http.max_batch_prepared_bytes = static_cast<std::size_t>(
      positive_number(value, "max_batch_prepared_bytes", 32U * 1024U * 1024U,
                      32U * 1024U * 1024U));
  result.http.descriptor.sink_id = "cli.export.opentelemetry-native";
  result.http.descriptor.destination_id = "configured.opentelemetry-native";
  result.process.working_directory =
      std::filesystem::u8path(value.at("working_directory").get<std::string>());
  if (value.contains("environment")) {
    const auto& environment = value.at("environment");
    if (!environment.is_object() || environment.size() > 128U)
      throw ResourceExhausted("Native OTLP child environment bound");
    result.process.environment =
        environment.get<std::map<std::string, std::string>>();
  }
  if (value.contains("inherit_environment"))
    result.process.inherit_environment =
        value.at("inherit_environment").get<bool>();
  result.process.operation_timeout = std::chrono::milliseconds(
      positive_number(value, "operation_timeout_ms", 10000U, 600000U));
  result.process.startup_timeout = std::chrono::milliseconds(
      positive_number(value, "startup_timeout_ms", 10000U, 600000U));
  result.process.cleanup_timeout = std::chrono::milliseconds(
      positive_number(value, "cleanup_timeout_ms", 2000U, 10000U));
  return result;
}

std::filesystem::path this_executable() {
#if defined(_WIN32)
  std::wstring buffer(32768U, L'\0');
  const auto length = GetModuleFileNameW(nullptr, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
  if (!length || length >= buffer.size())
    throw IoError("Cannot identify the native OTLP worker executable");
  buffer.resize(length);
  return std::filesystem::path(buffer);
#elif defined(__linux__)
  std::error_code error;
  const auto result = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || !result.is_absolute())
    throw IoError("Cannot identify the native OTLP worker executable");
  return result;
#else
  throw CompatibilityError("Native OTLP workers require Windows or Linux");
#endif
}

std::string utf8_path(const std::filesystem::path& path) {
  const auto value = path.u8string();
  return std::string(value.begin(), value.end());
}

std::string digest(std::string_view value) {
  return sha256_hex(
      {reinterpret_cast<const std::byte*>(value.data()), value.size()});
}

// No std::getline: allocation never follows an unbounded input frame. The
// trusted parent owns cancellation/deadline supervision even while stdin waits.
bool read_frame(std::string& frame) {
  frame.clear();
  while (true) {
    const auto byte = std::cin.get();
    if (byte == std::char_traits<char>::eof()) {
      if (std::cin.bad() || !frame.empty())
        throw SerializationError("Incomplete native OTLP request frame");
      return false;
    }
    if (byte == '\n') {
      if (frame.empty())
        throw SerializationError("Empty native OTLP request frame");
      return true;
    }
    if (byte == '\r' || byte == '\0')
      throw SerializationError("Invalid native OTLP request framing");
    if (frame.size() == kRequestBytes)
      throw ResourceExhausted("Native OTLP request frame bound");
    frame.push_back(static_cast<char>(byte));
  }
}
#endif
}  // namespace

SidecarProcessOptions native_otlp_worker_options(
    const Arguments& arguments, std::uint64_t snapshot_events) {
#ifdef JYPPX_LIUDING_CLI_NATIVE_OTLP_HTTP
  std::error_code error;
  const auto path = std::filesystem::canonical(
      std::filesystem::u8path(arguments.require("native-config")), error);
  if (error || !path.is_absolute())
    throw IoError("Could not resolve native OTLP configuration");
  const auto content = read_text(path, kConfigBytes);
  auto config =
      configuration(parse_native_telemetry_json(content, kConfigBytes));
  // The CLI scans a bounded canonical snapshot; reject the pessimistic number
  // of distinct identities up front, without relying on eviction/restarts.
  if (snapshot_events > config.http.max_seen)
    throw ResourceExhausted("Native OTLP snapshot exceeds the seen capacity",
                            "Select a smaller explicit snapshot.");
  config.process.executable = this_executable();
  config.process.arguments = {"__liuding_otlp_worker", "--config",
                              utf8_path(path), "--config-sha256",
                              digest(content)};
  return std::move(config.process);
#else
  (void)arguments;
  (void)snapshot_events;
  throw NotImplemented("Native OTLP HTTP is disabled in this build");
#endif
}

int native_otlp_worker_entry(
    std::span<const std::string_view> values) noexcept {
#ifdef JYPPX_LIUDING_CLI_NATIVE_OTLP_HTTP
  try {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1)
      throw IoError("Cannot set native OTLP protocol stream mode");
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    const Arguments arguments(values);
    arguments.allow({"config", "config-sha256"});
    if (arguments.words().size() != 1U ||
        arguments.words().front() != "__liuding_otlp_worker")
      throw InvalidArgument("Invalid native OTLP worker invocation");
    const auto content = read_text(
        std::filesystem::u8path(arguments.require("config")), kConfigBytes);
    if (digest(content) != arguments.require("config-sha256"))
      throw CompatibilityError(
          "Native OTLP configuration changed after launch");
    auto config =
        configuration(parse_native_telemetry_json(content, kConfigBytes));
    experiment::OtlpHttpTelemetrySink sink(std::move(config.http));
    NativeTelemetryProtocol protocol(sink);
    std::string frame;
    while (read_frame(frame)) {
      const auto deadline =
          std::chrono::steady_clock::now() + config.process.operation_timeout;
      const auto request = parse_native_telemetry_json(frame, kRequestBytes);
      const auto reply = protocol.exchange(request, deadline).dump();
      if (reply.size() > kReplyBytes)
        throw ResourceExhausted("Native OTLP response frame bound");
      std::cout.write(reply.data(), static_cast<std::streamsize>(reply.size()));
      std::cout.put('\n');
      std::cout.flush();
      if (!std::cout) throw IoError("Native OTLP response stream failed");
    }
    // EOF/destruction never flushes the network. The supervisor owns cleanup.
    return 0;
  } catch (...) {
    std::fputs("Native OTLP worker failed.\n", stderr);
    return 3;
  }
#else
  (void)values;
  std::fputs("Native OTLP HTTP is disabled in this build.\n", stderr);
  return 2;
#endif
}

}  // namespace jyppx::liuding::cli
