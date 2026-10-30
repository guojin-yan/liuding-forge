// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "native_tensorboard_worker.hpp"

#include "native_telemetry_protocol.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

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
#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
constexpr std::size_t kConfigBytes = 64U * 1024U;
constexpr std::size_t kRequestBytes = 16U * 1024U * 1024U;
constexpr std::size_t kReplyBytes = 128U * 1024U;

struct WorkerConfig {
  experiment::TensorBoardTelemetryOptions writer;
  SidecarProcessOptions process;
};

std::uint64_t positive_number(const Json& value, const char* name,
                              std::uint64_t fallback, std::uint64_t maximum) {
  const auto result = value.contains(name)
                          ? unsigned_number(value.at(name), maximum)
                          : fallback;
  if (!result)
    throw InvalidArgument("Native TensorBoard limits must be positive");
  return result;
}

std::string text(const Json& value) {
  if (!value.is_string())
    throw SerializationError("Native TensorBoard configuration requires text");
  auto result = value.get<std::string>();
  if (result.empty() || result.find('\0') != std::string::npos)
    throw InvalidArgument(
        "Native TensorBoard configuration text is empty or invalid");
  return result;
}

std::filesystem::path absolute_path(const Json& value) {
  const auto result = std::filesystem::u8path(text(value));
  if (!result.is_absolute())
    throw InvalidArgument("Native TensorBoard paths must be absolute");
  return result;
}

WorkerConfig configuration(const Json& value) {
  fields(value, {"schema_version", "directory", "working_directory"},
         {"environment", "inherit_environment", "operation_timeout_ms",
          "startup_timeout_ms", "cleanup_timeout_ms", "max_file_bytes",
          "max_record_bytes", "max_batch_encoded_bytes", "max_seen",
          "max_seen_bytes"});
  if (unsigned_number(value.at("schema_version"), 1U) != 1U)
    throw CompatibilityError(
        "Unsupported native TensorBoard configuration version");
  WorkerConfig result;
  const auto directory = absolute_path(value.at("directory"));
  if (directory.filename().empty() || directory.filename() == "." ||
      directory.filename() == "..")
    throw InvalidArgument(
        "Native TensorBoard output requires a new directory name");
  std::error_code error;
  const auto parent =
      std::filesystem::canonical(directory.parent_path(), error);
  if (error || !std::filesystem::is_directory(parent))
    throw IoError("Native TensorBoard output parent must exist");
  result.writer.directory = parent / directory.filename();
  const auto status =
      std::filesystem::symlink_status(result.writer.directory, error);
  if ((error && error != std::errc::no_such_file_or_directory) ||
      (!error && std::filesystem::exists(status)))
    throw IoError("Native TensorBoard requires a fresh output directory");
  result.writer.sink_id = "cli.export.tensorboard-native";
  result.writer.destination_id = "local.tensorboard-native";
  result.writer.max_file_bytes =
      positive_number(value, "max_file_bytes", 1ULL << 30U, 1ULL << 30U);
  result.writer.max_record_bytes = static_cast<std::size_t>(positive_number(
      value, "max_record_bytes", 4U * 1024U * 1024U, 4U * 1024U * 1024U));
  result.writer.max_batch_encoded_bytes = static_cast<std::size_t>(
      positive_number(value, "max_batch_encoded_bytes", 32U * 1024U * 1024U,
                      32U * 1024U * 1024U));
  if (result.writer.max_batch_encoded_bytes < 16U)
    throw InvalidArgument("Native TensorBoard batch budget is too small");
  result.writer.max_seen = static_cast<std::size_t>(
      positive_number(value, "max_seen", 100000U, 100000U));
  result.writer.max_seen_bytes = positive_number(
      value, "max_seen_bytes", 32U * 1024U * 1024U, 32U * 1024U * 1024U);
  error.clear();
  result.process.working_directory = std::filesystem::canonical(
      absolute_path(value.at("working_directory")), error);
  if (error || !std::filesystem::is_directory(result.process.working_directory))
    throw IoError("Native TensorBoard worker directory must exist");
  if (value.contains("environment")) {
    const auto& environment = value.at("environment");
    if (!environment.is_object() || environment.size() > 128U)
      throw ResourceExhausted("Native TensorBoard child environment bound");
    for (const auto& [name, field] : environment.items()) {
      if (name.empty() || name.find('=') != std::string::npos ||
          name.find('\0') != std::string::npos || !field.is_string() ||
          field.get_ref<const std::string&>().find('\0') != std::string::npos)
        throw InvalidArgument("Invalid native TensorBoard child environment");
    }
    result.process.environment =
        environment.get<std::map<std::string, std::string>>();
  }
  if (value.contains("inherit_environment")) {
    if (!value.at("inherit_environment").is_boolean())
      throw SerializationError(
          "Native TensorBoard inheritance requires a boolean");
    result.process.inherit_environment =
        value.at("inherit_environment").get<bool>();
  }
  result.process.operation_timeout = std::chrono::milliseconds(
      positive_number(value, "operation_timeout_ms", 10000U, 600000U));
  result.process.startup_timeout = std::chrono::milliseconds(
      positive_number(value, "startup_timeout_ms", 10000U, 600000U));
  result.process.cleanup_timeout = std::chrono::milliseconds(
      positive_number(value, "cleanup_timeout_ms", 2000U, 10000U));
  return result;
}

void outside_store(const std::filesystem::path& directory,
                   const std::filesystem::path& managed_root) {
  for (auto ancestor = directory; !ancestor.empty();) {
    std::error_code error;
    if (std::filesystem::equivalent(ancestor, managed_root, error) && !error)
      throw InvalidArgument(
          "Native TensorBoard output must be outside the run store");
    if (error && error != std::errc::no_such_file_or_directory)
      throw IoError("Cannot inspect native TensorBoard output ancestry");
    const auto parent = ancestor.parent_path();
    if (ancestor == parent) break;
    ancestor = parent;
  }
}

std::filesystem::path this_executable() {
#if defined(_WIN32)
  std::wstring buffer(32768U, L'\0');
  const auto length = GetModuleFileNameW(nullptr, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
  if (!length || length >= buffer.size())
    throw IoError("Cannot identify the native TensorBoard worker executable");
  buffer.resize(length);
  return std::filesystem::path(buffer);
#elif defined(__linux__)
  std::error_code error;
  const auto result = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || !result.is_absolute())
    throw IoError("Cannot identify the native TensorBoard worker executable");
  return result;
#else
  throw CompatibilityError(
      "Native TensorBoard workers require Windows or Linux");
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

// The parent supervises deadlines while this bounded reader waits for input.
bool read_frame(std::string& frame) {
  frame.clear();
  while (true) {
    const auto byte = std::cin.get();
    if (byte == std::char_traits<char>::eof()) {
      if (std::cin.bad() || !frame.empty())
        throw SerializationError("Incomplete native TensorBoard request frame");
      return false;
    }
    if (byte == '\n') {
      if (frame.empty())
        throw SerializationError("Empty native TensorBoard request frame");
      return true;
    }
    if (byte == '\r' || byte == '\0')
      throw SerializationError("Invalid native TensorBoard request framing");
    if (frame.size() == kRequestBytes)
      throw ResourceExhausted("Native TensorBoard request frame bound");
    frame.push_back(static_cast<char>(byte));
  }
}
#endif
}  // namespace

SidecarProcessOptions native_tensorboard_worker_options(
    const Arguments& arguments, std::uint64_t snapshot_events,
    const std::filesystem::path& managed_root) {
#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
  std::error_code error;
  const auto path = std::filesystem::canonical(
      std::filesystem::u8path(arguments.require("native-config")), error);
  if (error || !path.is_absolute())
    throw IoError("Could not resolve native TensorBoard configuration");
  const auto content = read_text(path, kConfigBytes);
  auto config =
      configuration(parse_native_telemetry_json(content, kConfigBytes));
  outside_store(config.writer.directory, managed_root);
  if (snapshot_events > config.writer.max_seen)
    throw ResourceExhausted(
        "Native TensorBoard snapshot exceeds the seen capacity");
  config.process.executable = this_executable();
  config.process.arguments = {"__liuding_tensorboard_worker",
                              "--config",
                              utf8_path(path),
                              "--config-sha256",
                              digest(content),
                              "--managed-root",
                              utf8_path(managed_root)};
  return std::move(config.process);
#else
  (void)arguments;
  (void)snapshot_events;
  (void)managed_root;
  throw NotImplemented("Native TensorBoard is disabled in this build");
#endif
}

int native_tensorboard_worker_entry(
    std::span<const std::string_view> values) noexcept {
#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
  try {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 ||
        _setmode(_fileno(stdout), _O_BINARY) == -1)
      throw IoError("Cannot set native TensorBoard protocol stream mode");
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    const Arguments arguments(values);
    arguments.allow({"config", "config-sha256", "managed-root"});
    if (arguments.words().size() != 1U ||
        arguments.words().front() != "__liuding_tensorboard_worker" ||
        arguments.optional("format"))
      throw InvalidArgument("Invalid native TensorBoard worker invocation");
    const auto config_path =
        std::filesystem::u8path(arguments.require("config"));
    if (!config_path.is_absolute())
      throw InvalidArgument(
          "Native TensorBoard configuration path must be absolute");
    const auto content = read_text(config_path, kConfigBytes);
    if (digest(content) != arguments.require("config-sha256"))
      throw CompatibilityError(
          "Native TensorBoard configuration changed after launch");
    auto config =
        configuration(parse_native_telemetry_json(content, kConfigBytes));
    std::error_code error;
    const auto managed_root = std::filesystem::canonical(
        absolute_path(arguments.require("managed-root")), error);
    if (error || !std::filesystem::is_directory(managed_root))
      throw IoError("Native TensorBoard run store must exist");
    outside_store(config.writer.directory, managed_root);
    experiment::TensorBoardTelemetrySink sink(std::move(config.writer));
    NativeTelemetryProtocol protocol(sink, {}, "tensorboard");
    std::string frame;
    while (read_frame(frame)) {
      const auto deadline =
          std::chrono::steady_clock::now() + config.process.operation_timeout;
      const auto request = parse_native_telemetry_json(frame, kRequestBytes);
      const auto reply = protocol.exchange(request, deadline).dump();
      if (reply.size() > kReplyBytes)
        throw ResourceExhausted("Native TensorBoard response frame bound");
      std::cout.write(reply.data(), static_cast<std::streamsize>(reply.size()));
      std::cout.put('\n');
      std::cout.flush();
      if (!std::cout)
        throw IoError("Native TensorBoard response stream failed");
    }
    // EOF/destruction releases file handles; only protocol close can finalize.
    return 0;
  } catch (...) {
    std::fputs("Native TensorBoard worker failed.\n", stderr);
    return 3;
  }
#else
  (void)values;
  std::fputs("Native TensorBoard is disabled in this build.\n", stderr);
  return 2;
#endif
}

}  // namespace jyppx::liuding::cli
