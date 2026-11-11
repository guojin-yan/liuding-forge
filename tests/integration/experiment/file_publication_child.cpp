// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/experiment/detail/artifact_publish.hpp"
#include "jyppx/liuding/experiment/detail/file_publication.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
using namespace jyppx::liuding;
namespace ex = jyppx::liuding::experiment;
using Json = nlohmann::json;
constexpr std::size_t kSize = 2U * 1024U * 1024U + 257U;
std::filesystem::path active_root;
std::string active_boundary;
bool preparing = false;
bool recovering = false;

std::string utf8(const std::filesystem::path& path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::string content() {
  std::string result(kSize, '\0');
  for (std::size_t i = 0U; i < result.size(); ++i)
    result[i] = static_cast<char>((i * 37U + 11U) & 255U);
  return result;
}
ArtifactDigest digest() {
  const auto value = content();
  return {.hex =
              sha256_hex(std::as_bytes(std::span(value.data(), value.size())))};
}
std::filesystem::path payload_object() {
  const auto hash = digest().hex;
  return active_root / "store" / "artifacts" / "objects" / hash.substr(0U, 2U) /
         hash.substr(2U);
}
void report(Json value) {
  value["schema_version"] = 1U;
  const auto text = value.dump();
  if (text.size() > 65536U) throw std::runtime_error("Oversized child report");
  std::cout << text << '\n' << std::flush;
  if (!std::cout) throw std::runtime_error("Could not flush child report");
}
[[noreturn]] void fence(std::string_view boundary) {
#ifdef _WIN32
  const auto pid = ::_getpid();
#else
  const auto pid = ::getpid();
#endif
  report({{"phase", "ready-for-forced-kill"},
          {"boundary", boundary},
          {"process_id", pid},
          {"source_file_exists",
           std::filesystem::exists(active_root / "input.bin")},
          {"file_exists", std::filesystem::exists(active_root / "result.bin")},
          {"payload_cas_exists", std::filesystem::exists(payload_object())}});
  char ignored = 0;
  std::cin.get(ignored);
  std::_Exit(91);
}
int before_commit(void*) noexcept {
  try {
    if (preparing && active_boundary == "prepare-before-commit")
      fence(active_boundary);
    if (recovering && active_boundary == "recovery-before-commit")
      fence(active_boundary);
  } catch (...) {
    std::_Exit(92);
  }
  return 0;
}
int install_hook(sqlite3* database, char**, const sqlite3_api_routines*) {
  sqlite3_commit_hook(database, before_commit, nullptr);
  return SQLITE_OK;
}
ex::FilePublicationRequest request() {
  ex::FilePublicationRequest result;
  result.publication_id = "publication-1";
  result.destination = active_root / "result.bin";
  auto& value = result.evidence;
  value.run_id = "run-1";
  value.reference_id = "result-1";
  value.path = active_root / "input.bin";
  value.descriptor = {.logical_name = "result.bin",
                      .media_type = "application/octet-stream",
                      .producer = "file-publication-process-test",
                      .expected_size_bytes = kSize};
  value.attached_at_unix_ms = 101U;
  value.expected_digest = digest();
  value.event_batch_key = "result-batch-1";
  value.event.identity = {.stream_id = "output-1",
                          .event_id = "result-event-1",
                          .idempotency_key = "result-key-1",
                          .sequence = 1U};
  value.event.correlation.project_id = "project-1";
  value.event.correlation.run_id = "run-1";
  value.event.payload = TelemetryArtifactReference{.role = "image.preview"};
  return result;
}
struct UnwoundMarker {
  ~UnwoundMarker() {
    std::ofstream(active_root / "unwound.marker") << "unwound\n";
  }
};

int execute(std::string_view mode) {
  if (sqlite3_auto_extension(reinterpret_cast<void (*)()>(install_hook)) !=
      SQLITE_OK)
    throw std::runtime_error("Could not install official SQLite test hook");
  ex::SQLiteRunStore store(active_root / "store");
  if (mode == "seed" || mode == "seed-bytes") {
    store.start({.run_id = "run-1",
                 .project_id = "project-1",
                 .experiment_id = "experiment-1",
                 .display_name = "publication",
                 .status = ex::RunStatus::running,
                 .created_at_unix_ms = 100U,
                 .updated_at_unix_ms = 100U});
    if (mode == "seed-bytes") {
      report({{"phase", "completed"},
              {"mode", mode},
              {"references", 0},
              {"events", 0},
              {"integrity_ok", store.integrity_check().ok()}});
      return 0;
    }
    const auto bytes = content();
    std::ofstream output(active_root / "input.bin", std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Could not write source fixture");
  } else if (mode == "prepare" || mode == "prepare-fence" ||
             mode == "prepare-bytes" || mode == "prepare-bytes-fence") {
    UnwoundMarker marker;
    preparing = mode == "prepare-fence" || mode == "prepare-bytes-fence";
    if (mode == "prepare-bytes" || mode == "prepare-bytes-fence") {
      const auto payload = content();
      auto value = request();
      value.evidence.path.reset();
      store.prepare_file_publication_bytes(
          value, std::as_bytes(std::span(payload.data(), payload.size())));
    } else {
      store.prepare_file_publication(request());
    }
    if (preparing && active_boundary == "after-prepare") fence(active_boundary);
    if (preparing)
      throw std::runtime_error("Preparation fence was not reached");
  } else if (mode == "recover" || mode == "recover-fence") {
    UnwoundMarker marker;
    recovering = mode == "recover-fence";
    if (recovering) {
      // No producer request survives this process boundary. Discover only the
      // run's persisted IDs, then ask the product to reconstruct everything.
      const auto ids = store.file_publications("run-1");
      if (ids != std::vector<std::string>{"publication-1"})
        throw std::runtime_error("Prepared intent was not discoverable");
      store.recover_file_publication("run-1", ids.front());
      if (active_boundary == "after-complete") fence(active_boundary);
      throw std::runtime_error("Recovery fence was not reached");
    }
    bool recovered = false;
    bool reference_replay = false;
    bool event_replay = false;
    try {
      const auto ids = store.file_publications("run-1");
      if (!ids.empty()) {
        const auto result =
            store.recover_file_publication("run-1", ids.front());
        recovered = true;
        reference_replay = result.evidence.reference_replay;
        event_replay = result.evidence.event.batch_replay;
      }
    } catch (const Error&) {
      throw;  // A discovered valid intent must not silently fail recovery.
    }
    report({{"phase", "completed"},
            {"mode", mode},
            {"recovered", recovered},
            {"reference_replay", reference_replay},
            {"event_replay", event_replay},
            {"references", store.references("run-1").size()},
            {"events", store.events("run-1").events.size()},
            {"integrity_ok", store.integrity_check().ok()},
            {"sha256", digest().hex},
            {"bytes", kSize}});
    return 0;
  } else if (mode == "gc") {
    const auto removed = store.collect_garbage(std::chrono::seconds::zero());
    report({{"phase", "completed"},
            {"mode", mode},
            {"gc_removed", removed.removed_files}});
    return 0;
  } else {
    return 2;
  }
  report({{"phase", "completed"},
          {"mode", mode},
          {"references", store.references("run-1").size()},
          {"events", store.events("run-1").events.size()},
          {"integrity_ok", store.integrity_check().ok()}});
  return 0;
}

int guarded(std::string_view mode, const std::filesystem::path& root,
            std::string_view boundary) {
  try {
    active_root = std::filesystem::absolute(root);
    active_boundary = boundary;
    return execute(mode);
  } catch (const std::exception& error) {
    std::cerr << "file-publication: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

namespace jyppx::liuding::experiment::detail {
std::error_code publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  const auto payload = preparing && destination == payload_object();
  if (payload && active_boundary == "before-payload-cas")
    fence(active_boundary);
  const auto error = native_publish_artifact_file(source, destination);
  if (!error && payload && active_boundary == "after-payload-cas")
    fence(active_boundary);
  return error;
}
std::error_code publish_output_file(const std::filesystem::path& staging,
                                    const std::filesystem::path& destination) {
  if (recovering && active_boundary == "before-file-publication")
    fence(active_boundary);
  const auto error = native_publish_output_file(staging, destination);
  if (!error && recovering && active_boundary == "after-file-publication")
    fence(active_boundary);
  return error;
}
}  // namespace jyppx::liuding::experiment::detail

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  if (argc != 3 && argc != 4) return 2;
  return guarded(utf8(std::filesystem::path(argv[1])),
                 std::filesystem::path(argv[2]),
                 argc == 4 ? utf8(std::filesystem::path(argv[3])) : "");
}
#else
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  return guarded(argv[1], std::filesystem::path(argv[2]),
                 argc == 4 ? argv[3] : "");
}
#endif
