// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Only the static test executable supplies the private forwarding symbol and
// official SQLite hook. Publication, recovery, locking and commits stay native.
#include "jyppx/liuding/experiment/detail/artifact_publish.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
namespace ex = jyppx::liuding::experiment;
using namespace jyppx::liuding;
using Json = nlohmann::json;
constexpr std::size_t kSize = 2U * 1024U * 1024U + 257U;
std::filesystem::path active_root;
std::string active_boundary;
bool publication_armed = false;
bool recovery_armed = false;
unsigned native_calls = 0U;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string utf8(const std::filesystem::path& path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::string content() {
  std::string value(kSize, '\0');
  for (std::size_t i = 0U; i < value.size(); ++i)
    value[i] = static_cast<char>((i * 37U + 11U) & 255U);
  return value;
}

ArtifactDigest digest() {
  const auto value = content();
  return {.hex =
              sha256_hex(std::as_bytes(std::span(value.data(), value.size())))};
}

std::filesystem::path object_path(const std::filesystem::path& root) {
  const auto hash = digest().hex;
  return root / "artifacts" / "objects" / hash.substr(0U, 2U) / hash.substr(2U);
}

void report(Json value) {
  value["schema_version"] = 1U;
  const auto text = value.dump();
  require(text.size() <= 65536U, "report exceeded its bound");
  std::cout << text << '\n' << std::flush;
  require(static_cast<bool>(std::cout), "report could not be flushed");
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
          {"native_calls", native_calls},
          {"cas_exists", std::filesystem::exists(object_path(active_root))}});
  char value = 0;
  std::cin.get(value);
  // Never throw across the official SQLite C callback. EOF is a failed test.
  std::_Exit(91);
}

int before_commit(void*) noexcept {
  if (recovery_armed) {
    try {
      fence("recovery-before-commit");
    } catch (...) {
      std::_Exit(92);
    }
  }
  return 0;
}

int install_hook(sqlite3* connection, char**, const sqlite3_api_routines*) {
  sqlite3_commit_hook(connection, &before_commit, nullptr);
  return SQLITE_OK;
}

ex::PublishedFileEvidenceRequest request(const std::filesystem::path& root) {
  ex::PublishedFileEvidenceRequest value;
  value.run_id = "run-1";
  value.reference_id = "published-1";
  value.path = root / "output.bin";
  value.descriptor = {.logical_name = "output.bin",
                      .media_type = "application/octet-stream",
                      .producer = "published-cas-recovery-test",
                      .expected_size_bytes = kSize};
  value.attached_at_unix_ms = 101U;
  value.expected_digest = digest();
  value.event_batch_key = "output-event-1";
  value.event.identity = {.stream_id = "output-1",
                          .event_id = "published-1",
                          .idempotency_key = "published-1",
                          .sequence = 1U};
  value.event.correlation.project_id = "project-1";
  value.event.correlation.run_id = "run-1";
  value.event.payload = TelemetryArtifactReference{.role = "output.file"};
  return value;
}

class CleanupMarker final {
 public:
  ~CleanupMarker() {
    std::ofstream file(active_root / "cleanup.marker", std::ios::binary);
    file << "operation unwound\n";
  }
};

int execute(std::string_view mode, const std::filesystem::path& root,
            std::string_view boundary) {
  active_root = root;
  require(std::filesystem::is_directory(root), "fixture root must exist");
  require(sqlite3_auto_extension(reinterpret_cast<void (*)()>(&install_hook)) ==
              SQLITE_OK,
          "official SQLite hook registration failed");
  ex::SQLiteRunStoreOptions options;
  options.busy_timeout = std::chrono::milliseconds(30);
  if (mode == "gc-busy") {
    try {
      ex::SQLiteRunStore store(root, options);
      (void)store.collect_garbage(std::chrono::seconds::zero());
    } catch (const NotReady&) {
      report(
          {{"phase", "completed"}, {"mode", mode}, {"writer_blocked", true}});
      return 0;
    }
    throw std::runtime_error("coordinated GC bypassed the writer lock");
  }
  ex::SQLiteRunStore store(root, options);
  auto value = request(root);
  Json result{{"phase", "completed"}, {"mode", mode}, {"root", utf8(root)}};
  if (mode == "seed") {
    store.start({.run_id = "run-1",
                 .project_id = "project-1",
                 .experiment_id = "experiment-1",
                 .display_name = "published CAS recovery",
                 .status = ex::RunStatus::running,
                 .created_at_unix_ms = 100U,
                 .updated_at_unix_ms = 100U});
    const auto bytes = content();
    std::ofstream file(*value.path, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    file.close();
    require(static_cast<bool>(file), "source fixture could not be written");
  } else if (mode == "crash") {
    require(boundary == "before-cas" || boundary == "after-cas",
            "unknown publication boundary");
    CleanupMarker cleanup;
    active_boundary = boundary;
    publication_armed = true;
    store.record_published_file_evidence_atomic(value);
    throw std::runtime_error("publication escaped the required fence");
  } else if (mode == "recover" || mode == "recover-fence") {
    value.path.reset();
    require(!std::filesystem::exists(root / "output.bin"),
            "recovery fixture must not have its original source");
    if (mode == "recover-fence") {
      CleanupMarker cleanup;
      recovery_armed = true;
      store.recover_published_file_evidence_atomic(value);
      throw std::runtime_error("recovery escaped the required commit fence");
    }
    try {
      const auto restored = store.recover_published_file_evidence_atomic(value);
      result["recovered"] = true;
      result["reference_replay"] = restored.reference_replay;
      result["event_replay"] = restored.event.batch_replay;
      require(restored.reference.artifact.digest == value.expected_digest &&
                  restored.reference.artifact.size_bytes == kSize &&
                  restored.event.receipts.size() == 1U &&
                  restored.event.receipts.front().run_sequence == 1U,
              "recovery changed the original evidence identity");
    } catch (const IoError&) {
      require(!std::filesystem::exists(object_path(root)),
              "unexpected IO failure with a published CAS object");
      result["recovered"] = false;
    }
  } else if (mode == "gc") {
    result["gc_removed"] =
        store.collect_garbage(std::chrono::seconds::zero()).removed_files;
  } else {
    require(mode == "inspect", "unknown child mode");
  }
  result["references"] = store.references("run-1").size();
  result["events"] = store.events("run-1").events.size();
  result["integrity_ok"] = store.integrity_check().ok();
  result["cas_exists"] = std::filesystem::exists(object_path(root));
  result["sha256"] = digest().hex;
  result["bytes"] = kSize;
  report(result);
  return 0;
}

int guarded(std::string_view mode, const std::filesystem::path& root,
            std::string_view boundary) {
  try {
    return execute(mode, std::filesystem::absolute(root), boundary);
  } catch (const std::exception& error) {
    std::cerr << "published-cas-recovery: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

namespace jyppx::liuding::experiment::detail {
std::error_code publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  if (!publication_armed)
    return native_publish_artifact_file(source, destination);
  require(destination == object_path(active_root),
          "unexpected CAS destination");
  if (active_boundary == "before-cas") fence("before-cas");
  const auto error = native_publish_artifact_file(source, destination);
  if (error) return error;
  ++native_calls;
  fence("after-cas");
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
