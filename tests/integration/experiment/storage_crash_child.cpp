// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// SQLite's official hooks observe actual production operations. This fixture
// never substitutes SQL for the event/artifact/finalize/migration under test.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
namespace ex = jyppx::liuding::experiment;
using Json = nlohmann::json;
constexpr auto kRun = "run.storage-crash";
constexpr auto kMetric = "storage.validation.loss";
constexpr auto kConfirmed = "Previously confirmed immutable checkpoint bytes";
constexpr auto kTarget = "Newly committed artifact at the interrupted boundary";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string utf8(const std::filesystem::path& path) {
  const auto value = path.u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

std::span<const std::byte> bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

std::uint64_t process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(::_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

void report(const Json& value) {
  const auto encoded = value.dump();
  require(encoded.size() <= 256U * 1024U, "storage report exceeded bound");
  std::cout << encoded << '\n' << std::flush;
  require(static_cast<bool>(std::cout), "could not publish storage report");
}

class CleanupMarker final {
 public:
  explicit CleanupMarker(std::filesystem::path path) : path_(std::move(path)) {}
  ~CleanupMarker() {
    std::ofstream marker(path_, std::ios::binary);
    marker << "storage operation stack unwound\n";
  }
  CleanupMarker(const CleanupMarker&) = delete;
  CleanupMarker& operator=(const CleanupMarker&) = delete;

 private:
  std::filesystem::path path_;
};

struct HookState {
  std::filesystem::path database_path;
  std::string operation;
  std::string boundary;
  bool armed = false;
  bool verified_wal_full = false;
  bool wal_hook_installed = false;
  sqlite3* connection = nullptr;
  std::vector<std::string> statements;
};

// One single-threaded test child owns this state. No parent or production
// process registers extensions, changes hooks, or shares a SQLite connection.
HookState* active_hooks = nullptr;

[[noreturn]] void kill_fence(sqlite3* database, std::string_view boundary) {
  const char* filename = sqlite3_db_filename(database, "main");
  require(filename != nullptr, "SQLite hook did not expose its database path");
  require(std::filesystem::equivalent(std::filesystem::u8path(filename),
                                      active_hooks->database_path),
          "SQLite hook observed an unrelated database");
  report({{"schema_version", 1U},
          {"phase", "ready-for-forced-kill"},
          {"process_id", process_id()},
          {"operation", active_hooks->operation},
          {"boundary", boundary},
          {"database_path", filename},
          {"verified_wal_full", active_hooks->verified_wal_full},
          {"sqlite_version", sqlite3_libversion()},
          {"hook",
           boundary == "before" ? "sqlite3_commit_hook" : "sqlite3_wal_hook"},
          {"statements", active_hooks->statements}});
  char input = 0;
  std::cin.get(input);
  // A C callback must not propagate an exception through SQLite. EOF/input is
  // a failed supervisor, not a successful crash; normal exit leaves no success.
  std::cerr << "storage supervisor fence was released without forced kill\n";
  std::exit(91);
}

int before_commit(void*) noexcept {
  if (active_hooks != nullptr && active_hooks->armed &&
      active_hooks->boundary == "before") {
    try {
      kill_fence(active_hooks->connection, "before");
    } catch (...) {
      std::_Exit(92);
    }
  }
  return 0;
}

int after_wal_commit(void*, sqlite3* database, const char* name,
                     int pages) noexcept {
  if (active_hooks != nullptr && active_hooks->armed &&
      active_hooks->boundary == "after") {
    try {
      require(std::string_view(name) == "main" && pages > 0,
              "expected a real main-database WAL commit");
      kill_fence(database, "after");
    } catch (...) {
      std::_Exit(93);
    }
  }
  return SQLITE_OK;
}

int trace_statement(unsigned kind, void*, void* statement, void*) noexcept {
  if (active_hooks != nullptr && kind == SQLITE_TRACE_STMT) {
    try {
      if (!active_hooks->wal_hook_installed) {
        // sqlite3_open installs its default WAL autocheckpoint after automatic
        // extensions run. The first real SQL statement is after open returns
        // and before the constructor's migration transaction can commit.
        sqlite3_wal_hook(active_hooks->connection, &after_wal_commit, nullptr);
        active_hooks->wal_hook_installed = true;
      }
      if (active_hooks->statements.size() >= 128U) return 0;
      const auto* sql = sqlite3_sql(static_cast<sqlite3_stmt*>(statement));
      if (sql != nullptr) {
        const std::string_view text(sql);
        if (text.size() <= 16384U) active_hooks->statements.emplace_back(text);
      }
    } catch (...) {
      std::_Exit(94);
    }
  }
  return 0;
}

int install_hooks(sqlite3* database, char**,
                  const sqlite3_api_routines*) noexcept {
  if (active_hooks == nullptr) return SQLITE_ERROR;
  active_hooks->connection = database;
  sqlite3_commit_hook(database, &before_commit, nullptr);
  return sqlite3_trace_v2(database, SQLITE_TRACE_STMT, &trace_statement,
                          nullptr);
}

class ScopedHooks final {
 public:
  explicit ScopedHooks(HookState& state) {
    require(active_hooks == nullptr, "only one fixture extension is allowed");
    active_hooks = &state;
    const auto result =
        sqlite3_auto_extension(reinterpret_cast<void (*)()>(&install_hooks));
    require(result == SQLITE_OK, "could not register SQLite auto-extension");
  }
  ~ScopedHooks() {
    sqlite3_cancel_auto_extension(reinterpret_cast<void (*)()>(&install_hooks));
    active_hooks = nullptr;
  }
  ScopedHooks(const ScopedHooks&) = delete;
  ScopedHooks& operator=(const ScopedHooks&) = delete;
};

class RawDatabase final {
 public:
  explicit RawDatabase(const std::filesystem::path& path) {
    const auto filename = utf8(path);
    require(sqlite3_open(filename.c_str(), &database_) == SQLITE_OK,
            "could not open fixture SQLite database");
  }
  ~RawDatabase() { sqlite3_close(database_); }
  RawDatabase(const RawDatabase&) = delete;
  RawDatabase& operator=(const RawDatabase&) = delete;
  void execute(const char* sql) {
    require(
        sqlite3_exec(database_, sql, nullptr, nullptr, nullptr) == SQLITE_OK,
        "fixture setup SQL failed");
  }

 private:
  sqlite3* database_ = nullptr;
};

std::string scalar(sqlite3* database, const char* sql) {
  sqlite3_stmt* statement = nullptr;
  require(
      sqlite3_prepare_v2(database, sql, -1, &statement, nullptr) == SQLITE_OK,
      "could not inspect production connection settings");
  std::string result;
  if (sqlite3_step(statement) == SQLITE_ROW) {
    const auto* value = sqlite3_column_text(statement, 0);
    if (value != nullptr) result = reinterpret_cast<const char*>(value);
  }
  sqlite3_finalize(statement);
  return result;
}

ex::RunMetadata metadata() {
  return {.run_id = kRun,
          .project_id = "project.storage-crash",
          .experiment_id = "experiment.storage-crash",
          .display_name = "Storage operation crash fixture",
          .status = ex::RunStatus::running,
          .created_at_unix_ms = 100U,
          .updated_at_unix_ms = 100U,
          .lineage = {.source_digest = "source.original",
                      .build_fingerprint = "build.original",
                      .config_digest = "config.original",
                      .data_manifest_digest = "data.original"},
          .tags = {{"fixture", "real-production-operation"}}};
}

ex::RunArtifactReference artifact(ex::SQLiteRunStore& store, bool confirmed) {
  const std::string_view content = confirmed ? kConfirmed : kTarget;
  return store.put_artifact(
      kRun, confirmed ? "artifact.confirmed" : "artifact.target",
      ex::RunArtifactKind::checkpoint,
      confirmed ? "last.confirmed" : "last.target", bytes(content),
      {.logical_name = confirmed ? "confirmed.bin" : "target.bin",
       .media_type = "application/octet-stream",
       .producer = "storage-crash-fixture",
       .expected_size_bytes = content.size()},
      confirmed ? 110U : 200U,
      jyppx::liuding::ArtifactDigest{.hex = jyppx::liuding::sha256_hex(bytes(content))});
}

jyppx::liuding::TelemetryEnvelope event(std::uint64_t sequence) {
  jyppx::liuding::TelemetryMetricPoint point;
  point.definition = {.metric_id = kMetric,
                      .unit = jyppx::liuding::TelemetryMetricUnit::dimensionless,
                      .aggregation = jyppx::liuding::TelemetryAggregation::mean,
                      .direction = jyppx::liuding::TelemetryMetricDirection::minimize};
  point.step.value = sequence;
  point.observation =
      jyppx::liuding::make_telemetry_metric_value(1.0 / static_cast<double>(sequence));
  point.sample_count = 8U;
  jyppx::liuding::TelemetryEnvelope envelope;
  envelope.identity = {
      .stream_id = "stream.storage-crash",
      .event_id = "event/" + std::to_string(sequence),
      .idempotency_key = "operation/" + std::to_string(sequence),
      .sequence = sequence};
  envelope.time = {.wall_time_unix_ns = 1700000000000000000LL +
                                        static_cast<std::int64_t>(sequence),
                   .monotonic_time_ns = 1000U + sequence,
                   .clock_id = "clock.original"};
  envelope.correlation = {.project_id = "project.storage-crash",
                          .run_id = kRun};
  envelope.attributes = {{"source", "confirmed-original-observation"}};
  envelope.payload = jyppx::liuding::TelemetryMetricBatch{.points = {point}};
  return envelope;
}

Json receipt(const ex::RunEventAppendResult& result) {
  auto receipts = Json::array();
  for (const auto& item : result.receipts)
    receipts.push_back(
        {{"run_sequence", item.run_sequence},
         {"disposition", static_cast<unsigned>(item.disposition)}});
  return {{"committed_through", result.committed_through},
          {"batch_replay", result.batch_replay},
          {"receipts", receipts}};
}

Json inspect(ex::SQLiteRunStore& store) {
  const auto run = store.find(kRun);
  require(run.has_value(), "previously confirmed run disappeared");
  auto references = Json::array();
  for (const auto& item : store.references(kRun))
    references.push_back({{"reference_id", item.reference_id},
                          {"digest", item.artifact.digest.hex},
                          {"size_bytes", item.artifact.size_bytes},
                          {"role", item.role}});
  auto events = Json::array();
  const auto page = store.events(kRun);
  require(!page.next_after, "fixture must fit one bounded page");
  for (const auto& item : page.events)
    events.push_back(
        {{"run_sequence", item.run_sequence},
         {"digest", jyppx::liuding::telemetry_content_digest(item.envelope).hex},
         {"envelope",
          Json::parse(ex::telemetry_projection_json(item.envelope))}});
  const auto integrity = store.integrity_check();
  return {{"status", ex::run_status_name(run->status)},
          {"updated_at_ms", run->updated_at_unix_ms},
          {"references", references},
          {"events", events},
          {"snapshot_through", page.snapshot_through},
          {"integrity_ok", integrity.ok()},
          {"issues", integrity.issues}};
}

void seed(const std::filesystem::path& root, std::string_view operation) {
  require(!std::filesystem::exists(root / "store"),
          "seed requires a fresh store");
  {
    ex::SQLiteRunStore store(root / "store");
    store.start(metadata());
    artifact(store, true);
    if (operation != "migration") {
      const std::vector<jyppx::liuding::TelemetryEnvelope> confirmed{event(1U)};
      store.append(kRun, "batch.confirmed", confirmed);
    }
    const auto state = inspect(store);
    report({{"schema_version", 1U},
            {"phase", "seeded"},
            {"operation", operation},
            {"confirmed_content", kConfirmed},
            {"target_content", kTarget},
            {"store", state}});
  }
  if (operation == "migration") {
    // Construct a v1 INPUT fixture, matching the existing migration tests.
    // The interrupted operation itself is the real production constructor.
    RawDatabase raw(root / "store" / "runs.sqlite3");
    raw.execute(
        "PRAGMA foreign_keys=OFF; BEGIN IMMEDIATE; "
        "DROP TABLE jyppx_event_batch_items; DROP TABLE jyppx_event_batches; "
        "DROP TABLE jyppx_metric_points; DROP TABLE jyppx_metric_definitions; "
        "DROP TABLE jyppx_events; DROP TABLE jyppx_event_streams; "
        "DROP TABLE jyppx_run_event_state; PRAGMA user_version=1; COMMIT;");
  }
}

Json perform(ex::SQLiteRunStore& store, std::string_view operation) {
  if (operation == "event") {
    const std::vector<jyppx::liuding::TelemetryEnvelope> target{event(2U), event(3U)};
    return receipt(store.append(kRun, "batch.target", target));
  }
  if (operation == "artifact") {
    const auto result = artifact(store, false);
    return {{"reference_id", result.reference_id},
            {"digest", result.artifact.digest.hex}};
  }
  if (operation == "finalize") {
    const auto result = store.transition(kRun, ex::RunStatus::completed, 300U);
    return {{"status", ex::run_status_name(result.status)},
            {"updated_at_ms", result.updated_at_unix_ms}};
  }
  require(operation == "migration", "unknown storage operation");
  return {{"schema_version", ex::SQLiteRunStore::schema_version}};
}

int execute(std::string_view mode, const std::filesystem::path& root,
            std::string_view operation, std::string_view boundary) {
  require(mode == "seed" || mode == "crash" || mode == "retry" ||
              mode == "inspect" || mode == "gc",
          "unknown storage child mode");
  require(operation == "event" || operation == "artifact" ||
              operation == "finalize" || operation == "migration",
          "unknown storage operation");
  require(std::filesystem::is_directory(root),
          "fixture root must already exist");
  if (mode == "seed") {
    seed(root, operation);
    return 0;
  }
  const auto database_path = root / "store" / "runs.sqlite3";
  require(std::filesystem::is_regular_file(database_path),
          "seeded database is missing");
  if (mode == "crash") {
    require(boundary == "before" || boundary == "after",
            "unknown crash boundary");
    CleanupMarker cleanup(root / "storage-cleanup.marker");
    HookState state{.database_path = database_path,
                    .operation = std::string(operation),
                    .boundary = std::string(boundary)};
    ScopedHooks hooks(state);
    if (operation == "migration") state.armed = true;
    ex::SQLiteRunStore store(root / "store");
    require(state.connection != nullptr,
            "production connection did not install fixture hooks");
    require(scalar(state.connection, "PRAGMA journal_mode") == "wal" &&
                scalar(state.connection, "PRAGMA synchronous") == "2",
            "real operation requires production WAL and FULL synchronous");
    state.verified_wal_full = true;
    state.statements.clear();
    state.armed = true;
    perform(store, operation);
    throw std::runtime_error("real operation escaped its required SQLite hook");
  }
  ex::SQLiteRunStore store(root / "store");
  Json result{{"schema_version", 1U},
              {"phase", "completed"},
              {"operation", operation},
              {"mode", mode}};
  if (mode == "retry") result["receipt"] = perform(store, operation);
  if (mode == "gc") {
    const auto gc = store.collect_garbage(std::chrono::seconds::zero());
    result["gc"] = {{"removed_files", gc.removed_files},
                    {"removed_bytes", gc.removed_bytes},
                    {"retained_files", gc.retained_files}};
  }
  result["store"] = inspect(store);
  report(result);
  return 0;
}

int guarded_execute(std::string_view mode, const std::filesystem::path& root,
                    std::string_view operation, std::string_view boundary) {
  try {
    return execute(mode, std::filesystem::absolute(root), operation, boundary);
  } catch (const std::exception& error) {
    std::cerr << "storage-crash-child: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  if (argc != 4 && argc != 5) return 2;
  return guarded_execute(utf8(std::filesystem::path(argv[1])),
                         std::filesystem::path(argv[2]),
                         utf8(std::filesystem::path(argv[3])),
                         argc == 5 ? utf8(std::filesystem::path(argv[4])) : "");
}
#else
int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) return 2;
  return guarded_execute(argv[1], std::filesystem::path(argv[2]), argv[3],
                         argc == 5 ? argv[4] : "");
}
#endif
