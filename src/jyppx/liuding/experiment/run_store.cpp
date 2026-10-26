// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/experiment/run_store.hpp"

#include "jyppx/liuding/foundation/error.hpp"
#include "jyppx/liuding/foundation/sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <sqlite3.h>
#include <stop_token>
#include <system_error>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace jyppx::liuding::experiment {
namespace {

constexpr int kApplicationId = 0x4a595250;
constexpr std::size_t kMaxIntegrityIssues = 64U;

[[noreturn]] void invalid(std::string operation, std::string expected,
                          std::string actual) {
  throw InvalidArgument(
      "Invalid durable run operation",
      ErrorDetails{
          .component = "experiment.run_store",
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = std::move(actual),
          .hint = "Use a valid immutable identity and lifecycle request."});
}

[[noreturn]] void exhausted(std::string operation, std::string expected,
                            std::string actual) {
  throw ResourceExhausted(
      "Run storage bound exceeded",
      ErrorDetails{
          .component = "experiment.run_store",
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = std::move(actual),
          .hint = "Reduce the batch or raise the explicit storage bound."});
}

[[noreturn]] void incompatible(std::string actual) {
  throw CompatibilityError(
      "Unsupported local run database schema",
      ErrorDetails{
          .component = "experiment.run_store",
          .operation = "open.schema",
          .expected = "JYPPX run schema version 1/2 or a new empty database",
          .actual = std::move(actual),
          .hint =
              "Use a compatible framework or an explicit database migration."});
}

[[noreturn]] void corrupt(std::string operation, std::string actual) {
  throw SerializationError(
      "Invalid persisted run data",
      ErrorDetails{
          .component = "experiment.run_store",
          .operation = std::move(operation),
          .expected = "valid bounded metadata using the supported schema",
          .actual = std::move(actual),
          .hint =
              "Restore a verified backup and preserve the damaged database."});
}

class OperationControl final {
 public:
  OperationControl(sqlite3* database, std::chrono::milliseconds timeout,
                   std::stop_token stop, TelemetryDeadline deadline)
      : database_(database),
        timeout_(timeout),
        stop_(stop),
        deadline_(deadline) {
    check();
    sqlite3_progress_handler(database_, 1000, &progress, this);
    sqlite3_busy_handler(database_, &busy, this);
  }
  ~OperationControl() {
    suspend();
    sqlite3_busy_timeout(database_, static_cast<int>(timeout_.count()));
  }
  OperationControl(const OperationControl&) = delete;
  OperationControl& operator=(const OperationControl&) = delete;
  void check() const { check_telemetry_operation(stop_, deadline_); }
  void suspend() const noexcept {
    sqlite3_progress_handler(database_, 0, nullptr, nullptr);
    sqlite3_busy_handler(database_, nullptr, nullptr);
  }

 private:
  bool interrupted() const noexcept {
    return stop_.stop_requested() ||
           (deadline_ && std::chrono::steady_clock::now() >= *deadline_);
  }
  static int progress(void* data) noexcept {
    return static_cast<OperationControl*>(data)->interrupted() ? 1 : 0;
  }
  static int busy(void* data, int attempts) noexcept {
    auto& self = *static_cast<OperationControl*>(data);
    const auto now = std::chrono::steady_clock::now();
    if (attempts == 0) self.busy_started_ = now;
    if (self.interrupted() || now - self.busy_started_ >= self.timeout_)
      return 0;
    auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
        self.timeout_ - (now - self.busy_started_));
    if (self.deadline_) {
      remaining = std::min(
          remaining,
          std::chrono::ceil<std::chrono::milliseconds>(*self.deadline_ - now));
    }
    if (remaining <= std::chrono::milliseconds::zero()) return 0;
    sqlite3_sleep(static_cast<int>(
        std::min<std::chrono::milliseconds::rep>(remaining.count(), 5)));
    return self.interrupted() ? 0 : 1;
  }
  sqlite3* database_;
  std::chrono::milliseconds timeout_;
  std::stop_token stop_;
  TelemetryDeadline deadline_;
  std::chrono::steady_clock::time_point busy_started_{};
};

[[noreturn]] void sqlite_failure(sqlite3* database, int result,
                                 std::string operation,
                                 const OperationControl* control = nullptr) {
  if (control != nullptr) control->check();
  const auto extended =
      database == nullptr ? result : sqlite3_extended_errcode(database);
  const auto primary = result & 0xff;
  ErrorDetails details{.component = "experiment.run_store",
                       .operation = std::move(operation),
                       .expected = "successful SQLite transaction",
                       .actual = std::string(sqlite3_errstr(result)) +
                                 " (SQLite " + std::to_string(extended) + ")",
                       .hint =
                           "Check storage capacity, permissions, database "
                           "integrity, and writer contention."};
  switch (primary) {
    case SQLITE_BUSY:
    case SQLITE_LOCKED:
      throw NotReady("Run database is busy", std::move(details));
    case SQLITE_NOMEM:
    case SQLITE_FULL:
    case SQLITE_TOOBIG:
      throw ResourceExhausted("SQLite storage resource exhausted",
                              std::move(details));
    case SQLITE_CORRUPT:
    case SQLITE_NOTADB:
    case SQLITE_FORMAT:
    case SQLITE_SCHEMA:
      throw SerializationError("SQLite database is invalid",
                               std::move(details));
    case SQLITE_IOERR:
    case SQLITE_CANTOPEN:
    case SQLITE_READONLY:
    case SQLITE_PERM:
      throw IoError("SQLite filesystem operation failed", std::move(details));
    case SQLITE_CONSTRAINT:
    case SQLITE_MISMATCH:
    case SQLITE_RANGE:
      throw InvalidArgument("SQLite rejected a run metadata write",
                            std::move(details));
    default:
      throw BackendError("SQLite operation failed", std::move(details));
  }
}

void execute(sqlite3* database, const char* sql, std::string_view operation,
             const OperationControl* control = nullptr) {
  if (control != nullptr) control->check();
  const int result = sqlite3_exec(database, sql, nullptr, nullptr, nullptr);
  if (result != SQLITE_OK)
    sqlite_failure(database, result, std::string(operation), control);
}

class Statement final {
 public:
  Statement(sqlite3* database, const char* sql, std::string operation,
            const OperationControl* control = nullptr)
      : database_(database),
        operation_(std::move(operation)),
        control_(control) {
    if (control_ != nullptr) control_->check();
    const int result =
        sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr);
    if (result != SQLITE_OK) {
      sqlite3_finalize(statement_);
      statement_ = nullptr;
      sqlite_failure(database_, result, operation_, control_);
    }
  }

  ~Statement() { sqlite3_finalize(statement_); }
  Statement(const Statement&) = delete;
  Statement& operator=(const Statement&) = delete;

  void bind(int index, std::string_view value) {
    if (value.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      exhausted(operation_, "representable SQLite string size",
                "oversized text");
    }
    const int result =
        sqlite3_bind_text(statement_, index, value.empty() ? "" : value.data(),
                          static_cast<int>(value.size()), SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
      sqlite_failure(database_, result, operation_, control_);
  }

  void bind(int index, const std::string& value) {
    bind(index, std::string_view(value));
  }

  void bind(int index, std::uint64_t value) {
    if (value >
        static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max())) {
      exhausted(operation_, "nonnegative signed 64-bit integer",
                "integer overflow");
    }
    const int result = sqlite3_bind_int64(statement_, index,
                                          static_cast<sqlite3_int64>(value));
    if (result != SQLITE_OK)
      sqlite_failure(database_, result, operation_, control_);
  }

  void bind(int index, const std::optional<std::string>& value) {
    if (value) {
      bind(index, std::string_view(*value));
      return;
    }
    const int result = sqlite3_bind_null(statement_, index);
    if (result != SQLITE_OK)
      sqlite_failure(database_, result, operation_, control_);
  }

  bool row() {
    if (control_ != nullptr) control_->check();
    const int result = sqlite3_step(statement_);
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    sqlite_failure(database_, result, operation_, control_);
  }

  void done() {
    if (row())
      corrupt(operation_, "unexpected result rows for a metadata write");
  }

  std::uint64_t integer(int column) const {
    if (sqlite3_column_type(statement_, column) != SQLITE_INTEGER ||
        sqlite3_column_int64(statement_, column) < 0) {
      corrupt(operation_, "expected nonnegative integer column");
    }
    return static_cast<std::uint64_t>(sqlite3_column_int64(statement_, column));
  }

  std::string text(int column, std::size_t max_bytes) const {
    if (sqlite3_column_type(statement_, column) != SQLITE_TEXT) {
      corrupt(operation_, "expected text column");
    }
    const auto count = sqlite3_column_bytes(statement_, column);
    if (count < 0 || static_cast<std::size_t>(count) > max_bytes) {
      exhausted(operation_, "bounded persisted text", "oversized text column");
    }
    const auto* value = sqlite3_column_text(statement_, column);
    if (value == nullptr && count != 0) {
      sqlite_failure(database_, SQLITE_NOMEM, operation_, control_);
    }
    return count == 0 ? std::string{}
                      : std::string(reinterpret_cast<const char*>(value),
                                    static_cast<std::size_t>(count));
  }

  std::optional<std::string> optional_text(int column,
                                           std::size_t max_bytes) const {
    if (sqlite3_column_type(statement_, column) == SQLITE_NULL)
      return std::nullopt;
    return text(column, max_bytes);
  }

  void bind_blob(int index, std::span<const std::byte> bytes) {
    const int result = sqlite3_bind_blob64(
        statement_, index, bytes.data(),
        static_cast<sqlite3_uint64>(bytes.size()), SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
      sqlite_failure(database_, result, operation_, control_);
  }

  std::vector<std::byte> blob(int column, std::uint64_t maximum) const {
    if (sqlite3_column_type(statement_, column) != SQLITE_BLOB)
      corrupt(operation_, "expected canonical blob column");
    const auto count = sqlite3_column_bytes(statement_, column);
    if (count < 0 || static_cast<std::uint64_t>(count) > maximum)
      exhausted(operation_, "bounded persisted blob", "oversized blob");
    const auto* data =
        static_cast<const std::byte*>(sqlite3_column_blob(statement_, column));
    if (count == 0) return {};
    if (data == nullptr)
      sqlite_failure(database_, SQLITE_NOMEM, operation_, control_);
    return {data, data + count};
  }

  void reset() {
    const auto result = sqlite3_reset(statement_);
    if (result != SQLITE_OK)
      sqlite_failure(database_, result, operation_, control_);
    const auto cleared = sqlite3_clear_bindings(statement_);
    if (cleared != SQLITE_OK)
      sqlite_failure(database_, cleared, operation_, control_);
  }

 private:
  sqlite3* database_;
  std::string operation_;
  const OperationControl* control_;
  sqlite3_stmt* statement_ = nullptr;
};

class Transaction final {
 public:
  explicit Transaction(sqlite3* database, bool write = true,
                       const OperationControl* control = nullptr)
      : database_(database), control_(control) {
    execute(database_, write ? "BEGIN IMMEDIATE" : "BEGIN", "transaction.begin",
            control_);
  }

  ~Transaction() {
    if (!committed_) {
      if (control_ != nullptr) control_->suspend();
      (void)sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
    }
  }

  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;

  void commit() {
    execute(database_, "COMMIT", "transaction.commit", control_);
    committed_ = true;
  }

 private:
  sqlite3* database_;
  const OperationControl* control_;
  bool committed_ = false;
};

std::uint64_t scalar_integer(sqlite3* database, const char* sql,
                             std::string operation,
                             const OperationControl* control = nullptr) {
  Statement statement(database, sql, std::move(operation), control);
  if (!statement.row()) corrupt("query", "missing scalar result");
  return statement.integer(0);
}

bool valid_utf8(std::string_view value) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index++]);
    if (first < 0x20U) return false;
    if (first < 0x80U) continue;
    unsigned continuation = 0U;
    std::uint32_t codepoint = 0U;
    std::uint32_t minimum = 0U;
    if ((first & 0xe0U) == 0xc0U) {
      continuation = 1U;
      codepoint = first & 0x1fU;
      minimum = 0x80U;
    } else if ((first & 0xf0U) == 0xe0U) {
      continuation = 2U;
      codepoint = first & 0x0fU;
      minimum = 0x800U;
    } else if ((first & 0xf8U) == 0xf0U) {
      continuation = 3U;
      codepoint = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (continuation > value.size() - index) return false;
    for (unsigned remaining = continuation; remaining != 0U; --remaining) {
      const auto byte = static_cast<unsigned char>(value[index++]);
      if ((byte & 0xc0U) != 0x80U) return false;
      codepoint = (codepoint << 6U) | (byte & 0x3fU);
    }
    if (codepoint < minimum || codepoint > 0x10ffffU ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU))
      return false;
  }
  return true;
}

void validate_text(std::string_view value, std::size_t limit,
                   std::string_view operation, bool required = true) {
  if (required && value.empty())
    invalid(std::string(operation), "nonempty text", "empty");
  if (value.size() > limit) {
    exhausted(std::string(operation),
              "at most " + std::to_string(limit) + " bytes",
              std::to_string(value.size()) + " bytes");
  }
  if (!valid_utf8(value)) {
    invalid(std::string(operation), "valid UTF-8 without control characters",
            "invalid text");
  }
}

void validate_integer(std::uint64_t value, std::string_view operation) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<sqlite3_int64>::max())) {
    exhausted(std::string(operation), "nonnegative signed 64-bit integer",
              "integer overflow");
  }
}

bool same_lineage(const RunLineage& left, const RunLineage& right) {
  return left.parent_run_id == right.parent_run_id &&
         left.source_digest == right.source_digest &&
         left.build_fingerprint == right.build_fingerprint &&
         left.config_digest == right.config_digest &&
         left.data_manifest_digest == right.data_manifest_digest;
}

bool same_metadata(const RunMetadata& left, const RunMetadata& right) {
  return left.run_id == right.run_id && left.project_id == right.project_id &&
         left.experiment_id == right.experiment_id &&
         left.display_name == right.display_name &&
         left.status == right.status &&
         left.created_at_unix_ms == right.created_at_unix_ms &&
         left.updated_at_unix_ms == right.updated_at_unix_ms &&
         same_lineage(left.lineage, right.lineage) && left.tags == right.tags;
}

bool same_reference(const RunArtifactReference& left,
                    const RunArtifactReference& right) {
  return left.reference_id == right.reference_id && left.kind == right.kind &&
         left.role == right.role &&
         left.artifact.digest == right.artifact.digest &&
         left.artifact.size_bytes == right.artifact.size_bytes &&
         left.artifact.logical_name == right.artifact.logical_name &&
         left.artifact.media_type == right.artifact.media_type &&
         left.artifact.producer == right.artifact.producer &&
         left.attached_at_unix_ms == right.attached_at_unix_ms;
}

std::string utf8_path(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

void reject_link(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory) return;
  if (error || std::filesystem::is_symlink(status) ||
      (std::filesystem::exists(status) &&
       !std::filesystem::is_regular_file(status))) {
    throw IoError("Run database path must be an ordinary file",
                  ErrorDetails{.component = "experiment.run_store",
                               .operation = "open.path",
                               .expected = "an unlinked regular database file",
                               .actual = utf8_path(path),
                               .hint = "Use a dedicated local directory "
                                       "without redirected database files."});
  }
}

std::filesystem::path canonical_root(std::filesystem::path root,
                                     const SQLiteRunStoreOptions& options,
                                     std::stop_token stop,
                                     TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  if (root.empty())
    invalid("root", "a dedicated local directory", "empty path");
  if (options.busy_timeout.count() < 0 ||
      options.busy_timeout.count() > std::numeric_limits<int>::max()) {
    invalid("options.busy_timeout", "0 to INT_MAX milliseconds",
            "out of range");
  }
  if (options.max_runs == 0U || options.max_references_per_run == 0U ||
      options.max_run_artifact_bytes == 0U ||
      options.max_reference_id_bytes == 0U || options.max_role_bytes == 0U) {
    invalid("options", "positive metadata and run quota bounds", "zero bound");
  }
  validate_integer(options.max_runs, "options.max_runs");
  if (options.max_runs >
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      options.max_references_per_run >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    exhausted("options", "run and reference counts at most INT_MAX",
              "count overflow");
  }
  validate_integer(options.max_run_artifact_bytes,
                   "options.max_run_artifact_bytes");
  validate_integer(options.artifact_limits.max_artifact_bytes,
                   "options.artifact_limits.max_artifact_bytes");
  if (options.max_backup_database_bytes == 0U ||
      options.max_backup_manifest_bytes == 0U)
    invalid("options.backup", "positive database/manifest bounds", "zero");
  validate_integer(options.max_backup_database_bytes,
                   "options.backup.database_bytes");
  validate_integer(options.max_backup_manifest_bytes,
                   "options.backup.manifest_bytes");
  const auto& events = options.event_limits;
  validate_telemetry_limits(events.telemetry);
  const std::array<std::uint64_t, 8U> positive_bounds{
      events.max_events_per_run,          events.max_metric_points_per_run,
      events.max_event_bytes_per_run,     events.max_batches_per_run,
      events.max_streams_per_run,         events.max_metric_definitions,
      events.max_metric_points_per_batch, events.max_page_bytes};
  for (const auto bound : positive_bounds) {
    if (bound == 0U) invalid("options.event_limits", "positive bounds", "zero");
    validate_integer(bound, "options.event_limits");
  }
  if (events.max_page_events == 0U || events.max_page_metric_points == 0U ||
      events.max_page_events >=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      events.max_page_metric_points >=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      events.max_page_bytes < events.telemetry.max_envelope_bytes ||
      events.max_event_bytes_per_run < events.telemetry.max_envelope_bytes) {
    invalid("options.event_limits",
            "representable page bounds containing one envelope",
            "inconsistent bounds");
  }
  std::error_code error;
  root = std::filesystem::absolute(root, error);
  if (!error) root = std::filesystem::weakly_canonical(root, error);
  if (error) {
    throw IoError(
        "Could not resolve the run storage root",
        ErrorDetails{
            .component = "experiment.run_store",
            .operation = "open.root",
            .expected = "accessible local directory",
            .actual = error.message(),
            .hint = "Check the root path and local filesystem permissions."});
  }
  check_telemetry_operation(stop, deadline);
  return root;
}

std::filesystem::path checked_artifact_root(const std::filesystem::path& root,
                                            std::stop_token stop,
                                            TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  return root / "artifacts";
}

void verify_event_integrity(sqlite3* database,
                            const SQLiteRunStoreOptions& options,
                            RunStoreIntegrityReport& report,
                            const OperationControl* control = nullptr);

std::unique_lock<std::timed_mutex> event_lock(std::timed_mutex& mutex,
                                              std::stop_token stop,
                                              TelemetryDeadline deadline);

constexpr const char* kCreateSchema = R"sql(
CREATE TABLE jyppx_runs (
 run_id TEXT PRIMARY KEY NOT NULL,
 project_id TEXT NOT NULL,
 experiment_id TEXT NOT NULL,
 display_name TEXT NOT NULL,
 status INTEGER NOT NULL CHECK(status BETWEEN 1 AND 6),
 created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
 updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= created_at_ms),
 start_status INTEGER NOT NULL CHECK(start_status IN (1, 2)),
 start_updated_at_ms INTEGER NOT NULL CHECK(start_updated_at_ms >= created_at_ms),
 parent_run_id TEXT,
 source_digest TEXT,
 build_fingerprint TEXT,
 config_digest TEXT,
 data_manifest_digest TEXT
) WITHOUT ROWID;
CREATE TABLE jyppx_run_tags (
 run_id TEXT NOT NULL REFERENCES jyppx_runs(run_id) ON DELETE CASCADE,
 name TEXT NOT NULL,
 value TEXT NOT NULL,
 PRIMARY KEY(run_id, name)
) WITHOUT ROWID;
CREATE TABLE jyppx_blobs (
 digest TEXT PRIMARY KEY NOT NULL
   CHECK(length(digest) = 64 AND digest NOT GLOB '*[^0-9a-f]*'),
 algorithm INTEGER NOT NULL CHECK(algorithm = 1),
 size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0)
) WITHOUT ROWID;
CREATE TABLE jyppx_run_artifacts (
 run_id TEXT NOT NULL REFERENCES jyppx_runs(run_id) ON DELETE CASCADE,
 reference_id TEXT NOT NULL,
 kind INTEGER NOT NULL CHECK(kind IN (1, 2)),
 role TEXT NOT NULL,
 digest TEXT NOT NULL REFERENCES jyppx_blobs(digest),
 logical_name TEXT NOT NULL,
 media_type TEXT NOT NULL,
 producer TEXT NOT NULL,
 attached_at_ms INTEGER NOT NULL CHECK(attached_at_ms >= 0),
 PRIMARY KEY(run_id, reference_id)
) WITHOUT ROWID;
CREATE INDEX jyppx_run_artifact_digest ON jyppx_run_artifacts(digest);
PRAGMA application_id = 1247367760;
PRAGMA user_version = 1;
)sql";

constexpr const char* kSelectRun =
    "SELECT run_id, project_id, experiment_id, display_name, status, "
    "created_at_ms, updated_at_ms, parent_run_id, source_digest, "
    "build_fingerprint, config_digest, data_manifest_digest "
    "FROM jyppx_runs WHERE run_id = ?1";

constexpr const char* kCreateEventSchema = R"sql(
CREATE TABLE jyppx_run_event_state (
 run_id TEXT PRIMARY KEY NOT NULL REFERENCES jyppx_runs(run_id),
 last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0),
 metric_count INTEGER NOT NULL CHECK(metric_count >= 0),
 content_bytes INTEGER NOT NULL CHECK(content_bytes >= 0),
 batch_count INTEGER NOT NULL CHECK(batch_count >= 0),
 stream_count INTEGER NOT NULL CHECK(stream_count >= 0)
) WITHOUT ROWID;
CREATE TABLE jyppx_event_streams (
 stream_id TEXT PRIMARY KEY NOT NULL,
 run_id TEXT NOT NULL REFERENCES jyppx_runs(run_id),
 last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0),
 terminal INTEGER NOT NULL CHECK(terminal IN (0, 1)),
 UNIQUE(stream_id, run_id)
) WITHOUT ROWID;
CREATE TABLE jyppx_events (
 run_id TEXT NOT NULL REFERENCES jyppx_runs(run_id),
 run_sequence INTEGER NOT NULL CHECK(run_sequence > 0),
 stream_id TEXT NOT NULL,
 producer_sequence INTEGER NOT NULL CHECK(producer_sequence > 0),
 event_id TEXT NOT NULL,
 idempotency_key TEXT NOT NULL,
 payload_kind INTEGER NOT NULL CHECK(payload_kind BETWEEN 1 AND 6),
 content_digest TEXT NOT NULL CHECK(length(content_digest) = 64
   AND content_digest NOT GLOB '*[^0-9a-f]*'),
 canonical BLOB NOT NULL CHECK(length(canonical) > 0),
 PRIMARY KEY(run_id, run_sequence),
 FOREIGN KEY(stream_id, run_id) REFERENCES jyppx_event_streams(stream_id, run_id),
 UNIQUE(stream_id, producer_sequence),
 UNIQUE(stream_id, event_id),
 UNIQUE(stream_id, idempotency_key)
) WITHOUT ROWID;
CREATE INDEX jyppx_events_kind ON jyppx_events(run_id, payload_kind, run_sequence);
CREATE TABLE jyppx_event_batches (
 run_id TEXT NOT NULL REFERENCES jyppx_runs(run_id),
 batch_key TEXT NOT NULL,
 request_digest TEXT NOT NULL CHECK(length(request_digest) = 64),
 input_count INTEGER NOT NULL CHECK(input_count > 0),
 committed_through INTEGER NOT NULL CHECK(committed_through > 0),
 PRIMARY KEY(run_id, batch_key)
) WITHOUT ROWID;
CREATE TABLE jyppx_event_batch_items (
 run_id TEXT NOT NULL,
 batch_key TEXT NOT NULL,
 ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
 run_sequence INTEGER NOT NULL CHECK(run_sequence > 0),
 PRIMARY KEY(run_id, batch_key, ordinal),
 FOREIGN KEY(run_id, batch_key) REFERENCES jyppx_event_batches(run_id, batch_key),
 FOREIGN KEY(run_id, run_sequence) REFERENCES jyppx_events(run_id, run_sequence)
) WITHOUT ROWID;
CREATE TABLE jyppx_metric_definitions (
 metric_id TEXT NOT NULL,
 definition_version INTEGER NOT NULL CHECK(definition_version > 0),
 definition_digest TEXT NOT NULL CHECK(length(definition_digest) = 64),
 canonical BLOB NOT NULL CHECK(length(canonical) > 0),
 PRIMARY KEY(metric_id, definition_version)
) WITHOUT ROWID;
CREATE TABLE jyppx_metric_points (
 run_id TEXT NOT NULL,
 run_sequence INTEGER NOT NULL,
 point_index INTEGER NOT NULL CHECK(point_index >= 0),
 metric_id TEXT NOT NULL,
 definition_version INTEGER NOT NULL,
 step_kind INTEGER NOT NULL CHECK(step_kind BETWEEN 1 AND 5),
 step_key BLOB NOT NULL CHECK(length(step_key) = 8),
 value_status INTEGER NOT NULL CHECK(value_status BETWEEN 1 AND 5),
 PRIMARY KEY(run_id, run_sequence, point_index),
 FOREIGN KEY(run_id, run_sequence) REFERENCES jyppx_events(run_id, run_sequence),
 FOREIGN KEY(metric_id, definition_version)
   REFERENCES jyppx_metric_definitions(metric_id, definition_version)
) WITHOUT ROWID;
CREATE INDEX jyppx_metric_scan ON
 jyppx_metric_points(run_id, metric_id, run_sequence, point_index);
CREATE INDEX jyppx_metric_version_scan ON
 jyppx_metric_points(run_id, metric_id, definition_version, run_sequence, point_index);
PRAGMA user_version = 2;
)sql";

constexpr const char* kSelectReferences =
    "SELECT r.reference_id, r.kind, r.role, b.algorithm, b.digest, "
    "b.size_bytes, "
    "r.logical_name, r.media_type, r.producer, r.attached_at_ms "
    "FROM jyppx_run_artifacts r JOIN jyppx_blobs b ON b.digest = r.digest "
    "WHERE r.run_id = ?1 ORDER BY r.reference_id";

}  // namespace

class SQLiteRunStore::Impl final {
 public:
  Impl(std::filesystem::path root, SQLiteRunStoreOptions options,
       std::stop_token stop, TelemetryDeadline deadline)
      : options_(options),
        root_(canonical_root(std::move(root), options_, stop, deadline)),
        database_path_(root_ / "runs.sqlite3"),
        artifacts_(checked_artifact_root(root_, stop, deadline),
                   options_.artifact_limits) {
    check_telemetry_operation(stop, deadline);
    reject_link(database_path_);
    reject_link(root_ / "runs.sqlite3-wal");
    reject_link(root_ / "runs.sqlite3-shm");
    const auto path = utf8_path(database_path_);
    const auto result = sqlite3_open_v2(
        path.c_str(), &database_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
        nullptr);
    if (result != SQLITE_OK) {
      try {
        sqlite_failure(database_, result, "open");
      } catch (...) {
        sqlite3_close(database_);
        database_ = nullptr;
        throw;
      }
    }
    try {
      sqlite3_extended_result_codes(database_, 1);
      const auto timeout_result = sqlite3_busy_timeout(
          database_, static_cast<int>(options_.busy_timeout.count()));
      if (timeout_result != SQLITE_OK)
        sqlite_failure(database_, timeout_result, "busy_timeout");
      OperationControl control(database_, options_.busy_timeout, stop,
                               deadline);
      initialize(&control);
    } catch (...) {
      sqlite3_close(database_);
      database_ = nullptr;
      check_telemetry_operation(stop, deadline);
      throw;
    }
  }

  ~Impl() { sqlite3_close(database_); }

  void initialize(const OperationControl* control) {
    const auto version = scalar_integer(database_, "PRAGMA user_version",
                                        "open.version", control);
    const auto application = scalar_integer(database_, "PRAGMA application_id",
                                            "open.application", control);
    if (version != 0U && ((version != 1U && version != schema_version) ||
                          application != kApplicationId)) {
      incompatible("schema " + std::to_string(version) + ", application " +
                   std::to_string(application));
    }
    if (version == 0U && application != 0U)
      incompatible("foreign application identifier");
    if (version == 1U && !options_.allow_v1_migration)
      incompatible("schema v1 migration disabled by caller");
    execute(database_, "PRAGMA foreign_keys = ON", "open.foreign_keys",
            control);
    execute(database_, "PRAGMA synchronous = FULL", "open.synchronous",
            control);
    execute(database_, "PRAGMA trusted_schema = OFF", "open.trusted_schema",
            control);
    if (scalar_integer(database_, "PRAGMA foreign_keys", "open.foreign_keys",
                       control) != 1U) {
      incompatible("SQLite foreign key enforcement is unavailable");
    }
    if (version < schema_version) {
      Transaction transaction(database_, true, control);
      const auto locked_version = scalar_integer(
          database_, "PRAGMA user_version", "open.version", control);
      if (locked_version == 0U) {
        if (scalar_integer(database_, "PRAGMA application_id",
                           "open.application", control) != 0U)
          incompatible("foreign application appeared during initialization");
        if (scalar_integer(database_,
                           "SELECT COUNT(*) FROM sqlite_master WHERE name NOT "
                           "GLOB 'sqlite_*'",
                           "open.empty", control) != 0U) {
          incompatible("unversioned nonempty database");
        }
        execute(database_, kCreateSchema, "open.create_schema", control);
      } else if ((locked_version != 1U && locked_version != schema_version) ||
                 scalar_integer(database_, "PRAGMA application_id",
                                "open.application",
                                control) != kApplicationId) {
        incompatible("database schema changed during initialization");
      }
      if (locked_version < schema_version) {
        if (locked_version == 1U && !options_.allow_v1_migration)
          incompatible("schema v1 migration disabled by caller");
        // Preparing legacy columns and checking FK integrity precedes any DDL.
        Statement legacy(database_, kSelectRun, "migration.v1_runs", control);
        Statement legacy_refs(database_, kSelectReferences, "migration.v1_refs",
                              control);
        Statement legacy_start(
            database_,
            "SELECT start_status, start_updated_at_ms FROM jyppx_runs LIMIT 0",
            "migration.v1_start", control);
        Statement legacy_tags(
            database_, "SELECT run_id, name, value FROM jyppx_run_tags LIMIT 0",
            "migration.v1_tags", control);
        Statement foreign_keys(database_, "PRAGMA foreign_key_check",
                               "migration.foreign_keys", control);
        if (foreign_keys.row())
          corrupt("migration", "broken legacy relationships");
        execute(database_, kCreateEventSchema, "migration.v1_to_v2", control);
      }
      transaction.commit();
    }
    {
      Statement journal(database_, "PRAGMA journal_mode = WAL", "open.wal",
                        control);
      if (!journal.row() || journal.text(0, 16U) != "wal") {
        incompatible("SQLite WAL mode is unavailable for this filesystem");
      }
    }
    execute(database_, "PRAGMA synchronous = FULL", "open.synchronous",
            control);
    execute(database_, "PRAGMA trusted_schema = OFF", "open.trusted_schema",
            control);
    // Preparing the expected columns rejects an incomplete schema before use.
    Statement validate(database_, kSelectRun, "open.validate_schema", control);
    Statement validate_references(database_, kSelectReferences,
                                  "open.validate_schema", control);
    Statement validate_events(
        database_,
        "SELECT run_sequence, stream_id, producer_sequence, event_id, "
        "idempotency_key, content_digest, canonical FROM jyppx_events LIMIT 0",
        "open.validate_events", control);
    Statement validate_metrics(
        database_,
        "SELECT run_sequence, point_index, metric_id, definition_version, "
        "step_kind, step_key, value_status FROM jyppx_metric_points LIMIT 0",
        "open.validate_metrics", control);
  }

  void validate_run(const RunMetadata& metadata) const {
    validate_run_metadata(metadata, options_.run_metadata_limits);
    validate_integer(metadata.created_at_unix_ms, "run.created_at_unix_ms");
    validate_integer(metadata.updated_at_unix_ms, "run.updated_at_unix_ms");
    const auto& limits = options_.run_metadata_limits;
    validate_text(metadata.run_id, limits.max_id_bytes, "run.run_id");
    validate_text(metadata.project_id, limits.max_id_bytes, "run.project_id");
    validate_text(metadata.experiment_id, limits.max_id_bytes,
                  "run.experiment_id");
    validate_text(metadata.display_name, limits.max_display_name_bytes,
                  "run.display_name");
    const std::array<const std::optional<std::string>*, 5> lineage{
        &metadata.lineage.parent_run_id, &metadata.lineage.source_digest,
        &metadata.lineage.build_fingerprint, &metadata.lineage.config_digest,
        &metadata.lineage.data_manifest_digest};
    for (const auto* value : lineage) {
      if (*value)
        validate_text(**value, limits.max_lineage_ref_bytes, "run.lineage");
    }
    for (const auto& [key, value] : metadata.tags) {
      validate_text(key, limits.max_tag_key_bytes, "run.tags.key");
      validate_text(value, limits.max_tag_value_bytes, "run.tags.value");
    }
  }

  void validate_run_id(std::string_view run_id) const {
    validate_text(run_id, options_.run_metadata_limits.max_id_bytes, "run_id");
  }

  void validate_reference(const RunArtifactReference& reference) const {
    validate_text(reference.reference_id, options_.max_reference_id_bytes,
                  "reference_id");
    validate_text(reference.role, options_.max_role_bytes, "reference.role");
    if (reference.kind != RunArtifactKind::artifact &&
        reference.kind != RunArtifactKind::checkpoint) {
      invalid("reference.kind", "artifact or checkpoint", "unknown enum");
    }
    validate_artifact_digest(reference.artifact.digest);
    const auto& limits = options_.artifact_limits;
    validate_text(reference.artifact.logical_name,
                  limits.max_logical_name_bytes, "reference.logical_name");
    validate_text(reference.artifact.media_type, limits.max_media_type_bytes,
                  "reference.media_type");
    validate_text(reference.artifact.producer, limits.max_producer_bytes,
                  "reference.producer", false);
    if (reference.artifact.size_bytes > limits.max_artifact_bytes) {
      exhausted("reference.size_bytes", "maximum artifact size",
                "oversized artifact");
    }
    validate_integer(reference.artifact.size_bytes, "reference.size_bytes");
    validate_integer(reference.attached_at_unix_ms,
                     "reference.attached_at_unix_ms");
  }

  std::optional<RunMetadata> find_unlocked(std::string_view run_id) const {
    Statement statement(database_, kSelectRun, "run.find");
    statement.bind(1, run_id);
    if (!statement.row()) return std::nullopt;
    const auto& limits = options_.run_metadata_limits;
    const auto status = statement.integer(4);
    if (status < 1U || status > 6U)
      corrupt("run.find", "unknown persisted run status");
    RunMetadata metadata{
        .run_id = statement.text(0, limits.max_id_bytes),
        .project_id = statement.text(1, limits.max_id_bytes),
        .experiment_id = statement.text(2, limits.max_id_bytes),
        .display_name = statement.text(3, limits.max_display_name_bytes),
        .status = static_cast<RunStatus>(status),
        .created_at_unix_ms = statement.integer(5),
        .updated_at_unix_ms = statement.integer(6),
        .lineage = {.parent_run_id = statement.optional_text(
                        7, limits.max_lineage_ref_bytes),
                    .source_digest = statement.optional_text(
                        8, limits.max_lineage_ref_bytes),
                    .build_fingerprint = statement.optional_text(
                        9, limits.max_lineage_ref_bytes),
                    .config_digest = statement.optional_text(
                        10, limits.max_lineage_ref_bytes),
                    .data_manifest_digest = statement.optional_text(
                        11, limits.max_lineage_ref_bytes)},
        .tags = {}};
    Statement tags(database_,
                   "SELECT name, value FROM jyppx_run_tags WHERE run_id = ?1 "
                   "ORDER BY name",
                   "run.tags");
    tags.bind(1, run_id);
    while (tags.row()) {
      if (metadata.tags.size() >= limits.max_tag_count) {
        exhausted("run.tags", "bounded stored tag count", "too many tags");
      }
      metadata.tags.emplace(tags.text(0, limits.max_tag_key_bytes),
                            tags.text(1, limits.max_tag_value_bytes));
    }
    validate_run(metadata);
    return metadata;
  }

  RunMetadata require_run(std::string_view run_id) const {
    auto metadata = find_unlocked(run_id);
    if (!metadata)
      invalid("run_id", "existing run identity", "unknown identity");
    return std::move(*metadata);
  }

  std::vector<RunArtifactReference> references_unlocked(
      std::string_view run_id) const {
    Statement statement(database_, kSelectReferences, "run.references");
    statement.bind(1, run_id);
    std::vector<RunArtifactReference> result;
    while (statement.row()) {
      if (result.size() >= options_.max_references_per_run) {
        exhausted("run.references", "bounded stored reference count",
                  "too many references");
      }
      const auto& limits = options_.artifact_limits;
      const auto kind = statement.integer(1);
      const auto algorithm = statement.integer(3);
      if ((kind != 1U && kind != 2U) || algorithm != 1U) {
        corrupt("run.references",
                "unknown persisted artifact kind or digest algorithm");
      }
      RunArtifactReference reference{
          .reference_id = statement.text(0, options_.max_reference_id_bytes),
          .kind = static_cast<RunArtifactKind>(kind),
          .role = statement.text(2, options_.max_role_bytes),
          .artifact =
              {.digest = {.algorithm =
                              static_cast<ArtifactDigestAlgorithm>(algorithm),
                          .hex = statement.text(4, 64U)},
               .size_bytes = statement.integer(5),
               .logical_name = statement.text(6, limits.max_logical_name_bytes),
               .media_type = statement.text(7, limits.max_media_type_bytes),
               .producer = statement.text(8, limits.max_producer_bytes)},
          .attached_at_unix_ms = statement.integer(9)};
      validate_reference(reference);
      result.push_back(std::move(reference));
    }
    return result;
  }

  std::vector<ArtifactRef> live_blobs() const {
    Statement statement(
        database_,
        "SELECT digest, size_bytes FROM jyppx_blobs b WHERE EXISTS "
        "(SELECT 1 FROM jyppx_run_artifacts r WHERE r.digest = b.digest) ORDER "
        "BY digest",
        "run.live_blobs");
    std::vector<ArtifactRef> result;
    while (statement.row()) {
      if (result.size() >= options_.artifact_limits.max_object_files) {
        exhausted("live_blobs", "bounded CAS object count",
                  "too many live objects");
      }
      ArtifactRef reference;
      reference.digest.hex = statement.text(0, 64U);
      reference.size_bytes = statement.integer(1);
      validate_artifact_digest(reference.digest);
      result.push_back(std::move(reference));
    }
    return result;
  }

  void attach_unlocked(std::string_view run_id,
                       std::span<const RunArtifactReference> references,
                       std::stop_token stop = {},
                       TelemetryDeadline deadline = std::nullopt) {
    auto run = require_run(run_id);
    auto stored = references_unlocked(run_id);
    std::unordered_map<std::string, std::size_t> reference_index;
    reference_index.reserve(stored.size() + references.size());
    std::uint64_t linked_bytes = 0U;
    for (std::size_t index = 0U; index < stored.size(); ++index) {
      const auto& reference = stored[index];
      reference_index.emplace(reference.reference_id, index);
      if (reference.artifact.size_bytes >
          options_.max_run_artifact_bytes -
              std::min(linked_bytes, options_.max_run_artifact_bytes)) {
        exhausted("run.artifact_bytes", "bounded linked bytes",
                  "stored quota exceeded");
      }
      linked_bytes += reference.artifact.size_bytes;
    }
    for (const auto& reference : references) {
      check_telemetry_operation(stop, deadline);
      validate_reference(reference);
      const auto existing = reference_index.find(reference.reference_id);
      if (existing != reference_index.end()) {
        if (!same_reference(stored[existing->second], reference)) {
          invalid("reference_id", "identical immutable relationship for replay",
                  "identity conflict");
        }
        continue;
      }
      if (is_terminal_run_status(run.status)) {
        invalid("attach", "created or running run for new relationships",
                "terminal run");
      }
      if (reference.attached_at_unix_ms < run.created_at_unix_ms) {
        invalid("reference.attached_at_unix_ms", "at least run creation time",
                "earlier timestamp");
      }
      if (stored.size() >= options_.max_references_per_run) {
        exhausted("run.references", "configured reference count",
                  "run reference quota exceeded");
      }
      if (reference.artifact.size_bytes >
          options_.max_run_artifact_bytes - linked_bytes) {
        exhausted("run.artifact_bytes", "configured linked byte count",
                  "run artifact quota exceeded");
      }
      const auto reader =
          artifacts_.open(reference.artifact.digest, stop, deadline);
      if (reader->size_bytes() != reference.artifact.size_bytes) {
        invalid("reference.size_bytes", "exact committed blob size",
                "size mismatch");
      }
      {
        Statement blob(database_,
                       "INSERT OR IGNORE INTO jyppx_blobs(digest, algorithm, "
                       "size_bytes) VALUES(?1, 1, ?2)",
                       "attach.blob");
        blob.bind(1, reference.artifact.digest.hex);
        blob.bind(2, reference.artifact.size_bytes);
        blob.done();
      }
      {
        Statement blob(database_,
                       "SELECT size_bytes FROM jyppx_blobs WHERE digest = ?1",
                       "attach.blob_size");
        blob.bind(1, reference.artifact.digest.hex);
        if (!blob.row() || blob.integer(0) != reference.artifact.size_bytes) {
          corrupt("attach.blob_size",
                  "content identity has conflicting persisted size");
        }
      }
      Statement link(
          database_,
          "INSERT INTO jyppx_run_artifacts(run_id, reference_id, kind, role, "
          "digest, logical_name, media_type, producer, attached_at_ms) "
          "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
          "attach.reference");
      link.bind(1, run_id);
      link.bind(2, reference.reference_id);
      link.bind(3, static_cast<std::uint64_t>(reference.kind));
      link.bind(4, reference.role);
      link.bind(5, reference.artifact.digest.hex);
      link.bind(6, reference.artifact.logical_name);
      link.bind(7, reference.artifact.media_type);
      link.bind(8, reference.artifact.producer);
      link.bind(9, reference.attached_at_unix_ms);
      link.done();
      linked_bytes += reference.artifact.size_bytes;
      run.updated_at_unix_ms =
          std::max(run.updated_at_unix_ms, reference.attached_at_unix_ms);
      reference_index.emplace(reference.reference_id, stored.size());
      stored.push_back(reference);
    }
    Statement activity(
        database_, "UPDATE jyppx_runs SET updated_at_ms = ?2 WHERE run_id = ?1",
        "attach.activity");
    activity.bind(1, run_id);
    activity.bind(2, run.updated_at_unix_ms);
    activity.done();
  }

  SQLiteRunStoreOptions options_;
  std::filesystem::path root_;
  std::filesystem::path database_path_;
  FileArtifactStore artifacts_;
  sqlite3* database_ = nullptr;
  mutable std::timed_mutex mutex_;
};

SQLiteRunStore::SQLiteRunStore(std::filesystem::path root,
                               SQLiteRunStoreOptions options,
                               std::stop_token stop, TelemetryDeadline deadline)
    : impl_(std::make_unique<Impl>(std::move(root), options, stop, deadline)) {}

SQLiteRunStore::~SQLiteRunStore() = default;

RunMetadata SQLiteRunStore::start(const RunMetadata& metadata) {
  impl_->validate_run(metadata);
  if (metadata.status != RunStatus::created &&
      metadata.status != RunStatus::running) {
    invalid("start.status", "created or running", "terminal status");
  }
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  if (const auto existing = impl_->find_unlocked(metadata.run_id)) {
    auto original = *existing;
    Statement initial(impl_->database_,
                      "SELECT start_status, start_updated_at_ms FROM "
                      "jyppx_runs WHERE run_id = ?1",
                      "start.original");
    initial.bind(1, metadata.run_id);
    if (!initial.row())
      corrupt("start.original", "missing original start identity");
    original.status = static_cast<RunStatus>(initial.integer(0));
    original.updated_at_unix_ms = initial.integer(1);
    if (!same_metadata(metadata, original)) {
      invalid("start.run_id",
              "identical original metadata for an idempotent replay",
              "identity conflict");
    }
    transaction.commit();
    return *existing;
  }
  if (scalar_integer(impl_->database_, "SELECT COUNT(*) FROM jyppx_runs",
                     "start.run_count") >= impl_->options_.max_runs) {
    exhausted("start", "configured maximum run identities",
              "run quota exceeded");
  }
  {
    Statement statement(
        impl_->database_,
        "INSERT INTO jyppx_runs(run_id, project_id, experiment_id, "
        "display_name, status, created_at_ms, updated_at_ms, "
        "start_status, start_updated_at_ms, parent_run_id, source_digest, "
        "build_fingerprint, config_digest, data_manifest_digest) "
        "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?5, ?7, ?8, ?9, ?10, ?11, ?12)",
        "start.insert");
    statement.bind(1, metadata.run_id);
    statement.bind(2, metadata.project_id);
    statement.bind(3, metadata.experiment_id);
    statement.bind(4, metadata.display_name);
    statement.bind(5, static_cast<std::uint64_t>(metadata.status));
    statement.bind(6, metadata.created_at_unix_ms);
    statement.bind(7, metadata.updated_at_unix_ms);
    statement.bind(8, metadata.lineage.parent_run_id);
    statement.bind(9, metadata.lineage.source_digest);
    statement.bind(10, metadata.lineage.build_fingerprint);
    statement.bind(11, metadata.lineage.config_digest);
    statement.bind(12, metadata.lineage.data_manifest_digest);
    statement.done();
  }
  for (const auto& [key, value] : metadata.tags) {
    Statement statement(
        impl_->database_,
        "INSERT INTO jyppx_run_tags(run_id, name, value) VALUES(?1, ?2, ?3)",
        "start.tags");
    statement.bind(1, metadata.run_id);
    statement.bind(2, key);
    statement.bind(3, value);
    statement.done();
  }
  transaction.commit();
  return metadata;
}

RunMetadata SQLiteRunStore::transition(std::string_view run_id,
                                       RunStatus status,
                                       std::uint64_t updated_at_unix_ms) {
  impl_->validate_run_id(run_id);
  validate_integer(updated_at_unix_ms, "transition.timestamp");
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  auto metadata = impl_->require_run(run_id);
  if (!is_valid_run_transition(metadata.status, status)) {
    invalid("transition", "a permitted monotonic run lifecycle transition",
            "invalid transition");
  }
  if (metadata.status == status) {
    transaction.commit();
    return metadata;
  }
  if (updated_at_unix_ms < metadata.updated_at_unix_ms) {
    invalid("transition.timestamp", "at least the last activity timestamp",
            "earlier timestamp");
  }
  Statement statement(
      impl_->database_,
      "UPDATE jyppx_runs SET status = ?2, updated_at_ms = ?3 WHERE run_id = ?1",
      "transition.update");
  statement.bind(1, run_id);
  statement.bind(2, static_cast<std::uint64_t>(status));
  statement.bind(3, updated_at_unix_ms);
  statement.done();
  metadata.status = status;
  metadata.updated_at_unix_ms = updated_at_unix_ms;
  transaction.commit();
  return metadata;
}

void SQLiteRunStore::attach(std::string_view run_id,
                            std::span<const RunArtifactReference> references) {
  impl_->validate_run_id(run_id);
  if (references.size() > impl_->options_.max_references_per_run) {
    exhausted("attach", "configured reference batch bound", "oversized batch");
  }
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  impl_->attach_unlocked(run_id, references);
  transaction.commit();
}

RunArtifactReference SQLiteRunStore::put_artifact(
    std::string_view run_id, std::string reference_id, RunArtifactKind kind,
    std::string role, std::span<const std::byte> bytes,
    const ArtifactDescriptor& descriptor, std::uint64_t attached_at_unix_ms,
    std::optional<ArtifactDigest> expected_digest) {
  impl_->validate_run_id(run_id);
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  (void)impl_->require_run(run_id);
  RunArtifactReference reference{
      .reference_id = std::move(reference_id),
      .kind = kind,
      .role = std::move(role),
      .artifact =
          impl_->artifacts_.put(bytes, descriptor, std::move(expected_digest)),
      .attached_at_unix_ms = attached_at_unix_ms};
  impl_->attach_unlocked(run_id,
                         std::span<const RunArtifactReference>(&reference, 1U));
  transaction.commit();
  return reference;
}

RunArtifactReference SQLiteRunStore::put_artifact_file(
    std::string_view run_id, std::string reference_id, RunArtifactKind kind,
    std::string role, const std::filesystem::path& path,
    const ArtifactDescriptor& descriptor, std::uint64_t attached_at_unix_ms,
    ArtifactDigest expected_digest, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  impl_->validate_run_id(run_id);
  validate_artifact_digest(expected_digest);
  std::error_code path_error;
  const auto status = std::filesystem::symlink_status(path, path_error);
  if (path_error || !std::filesystem::is_regular_file(status)) {
    throw IoError("Published artifact path is not an ordinary file",
                  ErrorDetails{.component = "experiment.run_store",
                               .operation = "put_artifact_file.open",
                               .expected = "an existing ordinary file",
                               .actual = utf8_path(path),
                               .hint = "Wait for successful file publication "
                                       "before registering its result."});
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 0) {
    throw IoError("Could not open published artifact file");
  }
  const auto size = static_cast<std::uint64_t>(input.tellg());
  if (size > impl_->options_.artifact_limits.max_artifact_bytes) {
    exhausted("put_artifact_file.size", "configured artifact byte bound",
              "oversized file");
  }
  if (descriptor.expected_size_bytes &&
      size != *descriptor.expected_size_bytes) {
    invalid("put_artifact_file.size", "the published byte count",
            std::to_string(size) + " bytes");
  }
  input.seekg(0);
  if (!input) throw IoError("Could not seek published artifact file");

  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  (void)impl_->require_run(run_id);
  auto upload = impl_->artifacts_.begin(descriptor);
  std::array<std::byte, 64U * 1024U> buffer{};
  std::uint64_t copied = 0U;
  while (copied < size) {
    check_telemetry_operation(stop, deadline);
    const auto requested = static_cast<std::streamsize>(
        std::min<std::uint64_t>(buffer.size(), size - copied));
    input.read(reinterpret_cast<char*>(buffer.data()), requested);
    const auto count = input.gcount();
    if (count != requested)
      throw IoError("Published artifact file changed while reading");
    upload->write(std::span<const std::byte>(buffer.data(),
                                             static_cast<std::size_t>(count)));
    copied += static_cast<std::uint64_t>(count);
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
    throw IoError("Published artifact file changed while reading");
  }
  RunArtifactReference reference{
      .reference_id = std::move(reference_id),
      .kind = kind,
      .role = std::move(role),
      .artifact = impl_->artifacts_.commit(*upload, std::move(expected_digest),
                                           stop, deadline),
      .attached_at_unix_ms = attached_at_unix_ms};
  impl_->attach_unlocked(run_id,
                         std::span<const RunArtifactReference>(&reference, 1U),
                         stop, deadline);
  transaction.commit();
  return reference;
}

std::optional<RunMetadata> SQLiteRunStore::find(std::string_view run_id) const {
  impl_->validate_run_id(run_id);
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_, false);
  auto result = impl_->find_unlocked(run_id);
  transaction.commit();
  return result;
}

std::vector<RunArtifactReference> SQLiteRunStore::references(
    std::string_view run_id) const {
  impl_->validate_run_id(run_id);
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_, false);
  (void)impl_->require_run(run_id);
  auto result = impl_->references_unlocked(run_id);
  transaction.commit();
  return result;
}

RunMetadata SQLiteRunStore::heartbeat(std::string_view run_id,
                                      std::uint64_t updated_at_unix_ms) {
  impl_->validate_run_id(run_id);
  validate_integer(updated_at_unix_ms, "heartbeat.timestamp");
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  auto metadata = impl_->require_run(run_id);
  if (metadata.status != RunStatus::running) {
    invalid("heartbeat", "running run", "inactive run");
  }
  if (updated_at_unix_ms < metadata.updated_at_unix_ms) {
    invalid("heartbeat.timestamp", "at least last activity time",
            "earlier timestamp");
  }
  Statement statement(
      impl_->database_,
      "UPDATE jyppx_runs SET updated_at_ms = ?2 WHERE run_id = ?1",
      "heartbeat.update");
  statement.bind(1, run_id);
  statement.bind(2, updated_at_unix_ms);
  statement.done();
  metadata.updated_at_unix_ms = updated_at_unix_ms;
  transaction.commit();
  return metadata;
}

std::size_t SQLiteRunStore::recover_interrupted(
    std::uint64_t stale_at_or_before_unix_ms,
    std::uint64_t interrupted_at_unix_ms) {
  validate_integer(stale_at_or_before_unix_ms, "recover.cutoff");
  validate_integer(interrupted_at_unix_ms, "recover.timestamp");
  if (interrupted_at_unix_ms < stale_at_or_before_unix_ms) {
    invalid("recover.timestamp", "at least the stale cutoff",
            "earlier timestamp");
  }
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  Statement statement(impl_->database_,
                      "UPDATE jyppx_runs SET status = 6, updated_at_ms = ?2 "
                      "WHERE status = 2 AND updated_at_ms <= ?1",
                      "recover.interrupted");
  statement.bind(1, stale_at_or_before_unix_ms);
  statement.bind(2, interrupted_at_unix_ms);
  statement.done();
  const auto changed = sqlite3_changes(impl_->database_);
  transaction.commit();
  return static_cast<std::size_t>(changed);
}

RunStoreIntegrityReport SQLiteRunStore::integrity_check(
    bool verify_artifacts, std::stop_token stop,
    TelemetryDeadline deadline) const {
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, true, &control);
    RunStoreIntegrityReport report;
    {
      Statement statement(impl_->database_, "PRAGMA integrity_check(64)",
                          "integrity.database", &control);
      while (statement.row()) {
        const auto issue = statement.text(0, 4096U);
        if (issue != "ok" && report.issues.size() < kMaxIntegrityIssues) {
          report.issues.push_back("SQLite: " + issue);
        }
      }
    }
    {
      Statement statement(impl_->database_, "PRAGMA foreign_key_check",
                          "integrity.foreign_keys", &control);
      while (statement.row()) {
        if (report.issues.size() < kMaxIntegrityIssues) {
          report.issues.push_back("Foreign key violation in " +
                                  statement.text(0, 256U));
        }
      }
    }
    if (verify_artifacts) {
      for (const auto& reference : impl_->live_blobs()) {
        control.check();
        ++report.checked_artifacts;
        try {
          const auto reader =
              impl_->artifacts_.open(reference.digest, stop, deadline);
          if (reader->size_bytes() != reference.size_bytes &&
              report.issues.size() < kMaxIntegrityIssues) {
            report.issues.push_back("Artifact size mismatch: " +
                                    reference.digest.hex);
          }
        } catch (const Error&) {
          control.check();
          if (report.issues.size() < kMaxIntegrityIssues) {
            report.issues.push_back("Missing or invalid artifact: " +
                                    reference.digest.hex);
          }
        }
      }
    }
    verify_event_integrity(impl_->database_, impl_->options_, report, &control);
    transaction.commit();
    return report;
  } catch (...) {
    control.check();
    throw;
  }
}

ArtifactGcReport SQLiteRunStore::collect_garbage(
    std::chrono::seconds grace_period) {
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  {
    Statement foreign_keys(impl_->database_, "PRAGMA foreign_key_check",
                           "gc.foreign_keys");
    if (foreign_keys.row()) {
      corrupt(
          "gc.foreign_keys",
          "refusing collection while a persisted relationship is inconsistent");
    }
  }
  const auto live = impl_->live_blobs();
  RunStoreIntegrityReport event_report;
  verify_event_integrity(impl_->database_, impl_->options_, event_report);
  if (!event_report.ok())
    corrupt("gc.events",
            "refusing collection while canonical event relationships are "
            "inconsistent");
  std::vector<ArtifactDigest> digests;
  digests.reserve(live.size());
  for (const auto& reference : live) digests.push_back(reference.digest);
  const auto report = impl_->artifacts_.collect_garbage(digests, grace_period);
  transaction.commit();
  return report;
}

const std::filesystem::path& SQLiteRunStore::root() const noexcept {
  return impl_->root_;
}

const std::filesystem::path& SQLiteRunStore::database_path() const noexcept {
  return impl_->database_path_;
}

const std::filesystem::path& SQLiteRunStore::artifact_root() const noexcept {
  return impl_->artifacts_.root();
}

const RunEventStoreLimits& SQLiteRunStore::event_limits() const noexcept {
  return impl_->options_.event_limits;
}

namespace {

[[noreturn]] void event_conflict(std::string operation, std::string actual) {
  throw CompatibilityError(
      "Conflicting durable event request",
      ErrorDetails{
          .component = "experiment.run_events",
          .operation = std::move(operation),
          .expected =
              "immutable identity, contiguous sequence, and exact retry",
          .actual = std::move(actual),
          .hint = "Retry the original request; use a new stream for a new "
                  "attempt."});
}

std::unique_lock<std::timed_mutex> event_lock(std::timed_mutex& mutex,
                                              std::stop_token stop,
                                              TelemetryDeadline deadline) {
  std::unique_lock<std::timed_mutex> lock(mutex, std::defer_lock);
  while (true) {
    check_telemetry_operation(stop, deadline);
    auto until =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
    if (deadline) until = std::min(until, *deadline);
    if (lock.try_lock_until(until)) {
      check_telemetry_operation(stop, deadline);
      return lock;
    }
  }
}

std::span<const TelemetryMetricPoint> event_metrics(
    const TelemetryEnvelope& envelope) {
  if (const auto* value = std::get_if<TelemetryLifecycle>(&envelope.payload))
    return value->metrics;
  if (const auto* value = std::get_if<TelemetryMetricBatch>(&envelope.payload))
    return value->points;
  if (const auto* value =
          std::get_if<TelemetryResourceSample>(&envelope.payload))
    return value->measurements;
  return {};
}

std::array<std::byte, 8U> unsigned_key(std::uint64_t value) noexcept {
  std::array<std::byte, 8U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index)
    bytes[index] =
        static_cast<std::byte>((value >> ((7U - index) * 8U)) & 0xffU);
  return bytes;
}

struct EncodedEvent {
  const TelemetryEnvelope* envelope = nullptr;
  std::vector<std::byte> canonical;
  std::string digest;
};

std::string batch_digest(std::span<const EncodedEvent> events) {
  Sha256 hash;
  constexpr std::string_view domain = "jyppx.run.event-batch/v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  hash.update(unsigned_key(events.size()));
  for (const auto& event : events) {
    hash.update(unsigned_key(event.canonical.size()));
    hash.update(event.canonical);
  }
  return sha256_digest_hex(hash.finish());
}

struct EventState {
  std::uint64_t last = 0U;
  std::uint64_t metrics = 0U;
  std::uint64_t bytes = 0U;
  std::uint64_t batches = 0U;
  std::uint64_t streams = 0U;
};

EventState read_event_state(sqlite3* database, std::string_view run_id,
                            const OperationControl* control) {
  Statement state(database,
                  "SELECT last_sequence, metric_count, content_bytes, "
                  "batch_count, stream_count "
                  "FROM jyppx_run_event_state WHERE run_id = ?1",
                  "events.state", control);
  state.bind(1, run_id);
  if (!state.row()) return {};
  return {.last = state.integer(0),
          .metrics = state.integer(1),
          .bytes = state.integer(2),
          .batches = state.integer(3),
          .streams = state.integer(4)};
}

void consume_quota(std::uint64_t& used, std::uint64_t added,
                   std::uint64_t maximum, std::string_view name) {
  if (used > maximum || added > maximum - used)
    exhausted(std::string(name), "configured durable quota", "quota exceeded");
  used += added;
}

TelemetryEnvelope decode_stored_event(std::span<const std::byte> canonical,
                                      std::string_view digest,
                                      std::string_view run_id,
                                      const TelemetryLimits& limits) {
  if (sha256_hex(canonical) != digest)
    corrupt("events.decode", "content digest mismatch");
  auto envelope = decode_telemetry_envelope(canonical, limits);
  if (envelope.correlation.run_id != run_id)
    corrupt("events.decode", "persisted run correlation mismatch");
  return envelope;
}

std::uint64_t snapshot_boundary(std::uint64_t current,
                                std::optional<std::uint64_t> requested,
                                std::uint64_t after) {
  const auto snapshot = requested.value_or(current);
  if (snapshot > current || after > snapshot)
    invalid("events.cursor", "cursor within the committed snapshot",
            "future cursor");
  return snapshot;
}

void verify_event_integrity(sqlite3* database,
                            const SQLiteRunStoreOptions& options,
                            RunStoreIntegrityReport& report,
                            const OperationControl* control) {
  const auto& limits = options.event_limits;
  const auto note = [&](std::string message) {
    if (report.issues.size() < kMaxIntegrityIssues)
      report.issues.push_back("Telemetry: " + std::move(message));
  };
  // This audit reads one canonical envelope at a time. Derived SQL indices
  // cannot hide a missing point, a changed identity, or an omitted CAS link.
  Statement events(
      database,
      "SELECT "
      "e.run_id,e.run_sequence,e.stream_id,e.producer_sequence,e.event_id,"
      "e.idempotency_key,e.payload_kind,e.content_digest,e.canonical,r.project_"
      "id "
      "FROM jyppx_events e JOIN jyppx_runs r ON r.run_id=e.run_id "
      "ORDER BY e.run_id,e.run_sequence",
      "integrity.events", control);
  Statement points(database,
                   "SELECT "
                   "point_index,metric_id,definition_version,step_kind,step_"
                   "key,value_status "
                   "FROM jyppx_metric_points WHERE run_id=?1 AND "
                   "run_sequence=?2 ORDER BY point_index",
                   "integrity.metric_points", control);
  Statement definition(
      database,
      "SELECT definition_digest,canonical FROM jyppx_metric_definitions "
      "WHERE metric_id=?1 AND definition_version=?2",
      "integrity.metric_definition", control);
  Statement artifact(
      database,
      "SELECT "
      "a.kind,a.role,a.digest,b.size_bytes,a.logical_name,a.media_type,a."
      "producer "
      "FROM jyppx_run_artifacts a JOIN jyppx_blobs b ON b.digest=a.digest "
      "WHERE a.run_id=?1 AND a.reference_id=?2",
      "integrity.event_artifact", control);
  std::string current_run;
  std::uint64_t expected_sequence = 1U;
  while (report.issues.size() < kMaxIntegrityIssues && events.row()) {
    try {
      const auto run_id =
          events.text(0, options.run_metadata_limits.max_id_bytes);
      if (run_id != current_run) {
        current_run = run_id;
        expected_sequence = 1U;
      }
      if (events.integer(1) != expected_sequence++)
        corrupt("integrity.events", "run commit sequence has a gap");
      const auto canonical =
          events.blob(8, limits.telemetry.max_envelope_bytes);
      const auto envelope = decode_stored_event(canonical, events.text(7, 64U),
                                                run_id, limits.telemetry);
      if (events.text(2, limits.telemetry.max_identifier_bytes) !=
              envelope.identity.stream_id ||
          events.integer(3) != envelope.identity.sequence ||
          events.text(4, limits.telemetry.max_identifier_bytes) !=
              envelope.identity.event_id ||
          events.text(5, limits.telemetry.max_identifier_bytes) !=
              envelope.identity.idempotency_key ||
          events.integer(6) !=
              static_cast<std::uint64_t>(*telemetry_payload_kind(envelope)) ||
          (!envelope.correlation.project_id.empty() &&
           envelope.correlation.project_id !=
               events.text(9, options.run_metadata_limits.max_id_bytes)))
        corrupt("integrity.events",
                "identity/payload/project index differs from source");
      const auto source_points = event_metrics(envelope);
      points.bind(1, run_id);
      points.bind(2, events.integer(1));
      std::size_t index = 0U;
      while (points.row()) {
        if (index >= source_points.size() || points.integer(0) != index)
          corrupt("integrity.metric_points", "extra or missing point index");
        const auto& source = source_points[index++];
        const auto expected_key = unsigned_key(source.step.value);
        const auto step_key = points.blob(4, 8U);
        if (points.text(1, limits.telemetry.max_identifier_bytes) !=
                source.definition.metric_id ||
            points.integer(2) != source.definition.version ||
            points.integer(3) != static_cast<std::uint64_t>(source.step.kind) ||
            step_key.size() != expected_key.size() ||
            !std::equal(step_key.begin(), step_key.end(),
                        expected_key.begin()) ||
            points.integer(5) !=
                static_cast<std::uint64_t>(source.observation.status))
          corrupt("integrity.metric_points",
                  "metric index differs from source");
        const auto encoded = encode_telemetry_metric_definition(
            source.definition, limits.telemetry);
        definition.bind(1, source.definition.metric_id);
        definition.bind(2,
                        static_cast<std::uint64_t>(source.definition.version));
        if (!definition.row() ||
            definition.text(0, 64U) != sha256_hex(encoded) ||
            definition.blob(1, limits.telemetry.max_envelope_bytes) != encoded)
          corrupt("integrity.metric_definition",
                  "missing or conflicting definition");
        definition.reset();
      }
      if (index != source_points.size())
        corrupt("integrity.metric_points", "source point omitted from index");
      points.reset();
      if (const auto* source =
              std::get_if<TelemetryArtifactReference>(&envelope.payload)) {
        artifact.bind(1, run_id);
        artifact.bind(2, "telemetry." + telemetry_idempotency_digest(
                                            envelope.identity, limits.telemetry)
                                            .hex);
        if (!artifact.row() ||
            artifact.integer(0) !=
                static_cast<std::uint64_t>(RunArtifactKind::artifact) ||
            artifact.text(1, options.max_role_bytes) != source->role ||
            artifact.text(2, 64U) != source->artifact.digest.hex ||
            artifact.integer(3) != source->artifact.size_bytes ||
            artifact.text(4, options.artifact_limits.max_logical_name_bytes) !=
                source->artifact.logical_name ||
            artifact.text(5, options.artifact_limits.max_media_type_bytes) !=
                source->artifact.media_type ||
            artifact.text(6, options.artifact_limits.max_producer_bytes) !=
                source->artifact.producer)
          corrupt(
              "integrity.event_artifact",
              "event's GC-visible artifact relationship is missing or changed");
        artifact.reset();
      }
    } catch (const Error& error) {
      if (control != nullptr) control->check();
      note(error.what());
      // Read/decoding failures are reported; resetting only reuses read-only
      // SQL.
      points.reset();
      definition.reset();
      artifact.reset();
    }
  }
  {
    Statement definitions(
        database,
        "SELECT metric_id,definition_version,definition_digest,canonical "
        "FROM jyppx_metric_definitions",
        "integrity.all_definitions", control);
    while (report.issues.size() < kMaxIntegrityIssues && definitions.row()) {
      try {
        const auto encoded =
            definitions.blob(3, limits.telemetry.max_envelope_bytes);
        const auto decoded =
            decode_telemetry_metric_definition(encoded, limits.telemetry);
        if (sha256_hex(encoded) != definitions.text(2, 64U) ||
            decoded.metric_id !=
                definitions.text(0, limits.telemetry.max_identifier_bytes) ||
            decoded.version != definitions.integer(1))
          corrupt("integrity.all_definitions",
                  "definition key/digest differs from canonical bytes");
      } catch (const Error& error) {
        if (control != nullptr) control->check();
        note(error.what());
      }
    }
  }
  {
    Statement states(
        database,
        "SELECT "
        "r.run_id,COALESCE(s.last_sequence,0),COALESCE(s.metric_count,0),"
        "COALESCE(s.content_bytes,0),COALESCE(s.batch_count,0),COALESCE(s."
        "stream_count,0),"
        "(SELECT COUNT(*) FROM jyppx_events e WHERE e.run_id=r.run_id),"
        "(SELECT COALESCE(MAX(e.run_sequence),0) FROM jyppx_events e WHERE "
        "e.run_id=r.run_id),"
        "(SELECT COUNT(*) FROM jyppx_metric_points p WHERE p.run_id=r.run_id),"
        "(SELECT COALESCE(SUM(length(e.canonical)),0) FROM jyppx_events e "
        "WHERE e.run_id=r.run_id),"
        "(SELECT COUNT(*) FROM jyppx_event_batches b WHERE b.run_id=r.run_id),"
        "(SELECT COUNT(*) FROM jyppx_event_streams t WHERE t.run_id=r.run_id) "
        "FROM jyppx_runs r LEFT JOIN jyppx_run_event_state s ON "
        "s.run_id=r.run_id",
        "integrity.event_state", control);
    while (report.issues.size() < kMaxIntegrityIssues && states.row()) {
      try {
        if (states.integer(1) != states.integer(6) ||
            states.integer(1) != states.integer(7) ||
            states.integer(2) != states.integer(8) ||
            states.integer(3) != states.integer(9) ||
            states.integer(4) != states.integer(10) ||
            states.integer(5) != states.integer(11))
          corrupt("integrity.event_state",
                  "run counters differ from durable rows");
      } catch (const Error& error) {
        if (control != nullptr) control->check();
        note(error.what());
      }
    }
  }
  {
    Statement streams(
        database,
        "SELECT "
        "s.stream_id,s.last_sequence,s.terminal,COUNT(e.producer_sequence),"
        "COALESCE(MIN(e.producer_sequence),0),COALESCE(MAX(e.producer_sequence)"
        ",0),"
        "(SELECT t.content_digest FROM jyppx_events t WHERE "
        "t.stream_id=s.stream_id "
        "AND t.producer_sequence=s.last_sequence),"
        "(SELECT t.canonical FROM jyppx_events t WHERE t.stream_id=s.stream_id "
        "AND t.producer_sequence=s.last_sequence),s.run_id "
        "FROM jyppx_event_streams s LEFT JOIN jyppx_events e ON "
        "e.stream_id=s.stream_id "
        "GROUP BY s.stream_id",
        "integrity.event_streams", control);
    while (report.issues.size() < kMaxIntegrityIssues && streams.row()) {
      try {
        const auto last = streams.integer(1);
        if (last == 0U || streams.integer(3) != last ||
            streams.integer(4) != 1U || streams.integer(5) != last)
          corrupt("integrity.event_streams",
                  "producer sequence gaps or inconsistent watermark");
        const auto encoded =
            streams.blob(7, limits.telemetry.max_envelope_bytes);
        const auto envelope = decode_stored_event(
            encoded, streams.text(6, 64U),
            streams.text(8, options.run_metadata_limits.max_id_bytes),
            limits.telemetry);
        const auto* lifecycle =
            std::get_if<TelemetryLifecycle>(&envelope.payload);
        const bool terminal = lifecycle != nullptr && lifecycle->terminal;
        if ((streams.integer(2) != 0U) != terminal)
          corrupt("integrity.event_streams",
                  "stream terminal flag differs from source");
        Statement earlier_terminal(
            database,
            "SELECT content_digest,canonical FROM jyppx_events WHERE "
            "stream_id=?1 "
            "AND producer_sequence<?2 AND payload_kind=1",
            "integrity.early_terminal", control);
        earlier_terminal.bind(
            1, streams.text(0, limits.telemetry.max_identifier_bytes));
        earlier_terminal.bind(2, last);
        while (earlier_terminal.row()) {
          const auto bytes =
              earlier_terminal.blob(1, limits.telemetry.max_envelope_bytes);
          const auto prior = decode_stored_event(
              bytes, earlier_terminal.text(0, 64U), envelope.correlation.run_id,
              limits.telemetry);
          const auto* value = std::get_if<TelemetryLifecycle>(&prior.payload);
          if (value != nullptr && value->terminal)
            corrupt("integrity.event_streams",
                    "events follow a terminal producer event");
        }
      } catch (const Error& error) {
        if (control != nullptr) control->check();
        note(error.what());
      }
    }
  }
  {
    Statement batches(database,
                      "SELECT "
                      "b.run_id,b.batch_key,b.request_digest,b.input_count,b."
                      "committed_through,s.last_sequence "
                      "FROM jyppx_event_batches b LEFT JOIN "
                      "jyppx_run_event_state s ON s.run_id=b.run_id",
                      "integrity.event_batches", control);
    Statement items(database,
                    "SELECT i.ordinal,i.run_sequence,e.canonical FROM "
                    "jyppx_event_batch_items i "
                    "JOIN jyppx_events e ON e.run_id=i.run_id AND "
                    "e.run_sequence=i.run_sequence "
                    "WHERE i.run_id=?1 AND i.batch_key=?2 ORDER BY i.ordinal",
                    "integrity.batch_items", control);
    while (report.issues.size() < kMaxIntegrityIssues && batches.row()) {
      try {
        const auto count = batches.integer(3);
        if (count == 0U || count > limits.telemetry.max_batch_events ||
            batches.integer(4) > batches.integer(5))
          corrupt("integrity.event_batches",
                  "invalid count or original watermark");
        items.bind(1,
                   batches.text(0, options.run_metadata_limits.max_id_bytes));
        items.bind(2, batches.text(1, limits.telemetry.max_identifier_bytes));
        std::vector<EncodedEvent> encoded;
        std::uint64_t bytes = 0U;
        while (items.row()) {
          if (items.integer(0) != encoded.size() || encoded.size() >= count ||
              items.integer(1) > batches.integer(4))
            corrupt("integrity.batch_items",
                    "invalid ordinal or original event receipt");
          auto canonical = items.blob(2, limits.telemetry.max_envelope_bytes);
          consume_quota(bytes, canonical.size(),
                        limits.telemetry.max_batch_bytes,
                        "integrity.batch_items");
          encoded.push_back({.canonical = std::move(canonical), .digest = {}});
        }
        if (encoded.size() != count ||
            batch_digest(encoded) != batches.text(2, 64U))
          corrupt("integrity.event_batches",
                  "batch receipt differs from original ordered content");
        items.reset();
      } catch (const Error& error) {
        if (control != nullptr) control->check();
        note(error.what());
        items.reset();
      }
    }
  }
}

}  // namespace

std::vector<RunMetadata> SQLiteRunStore::list(
    const RunListRequest& request, std::stop_token stop,
    TelemetryDeadline deadline) const {
  check_telemetry_operation(stop, deadline);
  if (request.limit == 0U || request.limit > 1024U)
    invalid("runs.list.limit", "a positive count at most 1024", "out of range");
  const auto max_id_bytes = impl_->options_.run_metadata_limits.max_id_bytes;
  validate_text(request.after_run_id, max_id_bytes, "runs.list.after", false);
  if (request.project_id)
    validate_text(*request.project_id, max_id_bytes, "runs.list.project");
  if (request.experiment_id)
    validate_text(*request.experiment_id, max_id_bytes, "runs.list.experiment");
  if (request.status && (static_cast<unsigned>(*request.status) < 1U ||
                         static_cast<unsigned>(*request.status) > 6U))
    invalid("runs.list.status", "known run status", "unknown enum");
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, false, &control);
    std::string sql = "SELECT run_id FROM jyppx_runs WHERE run_id>?1 ";
    if (request.project_id) sql += "AND project_id=?2 ";
    if (request.experiment_id) sql += "AND experiment_id=?3 ";
    if (request.status) sql += "AND status=?4 ";
    sql += "ORDER BY run_id LIMIT ?5";
    Statement query(impl_->database_, sql.c_str(), "runs.list", &control);
    query.bind(1, request.after_run_id);
    if (request.project_id) query.bind(2, *request.project_id);
    if (request.experiment_id) query.bind(3, *request.experiment_id);
    if (request.status)
      query.bind(4, static_cast<std::uint64_t>(*request.status));
    query.bind(5, static_cast<std::uint64_t>(request.limit));
    std::vector<RunMetadata> result;
    result.reserve(request.limit);
    while (query.row()) {
      control.check();
      result.push_back(impl_->require_run(query.text(0, max_id_bytes)));
    }
    transaction.commit();
    return result;
  } catch (...) {
    control.check();
    throw;
  }
}

RunEventAppendResult SQLiteRunStore::append(
    std::string_view run_id, std::string_view batch_key,
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  impl_->validate_run_id(run_id);
  const auto& limits = impl_->options_.event_limits;
  validate_text(batch_key, limits.telemetry.max_identifier_bytes,
                "events.batch_key");
  validate_telemetry_batch(envelopes, limits.telemetry);
  std::vector<EncodedEvent> encoded;
  encoded.reserve(envelopes.size());
  std::uint64_t batch_metric_count = 0U;
  for (const auto& envelope : envelopes) {
    check_telemetry_operation(stop, deadline);
    if (envelope.correlation.run_id != run_id)
      invalid("events.correlation.run_id", "the requested run identity",
              "mismatch");
    validate_integer(envelope.identity.sequence, "events.producer_sequence");
    consume_quota(batch_metric_count, event_metrics(envelope).size(),
                  limits.max_metric_points_per_batch, "events.batch_metrics");
    auto canonical = encode_telemetry_envelope(envelope, limits.telemetry);
    auto digest = sha256_hex(canonical);
    encoded.push_back({.envelope = &envelope,
                       .canonical = std::move(canonical),
                       .digest = std::move(digest)});
  }
  const auto request_digest = batch_digest(encoded);
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, true, &control);
    const auto run = impl_->require_run(run_id);
    for (const auto& envelope : envelopes) {
      if (!envelope.correlation.project_id.empty() &&
          envelope.correlation.project_id != run.project_id)
        invalid("events.project_id", "run project identity or empty",
                "mismatch");
    }
    auto state = read_event_state(impl_->database_, run_id, &control);
    RunEventAppendResult result;
    result.receipts.reserve(envelopes.size());
    result.committed_through = state.last;
    if (envelopes.empty()) {
      transaction.commit();
      return result;
    }
    {
      Statement batch(impl_->database_,
                      "SELECT request_digest, input_count, committed_through "
                      "FROM jyppx_event_batches "
                      "WHERE run_id = ?1 AND batch_key = ?2",
                      "events.batch_replay", &control);
      batch.bind(1, run_id);
      batch.bind(2, batch_key);
      if (batch.row()) {
        if (batch.text(0, 64U) != request_digest ||
            batch.integer(1) != envelopes.size())
          event_conflict("events.batch_key",
                         "different ordered request for batch key");
        result.committed_through = batch.integer(2);
        result.batch_replay = true;
        Statement items(
            impl_->database_,
            "SELECT i.ordinal, i.run_sequence, e.canonical FROM "
            "jyppx_event_batch_items i "
            "JOIN jyppx_events e ON e.run_id = i.run_id AND e.run_sequence = "
            "i.run_sequence "
            "WHERE i.run_id = ?1 AND i.batch_key = ?2 ORDER BY i.ordinal",
            "events.batch_items", &control);
        items.bind(1, run_id);
        items.bind(2, batch_key);
        while (items.row()) {
          const auto ordinal = items.integer(0);
          if (ordinal != result.receipts.size() || ordinal >= encoded.size() ||
              items.blob(2, limits.telemetry.max_envelope_bytes) !=
                  encoded[ordinal].canonical)
            corrupt("events.batch_replay",
                    "inconsistent original batch receipt");
          result.receipts.push_back(
              {.run_sequence = items.integer(1),
               .disposition = TelemetryDisposition::duplicate});
        }
        if (result.receipts.size() != envelopes.size())
          corrupt("events.batch_replay", "incomplete original batch receipt");
        transaction.commit();
        return result;
      }
    }

    consume_quota(state.batches, 1U, limits.max_batches_per_run,
                  "events.batches");
    Statement duplicate(
        impl_->database_,
        "SELECT run_id, run_sequence, event_id, producer_sequence, "
        "idempotency_key, "
        "content_digest, canonical FROM jyppx_events WHERE stream_id = ?1 AND "
        "(event_id = ?2 OR producer_sequence = ?3 OR idempotency_key = ?4)",
        "events.identity", &control);
    Statement stream(impl_->database_,
                     "SELECT run_id, last_sequence, terminal FROM "
                     "jyppx_event_streams WHERE stream_id = ?1",
                     "events.stream", &control);
    Statement new_stream(impl_->database_,
                         "INSERT INTO jyppx_event_streams(stream_id, run_id, "
                         "last_sequence, terminal) "
                         "VALUES(?1, ?2, 0, 0)",
                         "events.new_stream", &control);
    Statement advance_stream(impl_->database_,
                             "UPDATE jyppx_event_streams SET last_sequence = "
                             "?2, terminal = ?3 WHERE stream_id = ?1",
                             "events.advance_stream", &control);
    Statement insert(
        impl_->database_,
        "INSERT INTO jyppx_events(run_id, run_sequence, stream_id, "
        "producer_sequence, "
        "event_id, idempotency_key, payload_kind, content_digest, canonical) "
        "VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        "events.insert", &control);
    Statement find_definition(
        impl_->database_,
        "SELECT definition_digest, canonical FROM jyppx_metric_definitions "
        "WHERE metric_id = ?1 AND definition_version = ?2",
        "metrics.definition", &control);
    Statement insert_definition(
        impl_->database_,
        "INSERT INTO jyppx_metric_definitions(metric_id, definition_version, "
        "definition_digest, canonical) "
        "VALUES(?1, ?2, ?3, ?4)",
        "metrics.new_definition", &control);
    Statement insert_point(
        impl_->database_,
        "INSERT INTO jyppx_metric_points(run_id, run_sequence, point_index, "
        "metric_id, "
        "definition_version, step_kind, step_key, value_status) VALUES(?1, ?2, "
        "?3, ?4, ?5, ?6, ?7, ?8)",
        "metrics.insert", &control);
    std::optional<std::uint64_t> definition_count;
    for (const auto& item : encoded) {
      control.check();
      const auto& envelope = *item.envelope;
      const auto& identity = envelope.identity;
      duplicate.bind(1, identity.stream_id);
      duplicate.bind(2, identity.event_id);
      duplicate.bind(3, identity.sequence);
      duplicate.bind(4, identity.idempotency_key);
      std::optional<std::uint64_t> existing_sequence;
      while (duplicate.row()) {
        if (existing_sequence ||
            duplicate.text(0,
                           impl_->options_.run_metadata_limits.max_id_bytes) !=
                run_id ||
            duplicate.text(2, limits.telemetry.max_identifier_bytes) !=
                identity.event_id ||
            duplicate.integer(3) != identity.sequence ||
            duplicate.text(4, limits.telemetry.max_identifier_bytes) !=
                identity.idempotency_key ||
            duplicate.text(5, 64U) != item.digest ||
            duplicate.blob(6, limits.telemetry.max_envelope_bytes) !=
                item.canonical)
          event_conflict(
              "events.identity",
              "event ID, operation key, or producer sequence conflict");
        existing_sequence = duplicate.integer(1);
      }
      duplicate.reset();
      if (existing_sequence) {
        result.receipts.push_back(
            {.run_sequence = *existing_sequence,
             .disposition = TelemetryDisposition::duplicate});
        continue;
      }
      if (is_terminal_run_status(run.status))
        invalid("events.append", "nonterminal run for new evidence",
                "terminal run");
      stream.bind(1, identity.stream_id);
      std::uint64_t previous = 0U;
      if (stream.row()) {
        if (stream.text(0, impl_->options_.run_metadata_limits.max_id_bytes) !=
            run_id)
          event_conflict("events.stream",
                         "stream already belongs to a different run");
        previous = stream.integer(1);
        if (stream.integer(2) != 0U)
          invalid("events.stream", "open producer stream", "terminal stream");
      } else {
        consume_quota(state.streams, 1U, limits.max_streams_per_run,
                      "events.streams");
        new_stream.bind(1, identity.stream_id);
        new_stream.bind(2, run_id);
        new_stream.done();
        new_stream.reset();
      }
      stream.reset();
      if (previous == static_cast<std::uint64_t>(
                          std::numeric_limits<sqlite3_int64>::max()) ||
          identity.sequence != previous + 1U)
        event_conflict("events.sequence",
                       "producer sequence gap or regression");
      consume_quota(state.last, 1U, limits.max_events_per_run, "events.count");
      consume_quota(state.bytes, item.canonical.size(),
                    limits.max_event_bytes_per_run, "events.content_bytes");
      const auto points = event_metrics(envelope);
      consume_quota(state.metrics, points.size(),
                    limits.max_metric_points_per_run, "metrics.count");
      if (const auto* artifact =
              std::get_if<TelemetryArtifactReference>(&envelope.payload)) {
        RunArtifactReference reference;
        reference.reference_id =
            "telemetry." +
            telemetry_idempotency_digest(identity, limits.telemetry).hex;
        reference.role = artifact->role;
        reference.artifact = artifact->artifact;
        reference.attached_at_unix_ms = run.updated_at_unix_ms;
        impl_->attach_unlocked(
            run_id, std::span<const RunArtifactReference>(&reference, 1U), stop,
            deadline);
      }
      insert.bind(1, run_id);
      insert.bind(2, state.last);
      insert.bind(3, identity.stream_id);
      insert.bind(4, identity.sequence);
      insert.bind(5, identity.event_id);
      insert.bind(6, identity.idempotency_key);
      insert.bind(
          7, static_cast<std::uint64_t>(*telemetry_payload_kind(envelope)));
      insert.bind(8, item.digest);
      insert.bind_blob(9, item.canonical);
      insert.done();
      insert.reset();
      for (std::size_t point_index = 0U; point_index < points.size();
           ++point_index) {
        control.check();
        const auto& point = points[point_index];
        const auto definition_bytes = encode_telemetry_metric_definition(
            point.definition, limits.telemetry);
        const auto digest = sha256_hex(definition_bytes);
        find_definition.bind(1, point.definition.metric_id);
        find_definition.bind(
            2, static_cast<std::uint64_t>(point.definition.version));
        if (find_definition.row()) {
          if (find_definition.text(0, 64U) != digest ||
              find_definition.blob(1, limits.telemetry.max_envelope_bytes) !=
                  definition_bytes)
            event_conflict("metrics.definition",
                           "immutable definition ID/version was redefined");
        } else {
          if (!definition_count) {
            Statement count(impl_->database_,
                            "SELECT COUNT(*) FROM jyppx_metric_definitions",
                            "metrics.definition_count", &control);
            if (!count.row())
              corrupt("metrics.definition_count", "missing count");
            definition_count = count.integer(0);
          }
          consume_quota(*definition_count, 1U, limits.max_metric_definitions,
                        "metrics.definitions");
          insert_definition.bind(1, point.definition.metric_id);
          insert_definition.bind(
              2, static_cast<std::uint64_t>(point.definition.version));
          insert_definition.bind(3, digest);
          insert_definition.bind_blob(4, definition_bytes);
          insert_definition.done();
          insert_definition.reset();
        }
        find_definition.reset();
        insert_point.bind(1, run_id);
        insert_point.bind(2, state.last);
        insert_point.bind(3, static_cast<std::uint64_t>(point_index));
        insert_point.bind(4, point.definition.metric_id);
        insert_point.bind(5,
                          static_cast<std::uint64_t>(point.definition.version));
        insert_point.bind(6, static_cast<std::uint64_t>(point.step.kind));
        insert_point.bind_blob(7, unsigned_key(point.step.value));
        insert_point.bind(8,
                          static_cast<std::uint64_t>(point.observation.status));
        insert_point.done();
        insert_point.reset();
      }
      const auto* lifecycle =
          std::get_if<TelemetryLifecycle>(&envelope.payload);
      advance_stream.bind(1, identity.stream_id);
      advance_stream.bind(2, identity.sequence);
      advance_stream.bind(
          3, lifecycle != nullptr && lifecycle->terminal ? 1U : 0U);
      advance_stream.done();
      advance_stream.reset();
      result.receipts.push_back(
          {.run_sequence = state.last,
           .disposition = TelemetryDisposition::accepted});
    }
    {
      Statement save(
          impl_->database_,
          "INSERT INTO "
          "jyppx_run_event_state(run_id,last_sequence,metric_count,content_"
          "bytes,batch_count,stream_count) "
          "VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(run_id) DO UPDATE SET "
          "last_sequence=excluded.last_sequence,metric_count=excluded.metric_"
          "count,"
          "content_bytes=excluded.content_bytes,batch_count=excluded.batch_"
          "count,stream_count=excluded.stream_count",
          "events.save_state", &control);
      save.bind(1, run_id);
      save.bind(2, state.last);
      save.bind(3, state.metrics);
      save.bind(4, state.bytes);
      save.bind(5, state.batches);
      save.bind(6, state.streams);
      save.done();
      Statement batch(impl_->database_,
                      "INSERT INTO "
                      "jyppx_event_batches(run_id,batch_key,request_digest,"
                      "input_count,committed_through) "
                      "VALUES(?1,?2,?3,?4,?5)",
                      "events.save_batch", &control);
      batch.bind(1, run_id);
      batch.bind(2, batch_key);
      batch.bind(3, request_digest);
      batch.bind(4, static_cast<std::uint64_t>(envelopes.size()));
      batch.bind(5, state.last);
      batch.done();
      Statement receipt(impl_->database_,
                        "INSERT INTO "
                        "jyppx_event_batch_items(run_id,batch_key,ordinal,run_"
                        "sequence) VALUES(?1,?2,?3,?4)",
                        "events.save_receipt", &control);
      for (std::size_t index = 0U; index < result.receipts.size(); ++index) {
        receipt.bind(1, run_id);
        receipt.bind(2, batch_key);
        receipt.bind(3, static_cast<std::uint64_t>(index));
        receipt.bind(4, result.receipts[index].run_sequence);
        receipt.done();
        receipt.reset();
      }
    }
    result.committed_through = state.last;
    transaction.commit();
    // Do not check cancellation after a successful commit: its durable receipt
    // must win over a stop arriving after the transaction's decision point.
    return result;
  } catch (...) {
    control.check();
    throw;
  }
}

RunEventPage SQLiteRunStore::events(std::string_view run_id,
                                    const RunEventPageRequest& request,
                                    std::stop_token stop,
                                    TelemetryDeadline deadline) const {
  check_telemetry_operation(stop, deadline);
  impl_->validate_run_id(run_id);
  const auto& limits = impl_->options_.event_limits;
  if (request.limit == 0U || request.limit > limits.max_page_events)
    invalid("events.limit", "positive bounded page count", "out of range");
  validate_integer(request.after_sequence, "events.after_sequence");
  if (request.kind && (static_cast<unsigned>(*request.kind) < 1U ||
                       static_cast<unsigned>(*request.kind) > 6U))
    invalid("events.kind", "known neutral payload kind", "unknown kind");
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, false, &control);
    (void)impl_->require_run(run_id);
    RunEventPage page;
    const auto state = read_event_state(impl_->database_, run_id, &control);
    page.snapshot_through = snapshot_boundary(
        state.last, request.snapshot_through, request.after_sequence);
    std::string sql =
        "SELECT run_sequence, content_digest, canonical FROM jyppx_events "
        "WHERE run_id = ?1 AND run_sequence > ?2 AND run_sequence <= ?3 ";
    if (request.kind) sql += "AND payload_kind = ?4 ";
    sql += "ORDER BY run_sequence LIMIT ?5";
    Statement query(impl_->database_, sql.c_str(), "events.page", &control);
    query.bind(1, run_id);
    query.bind(2, request.after_sequence);
    query.bind(3, page.snapshot_through);
    if (request.kind) query.bind(4, static_cast<std::uint64_t>(*request.kind));
    query.bind(5, static_cast<std::uint64_t>(request.limit) + 1U);
    std::uint64_t bytes = 0U;
    bool more = false;
    while (query.row()) {
      if (page.events.size() == request.limit) {
        more = true;
        break;
      }
      auto canonical = query.blob(2, limits.telemetry.max_envelope_bytes);
      if (canonical.size() > limits.max_page_bytes - bytes) {
        if (page.events.empty())
          exhausted("events.page", "page byte bound", "single event too large");
        more = true;
        break;
      }
      auto envelope = decode_stored_event(canonical, query.text(1, 64U), run_id,
                                          limits.telemetry);
      if (request.kind && telemetry_payload_kind(envelope) != request.kind)
        corrupt("events.page",
                "payload-kind index disagrees with source envelope");
      bytes += canonical.size();
      page.events.push_back(
          {.run_sequence = query.integer(0), .envelope = std::move(envelope)});
    }
    if (more) page.next_after = page.events.back().run_sequence;
    transaction.commit();
    return page;
  } catch (...) {
    control.check();
    throw;
  }
}

RunMetricPage SQLiteRunStore::metrics(std::string_view run_id,
                                      const RunMetricPageRequest& request,
                                      std::stop_token stop,
                                      TelemetryDeadline deadline) const {
  check_telemetry_operation(stop, deadline);
  impl_->validate_run_id(run_id);
  const auto& limits = impl_->options_.event_limits;
  validate_text(request.metric_id, limits.telemetry.max_identifier_bytes,
                "metrics.metric_id");
  if (request.limit == 0U || request.limit > limits.max_page_metric_points ||
      (request.definition_version && *request.definition_version == 0U) ||
      (request.after.run_sequence == 0U && request.after.point_index != 0U))
    invalid("metrics.page",
            "valid positive limits/version and exclusive cursor",
            "invalid request");
  if (request.step_kind && (static_cast<unsigned>(*request.step_kind) < 1U ||
                            static_cast<unsigned>(*request.step_kind) > 5U))
    invalid("metrics.step_kind", "known neutral counter kind", "unknown kind");
  validate_integer(request.after.run_sequence, "metrics.after_sequence");
  validate_integer(request.after.point_index, "metrics.after_index");
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, false, &control);
    (void)impl_->require_run(run_id);
    RunMetricPage page;
    const auto state = read_event_state(impl_->database_, run_id, &control);
    page.snapshot_through = snapshot_boundary(
        state.last, request.snapshot_through, request.after.run_sequence);
    std::string sql =
        "SELECT p.run_sequence,p.point_index,p.metric_id,p.definition_version,"
        "p.step_kind,p.step_key,p.value_status,e.content_digest,e.canonical "
        "FROM jyppx_metric_points p JOIN jyppx_events e ON e.run_id=p.run_id "
        "AND e.run_sequence=p.run_sequence WHERE p.run_id=?1 AND "
        "p.metric_id=?2 "
        "AND (p.run_sequence>?3 OR (p.run_sequence=?3 AND p.point_index>?4)) "
        "AND p.run_sequence<=?5 ";
    if (request.definition_version) sql += "AND p.definition_version=?6 ";
    if (request.step_kind) sql += "AND p.step_kind=?7 ";
    sql += "ORDER BY p.run_sequence,p.point_index LIMIT ?8";
    Statement query(impl_->database_, sql.c_str(), "metrics.page", &control);
    query.bind(1, run_id);
    query.bind(2, request.metric_id);
    query.bind(3, request.after.run_sequence);
    query.bind(4, request.after.point_index);
    query.bind(5, page.snapshot_through);
    if (request.definition_version)
      query.bind(6, static_cast<std::uint64_t>(*request.definition_version));
    if (request.step_kind)
      query.bind(7, static_cast<std::uint64_t>(*request.step_kind));
    query.bind(8, static_cast<std::uint64_t>(request.limit) + 1U);
    std::optional<TelemetryEnvelope> cached_envelope;
    std::uint64_t cached_sequence = 0U;
    std::uint64_t cached_bytes = 0U;
    std::uint64_t page_bytes = 0U;
    bool more = false;
    while (query.row()) {
      if (page.points.size() == request.limit) {
        more = true;
        break;
      }
      const auto sequence = query.integer(0);
      if (!cached_envelope || cached_sequence != sequence) {
        auto canonical = query.blob(8, limits.telemetry.max_envelope_bytes);
        cached_bytes = canonical.size();
        cached_envelope = decode_stored_event(canonical, query.text(7, 64U),
                                              run_id, limits.telemetry);
        cached_sequence = sequence;
      }
      // Charge the complete source envelope for each returned projection. This
      // deliberately overestimates content bytes and bounds large definitions.
      if (cached_bytes > limits.max_page_bytes - page_bytes) {
        if (page.points.empty())
          exhausted("metrics.page", "page byte bound",
                    "single source too large");
        more = true;
        break;
      }
      const auto point_index = query.integer(1);
      const auto source_points = event_metrics(*cached_envelope);
      if (point_index >= source_points.size())
        corrupt("metrics.page", "point index outside source");
      const auto& point = source_points[point_index];
      const auto expected_key = unsigned_key(point.step.value);
      const auto key = query.blob(5, 8U);
      if (query.text(2, limits.telemetry.max_identifier_bytes) !=
              point.definition.metric_id ||
          query.integer(3) != point.definition.version ||
          query.integer(4) != static_cast<std::uint64_t>(point.step.kind) ||
          key.size() != expected_key.size() ||
          !std::equal(key.begin(), key.end(), expected_key.begin()) ||
          query.integer(6) !=
              static_cast<std::uint64_t>(point.observation.status))
        corrupt("metrics.page", "metric index disagrees with canonical source");
      StoredRunMetricPoint stored;
      stored.cursor = {.run_sequence = sequence, .point_index = point_index};
      stored.event_identity = cached_envelope->identity;
      stored.time = cached_envelope->time;
      stored.correlation = cached_envelope->correlation;
      stored.attributes = cached_envelope->attributes;
      if (const auto* resource =
              std::get_if<TelemetryResourceSample>(&cached_envelope->payload))
        stored.resource_id = resource->resource_id;
      stored.point = point;
      page_bytes += cached_bytes;
      page.points.push_back(std::move(stored));
    }
    if (more) page.next_after = page.points.back().cursor;
    transaction.commit();
    return page;
  } catch (...) {
    control.check();
    throw;
  }
}

namespace {

[[noreturn]] void backup_io(
    std::string operation, const std::filesystem::path& path,
    std::error_code code = std::make_error_code(std::errc::io_error)) {
  throw IoError(
      "Run backup filesystem operation failed",
      ErrorDetails{
          .component = "experiment.run_backup",
          .operation = std::move(operation),
          .expected = "a complete readable backup or new writable destination",
          .actual = utf8_path(path) + ": " + code.message(),
          .hint = "Preserve existing stores and retry with a new local "
                  "destination."});
}

void sync_backup_path(const std::filesystem::path& path, bool directory) {
#ifdef _WIN32
  if (directory) return;  // Final directory rename uses WRITE_THROUGH.
  const auto handle =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    backup_io("sync.open", path,
              {static_cast<int>(GetLastError()), std::system_category()});
  const auto result = FlushFileBuffers(handle);
  const auto code = GetLastError();
  CloseHandle(handle);
  if (!result)
    backup_io("sync.file", path,
              {static_cast<int>(code), std::system_category()});
#else
  const auto descriptor = ::open(
      path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
  if (descriptor < 0)
    backup_io("sync.open", path, {errno, std::generic_category()});
  const auto result = ::fsync(descriptor);
  const auto code = errno;
  ::close(descriptor);
  if (result != 0)
    backup_io("sync.path", path, {code, std::generic_category()});
#endif
}

void durable_backup_parent(const std::filesystem::path& leaf) {
  std::error_code error;
  std::vector<std::filesystem::path> missing;
  for (auto path = leaf; !path.empty(); path = path.parent_path()) {
    const auto present = std::filesystem::exists(path, error);
    if (error) backup_io("parent.exists", path, error);
    if (present) break;
    missing.push_back(path);
  }
  std::filesystem::create_directories(leaf, error);
  if (error) backup_io("parent.create", leaf, error);
  for (const auto& path : missing) {
    sync_backup_path(path, true);
    if (path.has_parent_path()) sync_backup_path(path.parent_path(), true);
  }
}

bool path_is_within(const std::filesystem::path& child,
                    const std::filesystem::path& parent) {
  auto left = child.begin();
  for (auto right = parent.begin(); right != parent.end(); ++right, ++left) {
    if (left == child.end()) return false;
#ifdef _WIN32
    const auto& a = left->native();
    const auto& b = right->native();
    if (CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(),
                             static_cast<int>(b.size()), TRUE) != CSTR_EQUAL)
      return false;
#else
    if (*left != *right) return false;
#endif
  }
  return true;
}

std::filesystem::path resolved_backup_path(const std::filesystem::path& path) {
  if (path.empty()) invalid("backup.path", "nonempty local path", "empty");
  std::error_code error;
  const auto absolute = std::filesystem::absolute(path, error);
  if (error) backup_io("path.absolute", path, error);
  auto resolved = std::filesystem::weakly_canonical(absolute, error);
  if (error) backup_io("path.resolve", path, error);
  if (resolved.filename().empty() || resolved == resolved.root_path())
    invalid("backup.path", "a dedicated named directory", "filesystem root");
  return resolved;
}

class BackupStage final {
 public:
  explicit BackupStage(std::filesystem::path target)
      : target_(std::move(target)) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(target_, error);
    if (error && error != std::errc::no_such_file_or_directory)
      backup_io("destination.inspect", target_, error);
    if (std::filesystem::exists(status))
      invalid("backup.destination", "a nonexistent destination",
              "already exists");
    durable_backup_parent(target_.parent_path());
    for (unsigned attempt = 0U; attempt < 64U; ++attempt) {
      auto candidate =
          target_.parent_path() /
          (".jyppx-backup-stage-" + std::to_string(std::random_device{}()) +
           "-" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()));
      if (std::filesystem::create_directory(candidate, error)) {
        path_ = std::move(candidate);
        sync_backup_path(path_.parent_path(), true);
        return;
      }
      if (error) backup_io("stage.create", candidate, error);
    }
    exhausted("backup.stage", "exclusive staging directory", "name collisions");
  }
  ~BackupStage() {
    if (!published_ && !path_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }
  BackupStage(const BackupStage&) = delete;
  BackupStage& operator=(const BackupStage&) = delete;
  const std::filesystem::path& path() const noexcept { return path_; }
  void publish() {
    sync_backup_path(path_, true);
#ifdef _WIN32
    if (!MoveFileExW(path_.c_str(), target_.c_str(), MOVEFILE_WRITE_THROUGH))
      backup_io("stage.publish", target_,
                {static_cast<int>(GetLastError()), std::system_category()});
#elif defined(__linux__)
    // Linux's no-replace primitive protects an existing/racing target
    // directory.
    if (::syscall(SYS_renameat2, AT_FDCWD, path_.c_str(), AT_FDCWD,
                  target_.c_str(), 1U /* RENAME_NOREPLACE */) != 0)
      backup_io("stage.publish", target_, {errno, std::generic_category()});
#else
    throw CompatibilityError(
        "Atomic no-replace backup publication requires Windows or Linux");
#endif
    published_ = true;
    sync_backup_path(target_.parent_path(), true);
  }

 private:
  std::filesystem::path target_;
  std::filesystem::path path_;
  bool published_ = false;
};

struct BackupFileIdentity {
  std::uint64_t size = 0U;
  std::string digest;
};

BackupFileIdentity hash_or_copy_backup_file(
    const std::filesystem::path& source,
    const std::optional<std::filesystem::path>& target, std::uint64_t maximum,
    std::stop_token stop, TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  reject_link(source);
  std::ifstream input(source, std::ios::binary | std::ios::ate);
  if (!input) backup_io("copy.open", source);
  const auto length = input.tellg();
  if (length < 0) backup_io("copy.size", source);
  const auto size = static_cast<std::uint64_t>(length);
  if (size > maximum)
    exhausted("backup.file", "configured byte bound", "oversized file");
  input.seekg(0);
  std::ofstream output;
  if (target) {
    output.open(*target, std::ios::binary | std::ios::trunc);
    if (!output) backup_io("copy.destination", *target);
  }
  Sha256 hash;
  std::vector<std::byte> buffer(1024U * 1024U);
  std::uint64_t total = 0U;
  while (input) {
    check_telemetry_operation(stop, deadline);
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count <= 0) break;
    const auto bytes = static_cast<std::uint64_t>(count);
    if (bytes > maximum - std::min(total, maximum))
      exhausted("backup.file", "configured byte bound", "growing file");
    total += bytes;
    hash.update(std::span<const std::byte>(buffer.data(),
                                           static_cast<std::size_t>(count)));
    if (target) {
      output.write(reinterpret_cast<const char*>(buffer.data()), count);
      if (!output) backup_io("copy.write", *target);
    }
  }
  if (!input.eof() || total != size) backup_io("copy.source_changed", source);
  if (target) {
    output.close();
    if (!output) backup_io("copy.close", *target);
    sync_backup_path(*target, false);
  }
  return {.size = size, .digest = sha256_digest_hex(hash.finish())};
}

class BackupDatabase final {
 public:
  BackupDatabase(const std::filesystem::path& path, bool writable) {
    const auto text = utf8_path(path);
    const auto result =
        sqlite3_open_v2(text.c_str(), &database_,
                        (writable ? SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
                                  : SQLITE_OPEN_READONLY) |
                            SQLITE_OPEN_FULLMUTEX,
                        nullptr);
    if (result != SQLITE_OK) {
      try {
        sqlite_failure(database_, result, "backup.open");
      } catch (...) {
        sqlite3_close(database_);
        database_ = nullptr;
        throw;
      }
    }
  }
  ~BackupDatabase() { sqlite3_close(database_); }
  BackupDatabase(const BackupDatabase&) = delete;
  BackupDatabase& operator=(const BackupDatabase&) = delete;
  sqlite3* get() const noexcept { return database_; }

 private:
  sqlite3* database_ = nullptr;
};

void sqlite_snapshot(const std::filesystem::path& source,
                     const std::filesystem::path& target,
                     const SQLiteRunStoreOptions& options, std::stop_token stop,
                     TelemetryDeadline deadline) {
  BackupDatabase from(source, false);
  BackupDatabase to(target, true);
  OperationControl control(to.get(), options.busy_timeout, stop, deadline);
  const auto page_size =
      scalar_integer(from.get(), "PRAGMA page_size", "backup.page_size");
  const auto page_count =
      scalar_integer(from.get(), "PRAGMA page_count", "backup.page_count");
  if (page_size == 0U ||
      page_count > options.max_backup_database_bytes / page_size)
    exhausted("backup.database", "configured snapshot byte bound",
              "oversized database");
  execute(to.get(), "PRAGMA synchronous=FULL", "backup.synchronous", &control);
  auto* copy = sqlite3_backup_init(to.get(), "main", from.get(), "main");
  if (copy == nullptr)
    sqlite_failure(to.get(), sqlite3_errcode(to.get()), "backup.init",
                   &control);
  try {
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
      control.check();
      const auto result = sqlite3_backup_step(copy, 128);
      if (result == SQLITE_DONE) break;
      if (result != SQLITE_OK && result != SQLITE_BUSY &&
          result != SQLITE_LOCKED)
        sqlite_failure(to.get(), result, "backup.step", &control);
      if (result == SQLITE_BUSY || result == SQLITE_LOCKED) {
        if (std::chrono::steady_clock::now() - started >= options.busy_timeout)
          sqlite_failure(to.get(), result, "backup.step", &control);
        sqlite3_sleep(1);
      }
      std::error_code error;
      const auto size = std::filesystem::file_size(target, error);
      if (error) backup_io("backup.size", target, error);
      if (size > options.max_backup_database_bytes)
        exhausted("backup.database", "configured snapshot byte bound",
                  "oversized database");
    }
    const auto result = sqlite3_backup_finish(copy);
    copy = nullptr;
    if (result != SQLITE_OK)
      sqlite_failure(to.get(), result, "backup.finish", &control);
    Statement journal(to.get(), "PRAGMA journal_mode=DELETE", "backup.journal",
                      &control);
    if (!journal.row() || journal.text(0, 16U) != "delete")
      incompatible("standalone backup journal mode unavailable");
  } catch (...) {
    if (copy != nullptr) sqlite3_backup_finish(copy);
    throw;
  }
}

struct BackupManifest {
  BackupFileIdentity database;
  std::vector<ArtifactRef> artifacts;
};

void finalize_standalone_database(const std::filesystem::path& path,
                                  const SQLiteRunStoreOptions& options,
                                  std::stop_token stop,
                                  TelemetryDeadline deadline) {
  BackupDatabase database(path, true);
  OperationControl control(database.get(), options.busy_timeout, stop,
                           deadline);
  Statement journal(database.get(), "PRAGMA journal_mode=DELETE",
                    "backup.finalize_journal", &control);
  if (!journal.row() || journal.text(0, 16U) != "delete")
    incompatible("backup cannot finalize a standalone database");
}

std::string encode_backup_manifest(const BackupManifest& manifest,
                                   std::uint64_t maximum) {
  std::string bytes;
  const auto add = [&](std::string line) {
    if (bytes.size() > maximum || line.size() > maximum - bytes.size())
      exhausted("backup.manifest", "configured manifest byte bound",
                "oversized manifest");
    bytes += line;
  };
  add("JYPPX-RUN-BACKUP\n1\n2\n");
  add(std::to_string(manifest.database.size) + " " + manifest.database.digest +
      "\n");
  add(std::to_string(manifest.artifacts.size()) + "\n");
  for (const auto& artifact : manifest.artifacts)
    add(std::to_string(artifact.size_bytes) + " " + artifact.digest.hex + "\n");
  const auto checksum =
      sha256_hex(std::as_bytes(std::span(bytes.data(), bytes.size())));
  add(checksum + "\n");
  return bytes;
}

BackupManifest decode_backup_manifest(const std::filesystem::path& root,
                                      const SQLiteRunStoreOptions& options,
                                      std::stop_token stop,
                                      TelemetryDeadline deadline) {
  const auto path = root / "manifest.jyppx-backup";
  const auto identity = hash_or_copy_backup_file(
      path, std::nullopt, options.max_backup_manifest_bytes, stop, deadline);
  if (identity.size >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    exhausted("backup.manifest", "representable input", "oversized manifest");
  std::ifstream input(path, std::ios::binary);
  std::string bytes(static_cast<std::size_t>(identity.size), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    backup_io("manifest.read", path);
  if (sha256_hex(std::as_bytes(std::span(bytes.data(), bytes.size()))) !=
      identity.digest)
    corrupt("backup.manifest", "manifest changed while loading");
  if (bytes.size() < 65U || bytes.back() != '\n')
    corrupt("backup.manifest", "truncated checksum");
  const auto body = std::string_view(bytes).substr(0U, bytes.size() - 65U);
  if (sha256_hex(std::as_bytes(std::span(body.data(), body.size()))) !=
      std::string_view(bytes).substr(bytes.size() - 65U, 64U))
    corrupt("backup.manifest", "checksum mismatch");
  std::size_t offset = 0U;
  const auto line = [&]() {
    const auto end = body.find('\n', offset);
    if (end == std::string_view::npos || end - offset > 128U)
      corrupt("backup.manifest", "missing or oversized field");
    const auto result = body.substr(offset, end - offset);
    offset = end + 1U;
    return result;
  };
  const auto number = [&](std::string_view value) {
    std::uint64_t result = 0U;
    const auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != value.data() + value.size() ||
        std::to_string(result) != value)
      corrupt("backup.manifest", "noncanonical unsigned number");
    return result;
  };
  const auto file = [&](std::string_view value) {
    const auto space = value.find(' ');
    if (space == std::string_view::npos)
      corrupt("backup.manifest", "invalid file record");
    BackupFileIdentity result{.size = number(value.substr(0U, space)),
                              .digest = std::string(value.substr(space + 1U))};
    validate_artifact_digest({.hex = result.digest});
    return result;
  };
  if (line() != "JYPPX-RUN-BACKUP" || line() != "1" || line() != "2")
    incompatible("unsupported backup format or database schema");
  BackupManifest manifest;
  manifest.database = file(line());
  if (manifest.database.size == 0U ||
      manifest.database.size > options.max_backup_database_bytes)
    exhausted("backup.database", "positive bounded database bytes",
              "out of range");
  const auto count = number(line());
  if (count > options.artifact_limits.max_object_files ||
      count > body.size() / 67U)
    exhausted("backup.artifacts", "bounded artifact manifest",
              "oversized object count");
  manifest.artifacts.reserve(static_cast<std::size_t>(count));
  std::uint64_t total = 0U;
  std::string previous;
  for (std::uint64_t index = 0U; index < count; ++index) {
    check_telemetry_operation(stop, deadline);
    const auto record = file(line());
    if (record.digest <= previous)
      corrupt("backup.manifest", "unordered or duplicate digest");
    if (record.size > options.artifact_limits.max_artifact_bytes)
      exhausted("backup.artifact", "configured artifact bound",
                "oversized object");
    consume_quota(total, record.size, options.artifact_limits.max_total_bytes,
                  "backup.artifact_bytes");
    manifest.artifacts.push_back({.digest = {.hex = record.digest},
                                  .size_bytes = record.size,
                                  .logical_name = {},
                                  .media_type = "application/octet-stream",
                                  .producer = {}});
    previous = record.digest;
  }
  if (offset != body.size()) corrupt("backup.manifest", "trailing records");
  return manifest;
}

RunBackupReport copy_backup_blobs(FileArtifactStore& source,
                                  FileArtifactStore& target,
                                  std::span<const ArtifactRef> artifacts,
                                  std::stop_token stop,
                                  TelemetryDeadline deadline) {
  RunBackupReport report;
  std::vector<std::byte> buffer(1024U * 1024U);
  for (const auto& artifact : artifacts) {
    check_telemetry_operation(stop, deadline);
    auto reader = source.open(artifact.digest, stop, deadline);
    if (reader->size_bytes() != artifact.size_bytes)
      corrupt("backup.artifact", "manifest size mismatch");
    auto upload = target.begin({.logical_name = "backup-content",
                                .media_type = "application/octet-stream",
                                .producer = "experiment.run_backup",
                                .expected_size_bytes = artifact.size_bytes});
    for (;;) {
      check_telemetry_operation(stop, deadline);
      const auto count = reader->read(buffer);
      if (count == 0U) break;
      upload->write(std::span<const std::byte>(buffer.data(), count));
    }
    check_telemetry_operation(stop, deadline);
    target.commit(*upload, artifact.digest, stop, deadline);
    ++report.artifact_count;
    report.artifact_bytes += artifact.size_bytes;
  }
  return report;
}

void validate_staged_backup(const std::filesystem::path& root,
                            const SQLiteRunStoreOptions& options,
                            std::span<const ArtifactRef> expected,
                            std::stop_token stop, TelemetryDeadline deadline) {
  SQLiteRunStore store(root, options, stop, deadline);
  if (!store.integrity_check(true, stop, deadline).ok())
    corrupt("backup.integrity", "staged snapshot failed validation");
  check_telemetry_operation(stop, deadline);
  // Compare exact DB-visible content membership, not just copied file count.
  BackupDatabase database(root / "runs.sqlite3", false);
  OperationControl control(database.get(), options.busy_timeout, stop,
                           deadline);
  Statement live(database.get(),
                 "SELECT b.digest,b.size_bytes FROM jyppx_blobs b WHERE EXISTS "
                 "(SELECT 1 FROM jyppx_run_artifacts a WHERE "
                 "a.digest=b.digest) ORDER BY b.digest",
                 "backup.live_set", &control);
  std::size_t index = 0U;
  while (live.row()) {
    if (index >= expected.size() ||
        live.text(0, 64U) != expected[index].digest.hex ||
        live.integer(1) != expected[index].size_bytes)
      corrupt("backup.live_set",
              "manifest differs from referenced database content");
    ++index;
  }
  if (index != expected.size())
    corrupt("backup.live_set",
            "manifest contains unreferenced or omitted content");
}

}  // namespace

RunBackupReport SQLiteRunStore::backup(const std::filesystem::path& destination,
                                       std::stop_token stop,
                                       TelemetryDeadline deadline) const {
  check_telemetry_operation(stop, deadline);
  const auto target = resolved_backup_path(destination);
  if (path_is_within(target, impl_->root_) ||
      path_is_within(impl_->root_, target))
    invalid("backup.destination", "a destination separate from the live store",
            "overlapping paths");
  auto lock = event_lock(impl_->mutex_, stop, deadline);
  OperationControl control(impl_->database_, impl_->options_.busy_timeout, stop,
                           deadline);
  try {
    Transaction transaction(impl_->database_, true, &control);
    RunStoreIntegrityReport audit;
    verify_event_integrity(impl_->database_, impl_->options_, audit, &control);
    if (!audit.ok())
      corrupt("backup.integrity", "event relationships are inconsistent");
    {
      Statement integrity(impl_->database_, "PRAGMA integrity_check(1)",
                          "backup.database_integrity", &control);
      if (!integrity.row() || integrity.text(0, 4096U) != "ok")
        corrupt("backup.database_integrity",
                "source SQLite integrity check failed");
      Statement foreign_keys(impl_->database_, "PRAGMA foreign_key_check",
                             "backup.foreign_keys", &control);
      if (foreign_keys.row())
        corrupt("backup.foreign_keys", "source relationships are inconsistent");
    }
    BackupManifest manifest;
    manifest.artifacts = impl_->live_blobs();
    BackupStage stage(target);
    // The source writer transaction prevents append, attach, and coordinated
    // CAS collection. A separate read connection can use sqlite3_backup while
    // that writer reservation is held, including WAL-resident committed pages.
    sqlite_snapshot(impl_->database_path_, stage.path() / "runs.sqlite3",
                    impl_->options_, stop, deadline);
    FileArtifactStore target_cas(stage.path() / "artifacts",
                                 impl_->options_.artifact_limits);
    auto report = copy_backup_blobs(impl_->artifacts_, target_cas,
                                    manifest.artifacts, stop, deadline);
    validate_staged_backup(stage.path(), impl_->options_, manifest.artifacts,
                           stop, deadline);
    finalize_standalone_database(stage.path() / "runs.sqlite3", impl_->options_,
                                 stop, deadline);
    manifest.database = hash_or_copy_backup_file(
        stage.path() / "runs.sqlite3", std::nullopt,
        impl_->options_.max_backup_database_bytes, stop, deadline);
    report.database_bytes = manifest.database.size;
    const auto encoded = encode_backup_manifest(
        manifest, impl_->options_.max_backup_manifest_bytes);
    const auto manifest_path = stage.path() / "manifest.jyppx-backup";
    std::ofstream output(manifest_path, std::ios::binary | std::ios::trunc);
    output.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    output.close();
    if (!output) backup_io("manifest.write", manifest_path);
    sync_backup_path(stage.path() / "runs.sqlite3", false);
    sync_backup_path(manifest_path, false);
    transaction.commit();
    control.check();
    stage.publish();
    return report;
  } catch (...) {
    control.check();
    throw;
  }
}

RunBackupReport SQLiteRunStore::restore_backup(
    const std::filesystem::path& backup_root,
    const std::filesystem::path& target_root, SQLiteRunStoreOptions options,
    std::stop_token stop, TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  // Validate all option relationships before creating the target/staging root.
  const auto source = canonical_root(backup_root, options, stop, deadline);
  const auto target = resolved_backup_path(target_root);
  if (path_is_within(source, target) || path_is_within(target, source))
    invalid("restore.destination", "a separate new destination",
            "overlapping paths");
  const auto manifest = decode_backup_manifest(source, options, stop, deadline);
  const auto identity = hash_or_copy_backup_file(
      source / "runs.sqlite3", std::nullopt, options.max_backup_database_bytes,
      stop, deadline);
  if (identity.size != manifest.database.size ||
      identity.digest != manifest.database.digest)
    corrupt("restore.database", "database differs from backup manifest");
  // Published backups are standalone DB files; sidecars would be unmanifested
  // changes. Refuse them instead of letting SQLite silently replay new state.
  for (const auto* suffix :
       {"runs.sqlite3-wal", "runs.sqlite3-shm", "runs.sqlite3-journal"}) {
    std::error_code error;
    const auto present = std::filesystem::exists(source / suffix, error);
    if (error) backup_io("restore.sidecar", source / suffix, error);
    if (present)
      corrupt("restore.sidecar",
              "backup contains an unmanifested database sidecar");
  }
  {
    BackupDatabase database(source / "runs.sqlite3", false);
    OperationControl control(database.get(), options.busy_timeout, stop,
                             deadline);
    if (scalar_integer(database.get(), "PRAGMA user_version", "restore.version",
                       &control) != schema_version ||
        scalar_integer(database.get(), "PRAGMA application_id",
                       "restore.application", &control) != kApplicationId)
      incompatible(
          "backup database identity does not match the manifest schema");
  }
  BackupStage stage(target);
  const auto copied = hash_or_copy_backup_file(
      source / "runs.sqlite3", stage.path() / "runs.sqlite3",
      options.max_backup_database_bytes, stop, deadline);
  if (copied.size != manifest.database.size ||
      copied.digest != manifest.database.digest)
    corrupt("restore.database", "database changed during restore copy");
  for (const auto& directory :
       {source / "artifacts", source / "artifacts" / "objects",
        source / "artifacts" / "tmp"}) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(directory, error);
    if (error || std::filesystem::is_symlink(status) ||
        !std::filesystem::is_directory(status))
      backup_io(
          "restore.artifact_directory", directory,
          error ? error : std::make_error_code(std::errc::not_a_directory));
  }
  FileArtifactStore source_cas(source / "artifacts", options.artifact_limits);
  FileArtifactStore target_cas(stage.path() / "artifacts",
                               options.artifact_limits);
  auto report = copy_backup_blobs(source_cas, target_cas, manifest.artifacts,
                                  stop, deadline);
  report.database_bytes = copied.size;
  validate_staged_backup(stage.path(), options, manifest.artifacts, stop,
                         deadline);
  sync_backup_path(stage.path() / "runs.sqlite3", false);
  check_telemetry_operation(stop, deadline);
  stage.publish();
  return report;
}

}  // namespace jyppx::liuding::experiment
