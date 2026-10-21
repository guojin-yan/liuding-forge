// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/experiment/run_store.hpp"

#include "jyppx/foundation/error.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <sqlite3.h>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace jyppx::experiment {
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
          .expected = "JYPPX run schema version 1 or a new empty database",
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

[[noreturn]] void sqlite_failure(sqlite3* database, int result,
                                 std::string operation) {
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

void execute(sqlite3* database, const char* sql, std::string_view operation) {
  const int result = sqlite3_exec(database, sql, nullptr, nullptr, nullptr);
  if (result != SQLITE_OK)
    sqlite_failure(database, result, std::string(operation));
}

class Statement final {
 public:
  Statement(sqlite3* database, const char* sql, std::string operation)
      : database_(database), operation_(std::move(operation)) {
    const int result =
        sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr);
    if (result != SQLITE_OK) {
      sqlite3_finalize(statement_);
      statement_ = nullptr;
      sqlite_failure(database_, result, operation_);
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
    if (result != SQLITE_OK) sqlite_failure(database_, result, operation_);
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
    if (result != SQLITE_OK) sqlite_failure(database_, result, operation_);
  }

  void bind(int index, const std::optional<std::string>& value) {
    if (value) {
      bind(index, std::string_view(*value));
      return;
    }
    const int result = sqlite3_bind_null(statement_, index);
    if (result != SQLITE_OK) sqlite_failure(database_, result, operation_);
  }

  bool row() {
    const int result = sqlite3_step(statement_);
    if (result == SQLITE_ROW) return true;
    if (result == SQLITE_DONE) return false;
    sqlite_failure(database_, result, operation_);
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
      sqlite_failure(database_, SQLITE_NOMEM, operation_);
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

 private:
  sqlite3* database_;
  std::string operation_;
  sqlite3_stmt* statement_ = nullptr;
};

class Transaction final {
 public:
  explicit Transaction(sqlite3* database, bool write = true)
      : database_(database) {
    execute(database_, write ? "BEGIN IMMEDIATE" : "BEGIN",
            "transaction.begin");
  }

  ~Transaction() {
    if (!committed_)
      (void)sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
  }

  Transaction(const Transaction&) = delete;
  Transaction& operator=(const Transaction&) = delete;

  void commit() {
    execute(database_, "COMMIT", "transaction.commit");
    committed_ = true;
  }

 private:
  sqlite3* database_;
  bool committed_ = false;
};

std::uint64_t scalar_integer(sqlite3* database, const char* sql,
                             std::string operation) {
  Statement statement(database, sql, std::move(operation));
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
                                     const SQLiteRunStoreOptions& options) {
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
  return root;
}

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

constexpr const char* kSelectReferences =
    "SELECT r.reference_id, r.kind, r.role, b.algorithm, b.digest, "
    "b.size_bytes, "
    "r.logical_name, r.media_type, r.producer, r.attached_at_ms "
    "FROM jyppx_run_artifacts r JOIN jyppx_blobs b ON b.digest = r.digest "
    "WHERE r.run_id = ?1 ORDER BY r.reference_id";

}  // namespace

class SQLiteRunStore::Impl final {
 public:
  Impl(std::filesystem::path root, SQLiteRunStoreOptions options)
      : options_(std::move(options)),
        root_(canonical_root(std::move(root), options_)),
        database_path_(root_ / "runs.sqlite3"),
        artifacts_(root_ / "artifacts", options_.artifact_limits) {
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
      initialize();
    } catch (...) {
      sqlite3_close(database_);
      database_ = nullptr;
      throw;
    }
  }

  ~Impl() { sqlite3_close(database_); }

  void initialize() {
    const auto version =
        scalar_integer(database_, "PRAGMA user_version", "open.version");
    const auto application =
        scalar_integer(database_, "PRAGMA application_id", "open.application");
    if (version != 0U &&
        (version != schema_version || application != kApplicationId)) {
      incompatible("schema " + std::to_string(version) + ", application " +
                   std::to_string(application));
    }
    if (version == 0U && application != 0U)
      incompatible("foreign application identifier");
    execute(database_, "PRAGMA foreign_keys = ON", "open.foreign_keys");
    if (scalar_integer(database_, "PRAGMA foreign_keys", "open.foreign_keys") !=
        1U) {
      incompatible("SQLite foreign key enforcement is unavailable");
    }
    if (version == 0U) {
      Transaction transaction(database_);
      const auto locked_version =
          scalar_integer(database_, "PRAGMA user_version", "open.version");
      if (locked_version == 0U) {
        if (scalar_integer(database_,
                           "SELECT COUNT(*) FROM sqlite_master WHERE name NOT "
                           "GLOB 'sqlite_*'",
                           "open.empty") != 0U) {
          incompatible("unversioned nonempty database");
        }
        execute(database_, kCreateSchema, "open.create_schema");
      } else if (locked_version != schema_version ||
                 scalar_integer(database_, "PRAGMA application_id",
                                "open.application") != kApplicationId) {
        incompatible("database schema changed during initialization");
      }
      transaction.commit();
    }
    {
      Statement journal(database_, "PRAGMA journal_mode = WAL", "open.wal");
      if (!journal.row() || journal.text(0, 16U) != "wal") {
        incompatible("SQLite WAL mode is unavailable for this filesystem");
      }
    }
    execute(database_, "PRAGMA synchronous = FULL", "open.synchronous");
    execute(database_, "PRAGMA trusted_schema = OFF", "open.trusted_schema");
    // Preparing the expected columns rejects an incomplete schema before use.
    Statement validate(database_, kSelectRun, "open.validate_schema");
    Statement validate_references(database_, kSelectReferences,
                                  "open.validate_schema");
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
                       std::span<const RunArtifactReference> references) {
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
      const auto reader = artifacts_.open(reference.artifact.digest);
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
  mutable std::mutex mutex_;
};

SQLiteRunStore::SQLiteRunStore(std::filesystem::path root,
                               SQLiteRunStoreOptions options)
    : impl_(std::make_unique<Impl>(std::move(root), std::move(options))) {}

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
    bool verify_artifacts) const {
  std::lock_guard lock(impl_->mutex_);
  Transaction transaction(impl_->database_);
  RunStoreIntegrityReport report;
  {
    Statement statement(impl_->database_, "PRAGMA integrity_check(64)",
                        "integrity.database");
    while (statement.row()) {
      const auto issue = statement.text(0, 4096U);
      if (issue != "ok" && report.issues.size() < kMaxIntegrityIssues) {
        report.issues.push_back("SQLite: " + issue);
      }
    }
  }
  {
    Statement statement(impl_->database_, "PRAGMA foreign_key_check",
                        "integrity.foreign_keys");
    while (statement.row()) {
      if (report.issues.size() < kMaxIntegrityIssues) {
        report.issues.push_back("Foreign key violation in " +
                                statement.text(0, 256U));
      }
    }
  }
  if (verify_artifacts) {
    for (const auto& reference : impl_->live_blobs()) {
      ++report.checked_artifacts;
      try {
        const auto reader = impl_->artifacts_.open(reference.digest);
        if (reader->size_bytes() != reference.size_bytes &&
            report.issues.size() < kMaxIntegrityIssues) {
          report.issues.push_back("Artifact size mismatch: " +
                                  reference.digest.hex);
        }
      } catch (const Error&) {
        if (report.issues.size() < kMaxIntegrityIssues) {
          report.issues.push_back("Missing or invalid artifact: " +
                                  reference.digest.hex);
        }
      }
    }
  }
  transaction.commit();
  return report;
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

}  // namespace jyppx::experiment
