// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using jyppx::liuding::experiment::ArtifactDescriptor;
using jyppx::liuding::experiment::FileArtifactStore;
using jyppx::liuding::experiment::RunArtifactKind;
using jyppx::liuding::experiment::RunArtifactReference;
using jyppx::liuding::experiment::RunMetadata;
using jyppx::liuding::experiment::RunSink;
using jyppx::liuding::experiment::RunStatus;
using jyppx::liuding::experiment::SQLiteRunStore;
using jyppx::liuding::experiment::SQLiteRunStoreOptions;

std::span<const std::byte> bytes_of(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string utf8_path(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

class TempRunRoot final {
 public:
  TempRunRoot() {
    static std::atomic<std::uint64_t> counter{0U};
    const auto temporary = std::filesystem::temp_directory_path();
    for (unsigned attempt = 0U; attempt != 64U; ++attempt) {
      const auto now =
          std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate =
          temporary / ("jyppx-run-store-" + std::to_string(now) + "-" +
                       std::to_string(std::random_device{}()) + "-" +
                       std::to_string(++counter));
      if (std::filesystem::create_directory(candidate)) {
        path_ = std::move(candidate);
        return;
      }
    }
    throw std::runtime_error("could not reserve a run storage test directory");
  }

  ~TempRunRoot() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TempRunRoot(const TempRunRoot&) = delete;
  TempRunRoot& operator=(const TempRunRoot&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

#if GTEST_HAS_DEATH_TEST
// Windows death tests re-execute this setup. The child borrows the parent's
// atomically reserved root; only the parent restores the environment and owns
// cleanup. The marker rejects an unrelated preexisting environment path.
class AbruptExitRoot final {
 public:
  AbruptExitRoot() {
#ifdef _WIN32
    wchar_t* inherited = nullptr;
    std::size_t inherited_size = 0U;
    if (_wdupenv_s(&inherited, &inherited_size, kEnvironment) != 0) {
      throw std::runtime_error(
          "could not read abrupt-exit fixture environment");
    }
    const std::unique_ptr<wchar_t, decltype(&std::free)> inherited_owner(
        inherited, &std::free);
#else
    const auto* inherited = std::getenv(kEnvironment);
#endif
    if (inherited != nullptr) previous_ = inherited;
    if (previous_ && !previous_->empty()) {
      path_ = *previous_;
      if (!std::filesystem::is_regular_file(path_ / kMarker)) {
        throw std::runtime_error("invalid inherited abrupt-exit fixture root");
      }
      return;
    }
    owner_ = std::make_unique<TempRunRoot>();
    path_ = owner_->path();
    {
      std::ofstream marker(path_ / kMarker, std::ios::binary);
      marker << "SQLiteRunStore abrupt-exit fixture";
      marker.close();
      if (!marker) throw std::runtime_error("could not mark abrupt-exit root");
    }
#ifdef _WIN32
    const auto result = _wputenv_s(kEnvironment, path_.c_str());
#else
    const auto result = ::setenv(kEnvironment, path_.c_str(), 1);
#endif
    if (result != 0) {
      throw std::runtime_error("could not publish abrupt-exit fixture root");
    }
  }

  ~AbruptExitRoot() {
    if (!owner_) return;
#ifdef _WIN32
    (void)_wputenv_s(kEnvironment, previous_ ? previous_->c_str() : L"");
#else
    if (previous_) {
      (void)::setenv(kEnvironment, previous_->c_str(), 1);
    } else {
      (void)::unsetenv(kEnvironment);
    }
#endif
  }

  AbruptExitRoot(const AbruptExitRoot&) = delete;
  AbruptExitRoot& operator=(const AbruptExitRoot&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
#ifdef _WIN32
  static constexpr const wchar_t* kEnvironment =
      L"JYPPX_LIUDING_TEST_SQLITE_ABRUPT_EXIT_ROOT";
#else
  static constexpr const char* kEnvironment =
      "JYPPX_LIUDING_TEST_SQLITE_ABRUPT_EXIT_ROOT";
#endif
  static constexpr const char* kMarker = ".jyppx-sqlite-abrupt-exit-fixture";
  std::optional<std::filesystem::path::string_type> previous_;
  std::unique_ptr<TempRunRoot> owner_;
  std::filesystem::path path_;
};
#endif

class RawDatabase final {
 public:
  explicit RawDatabase(const std::filesystem::path& path) {
    const auto text = utf8_path(path);
    if (sqlite3_open(text.c_str(), &database_) != SQLITE_OK) {
      sqlite3_close(database_);
      throw std::runtime_error("could not open SQLite fixture");
    }
  }

  ~RawDatabase() { sqlite3_close(database_); }
  RawDatabase(const RawDatabase&) = delete;
  RawDatabase& operator=(const RawDatabase&) = delete;

  void execute(const char* sql) {
    if (sqlite3_exec(database_, sql, nullptr, nullptr, nullptr) != SQLITE_OK) {
      throw std::runtime_error("could not execute SQLite fixture statement");
    }
  }

  std::string scalar_text(const char* sql) {
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(database_, sql, -1, &statement, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("could not prepare SQLite fixture query");
    }
    if (sqlite3_step(statement) != SQLITE_ROW) {
      sqlite3_finalize(statement);
      throw std::runtime_error("missing SQLite fixture result");
    }
    const auto* text = sqlite3_column_text(statement, 0);
    const std::string result =
        text == nullptr ? "" : reinterpret_cast<const char*>(text);
    sqlite3_finalize(statement);
    return result;
  }

 private:
  sqlite3* database_ = nullptr;
};

RunMetadata make_run(std::string run_id = "run-1") {
  return {.run_id = std::move(run_id),
          .project_id = "project-1",
          .experiment_id = "experiment-1",
          .display_name = "离线训练",
          .status = RunStatus::running,
          .created_at_unix_ms = 100U,
          .updated_at_unix_ms = 100U,
          .lineage = {.parent_run_id = "parent-attempt",
                      .source_digest = "source-identity",
                      .build_fingerprint = "build-identity",
                      .config_digest = "redacted-config-digest",
                      .data_manifest_digest = "data-manifest-digest"},
          .tags = {{"model", "baseline"}, {"recipe", "cpu-smoke"}}};
}

ArtifactDescriptor make_descriptor(std::size_t size) {
  return {.logical_name = "weights.bin",
          .media_type = "application/octet-stream",
          .producer = "run-store-test",
          .expected_size_bytes = size};
}

RunArtifactReference publish(SQLiteRunStore& store, std::string id,
                             std::string_view content = "weights") {
  return store.put_artifact("run-1", std::move(id), RunArtifactKind::checkpoint,
                            "last", bytes_of(content),
                            make_descriptor(content.size()), 110U);
}

#if GTEST_HAS_DEATH_TEST
TEST(SQLiteRunStoreDeathTest,
     RecoversConfirmedReferenceFromWalAfterAbruptExit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  AbruptExitRoot root;
  ASSERT_EXIT(([&] {
                SQLiteRunStore store(root.path());
                store.start(make_run());
                publish(store, "confirmed-before-exit", "durable-weights");
                // Deliberately bypass store, SQLite connection, and fixture
                // destructors.
                std::_Exit(0);
              }()),
              ::testing::ExitedWithCode(0), "");

  const auto wal = root.path() / "runs.sqlite3-wal";
  ASSERT_TRUE(std::filesystem::exists(wal));
  ASSERT_GT(std::filesystem::file_size(wal), 0U);
  SQLiteRunStore reopened(root.path());
  const auto metadata = reopened.find("run-1");
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->status, RunStatus::running);
  EXPECT_EQ(metadata->updated_at_unix_ms, 110U);
  EXPECT_EQ(metadata->lineage.config_digest, make_run().lineage.config_digest);
  EXPECT_EQ(metadata->tags, make_run().tags);
  const auto references = reopened.references("run-1");
  ASSERT_EQ(references.size(), 1U);
  EXPECT_EQ(references.front().reference_id, "confirmed-before-exit");
  EXPECT_EQ(references.front().kind, RunArtifactKind::checkpoint);
  EXPECT_EQ(references.front().artifact.digest.hex,
            jyppx::liuding::sha256_hex(bytes_of("durable-weights")));
  EXPECT_TRUE(reopened.integrity_check().ok());
  EXPECT_EQ(
      reopened.collect_garbage(std::chrono::seconds::zero()).removed_files, 0U);
  FileArtifactStore cas(reopened.artifact_root());
  EXPECT_TRUE(cas.exists(references.front().artifact.digest));
  EXPECT_TRUE(reopened.integrity_check().ok());
}

TEST(SQLiteRunStoreDeathTest, DiscardsUncommittedTransactionAfterAbruptExit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  AbruptExitRoot root;
  ASSERT_EXIT(
      ([&] {
        SQLiteRunStore store(root.path());
        store.start(make_run());
        publish(store, "confirmed-before-transaction", "durable-weights");
        RawDatabase writer(store.database_path());
        writer.execute("PRAGMA cache_size = 1");
        writer.execute("BEGIN IMMEDIATE");
        writer.execute(
            "UPDATE jyppx_runs SET status = 3, updated_at_ms = 999 "
            "WHERE run_id = 'run-1'");
        writer.execute(
            "INSERT INTO jyppx_runs(run_id, project_id, experiment_id, "
            "display_name, status, created_at_ms, updated_at_ms, start_status, "
            "start_updated_at_ms) VALUES('uncommitted', 'project-1', "
            "'experiment-1', 'uncommitted identity', 2, 100, 100, 2, 100)");
        std::_Exit(0);
      }()),
      ::testing::ExitedWithCode(0), "");

  SQLiteRunStore reopened(root.path());
  EXPECT_FALSE(reopened.find("uncommitted"));
  const auto metadata = reopened.find("run-1");
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->status, RunStatus::running);
  EXPECT_EQ(metadata->updated_at_unix_ms, 110U);
  const auto references = reopened.references("run-1");
  ASSERT_EQ(references.size(), 1U);
  EXPECT_EQ(references.front().reference_id, "confirmed-before-transaction");
  EXPECT_TRUE(reopened.integrity_check().ok());
  EXPECT_EQ(
      reopened.collect_garbage(std::chrono::seconds::zero()).removed_files, 0U);
  EXPECT_TRUE(reopened.integrity_check().ok());
}
#endif

TEST(SQLiteRunStoreTest, RestartsWithRunLineageTagsCheckpointAndWalIntact) {
  TempRunRoot root;
  RunArtifactReference checkpoint;
  {
    SQLiteRunStore store(root.path());
    RunSink& sink = store;
    EXPECT_EQ(sink.start(make_run()).run_id, "run-1");
    checkpoint = publish(store, "checkpoint-1");
    EXPECT_EQ(sink.transition("run-1", RunStatus::completed, 120U).status,
              RunStatus::completed);
    RawDatabase raw(store.database_path());
    EXPECT_EQ(raw.scalar_text("PRAGMA journal_mode"), "wal");
    EXPECT_EQ(raw.scalar_text("PRAGMA user_version"),
              std::to_string(
                  jyppx::liuding::experiment::SQLiteRunStore::schema_version));
  }
  SQLiteRunStore reopened(root.path());
  const auto metadata = reopened.find("run-1");
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->status, RunStatus::completed);
  EXPECT_EQ(metadata->display_name, "离线训练");
  EXPECT_EQ(metadata->lineage.parent_run_id, "parent-attempt");
  EXPECT_EQ(metadata->lineage.source_digest, "source-identity");
  EXPECT_EQ(metadata->lineage.build_fingerprint, "build-identity");
  EXPECT_EQ(metadata->lineage.config_digest, "redacted-config-digest");
  EXPECT_EQ(metadata->lineage.data_manifest_digest, "data-manifest-digest");
  EXPECT_EQ(metadata->tags, make_run().tags);
  const auto references = reopened.references("run-1");
  ASSERT_EQ(references.size(), 1U);
  EXPECT_EQ(references.front().kind, RunArtifactKind::checkpoint);
  EXPECT_EQ(references.front().artifact.digest, checkpoint.artifact.digest);
  EXPECT_TRUE(reopened.integrity_check().ok());
}

TEST(SQLiteRunStoreTest, ReplaysOriginalStartAndRelationshipIdsAfterFinalize) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  const auto original = make_run();
  store.start(original);
  const auto reference = publish(store, "checkpoint-1");
  publish(store, "checkpoint-1");
  store.transition("run-1", RunStatus::completed, 120U);
  EXPECT_EQ(store.start(original).status, RunStatus::completed);
  EXPECT_EQ(
      store.transition("run-1", RunStatus::completed, 999U).updated_at_unix_ms,
      120U);
  const std::array<RunArtifactReference, 1U> replay{reference};
  EXPECT_NO_THROW(store.attach("run-1", replay));
  EXPECT_EQ(store.references("run-1").size(), 1U);
  auto conflict = original;
  conflict.lineage.config_digest = "different-config";
  EXPECT_THROW(store.start(conflict), jyppx::liuding::InvalidArgument);
  auto changed_reference = reference;
  changed_reference.role = "best";
  const std::array<RunArtifactReference, 1U> changed{changed_reference};
  EXPECT_THROW(store.attach("run-1", changed), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(store.transition("run-1", RunStatus::running, 130U),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(publish(store, "new-after-finalize"),
               jyppx::liuding::InvalidArgument);
}

TEST(SQLiteRunStoreTest,
     RollsBackWholeReferenceBatchWhenACommittedBlobIsMissing) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  FileArtifactStore cas(store.artifact_root());
  const auto artifact = cas.put(bytes_of("valid"), make_descriptor(5U));
  RunArtifactReference valid{.reference_id = "valid",
                             .kind = RunArtifactKind::artifact,
                             .role = "model",
                             .artifact = artifact,
                             .attached_at_unix_ms = 110U};
  auto missing = valid;
  missing.reference_id = "missing";
  missing.artifact.digest.hex = std::string(64U, '0');
  const std::array<RunArtifactReference, 2U> batch{valid, missing};
  EXPECT_THROW(store.attach("run-1", batch), jyppx::liuding::IoError);
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_EQ(store.find("run-1")->updated_at_unix_ms, 100U);
  RawDatabase raw(store.database_path());
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_blobs"), "0");
  EXPECT_TRUE(cas.exists(artifact.digest));
}

TEST(SQLiteRunStoreTest,
     FailedMetadataCommitLeavesOrphanAndPreservesConfirmedContent) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto confirmed = publish(store, "confirmed", "keep");
  {
    RawDatabase raw(store.database_path());
    raw.execute(
        "CREATE TRIGGER fail_reference BEFORE INSERT ON jyppx_run_artifacts "
        "BEGIN SELECT RAISE(ABORT, 'injected'); END");
    EXPECT_THROW(publish(store, "unconfirmed", "orphan"),
                 jyppx::liuding::InvalidArgument);
    raw.execute("DROP TRIGGER fail_reference");
  }
  FileArtifactStore cas(store.artifact_root());
  const jyppx::liuding::ArtifactDigest orphan{
      .hex = jyppx::liuding::sha256_hex(bytes_of("orphan"))};
  EXPECT_TRUE(cas.exists(orphan));
  ASSERT_EQ(store.references("run-1").size(), 1U);
  const auto collected = store.collect_garbage(std::chrono::seconds::zero());
  EXPECT_EQ(collected.removed_files, 1U);
  EXPECT_FALSE(cas.exists(orphan));
  EXPECT_TRUE(cas.exists(confirmed.artifact.digest));
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST(SQLiteRunStoreTest, RejectsBusyWriterThenRetriesWithoutPartialIdentity) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.busy_timeout = std::chrono::milliseconds::zero();
  SQLiteRunStore store(root.path(), options);
  RawDatabase competing(store.database_path());
  competing.execute("BEGIN IMMEDIATE");
  EXPECT_THROW(store.start(make_run()), jyppx::liuding::NotReady);
  EXPECT_FALSE(store.find("run-1"));
  competing.execute("ROLLBACK");
  EXPECT_NO_THROW(store.start(make_run()));
  EXPECT_TRUE(store.find("run-1"));
}

TEST(SQLiteRunStoreTest,
     CoordinatesIndependentHandlesAndGcUsingOneDatabaseLock) {
  TempRunRoot root;
  SQLiteRunStore first(root.path());
  SQLiteRunStore second(root.path());
  first.start(make_run());
  const auto reference = publish(second, "checkpoint-1");
  EXPECT_EQ(first.references("run-1").front().artifact.digest,
            reference.artifact.digest);
  const auto collected = first.collect_garbage(std::chrono::seconds::zero());
  EXPECT_EQ(collected.removed_files, 0U);
  EXPECT_TRUE(second.integrity_check().ok());
}

TEST(SQLiteRunStoreTest, RecoversOnlyExplicitlyStaleRunningRuns) {
  TempRunRoot root;
  {
    SQLiteRunStore store(root.path());
    store.start(make_run("stale"));
    store.start(make_run("active"));
    store.heartbeat("active", 200U);
    auto created = make_run("not-started");
    created.status = RunStatus::created;
    store.start(created);
    store.start(make_run("completed"));
    store.transition("completed", RunStatus::completed, 130U);
  }
  SQLiteRunStore reopened(root.path());
  EXPECT_EQ(reopened.find("stale")->status, RunStatus::running);
  EXPECT_EQ(reopened.recover_interrupted(150U, 300U), 1U);
  EXPECT_EQ(reopened.find("stale")->status, RunStatus::interrupted);
  EXPECT_EQ(reopened.find("active")->status, RunStatus::running);
  EXPECT_EQ(reopened.find("not-started")->status, RunStatus::created);
  EXPECT_EQ(reopened.find("completed")->status, RunStatus::completed);
  EXPECT_EQ(reopened.recover_interrupted(150U, 300U), 0U);
  EXPECT_THROW(reopened.heartbeat("stale", 310U),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(reopened.heartbeat("active", 1U),
               jyppx::liuding::InvalidArgument);
}

TEST(SQLiteRunStoreTest,
     EnforcesRunAndReferenceQuotasWithoutLosingAcceptedReferences) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.max_runs = 1U;
  options.max_references_per_run = 2U;
  options.max_run_artifact_bytes = 6U;
  SQLiteRunStore store(root.path(), options);
  store.start(make_run());
  EXPECT_THROW(store.start(make_run("second-run")),
               jyppx::liuding::ResourceExhausted);
  const auto reference = publish(store, "one", "four");
  EXPECT_THROW(publish(store, "two", "more"),
               jyppx::liuding::ResourceExhausted);
  ASSERT_EQ(store.references("run-1").size(), 1U);
  EXPECT_EQ(store.references("run-1").front().artifact.digest,
            reference.artifact.digest);
  EXPECT_NO_THROW(publish(store, "small-second", "x"));
  EXPECT_THROW(publish(store, "count-exceeded", ""),
               jyppx::liuding::ResourceExhausted);
  auto invalid = make_run();
  invalid.created_at_unix_ms = std::numeric_limits<std::uint64_t>::max();
  invalid.updated_at_unix_ms = invalid.created_at_unix_ms;
  EXPECT_THROW(store.start(invalid), jyppx::liuding::ResourceExhausted);
}

TEST(SQLiteRunStoreTest, RollsBackNewRunWhenItsTagTransactionFails) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  RawDatabase raw(store.database_path());
  raw.execute(
      "CREATE TRIGGER fail_tag BEFORE INSERT ON jyppx_run_tags BEGIN SELECT "
      "RAISE(ABORT, 'injected'); END");
  EXPECT_THROW(store.start(make_run()), jyppx::liuding::InvalidArgument);
  EXPECT_FALSE(store.find("run-1"));
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_run_tags"), "0");
  raw.execute("DROP TRIGGER fail_tag");
  EXPECT_NO_THROW(store.start(make_run()));
  EXPECT_EQ(store.find("run-1")->tags, make_run().tags);
}

TEST(SQLiteRunStoreTest,
     RefusesFutureAndForeignSchemasWithoutRewritingVersion) {
  TempRunRoot future;
  {
    RawDatabase raw(future.path() / "runs.sqlite3");
    raw.execute("PRAGMA user_version = 999");
  }
  EXPECT_THROW(SQLiteRunStore(future.path()),
               jyppx::liuding::CompatibilityError);
  {
    RawDatabase raw(future.path() / "runs.sqlite3");
    EXPECT_EQ(raw.scalar_text("PRAGMA user_version"), "999");
  }
  TempRunRoot foreign;
  {
    RawDatabase raw(foreign.path() / "runs.sqlite3");
    raw.execute("CREATE TABLE unrelated(value TEXT)");
  }
  EXPECT_THROW(SQLiteRunStore(foreign.path()),
               jyppx::liuding::CompatibilityError);
  RawDatabase raw(foreign.path() / "runs.sqlite3");
  EXPECT_EQ(raw.scalar_text(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'unrelated'"),
            "1");
}

TEST(SQLiteRunStoreTest,
     ReportsCorruptContentAndFailsGcClosedOnBrokenForeignKeys) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto reference = publish(store, "confirmed", "keep");
  const auto digest = reference.artifact.digest.hex;
  const auto path = store.artifact_root() / "objects" / digest.substr(0U, 2U) /
                    digest.substr(2U);
  std::ofstream(path, std::ios::binary | std::ios::trunc) << "damage";
  EXPECT_FALSE(store.integrity_check().ok());
  EXPECT_TRUE(store.integrity_check(false).ok());
  {
    RawDatabase raw(store.database_path());
    raw.execute("PRAGMA foreign_keys = OFF");
    raw.execute("DELETE FROM jyppx_blobs");
  }
  EXPECT_FALSE(store.integrity_check(false).ok());
  EXPECT_THROW(store.collect_garbage(std::chrono::seconds::zero()),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(std::filesystem::exists(path));
}

TEST(SQLiteRunStoreTest, RejectsCorruptDatabaseAndInvalidUtf8Metadata) {
  TempRunRoot corrupt;
  std::ofstream(corrupt.path() / "runs.sqlite3", std::ios::binary)
      << "not a SQLite database";
  EXPECT_THROW(SQLiteRunStore(corrupt.path()),
               jyppx::liuding::SerializationError);
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  auto invalid = make_run();
  invalid.display_name = std::string(1U, static_cast<char>(0xff));
  EXPECT_THROW(store.start(invalid), jyppx::liuding::InvalidArgument);
  EXPECT_FALSE(store.find("run-1"));
}

}  // namespace
