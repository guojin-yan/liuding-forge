// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/experiment/run_telemetry_sink.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sqlite3.h>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

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
          temporary / ("jyppx-run-events-" + std::to_string(now) + "-" +
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
      L"JYPPX_LIUDING_TEST_EVENT_ABRUPT_EXIT_ROOT";
#else
  static constexpr const char* kEnvironment =
      "JYPPX_LIUDING_TEST_EVENT_ABRUPT_EXIT_ROOT";
#endif
  static constexpr const char* kMarker = ".jyppx-event-abrupt-exit-fixture";
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

using jyppx::liuding::TelemetryDisposition;
using jyppx::liuding::TelemetryEnvelope;
using jyppx::liuding::TelemetryLifecycle;
using jyppx::liuding::TelemetryLog;
using jyppx::liuding::TelemetryMetricBatch;
using jyppx::liuding::TelemetryMetricPoint;
using jyppx::liuding::TelemetryPayloadKind;
using jyppx::liuding::experiment::RunEventPageRequest;
using jyppx::liuding::experiment::RunMetricPageRequest;
using jyppx::liuding::experiment::SQLiteRunTelemetrySink;

TelemetryMetricPoint metric(std::uint64_t step = 0U, double value = 0.5) {
  TelemetryMetricPoint point;
  point.definition.metric_id = "eval.accuracy";
  point.definition.unit = jyppx::liuding::TelemetryMetricUnit::ratio;
  point.definition.aggregation = jyppx::liuding::TelemetryAggregation::mean;
  point.definition.direction =
      jyppx::liuding::TelemetryMetricDirection::maximize;
  point.definition.parameters = {{"average", "macro"}};
  point.definition.context_keys = {"dataset", "split"};
  point.context = {{"dataset", "validation-v1"}, {"split", "val"}};
  point.step.value = step;
  point.observation = jyppx::liuding::make_telemetry_metric_value(value);
  point.sample_count = 32U;
  return point;
}

TelemetryEnvelope log_event(std::uint64_t sequence = 1U,
                            std::string stream = "producer-1",
                            std::string run_id = "run-1") {
  TelemetryEnvelope envelope;
  envelope.identity = {
      .stream_id = std::move(stream),
      .event_id = "event-" + std::to_string(sequence),
      .idempotency_key = "operation-" + std::to_string(sequence),
      .sequence = sequence};
  envelope.correlation.project_id = "project-1";
  envelope.correlation.run_id = std::move(run_id);
  envelope.correlation.attempt_id = "attempt-original";
  envelope.attributes = {{"source", "unit-test"}};
  envelope.payload = TelemetryLog{.body = "captured original log"};
  return envelope;
}

TelemetryEnvelope metric_event(std::uint64_t sequence = 1U,
                               std::size_t count = 1U,
                               std::string stream = "producer-1") {
  auto envelope = log_event(sequence, std::move(stream));
  TelemetryMetricBatch batch;
  for (std::size_t index = 0U; index < count; ++index)
    batch.points.push_back(
        metric(static_cast<std::uint64_t>(index), static_cast<double>(index)));
  envelope.payload = std::move(batch);
  return envelope;
}

jyppx::liuding::experiment::RunEventAppendResult append_one(
    SQLiteRunStore& store, std::string_view key, const TelemetryEnvelope& event,
    std::stop_token stop = {},
    jyppx::liuding::TelemetryDeadline deadline = std::nullopt) {
  return store.append(event.correlation.run_id, key,
                      std::span<const TelemetryEnvelope>(&event, 1U), stop,
                      deadline);
}

jyppx::liuding::ArtifactRef publish_unattached(
    SQLiteRunStore& store, std::string_view content = "event-artifact") {
  FileArtifactStore cas(store.artifact_root());
  return cas.put(bytes_of(content), {.logical_name = "report.bin",
                                     .media_type = "application/octet-stream",
                                     .producer = "event-test",
                                     .expected_size_bytes = content.size()});
}

TEST(SQLiteRunEventTest,
     ReplaysCompleteBatchWithOriginalWatermarkAfterLaterCommit) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const std::array batch{log_event(), metric_event(2U, 2U)};
  const auto first = store.append("run-1", "batch-original", batch);
  ASSERT_EQ(first.receipts.size(), 2U);
  EXPECT_EQ(first.receipts[0].run_sequence, 1U);
  EXPECT_EQ(first.receipts[1].run_sequence, 2U);
  EXPECT_EQ(first.receipts[0].disposition, TelemetryDisposition::accepted);
  EXPECT_EQ(first.committed_through, 2U);
  append_one(store, "later", log_event(3U));
  const auto replay = store.append("run-1", "batch-original", batch);
  EXPECT_TRUE(replay.batch_replay);
  EXPECT_EQ(replay.committed_through, 2U);
  ASSERT_EQ(replay.receipts.size(), 2U);
  EXPECT_EQ(replay.receipts[0].run_sequence, 1U);
  EXPECT_EQ(replay.receipts[1].run_sequence, 2U);
  EXPECT_EQ(replay.receipts[1].disposition, TelemetryDisposition::duplicate);
  auto changed = batch;
  std::get<TelemetryLog>(changed[0].payload).body = "different bytes";
  EXPECT_THROW(store.append("run-1", "batch-original", changed),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(store.events("run-1").events.size(), 3U);
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest, ChecksEveryProducerIdentityCoordinateAcrossBatches) {
  for (unsigned conflict = 0U; conflict < 4U; ++conflict) {
    SCOPED_TRACE(conflict);
    TempRunRoot root;
    SQLiteRunStore store(root.path());
    store.start(make_run());
    const auto original = log_event();
    append_one(store, "original", original);
    auto changed = original;
    if (conflict == 0U) changed.identity.event_id = "changed-event";
    if (conflict == 1U) changed.identity.idempotency_key = "changed-operation";
    if (conflict == 2U) changed.identity.sequence = 2U;
    if (conflict == 3U)
      std::get<TelemetryLog>(changed.payload).body = "changed";
    EXPECT_THROW(append_one(store, "conflict", changed),
                 jyppx::liuding::CompatibilityError);
    EXPECT_EQ(store.events("run-1").events.size(), 1U);
    EXPECT_TRUE(store.integrity_check(false).ok());
  }
}

TEST(SQLiteRunEventTest,
     AtomicallyCombinesDuplicateInputWithNewEventsAndStreams) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto first = log_event();
  append_one(store, "first", first);
  const std::array batch{first, log_event(1U, "producer-2"), log_event(2U)};
  const auto result = store.append("run-1", "mixed", batch);
  ASSERT_EQ(result.receipts.size(), 3U);
  EXPECT_EQ(result.receipts[0].disposition, TelemetryDisposition::duplicate);
  EXPECT_EQ(result.receipts[1].run_sequence, 2U);
  EXPECT_EQ(result.receipts[2].run_sequence, 3U);
  EXPECT_EQ(result.committed_through, 3U);
  const auto page = store.events("run-1");
  ASSERT_EQ(page.events.size(), 3U);
  EXPECT_EQ(page.events[1].envelope.identity.sequence, 1U);
  EXPECT_EQ(page.events[2].envelope.identity.sequence, 2U);
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest, RejectsSequenceGapsAndRollsBackPrecedingNewEvidence) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const std::array gap{metric_event(1U), metric_event(3U)};
  EXPECT_THROW(store.append("run-1", "gap", gap),
               jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_TRUE(
      store.metrics("run-1", {.metric_id = "eval.accuracy"}).points.empty());
  RawDatabase raw(store.database_path());
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_event_streams"), "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_metric_definitions"),
            "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_event_batches"), "0");
  EXPECT_NO_THROW(append_one(store, "gap", log_event()));
}

TEST(SQLiteRunEventTest,
     RejectsCrossRunStreamsAndIncorrectCorrelationWithoutWrites) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  store.start(make_run("run-2"));
  append_one(store, "first", log_event());
  auto moved = log_event(2U, "producer-1", "run-2");
  EXPECT_THROW(append_one(store, "moved", moved),
               jyppx::liuding::CompatibilityError);
  auto wrong_project = log_event(1U, "fresh");
  wrong_project.correlation.project_id = "project-other";
  EXPECT_THROW(append_one(store, "wrong-project", wrong_project),
               jyppx::liuding::InvalidArgument);
  const std::array wrong_run{log_event(1U, "fresh", "run-2")};
  EXPECT_THROW(store.append("run-1", "wrong-run", wrong_run),
               jyppx::liuding::InvalidArgument);
  EXPECT_TRUE(store.events("run-2").events.empty());
  EXPECT_EQ(store.events("run-1").events.size(), 1U);
}

TEST(SQLiteRunEventTest,
     TerminalStreamsRejectNewEvidenceAndAllowIdenticalRetries) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  auto terminal = log_event();
  terminal.payload =
      TelemetryLifecycle{.name = "training.finished", .terminal = true};
  append_one(store, "terminal", terminal);
  EXPECT_EQ(store.find("run-1")->status, RunStatus::running);
  EXPECT_THROW(append_one(store, "after-terminal", log_event(2U)),
               jyppx::liuding::InvalidArgument);
  EXPECT_TRUE(append_one(store, "terminal", terminal).batch_replay);
  EXPECT_EQ(
      append_one(store, "new-retry-key", terminal).receipts[0].disposition,
      TelemetryDisposition::duplicate);
  EXPECT_NO_THROW(
      append_one(store, "another-producer", log_event(1U, "producer-2")));
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     TerminalRunsPermitOldRetriesButRejectNewProducerAttempts) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto original = log_event();
  append_one(store, "original", original);
  store.transition("run-1", RunStatus::completed, 150U);
  EXPECT_TRUE(append_one(store, "original", original).batch_replay);
  EXPECT_EQ(append_one(store, "same-event-new-batch", original)
                .receipts[0]
                .disposition,
            TelemetryDisposition::duplicate);
  EXPECT_THROW(append_one(store, "new-attempt", log_event(1U, "new-stream")),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(store.events("run-1").events.size(), 1U);
}

TEST(SQLiteRunEventTest,
     RejectsMetricRedefinitionAndAcceptsExplicitNewVersion) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  append_one(store, "first-definition", metric_event());
  auto changed = metric_event(2U);
  auto& point = std::get<TelemetryMetricBatch>(changed.payload).points[0];
  point.definition.unit = jyppx::liuding::TelemetryMetricUnit::percent;
  EXPECT_THROW(append_one(store, "redefinition", changed),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(store.events("run-1").events.size(), 1U);
  point.definition.version = 2U;
  append_one(store, "new-version", changed);
  const auto all = store.metrics("run-1", {.metric_id = "eval.accuracy"});
  ASSERT_EQ(all.points.size(), 2U);
  const auto v2 = store.metrics(
      "run-1", {.metric_id = "eval.accuracy", .definition_version = 2U});
  ASSERT_EQ(v2.points.size(), 1U);
  EXPECT_EQ(v2.points[0].point.definition.unit,
            jyppx::liuding::TelemetryMetricUnit::percent);
  EXPECT_EQ(v2.points[0].point.context, metric().context);
  EXPECT_EQ(v2.points[0].point.definition.parameters,
            metric().definition.parameters);
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     PreservesSymbolicValuesUnsignedStepAndMissingHistoricalClocks) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  auto envelope = metric_event(1U, 6U);
  auto& points = std::get<TelemetryMetricBatch>(envelope.payload).points;
  points[0].observation = jyppx::liuding::make_telemetry_metric_value(-0.0);
  points[1].observation = jyppx::liuding::make_telemetry_metric_value(
      std::numeric_limits<double>::quiet_NaN());
  points[2].observation = jyppx::liuding::make_telemetry_metric_value(
      std::numeric_limits<double>::infinity());
  points[3].observation = jyppx::liuding::make_telemetry_metric_value(
      -std::numeric_limits<double>::infinity());
  points[4].observation = jyppx::liuding::make_telemetry_metric_value(
      std::nullopt, "no valid population");
  points[5].step.value = std::numeric_limits<std::uint64_t>::max();
  envelope.attributes["clock_provenance"] = "historical-unavailable";
  append_one(store, "values", envelope);
  const auto page = store.metrics("run-1", {.metric_id = "eval.accuracy"});
  ASSERT_EQ(page.points.size(), 6U);
  EXPECT_FALSE(std::signbit(*page.points[0].point.observation.value));
  EXPECT_EQ(page.points[1].point.observation.status,
            jyppx::liuding::TelemetryValueStatus::not_a_number);
  EXPECT_EQ(page.points[2].point.observation.status,
            jyppx::liuding::TelemetryValueStatus::positive_infinity);
  EXPECT_EQ(page.points[3].point.observation.status,
            jyppx::liuding::TelemetryValueStatus::negative_infinity);
  EXPECT_EQ(page.points[4].point.observation.status,
            jyppx::liuding::TelemetryValueStatus::missing);
  EXPECT_EQ(page.points[4].point.observation.reason, "no valid population");
  EXPECT_EQ(page.points[5].point.step.value,
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_FALSE(page.points[0].time.wall_time_unix_ns);
  EXPECT_FALSE(page.points[0].time.monotonic_time_ns);
  EXPECT_EQ(page.points[0].attributes, envelope.attributes);
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     EventPagesKeepFixedSnapshotAcrossNewCommitsAndPayloadFilters) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const std::array initial{log_event(), metric_event(2U), log_event(3U)};
  store.append("run-1", "initial", initial);
  const auto first = store.events("run-1", {.limit = 1U});
  ASSERT_EQ(first.events.size(), 1U);
  ASSERT_TRUE(first.next_after);
  EXPECT_EQ(first.snapshot_through, 3U);
  SQLiteRunStore concurrent(root.path());
  append_one(concurrent, "concurrent", log_event(4U));
  const auto remainder =
      store.events("run-1", {.after_sequence = *first.next_after,
                             .snapshot_through = first.snapshot_through,
                             .limit = 8U});
  ASSERT_EQ(remainder.events.size(), 2U);
  EXPECT_EQ(remainder.events.back().run_sequence, 3U);
  EXPECT_FALSE(remainder.next_after);
  const auto logs = store.events("run-1", {.kind = TelemetryPayloadKind::log});
  ASSERT_EQ(logs.events.size(), 3U);
  EXPECT_EQ(logs.events.back().run_sequence, 4U);
  EXPECT_THROW(
      store.events("run-1", {.after_sequence = 4U, .snapshot_through = 3U}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(store.events("run-1", {.snapshot_through = 5U}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(store.events("run-1", {.limit = 0U}),
               jyppx::liuding::InvalidArgument);
}

TEST(SQLiteRunEventTest,
     MetricPagesUseEnvelopeOrderDespiteStepResetsAndLateAppend) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  auto initial = metric_event(1U, 3U);
  auto& points = std::get<TelemetryMetricBatch>(initial.payload).points;
  points[0].step.value = 100U;
  points[1].step.value = 0U;
  points[2].step.value = 100U;
  append_one(store, "initial", initial);
  const auto first =
      store.metrics("run-1", {.metric_id = "eval.accuracy", .limit = 2U});
  ASSERT_EQ(first.points.size(), 2U);
  ASSERT_TRUE(first.next_after);
  EXPECT_EQ(first.points[0].point.step.value, 100U);
  EXPECT_EQ(first.points[1].point.step.value, 0U);
  append_one(store, "late", metric_event(2U));
  const auto rest =
      store.metrics("run-1", {.metric_id = "eval.accuracy",
                              .after = *first.next_after,
                              .snapshot_through = first.snapshot_through,
                              .limit = 2U});
  ASSERT_EQ(rest.points.size(), 1U);
  EXPECT_EQ(rest.points[0].cursor.point_index, 2U);
  EXPECT_FALSE(rest.next_after);
  EXPECT_EQ(
      store.metrics("run-1", {.metric_id = "eval.accuracy"}).points.size(), 4U);
  EXPECT_TRUE(
      store
          .metrics("run-1",
                   {.metric_id = "eval.accuracy",
                    .step_kind = jyppx::liuding::TelemetryStepKind::epoch})
          .points.empty());
  EXPECT_THROW(store.metrics(
                   "run-1", {.metric_id = "eval.accuracy",
                             .after = {.run_sequence = 0U, .point_index = 1U}}),
               jyppx::liuding::InvalidArgument);
}

TEST(SQLiteRunEventTest,
     ProjectsResourceMetricIdentityAndLifecycleMetricDefinitions) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  auto resource = log_event();
  resource.payload = jyppx::liuding::TelemetryResourceSample{
      .resource_id = "cuda.device-0", .measurements = {metric()}};
  auto lifecycle = log_event(2U);
  lifecycle.payload =
      TelemetryLifecycle{.name = "epoch.finished", .metrics = {metric(2U)}};
  const std::array batch{resource, lifecycle};
  store.append("run-1", "typed-payloads", batch);
  const auto page = store.metrics("run-1", {.metric_id = "eval.accuracy"});
  ASSERT_EQ(page.points.size(), 2U);
  EXPECT_EQ(page.points[0].resource_id, "cuda.device-0");
  EXPECT_FALSE(page.points[1].resource_id);
  EXPECT_EQ(page.points[0].event_identity.event_id, resource.identity.event_id);
  EXPECT_EQ(page.points[1].correlation.attempt_id, "attempt-original");
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     EventPagesHonorContentBudgetWithoutSkippingAnEnvelope) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.event_limits.telemetry.max_text_bytes = 1100U;
  options.event_limits.telemetry.max_envelope_bytes = 2048U;
  options.event_limits.telemetry.max_batch_bytes = 8192U;
  options.event_limits.max_page_bytes = 2048U;
  SQLiteRunStore store(root.path(), options);
  store.start(make_run());
  auto first = log_event();
  auto second = log_event(2U);
  std::get<TelemetryLog>(first.payload).body = std::string(1000U, 'a');
  std::get<TelemetryLog>(second.payload).body = std::string(1000U, 'b');
  const std::array batch{first, second};
  store.append("run-1", "bounded", batch);
  const auto page = store.events("run-1", {.limit = 2U});
  ASSERT_EQ(page.events.size(), 1U);
  ASSERT_TRUE(page.next_after);
  const auto next =
      store.events("run-1", {.after_sequence = *page.next_after,
                             .snapshot_through = page.snapshot_through,
                             .limit = 2U});
  ASSERT_EQ(next.events.size(), 1U);
  EXPECT_EQ(next.events[0].run_sequence, 2U);
  EXPECT_FALSE(next.next_after);
}

TEST(SQLiteRunEventTest,
     ArtifactEvidenceCreatesGcVisibleRelationshipInSameCommit) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto artifact = publish_unattached(store);
  auto event = log_event();
  event.payload = jyppx::liuding::TelemetryArtifactReference{
      .role = "evaluation.report", .artifact = artifact};
  append_one(store, "artifact-event", event);
  const auto references = store.references("run-1");
  ASSERT_EQ(references.size(), 1U);
  EXPECT_EQ(references[0].artifact.digest, artifact.digest);
  EXPECT_EQ(
      references[0].reference_id,
      "telemetry." +
          jyppx::liuding::telemetry_idempotency_digest(event.identity).hex);
  EXPECT_EQ(references[0].attached_at_unix_ms, 100U);
  EXPECT_EQ(store.find("run-1")->updated_at_unix_ms, 100U);
  EXPECT_EQ(store.collect_garbage(std::chrono::seconds(0)).removed_files, 0U);
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST(SQLiteRunEventTest,
     MissingArtifactRejectsWholeBatchAndLeavesProducerSequenceReusable) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  auto event = log_event(2U);
  event.payload = jyppx::liuding::TelemetryArtifactReference{
      .role = "report",
      .artifact = {.digest = {.hex = std::string(64U, 'a')},
                   .size_bytes = 5U,
                   .logical_name = "missing.bin",
                   .media_type = "application/octet-stream",
                   .producer = "event-test"}};
  const std::array batch{metric_event(), event};
  EXPECT_THROW(store.append("run-1", "missing", batch),
               jyppx::liuding::IoError);
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_NO_THROW(append_one(store, "reuse", log_event()));
}

TEST(SQLiteRunEventTest,
     LateSqlFailureRollsBackEventsDefinitionsReferencesAndReceipts) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto artifact = publish_unattached(store);
  auto referenced = log_event(2U);
  referenced.payload = jyppx::liuding::TelemetryArtifactReference{
      .role = "report", .artifact = artifact};
  const std::array batch{metric_event(), referenced};
  RawDatabase raw(store.database_path());
  raw.execute(
      "CREATE TRIGGER fail_late BEFORE INSERT ON jyppx_event_batches "
      "BEGIN SELECT RAISE(ABORT,'injected batch receipt failure'); END;");
  EXPECT_THROW(store.append("run-1", "atomic", batch), jyppx::liuding::Error);
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_metric_definitions"),
            "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_metric_points"), "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_event_streams"), "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_event_batches"), "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_run_event_state"), "0");
  raw.execute("DROP TRIGGER fail_late");
  EXPECT_NO_THROW(store.append("run-1", "atomic", batch));
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST(SQLiteRunEventTest,
     EnforcesMetricEventStreamAndBatchQuotaAtCommitBoundary) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.event_limits.max_metric_points_per_run = 3U;
  options.event_limits.max_events_per_run = 2U;
  options.event_limits.max_batches_per_run = 2U;
  options.event_limits.max_streams_per_run = 1U;
  SQLiteRunStore store(root.path(), options);
  store.start(make_run());
  const auto initial = metric_event(1U, 2U);
  append_one(store, "initial", initial);
  EXPECT_THROW(append_one(store, "too-many-points", metric_event(2U, 2U)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(append_one(store, "new-stream", log_event(1U, "another-stream")),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(store.events("run-1").events.size(), 1U);
  append_one(store, "within-quota", metric_event(2U));
  EXPECT_THROW(append_one(store, "extra-event", log_event(3U)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(append_one(store, "extra-receipt", initial),
               jyppx::liuding::ResourceExhausted);
  EXPECT_TRUE(append_one(store, "initial", initial).batch_replay);
  EXPECT_EQ(
      store.metrics("run-1", {.metric_id = "eval.accuracy"}).points.size(), 3U);
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     EmptyBatchIsNoOpAndInvalidOperationsHaveNoDirectorySideEffects) {
  TempRunRoot root;
  const auto unused = root.path() / "must-not-exist";
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW({ SQLiteRunStore blocked(unused, {}, stopped.get_token()); },
               jyppx::liuding::Cancelled);
  EXPECT_FALSE(std::filesystem::exists(unused));
  EXPECT_THROW(
      {
        SQLiteRunStore blocked(unused, {}, {},
                               std::chrono::steady_clock::now());
      },
      jyppx::liuding::DeadlineExceeded);
  EXPECT_FALSE(std::filesystem::exists(unused));
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto empty = store.append("run-1", "empty", {});
  EXPECT_TRUE(empty.receipts.empty());
  EXPECT_EQ(empty.committed_through, 0U);
  EXPECT_FALSE(empty.batch_replay);
  RawDatabase raw(store.database_path());
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_event_batches"), "0");
  const auto event = log_event();
  EXPECT_THROW(append_one(store, "cancelled", event, stopped.get_token()),
               jyppx::liuding::Cancelled);
  EXPECT_THROW(store.events("run-1", {}, stopped.get_token()),
               jyppx::liuding::Cancelled);
  EXPECT_TRUE(store.events("run-1").events.empty());
}

TEST(SQLiteRunEventTest,
     DeadlineAndCancellationInterruptCompetingSqliteWriterWait) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.busy_timeout = std::chrono::seconds(5);
  SQLiteRunStore store(root.path(), options);
  store.start(make_run());
  const auto event = log_event();
  RawDatabase competing(store.database_path());
  competing.execute("BEGIN IMMEDIATE");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
  EXPECT_THROW(append_one(store, "deadline", event, {}, deadline),
               jyppx::liuding::DeadlineExceeded);
  std::stop_source stop;
  std::latch entered(1);
  auto waiting = std::async(std::launch::async, [&] {
    entered.count_down();
    return append_one(store, "cancelled", event, stop.get_token());
  });
  entered.wait();
  stop.request_stop();
  EXPECT_THROW(waiting.get(), jyppx::liuding::Cancelled);
  competing.execute("ROLLBACK");
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_NO_THROW(append_one(store, "deadline", event));
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunEventTest,
     IntegrityDetectsOmittedMetricIndexAndChangedStateCounters) {
  for (unsigned damage = 0U; damage < 3U; ++damage) {
    SCOPED_TRACE(damage);
    TempRunRoot root;
    SQLiteRunStore store(root.path());
    store.start(make_run());
    append_one(store, "metrics", metric_event(1U, 3U));
    RawDatabase raw(store.database_path());
    if (damage == 0U)
      raw.execute("DELETE FROM jyppx_metric_points WHERE point_index=1");
    if (damage == 1U)
      raw.execute(
          "UPDATE jyppx_run_event_state SET content_bytes=content_bytes+1");
    if (damage == 2U)
      raw.execute("UPDATE jyppx_events SET event_id='tampered-event-id'");
    EXPECT_FALSE(store.integrity_check(false).ok());
    EXPECT_THROW(store.collect_garbage(std::chrono::seconds(0)),
                 jyppx::liuding::SerializationError);
  }
}

TEST(SQLiteRunEventTest, IntegrityAndGcRejectMissingEventArtifactRelationship) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  const auto artifact = publish_unattached(store);
  auto event = log_event();
  event.payload = jyppx::liuding::TelemetryArtifactReference{
      .role = "report", .artifact = artifact};
  append_one(store, "artifact", event);
  RawDatabase raw(store.database_path());
  raw.execute("DELETE FROM jyppx_run_artifacts");
  EXPECT_FALSE(store.integrity_check(false).ok());
  EXPECT_THROW(store.collect_garbage(std::chrono::seconds(0)),
               jyppx::liuding::SerializationError);
  FileArtifactStore cas(store.artifact_root());
  EXPECT_TRUE(cas.exists(artifact.digest));
}

void convert_empty_v2_to_v1(const std::filesystem::path& root) {
  // The v1 tables are copied unchanged by the v2 migration. This fixture first
  // creates real metadata/CAS with the public API, then removes only v2
  // objects.
  RawDatabase raw(root / "runs.sqlite3");
  raw.execute(
      "PRAGMA foreign_keys=OFF; BEGIN IMMEDIATE; "
      "DROP TABLE jyppx_event_batch_items; DROP TABLE jyppx_event_batches; "
      "DROP TABLE jyppx_metric_points; DROP TABLE jyppx_metric_definitions; "
      "DROP TABLE jyppx_events; DROP TABLE jyppx_event_streams; "
      "DROP TABLE jyppx_run_event_state; PRAGMA user_version=1; COMMIT;");
}

void make_v1_fixture(const std::filesystem::path& root) {
  {
    SQLiteRunStore initial(root);
    initial.start(make_run());
    initial.put_artifact("run-1", "legacy-weights", RunArtifactKind::checkpoint,
                         "last", bytes_of("v1-checkpoint"),
                         {.logical_name = "weights.bin",
                          .media_type = "application/octet-stream",
                          .producer = "v1"},
                         110U);
  }
  convert_empty_v2_to_v1(root);
}

TEST(SQLiteRunEventTest, MigratesV1InOneTransactionPreservingMetadataAndCas) {
  TempRunRoot root;
  make_v1_fixture(root.path());
  SQLiteRunStore migrated(root.path());
  const auto metadata = migrated.find("run-1");
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->tags, make_run().tags);
  EXPECT_EQ(metadata->lineage.config_digest, make_run().lineage.config_digest);
  const auto references = migrated.references("run-1");
  ASSERT_EQ(references.size(), 1U);
  EXPECT_EQ(references[0].reference_id, "legacy-weights");
  append_one(migrated, "after-migration", metric_event());
  EXPECT_TRUE(migrated.integrity_check().ok());
  RawDatabase raw(migrated.database_path());
  EXPECT_EQ(raw.scalar_text("PRAGMA user_version"), "2");
  EXPECT_EQ(raw.scalar_text("PRAGMA application_id"), "1247367760");
}

TEST(SQLiteRunEventTest, CanRefuseMigrationAndRejectsFutureOrForeignSchemas) {
  TempRunRoot root;
  make_v1_fixture(root.path());
  SQLiteRunStoreOptions options;
  options.allow_v1_migration = false;
  EXPECT_THROW({ SQLiteRunStore refused(root.path(), options); },
               jyppx::liuding::CompatibilityError);
  RawDatabase raw(root.path() / "runs.sqlite3");
  EXPECT_EQ(raw.scalar_text("PRAGMA user_version"), "1");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_runs"), "1");
  raw.execute("PRAGMA user_version=99");
  EXPECT_THROW({ SQLiteRunStore refused(root.path()); },
               jyppx::liuding::CompatibilityError);
  raw.execute("PRAGMA user_version=1; PRAGMA application_id=17;");
  EXPECT_THROW({ SQLiteRunStore refused(root.path()); },
               jyppx::liuding::CompatibilityError);
}

TEST(SQLiteRunEventTest, LateMigrationFailureRollsBackAllDdlAndVersionChanges) {
  TempRunRoot root;
  make_v1_fixture(root.path());
  {
    RawDatabase raw(root.path() / "runs.sqlite3");
    // Conflict near the end, after every new table would have been created.
    raw.execute("CREATE INDEX jyppx_metric_version_scan ON jyppx_runs(status)");
  }
  EXPECT_THROW({ SQLiteRunStore failed(root.path()); }, jyppx::liuding::Error);
  RawDatabase raw(root.path() / "runs.sqlite3");
  EXPECT_EQ(raw.scalar_text("PRAGMA user_version"), "1");
  EXPECT_EQ(
      raw.scalar_text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                      "AND name='jyppx_events'"),
      "0");
  EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM jyppx_run_artifacts"), "1");
  raw.execute("DROP INDEX jyppx_metric_version_scan");
  SQLiteRunStore retried(root.path());
  EXPECT_TRUE(retried.integrity_check().ok());
}

TEST(SQLiteRunEventTest, RunListIsBoundedFilteredAndOrderedByImmutableId) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run("run-c"));
  store.start(make_run("run-a"));
  auto other = make_run("run-b");
  other.project_id = "project-other";
  store.start(other);
  store.transition("run-c", RunStatus::completed, 150U);
  const auto first = store.list({.limit = 2U});
  ASSERT_EQ(first.size(), 2U);
  EXPECT_EQ(first[0].run_id, "run-a");
  EXPECT_EQ(first[1].run_id, "run-b");
  const auto next =
      store.list({.after_run_id = first.back().run_id, .limit = 2U});
  ASSERT_EQ(next.size(), 1U);
  EXPECT_EQ(next[0].run_id, "run-c");
  EXPECT_EQ(store.list({.project_id = "project-1"}).size(), 2U);
  EXPECT_EQ(store.list({.status = RunStatus::completed}).size(), 1U);
  EXPECT_TRUE(store.list({.experiment_id = "missing"}).empty());
  EXPECT_THROW(store.list({.limit = 0U}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(store.list({.limit = 1025U}), jyppx::liuding::InvalidArgument);
}

TEST(SQLiteRunEventTest,
     NeutralSinkReturnsDurablePositionalReceiptsAndClosesIdempotently) {
  TempRunRoot root;
  auto store = std::make_shared<SQLiteRunStore>(root.path());
  store->start(make_run());
  SQLiteRunTelemetrySink sink(store, "run-1", "evidence-1", "local-run-db");
  EXPECT_EQ(sink.descriptor().type_id, "jyppx.telemetry-sink.sqlite/v1");
  const std::array batch{metric_event(), log_event(2U)};
  const auto first = sink.emit(batch);
  jyppx::liuding::validate_telemetry_emit_result(sink.descriptor(), batch,
                                                 first);
  EXPECT_EQ(first.dispositions[0], TelemetryDisposition::accepted);
  const auto retried = sink.emit(batch);
  EXPECT_EQ(retried.dispositions[1], TelemetryDisposition::duplicate);
  auto changed = batch;
  std::get<TelemetryLog>(changed[1].payload).body =
      "same operation changed content";
  EXPECT_THROW(sink.emit(changed), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(sink.flush().exported_events, 2U);
  EXPECT_EQ(sink.close().exported_events, 2U);
  EXPECT_EQ(sink.close().exported_events, 2U);
  EXPECT_THROW(sink.emit(batch), jyppx::liuding::NotReady);
  SQLiteRunStore reopened(root.path());
  EXPECT_EQ(reopened.events("run-1").events.size(), 2U);
  EXPECT_TRUE(reopened.integrity_check(false).ok());
}

TEST(SQLiteRunEventCapacityTest,
     PersistsAndPagesOneHundredThousandMetricPoints) {
  TempRunRoot root;
  SQLiteRunStore store(root.path());
  store.start(make_run());
  constexpr std::size_t kPoints = 100000U;
  constexpr std::size_t kPointsPerEnvelope = 100U;
  constexpr std::size_t kEnvelopesPerBatch = 20U;
  const auto begin = std::chrono::steady_clock::now();
  std::uint64_t sequence = 1U;
  for (std::size_t offset = 0U; offset < kPoints;) {
    std::vector<TelemetryEnvelope> batch;
    batch.reserve(kEnvelopesPerBatch);
    for (std::size_t index = 0U; index < kEnvelopesPerBatch && offset < kPoints;
         ++index) {
      auto event = metric_event(sequence++, kPointsPerEnvelope);
      auto& points = std::get<TelemetryMetricBatch>(event.payload).points;
      for (std::size_t point = 0U; point < points.size(); ++point) {
        points[point].step.value = offset + point;
        points[point].observation = jyppx::liuding::make_telemetry_metric_value(
            static_cast<double>(offset + point));
      }
      offset += points.size();
      batch.push_back(std::move(event));
    }
    const auto receipt =
        store.append("run-1", "capacity-" + std::to_string(offset), batch);
    ASSERT_EQ(receipt.receipts.size(), batch.size());
  }
  const auto committed = std::chrono::steady_clock::now();
  RunMetricPageRequest request{.metric_id = "eval.accuracy", .limit = 4096U};
  std::uint64_t count = 0U;
  for (;;) {
    const auto page = store.metrics("run-1", request);
    ASSERT_FALSE(page.points.empty());
    request.snapshot_through = page.snapshot_through;
    for (const auto& point : page.points) {
      EXPECT_EQ(point.point.step.value, count);
      EXPECT_EQ(point.point.observation.value, static_cast<double>(count));
      EXPECT_EQ(point.point.definition.version, 1U);
      EXPECT_EQ(point.point.definition.unit,
                jyppx::liuding::TelemetryMetricUnit::ratio);
      ++count;
    }
    if (!page.next_after) break;
    request.after = *page.next_after;
  }
  EXPECT_EQ(count, kPoints);
  EXPECT_EQ(request.snapshot_through, kPoints / kPointsPerEnvelope);
  const auto end = std::chrono::steady_clock::now();
  RecordProperty(
      "append_ms",
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                         committed - begin)
                         .count()));
  RecordProperty(
      "page_ms",
      std::to_string(
          std::chrono::duration_cast<std::chrono::milliseconds>(end - committed)
              .count()));
  EXPECT_TRUE(store.integrity_check(false).ok());
}

TEST(SQLiteRunBackupTest,
     RestoresCompleteWalSnapshotAndExactReferencedCasMembership) {
  TempRunRoot root;
  SQLiteRunStore source(root.path() / "source");
  source.start(make_run());
  auto artifact_event = log_event(2U);
  const auto live = publish_unattached(source, "confirmed-live-object");
  artifact_event.payload = jyppx::liuding::TelemetryArtifactReference{
      .role = "report", .artifact = live};
  const std::array batch{metric_event(1U, 4U), artifact_event};
  source.append("run-1", "snapshot-batch", batch);
  const auto orphan =
      publish_unattached(source, "unreferenced-object-is-not-backup-evidence");
  const auto backup = root.path() / "backups" / "snapshot-1";
  const auto saved = source.backup(backup);
  EXPECT_EQ(saved.artifact_count, 1U);
  EXPECT_EQ(saved.artifact_bytes, live.size_bytes);
  EXPECT_GT(saved.database_bytes, 0U);
  EXPECT_TRUE(std::filesystem::exists(backup / "manifest.jyppx-backup"));
  EXPECT_FALSE(std::filesystem::exists(backup / "runs.sqlite3-wal"));
  EXPECT_FALSE(std::filesystem::exists(backup / "runs.sqlite3-shm"));
  append_one(source, "after-snapshot", log_event(3U));
  const auto restored_path = root.path() / "restored" / "run-store";
  const auto restored = SQLiteRunStore::restore_backup(backup, restored_path);
  EXPECT_EQ(restored.artifact_count, saved.artifact_count);
  EXPECT_EQ(restored.artifact_bytes, saved.artifact_bytes);
  SQLiteRunStore recovered(restored_path);
  EXPECT_EQ(recovered.events("run-1").events.size(), 2U);
  EXPECT_EQ(
      recovered.metrics("run-1", {.metric_id = "eval.accuracy"}).points.size(),
      4U);
  EXPECT_TRUE(recovered.append("run-1", "snapshot-batch", batch).batch_replay);
  EXPECT_EQ(recovered.find("run-1")->tags, make_run().tags);
  EXPECT_EQ(recovered.find("run-1")->lineage.build_fingerprint,
            make_run().lineage.build_fingerprint);
  EXPECT_TRUE(recovered.integrity_check().ok());
  FileArtifactStore cas(recovered.artifact_root());
  EXPECT_TRUE(cas.exists(live.digest));
  EXPECT_FALSE(cas.exists(orphan.digest));
  EXPECT_EQ(recovered.collect_garbage(std::chrono::seconds(0)).removed_files,
            0U);
  EXPECT_EQ(source.events("run-1").events.size(), 3U);
}

TEST(SQLiteRunBackupTest,
     NeverOverwritesLiveStoreExistingDestinationOrExistingRestore) {
  TempRunRoot root;
  SQLiteRunStore source(root.path() / "source");
  source.start(make_run());
  const auto backup = root.path() / "backup";
  source.backup(backup);
  EXPECT_THROW(source.backup(backup), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(source.backup(source.root() / "nested-backup"),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(SQLiteRunStore::restore_backup(backup, source.root()),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      SQLiteRunStore::restore_backup(backup, backup / "nested-restore"),
      jyppx::liuding::InvalidArgument);
  EXPECT_TRUE(source.find("run-1"));
  EXPECT_TRUE(source.integrity_check().ok());
}

TEST(SQLiteRunBackupTest,
     RejectsCorruptDatabaseManifestArtifactAndUnmanifestedWal) {
  for (unsigned damage = 0U; damage < 4U; ++damage) {
    SCOPED_TRACE(damage);
    TempRunRoot root;
    SQLiteRunStore source(root.path() / "source");
    source.start(make_run());
    const auto live = publish_unattached(source);
    auto event = log_event();
    event.payload = jyppx::liuding::TelemetryArtifactReference{
        .role = "report", .artifact = live};
    append_one(source, "artifact", event);
    const auto backup = root.path() / "backup";
    source.backup(backup);
    std::filesystem::path corrupt;
    if (damage == 0U) corrupt = backup / "runs.sqlite3";
    if (damage == 1U) corrupt = backup / "manifest.jyppx-backup";
    if (damage == 2U)
      corrupt = backup / "artifacts" / "objects" /
                live.digest.hex.substr(0U, 2U) / live.digest.hex.substr(2U);
    if (damage == 3U) corrupt = backup / "runs.sqlite3-wal";
    {
      std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
      output << "corrupt-backup-evidence";
    }
    const auto target = root.path() / "target";
    EXPECT_THROW(SQLiteRunStore::restore_backup(backup, target),
                 jyppx::liuding::Error);
    EXPECT_FALSE(std::filesystem::exists(target));
    EXPECT_TRUE(source.integrity_check().ok());
  }
}

TEST(SQLiteRunBackupTest,
     CancellationAndDatabaseQuotaLeaveNoPublishedDestination) {
  TempRunRoot root;
  SQLiteRunStoreOptions options;
  options.max_backup_database_bytes = 1U;
  SQLiteRunStore bounded(root.path() / "source", options);
  bounded.start(make_run());
  EXPECT_THROW(bounded.backup(root.path() / "quota-rejected"),
               jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(std::filesystem::exists(root.path() / "quota-rejected"));
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(bounded.backup(root.path() / "cancelled", stop.get_token()),
               jyppx::liuding::Cancelled);
  EXPECT_FALSE(std::filesystem::exists(root.path() / "cancelled"));
  for (const auto& entry : std::filesystem::directory_iterator(root.path()))
    EXPECT_FALSE(
        entry.path().filename().string().starts_with(".jyppx-backup-stage-"));
}

TEST(SQLiteRunEventCasTest,
     VerifiedReaderRetainsOriginalHandleAfterPathReplacement) {
  TempRunRoot root;
  FileArtifactStore cas(root.path() / "new" / "nested" / "artifacts");
  const auto original = cas.put(
      bytes_of("verified-original"),
      {.logical_name = "content", .media_type = "application/octet-stream"});
  auto reader = cas.open(original.digest);
  const auto path = cas.root() / "objects" /
                    original.digest.hex.substr(0U, 2U) /
                    original.digest.hex.substr(2U);
  const auto retired = root.path() / "retired-original";
  std::filesystem::rename(path, retired);
  {
    std::ofstream replacement(path, std::ios::binary);
    replacement << "replacement-that-does-not-match-address";
  }
  std::vector<std::byte> result(static_cast<std::size_t>(original.size_bytes));
  ASSERT_EQ(reader->read(result), result.size());
  EXPECT_EQ(jyppx::liuding::sha256_hex(result), original.digest.hex);
  EXPECT_THROW(cas.open(original.digest), jyppx::liuding::SerializationError);
}

#if GTEST_HAS_DEATH_TEST
TEST(SQLiteRunEventDeathTest,
     AbruptExitAfterAcknowledgmentRetainsBatchMetricsAndCas) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  AbruptExitRoot root;
  ASSERT_EXIT(([&] {
                SQLiteRunStore store(root.path());
                store.start(make_run());
                auto artifact = log_event(2U);
                artifact.payload = jyppx::liuding::TelemetryArtifactReference{
                    .role = "report", .artifact = publish_unattached(store)};
                const std::array batch{metric_event(1U, 4U), artifact};
                store.append("run-1", "confirmed", batch);
                std::_Exit(0);
              }()),
              ::testing::ExitedWithCode(0), "");
  ASSERT_TRUE(std::filesystem::exists(root.path() / "runs.sqlite3-wal"));
  SQLiteRunStore recovered(root.path());
  const auto original = recovered.events("run-1");
  ASSERT_EQ(original.events.size(), 2U);
  std::vector<TelemetryEnvelope> replay;
  for (const auto& event : original.events) replay.push_back(event.envelope);
  const auto receipt = recovered.append("run-1", "confirmed", replay);
  EXPECT_TRUE(receipt.batch_replay);
  EXPECT_EQ(receipt.committed_through, 2U);
  EXPECT_EQ(receipt.receipts[0].disposition, TelemetryDisposition::duplicate);
  EXPECT_EQ(
      recovered.metrics("run-1", {.metric_id = "eval.accuracy"}).points.size(),
      4U);
  EXPECT_EQ(recovered.references("run-1").size(), 1U);
  EXPECT_EQ(recovered.collect_garbage(std::chrono::seconds(0)).removed_files,
            0U);
  EXPECT_TRUE(recovered.integrity_check().ok());
}

TEST(SQLiteRunEventDeathTest,
     AbruptExitDuringUncommittedEventMutationDiscardsChanges) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  AbruptExitRoot root;
  ASSERT_EXIT(
      ([&] {
        SQLiteRunStore store(root.path());
        store.start(make_run());
        append_one(store, "before-crash", metric_event(1U, 3U));
        RawDatabase uncommitted(store.database_path());
        uncommitted.execute(
            "BEGIN IMMEDIATE; DELETE FROM jyppx_metric_points; "
            "DELETE FROM jyppx_event_batch_items; DELETE FROM "
            "jyppx_event_batches; "
            "UPDATE jyppx_run_event_state SET metric_count=0,batch_count=0;");
        std::_Exit(17);
      }()),
      ::testing::ExitedWithCode(17), "");
  SQLiteRunStore recovered(root.path());
  EXPECT_EQ(recovered.events("run-1").events.size(), 1U);
  EXPECT_EQ(
      recovered.metrics("run-1", {.metric_id = "eval.accuracy"}).points.size(),
      3U);
  EXPECT_TRUE(
      append_one(recovered, "before-crash", metric_event(1U, 3U)).batch_replay);
  EXPECT_TRUE(recovered.integrity_check(false).ok());
}

TEST(SQLiteRunEventDeathTest,
     AbruptExitDuringUncommittedMigrationLeavesV1Reusable) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  AbruptExitRoot root;
  ASSERT_EXIT(
      ([&] {
        make_v1_fixture(root.path());
        RawDatabase raw(root.path() / "runs.sqlite3");
        raw.execute(
            "BEGIN IMMEDIATE; CREATE TABLE jyppx_run_event_state(run_id TEXT); "
            "CREATE TABLE jyppx_event_streams(stream_id TEXT); PRAGMA "
            "user_version=2;");
        std::_Exit(17);
      }()),
      ::testing::ExitedWithCode(17), "");
  {
    RawDatabase raw(root.path() / "runs.sqlite3");
    EXPECT_EQ(raw.scalar_text("PRAGMA user_version"), "1");
    EXPECT_EQ(raw.scalar_text("SELECT COUNT(*) FROM sqlite_master WHERE "
                              "name='jyppx_event_streams'"),
              "0");
  }
  SQLiteRunStore migrated(root.path());
  EXPECT_EQ(migrated.references("run-1").size(), 1U);
  EXPECT_TRUE(migrated.integrity_check().ok());
}
#endif

}  // namespace
