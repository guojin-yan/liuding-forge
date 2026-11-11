// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::experiment;

std::span<const std::byte> bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

void write(const std::filesystem::path& path, std::string_view value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  output.close();
  if (!output) throw std::runtime_error("Could not write publication fixture");
}

std::string read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Could not read publication fixture");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

class FilePublicationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<unsigned> sequence{0U};
    root = std::filesystem::temp_directory_path() /
           ("liuding-publication-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1U)));
    ASSERT_TRUE(std::filesystem::create_directory(root));
    payload.resize(64U * 1024U + 257U);
    for (std::size_t i = 0U; i < payload.size(); ++i)
      payload[i] = static_cast<char>((i * 37U + 11U) & 255U);
  }
  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }
  void start(SQLiteRunStore& store) {
    store.start({.run_id = "run-1",
                 .project_id = "project-1",
                 .experiment_id = "experiment-1",
                 .display_name = "publication",
                 .status = RunStatus::running,
                 .created_at_unix_ms = 100U,
                 .updated_at_unix_ms = 100U});
  }
  FilePublicationRequest request(std::string id = "intent",
                                 bool file_source = true) {
    FilePublicationRequest result;
    result.publication_id = id;
    result.destination = root / (id + ".bin");
    auto& evidence = result.evidence;
    evidence.run_id = "run-1";
    evidence.reference_id = "result-" + id;
    if (file_source) {
      evidence.path = root / (id + ".input");
      write(*evidence.path, payload);
    }
    evidence.descriptor = {.logical_name = "preview.bin",
                           .media_type = "application/octet-stream",
                           .producer = "publication-test",
                           .expected_size_bytes = payload.size()};
    evidence.attached_at_unix_ms = 101U;
    evidence.expected_digest = {.hex = sha256_hex(bytes(payload))};
    evidence.event_batch_key = "batch-" + id;
    evidence.event.identity = {.stream_id = "stream-" + id,
                               .event_id = id,
                               .idempotency_key = id,
                               .sequence = 1U};
    evidence.event.correlation.project_id = "project-1";
    evidence.event.correlation.run_id = "run-1";
    evidence.event.payload =
        TelemetryArtifactReference{.role = "image.preview"};
    return result;
  }
  std::filesystem::path object(SQLiteRunStore& store,
                               const ArtifactDigest& digest) {
    return store.artifact_root() / "objects" / digest.hex.substr(0U, 2U) /
           digest.hex.substr(2U);
  }
  std::filesystem::path root;
  std::string payload;
};

TEST_F(FilePublicationTest, BorrowedBytesRecoverAfterProducerStorageIsGone) {
  const auto value = request("intent", false);
  {
    SQLiteRunStore store(root / "store");
    start(store);
    std::string owned = payload;
    store.prepare_file_publication_bytes(value, bytes(owned));
    std::string().swap(owned);
    EXPECT_FALSE(std::filesystem::exists(root / "intent.input"));
    EXPECT_FALSE(std::filesystem::exists(value.destination));
    EXPECT_EQ(store.references("run-1").size(), 2U);
    EXPECT_TRUE(store.events("run-1").events.empty());
  }
  SQLiteRunStore reopened(root / "store");
  EXPECT_EQ(reopened.file_publications("run-1"),
            std::vector<std::string>{"intent"});
  const auto result = reopened.recover_file_publication("run-1", "intent");
  EXPECT_EQ(read(result.path), payload);
  EXPECT_EQ(reopened.references("run-1").size(), 4U);
  EXPECT_EQ(reopened.events("run-1").events.size(), 1U);
  EXPECT_TRUE(reopened.integrity_check().ok());
}

TEST_F(FilePublicationTest, ByteAndFilePreparationShareImmutableReplay) {
  SQLiteRunStore store(root / "store");
  start(store);
  auto value = request("intent", false);
  const auto first =
      store.prepare_file_publication_bytes(value, bytes(payload));
  EXPECT_EQ(store.prepare_file_publication_bytes(value, {}).artifact.digest,
            first.artifact.digest);
  EXPECT_EQ(store.prepare_file_publication(value).artifact.digest,
            first.artifact.digest);
  auto changed = value;
  changed.evidence.event.identity.event_id = "different-event";
  EXPECT_THROW(store.prepare_file_publication_bytes(changed, {}),
               InvalidArgument);
  const auto second = request("file");
  const auto file = store.prepare_file_publication(second);
  value = second;
  value.evidence.path.reset();
  EXPECT_EQ(store.prepare_file_publication_bytes(value, {}).artifact.digest,
            file.artifact.digest);
  EXPECT_EQ(store.references("run-1").size(), 4U);
}

TEST_F(FilePublicationTest, BytePreparationRejectsAmbiguousOrInvalidContent) {
  SQLiteRunStore store(root / "store");
  start(store);
  auto value = request("intent", false);
  EXPECT_THROW(store.prepare_file_publication_bytes(value, {}),
               InvalidArgument);
  std::string wrong(payload.size(), 'x');
  EXPECT_THROW(store.prepare_file_publication_bytes(value, bytes(wrong)),
               Error);
  auto ambiguous = value;
  ambiguous.evidence.path = root / "absent.input";
  EXPECT_THROW(store.prepare_file_publication_bytes(ambiguous, bytes(payload)),
               InvalidArgument);
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_TRUE(store.file_publications("run-1").empty());
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_FALSE(std::filesystem::exists(value.destination));
  EXPECT_NO_THROW(store.prepare_file_publication_bytes(value, bytes(payload)));
  write(object(store, value.evidence.expected_digest), wrong);
  EXPECT_THROW(store.prepare_file_publication_bytes(value, {}), Error);
  EXPECT_FALSE(std::filesystem::exists(value.destination));
}

TEST_F(FilePublicationTest, BytePreparationKeepsCancellationAndQuotasAtomic) {
  SQLiteRunStoreOptions options;
  options.max_references_per_run = 1U;
  SQLiteRunStore store(root / "store", options);
  start(store);
  const auto value = request("intent", false);
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(store.prepare_file_publication_bytes(value, bytes(payload),
                                                    stopped.get_token()),
               Error);
  EXPECT_THROW(store.prepare_file_publication_bytes(
                   value, bytes(payload), {},
                   std::chrono::steady_clock::now() - std::chrono::seconds(1)),
               Error);
  EXPECT_THROW(store.prepare_file_publication_bytes(value, bytes(payload)),
               ResourceExhausted);
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_TRUE(store.file_publications("run-1").empty());
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_FALSE(std::filesystem::exists(value.destination));
}

TEST_F(FilePublicationTest, BytePreparationSupportsEmptyAndConcurrentInputs) {
  SQLiteRunStore first(root / "store");
  start(first);
  SQLiteRunStore second(root / "store");
  const auto value = request("intent", false);
  auto a = std::async(std::launch::async, [&] {
    return first.prepare_file_publication_bytes(value, bytes(payload));
  });
  auto b = std::async(std::launch::async, [&] {
    return second.prepare_file_publication_bytes(value, bytes(payload));
  });
  EXPECT_EQ(a.get().artifact.digest, b.get().artifact.digest);
  EXPECT_EQ(first.references("run-1").size(), 2U);
  auto empty = request("empty", false);
  empty.evidence.descriptor.expected_size_bytes = 0;
  empty.evidence.expected_digest = {.hex = sha256_hex(bytes(""))};
  first.prepare_file_publication_bytes(empty, {});
  const auto result = first.recover_file_publication("run-1", "empty");
  EXPECT_EQ(std::filesystem::file_size(result.path), 0U);
  EXPECT_EQ(first.references("run-1").size(), 6U);
}

TEST_F(FilePublicationTest, DurableIntentRecoversWithoutInputOrProducerState) {
  auto value = request();
  {
    SQLiteRunStore store(root / "store");
    start(store);
    store.prepare_file_publication(value);
    EXPECT_EQ(store.references("run-1").size(), 2U);
    EXPECT_TRUE(store.events("run-1").events.empty());
    EXPECT_FALSE(std::filesystem::exists(value.destination));
  }
  ASSERT_TRUE(std::filesystem::remove(*value.evidence.path));
  SQLiteRunStore reopened(root / "store");
  ASSERT_EQ(reopened.file_publications("run-1"),
            std::vector<std::string>{"intent"});
  EXPECT_EQ(
      reopened.collect_garbage(std::chrono::seconds::zero()).removed_files, 0U);
  const auto first = reopened.recover_file_publication("run-1", "intent");
  EXPECT_EQ(first.path, std::filesystem::canonical(value.destination));
  EXPECT_FALSE(first.evidence.reference_replay);
  EXPECT_FALSE(first.evidence.event.batch_replay);
  EXPECT_EQ(read(first.path), payload);
  EXPECT_EQ(reopened.references("run-1").size(), 4U);
  EXPECT_EQ(reopened.events("run-1").events.size(), 1U);
  const auto timestamp = std::filesystem::last_write_time(first.path);
  const auto replay = reopened.recover_file_publication("run-1", "intent");
  EXPECT_TRUE(replay.evidence.reference_replay);
  EXPECT_TRUE(replay.evidence.event.batch_replay);
  EXPECT_EQ(std::filesystem::last_write_time(first.path), timestamp);
  ASSERT_TRUE(std::filesystem::remove(first.path));
  EXPECT_TRUE(reopened.recover_file_publication("run-1", "intent")
                  .evidence.event.batch_replay);
  EXPECT_EQ(read(first.path), payload);
  EXPECT_TRUE(reopened.integrity_check().ok());
}

TEST_F(FilePublicationTest, PreparedIdentityRejectsChangedDestinationAndEvent) {
  SQLiteRunStore store(root / "store");
  start(store);
  auto value = request();
  const auto intent = store.prepare_file_publication(value);
  ASSERT_TRUE(std::filesystem::remove(*value.evidence.path));
  value.evidence.path.reset();
  EXPECT_EQ(store.prepare_file_publication(value).artifact.digest,
            intent.artifact.digest);
  auto changed = value;
  changed.destination = root / "different.bin";
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  changed = value;
  changed.evidence.event.identity.event_id = "different-event";
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  changed = value;
  changed.evidence.attached_at_unix_ms = 102U;
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  changed = value;
  changed.evidence.reference_id = changed.publication_id;
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(value.destination));
  EXPECT_EQ(store.references("run-1").size(), 2U);
}

TEST_F(FilePublicationTest, ExistingDifferentBytesArePreservedAndSqlRollsBack) {
  SQLiteRunStore store(root / "store");
  start(store);
  const auto value = request();
  store.prepare_file_publication(value);
  const std::string conflicting(payload.size(), 'x');
  write(value.destination, conflicting);
  const auto time = std::filesystem::last_write_time(value.destination);
  EXPECT_THROW(store.recover_file_publication("run-1", "intent"), Error);
  EXPECT_EQ(read(value.destination), conflicting);
  EXPECT_EQ(std::filesystem::last_write_time(value.destination), time);
  EXPECT_EQ(store.references("run-1").size(), 2U);
  EXPECT_TRUE(store.events("run-1").events.empty());
  ASSERT_TRUE(std::filesystem::remove(value.destination));
  EXPECT_FALSE(store.recover_file_publication("run-1", "intent")
                   .evidence.reference_replay);
  EXPECT_EQ(read(value.destination), payload);
}

TEST_F(FilePublicationTest, InvalidEventCannotPublishAFile) {
  SQLiteRunStore store(root / "store");
  start(store);
  auto value = request();
  value.evidence.event.identity.sequence = 2U;
  store.prepare_file_publication(value);
  EXPECT_THROW(store.recover_file_publication("run-1", "intent"), Error);
  EXPECT_FALSE(std::filesystem::exists(value.destination));
  EXPECT_EQ(store.references("run-1").size(), 2U);
  EXPECT_TRUE(store.events("run-1").events.empty());
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST_F(FilePublicationTest,
       PreparationAndCompletionRespectWholeReferenceQuotas) {
  const auto value = request();
  {
    SQLiteRunStoreOptions options;
    options.max_references_per_run = 1U;
    SQLiteRunStore store(root / "too-small", options);
    start(store);
    EXPECT_THROW(store.prepare_file_publication(value), ResourceExhausted);
    EXPECT_TRUE(store.references("run-1").empty());
    EXPECT_TRUE(store.file_publications("run-1").empty());
    EXPECT_FALSE(std::filesystem::exists(value.destination));
    EXPECT_EQ(store.collect_garbage(std::chrono::seconds::zero()).removed_files,
              2U);
  }
  {
    SQLiteRunStoreOptions options;
    options.max_references_per_run = 3U;
    SQLiteRunStore store(root / "store", options);
    start(store);
    store.prepare_file_publication(value);
    EXPECT_THROW(store.recover_file_publication("run-1", "intent"),
                 ResourceExhausted);
    EXPECT_FALSE(std::filesystem::exists(value.destination));
    EXPECT_EQ(store.references("run-1").size(), 2U);
    EXPECT_TRUE(store.events("run-1").events.empty());
  }
  SQLiteRunStore reopened(root / "store");
  EXPECT_FALSE(reopened.recover_file_publication("run-1", "intent")
                   .evidence.reference_replay);
}

TEST_F(FilePublicationTest, MissingOrCorruptCasNeverCreatesFinalFile) {
  SQLiteRunStore store(root / "store");
  start(store);
  const auto value = request();
  const auto intent = store.prepare_file_publication(value);
  const auto content = object(store, value.evidence.expected_digest);
  ASSERT_TRUE(std::filesystem::remove(content));
  EXPECT_THROW(store.recover_file_publication("run-1", "intent"), Error);
  write(content, std::string(payload.size(), 'x'));
  EXPECT_THROW(store.recover_file_publication("run-1", "intent"), Error);
  write(content, payload);
  const auto journal = object(store, intent.artifact.digest);
  const auto original = read(journal);
  write(journal, std::string(original.size(), 'x'));
  EXPECT_THROW(store.recover_file_publication("run-1", "intent"), Error);
  EXPECT_FALSE(std::filesystem::exists(value.destination));
  EXPECT_EQ(store.references("run-1").size(), 2U);
  write(journal, original);
  EXPECT_NO_THROW(store.recover_file_publication("run-1", "intent"));
}

TEST_F(FilePublicationTest, PendingIntentAndContentSurviveBackupRestore) {
  const auto value = request();
  {
    SQLiteRunStore store(root / "store");
    start(store);
    store.prepare_file_publication(value);
    const auto backup = store.backup(root / "backup");
    EXPECT_EQ(backup.artifact_count, 2U);
  }
  ASSERT_TRUE(std::filesystem::remove(*value.evidence.path));
  const auto restored =
      SQLiteRunStore::restore_backup(root / "backup", root / "restored");
  EXPECT_EQ(restored.artifact_count, 2U);
  SQLiteRunStore store(root / "restored");
  ASSERT_EQ(store.file_publications("run-1"),
            std::vector<std::string>{"intent"});
  EXPECT_NO_THROW(store.recover_file_publication("run-1", "intent"));
  EXPECT_EQ(read(value.destination), payload);
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST_F(FilePublicationTest, IntentDiscoveryUsesExclusiveBoundedPages) {
  SQLiteRunStore store(root / "store");
  start(store);
  for (const auto* id : {"c", "a", "b"})
    store.prepare_file_publication(request(id));
  EXPECT_EQ(store.file_publications("run-1", {}, 2U),
            (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(store.file_publications("run-1", "b", 2U),
            std::vector<std::string>{"c"});
  EXPECT_TRUE(store.file_publications("run-1", "c", 2U).empty());
  EXPECT_THROW(store.file_publications("run-1", {}, 0U), InvalidArgument);
  EXPECT_THROW(store.file_publications("run-1", {}, 1025U), InvalidArgument);
  EXPECT_THROW(store.file_publications("missing"), Error);
}

TEST_F(FilePublicationTest,
       IndependentConnectionsPrepareAndRecoverIdempotently) {
  SQLiteRunStore first(root / "store");
  start(first);
  SQLiteRunStore second(root / "store");
  const auto value = request();
  auto a = std::async(std::launch::async,
                      [&] { return first.prepare_file_publication(value); });
  auto b = std::async(std::launch::async,
                      [&] { return second.prepare_file_publication(value); });
  EXPECT_EQ(a.get().artifact.digest, b.get().artifact.digest);
  auto c = std::async(std::launch::async, [&] {
    return first.recover_file_publication("run-1", "intent");
  });
  auto d = std::async(std::launch::async, [&] {
    return second.recover_file_publication("run-1", "intent");
  });
  EXPECT_NE(c.get().evidence.reference_replay,
            d.get().evidence.reference_replay);
  EXPECT_EQ(first.references("run-1").size(), 4U);
  EXPECT_EQ(first.events("run-1").events.size(), 1U);
  EXPECT_EQ(read(value.destination), payload);
}

TEST_F(FilePublicationTest,
       CancellationDeadlineAndManagedPathFailuresAreNonPublishing) {
  SQLiteRunStore store(root / "store");
  start(store);
  auto value = request();
  std::stop_source stopped;
  stopped.request_stop();
  const auto expired =
      std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
  EXPECT_THROW(store.prepare_file_publication(value, stopped.get_token()),
               Error);
  EXPECT_THROW(store.prepare_file_publication(value, {}, expired), Error);
  auto changed = value;
  changed.destination = root / "store" / "output.bin";
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  changed.destination = root / "missing" / "output.bin";
  EXPECT_THROW(store.prepare_file_publication(changed), InvalidArgument);
  EXPECT_TRUE(store.references("run-1").empty());
  store.prepare_file_publication(value);
  EXPECT_THROW(
      store.recover_file_publication("run-1", "intent", stopped.get_token()),
      Error);
  EXPECT_THROW(store.recover_file_publication("run-1", "intent", {}, expired),
               Error);
  EXPECT_THROW(store.file_publications("run-1", {}, 64U, stopped.get_token()),
               Error);
  EXPECT_FALSE(std::filesystem::exists(value.destination));
  EXPECT_EQ(store.references("run-1").size(), 2U);
}

TEST_F(FilePublicationTest, EmptyContentCanBePublishedAndReplayed) {
  payload.clear();
  SQLiteRunStore store(root / "store");
  start(store);
  const auto value = request();
  store.prepare_file_publication(value);
  store.recover_file_publication("run-1", "intent");
  EXPECT_EQ(std::filesystem::file_size(value.destination), 0U);
  EXPECT_TRUE(store.recover_file_publication("run-1", "intent")
                  .evidence.event.batch_replay);
  EXPECT_TRUE(store.integrity_check().ok());
}

TEST_F(FilePublicationTest, ExistingIdenticalOrdinaryFileKeepsItsTimestamp) {
  SQLiteRunStore store(root / "store");
  start(store);
  const auto value = request();
  store.prepare_file_publication(value);
  write(value.destination, payload);
  const auto timestamp = std::filesystem::last_write_time(value.destination);
  EXPECT_NO_THROW(store.recover_file_publication("run-1", "intent"));
  EXPECT_EQ(std::filesystem::last_write_time(value.destination), timestamp);
  EXPECT_EQ(read(value.destination), payload);
}

TEST_F(FilePublicationTest,
       ReclaimsPartialStageAndRefusesToTruncateHardLinkedStage) {
  SQLiteRunStore store(root / "store");
  start(store);
  const auto value = request();
  const auto intent = store.prepare_file_publication(value);
  const auto stage =
      root / (".liuding-publication-" + intent.artifact.digest.hex + ".stage");
  write(stage, "interrupted partial copy");
  ASSERT_TRUE(std::filesystem::remove(*value.evidence.path));
  store.recover_file_publication("run-1", "intent");
  EXPECT_EQ(read(value.destination), payload);
  EXPECT_FALSE(std::filesystem::exists(stage));
  const auto other = request("second");
  const auto pending = store.prepare_file_publication(other);
  const auto linked_stage =
      root / (".liuding-publication-" + pending.artifact.digest.hex + ".stage");
  const auto ordinary = root / "ordinary.bin";
  write(ordinary, "preserve ordinary bytes");
  std::filesystem::create_hard_link(ordinary, linked_stage);
  EXPECT_THROW(store.recover_file_publication("run-1", "second"), Error);
  EXPECT_EQ(read(ordinary), "preserve ordinary bytes");
  EXPECT_FALSE(std::filesystem::exists(other.destination));
  EXPECT_EQ(store.events("run-1").events.size(), 1U);
}

TEST_F(FilePublicationTest, ValidCasWithMalformedIntentCannotPublish) {
  SQLiteRunStore store(root / "store");
  start(store);
  const std::string malformed = "invalid intent";
  store.put_artifact("run-1", "bad", RunArtifactKind::artifact,
                     "file.publication.intent", bytes(malformed),
                     {.logical_name = "file-publication-v1",
                      .media_type = "application/vnd.jyppx.file-publication-v1",
                      .producer = "experiment.file-publication",
                      .expected_size_bytes = malformed.size()},
                     101U);
  EXPECT_EQ(store.file_publications("run-1"), std::vector<std::string>{"bad"});
  EXPECT_THROW(store.recover_file_publication("run-1", "bad"), Error);
  EXPECT_EQ(store.references("run-1").size(), 1U);
  EXPECT_TRUE(store.events("run-1").events.empty());
}
}  // namespace
