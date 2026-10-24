// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/artifact_store.hpp>
#include <jyppx/liuding/experiment/run.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using jyppx::liuding::ArtifactDigest;
using jyppx::liuding::ArtifactDigestAlgorithm;
using jyppx::liuding::experiment::ArtifactDescriptor;
using jyppx::liuding::experiment::ArtifactStoreLimits;
using jyppx::liuding::experiment::FileArtifactStore;

std::span<const std::byte> bytes_of(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

class TempArtifactRoot {
 public:
  TempArtifactRoot() {
    static std::atomic<std::uint64_t> counter{0U};
    const auto temporary = std::filesystem::temp_directory_path();
    for (unsigned attempt = 0U; attempt != 64U; ++attempt) {
      const auto now =
          std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate =
          temporary / ("jyppx-artifact-" + std::to_string(now) + "-" +
                       std::to_string(std::random_device{}()) + "-" +
                       std::to_string(++counter));
      if (std::filesystem::create_directory(candidate)) {
        path_ = std::move(candidate);
        return;
      }
    }
    throw std::runtime_error("could not reserve an artifact test directory");
  }

  ~TempArtifactRoot() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TempArtifactRoot(const TempArtifactRoot&) = delete;
  TempArtifactRoot& operator=(const TempArtifactRoot&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

ArtifactDescriptor descriptor(std::uint64_t expected_size = 0U) {
  return {.logical_name = "weights.bin",
          .media_type = "application/octet-stream",
          .producer = "unit-test",
          .expected_size_bytes = expected_size};
}

TEST(ArtifactStoreTest, ComputesKnownSha256AndRejectsNonCanonicalDigest) {
  EXPECT_EQ(jyppx::liuding::sha256_hex(bytes_of("abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  ArtifactDigest uppercase{.algorithm = ArtifactDigestAlgorithm::sha256,
                           .hex = std::string(64U, 'A')};
  EXPECT_THROW(jyppx::liuding::validate_artifact_digest(uppercase),
               jyppx::liuding::InvalidArgument);
}

TEST(ArtifactStoreTest, StreamsPublishesAndReopensImmutableBlob) {
  TempArtifactRoot root;
  FileArtifactStore store(root.path());
  auto upload = store.begin(descriptor(11U));
  upload->write(bytes_of("hello "));
  upload->write(bytes_of("world"));
  const auto reference = store.commit(*upload);

  EXPECT_EQ(reference.size_bytes, 11U);
  EXPECT_TRUE(store.exists(reference.digest));
  const auto expected =
      "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
  EXPECT_EQ(reference.digest.hex, expected);

  auto reader = store.open(reference.digest);
  std::array<std::byte, 32> buffer{};
  const auto count = reader->read(buffer);
  EXPECT_EQ(count, 11U);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(buffer.data()), count),
            "hello world");
  EXPECT_EQ(reader->read(buffer), 0U);
  EXPECT_TRUE(std::filesystem::exists(root.path() / "objects" / "b9" /
                                      reference.digest.hex.substr(2U)));
}

TEST(ArtifactStoreTest, VerifiesMultipartBlobAcrossHashBufferBoundaries) {
  TempArtifactRoot root;
  FileArtifactStore store(root.path());
  std::vector<std::byte> payload(2U * 1024U * 1024U + 37U);
  for (std::size_t index = 0U; index < payload.size(); ++index) {
    payload[index] = static_cast<std::byte>(index % 251U);
  }
  const std::span<const std::byte> bytes(payload);
  auto upload = store.begin(descriptor(payload.size()));
  upload->write(bytes.first(701U));
  upload->write(bytes.subspan(701U, 1024U * 1024U));
  upload->write(bytes.subspan(701U + 1024U * 1024U));
  const auto reference = store.commit(*upload);
  EXPECT_EQ(reference.size_bytes, payload.size());
  EXPECT_EQ(reference.digest.hex, jyppx::liuding::sha256_hex(bytes));

  auto reader = store.open(reference.digest);
  std::vector<std::byte> buffer(65537U);
  std::size_t offset = 0U;
  while (const auto count = reader->read(buffer)) {
    ASSERT_LE(count, payload.size() - offset);
    EXPECT_TRUE(std::equal(buffer.begin(), buffer.begin() + count,
                           payload.begin() + offset));
    offset += count;
  }
  EXPECT_EQ(offset, payload.size());
  EXPECT_TRUE(store.verify().ok());
}

TEST(ArtifactStoreTest, DuplicateUploadIsIdempotentAndExpectedHashIsChecked) {
  TempArtifactRoot root;
  FileArtifactStore store(root.path());
  const auto first = store.put(bytes_of("same"), descriptor(4U));
  const auto second = store.put(bytes_of("same"), descriptor(4U), first.digest);
  EXPECT_EQ(first.digest, second.digest);
  EXPECT_THROW(store.put(bytes_of("other"), descriptor(5U), first.digest),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(store.put(bytes_of("too long"), descriptor(2U)),
               jyppx::liuding::ResourceExhausted);
}

TEST(ArtifactStoreTest, BoundsArtifactStoreAndRootMetadata) {
  TempArtifactRoot root;
  ArtifactStoreLimits limits;
  limits.max_artifact_bytes = 4U;
  limits.max_total_bytes = 6U;
  limits.max_object_files = 2U;
  FileArtifactStore store(root.path(), limits);
  const auto first = store.put(bytes_of("four"), descriptor(4U));
  EXPECT_THROW(store.put(bytes_of("more"), descriptor(4U)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_TRUE(store.exists(first.digest));

  limits.max_root_path_bytes = 1U;
  EXPECT_THROW(FileArtifactStore(root.path(), limits),
               jyppx::liuding::ResourceExhausted);
  auto invalid = descriptor(1U);
  invalid.logical_name.clear();
  EXPECT_THROW(store.begin(invalid), jyppx::liuding::InvalidArgument);
}

TEST(ArtifactStoreTest, RejectsForeignSessionsAndCleansAbandonedUpload) {
  TempArtifactRoot first_root;
  TempArtifactRoot second_root;
  FileArtifactStore first(first_root.path());
  FileArtifactStore second(second_root.path());
  {
    auto upload = first.begin(descriptor(4U));
    upload->write(bytes_of("data"));
    EXPECT_THROW(second.commit(*upload), jyppx::liuding::InvalidArgument);
  }
  EXPECT_TRUE(std::filesystem::is_empty(first_root.path() / "tmp"));
}

TEST(ArtifactStoreTest, RecoversOnlyStaleTemporaryUploads) {
  TempArtifactRoot root;
  ArtifactStoreLimits limits;
  limits.stale_temp_age = std::chrono::seconds(60);
  FileArtifactStore store(root.path(), limits);

  const auto stale = root.path() / "tmp" / ".upload-stale.part";
  const auto fresh = root.path() / "tmp" / ".upload-fresh.part";
  std::ofstream(stale, std::ios::binary) << "stale";
  std::ofstream(fresh, std::ios::binary) << "fresh";
  const auto old_time =
      std::filesystem::file_time_type::clock::now() - std::chrono::hours(2);
  std::error_code error;
  std::filesystem::last_write_time(stale, old_time, error);
  ASSERT_FALSE(error);

  const auto report = store.recover();
  EXPECT_EQ(report.removed_files, 1U);
  EXPECT_EQ(report.retained_files, 1U);
  EXPECT_FALSE(std::filesystem::exists(stale));
  EXPECT_TRUE(std::filesystem::exists(fresh));
}

TEST(ArtifactStoreTest, VerifyDetectsMutationAndGcRespectsLiveDigestAndGrace) {
  TempArtifactRoot root;
  FileArtifactStore store(root.path());
  const auto live = store.put(bytes_of("keep"), descriptor(4U));
  const auto dead = store.put(bytes_of("delete"), descriptor(6U));

  const auto dead_path = root.path() / "objects" /
                         dead.digest.hex.substr(0U, 2U) /
                         dead.digest.hex.substr(2U);
  std::ofstream(dead_path, std::ios::binary | std::ios::trunc) << "mutated";
  const auto verification = store.verify();
  EXPECT_FALSE(verification.ok());
  ASSERT_EQ(verification.corrupt_files.size(), 1U);
  EXPECT_EQ(verification.corrupt_files.front(), dead_path);

  std::ofstream(dead_path, std::ios::binary | std::ios::trunc) << "delete";
  const auto old_time =
      std::filesystem::file_time_type::clock::now() - std::chrono::hours(2);
  std::error_code error;
  std::filesystem::last_write_time(dead_path, old_time, error);
  ASSERT_FALSE(error);
  const std::array<ArtifactDigest, 1> live_set{live.digest};
  const auto gc = store.collect_garbage(live_set, std::chrono::seconds::zero());
  EXPECT_EQ(gc.removed_files, 1U);
  EXPECT_TRUE(store.exists(live.digest));
  EXPECT_FALSE(store.exists(dead.digest));
}

TEST(RunMetadataTest, ValidatesLifecycleAndLineageBounds) {
  jyppx::liuding::experiment::RunMetadata metadata{
      .run_id = "run-1",
      .project_id = "project",
      .experiment_id = "experiment",
      .display_name = "smoke",
      .status = jyppx::liuding::experiment::RunStatus::running,
      .created_at_unix_ms = 10U,
      .updated_at_unix_ms = 11U,
      .lineage = {.parent_run_id = "parent",
                  .source_digest = std::nullopt,
                  .build_fingerprint = std::nullopt,
                  .config_digest = "cfg",
                  .data_manifest_digest = std::nullopt},
      .tags = {{"model", "baseline"}}};
  EXPECT_NO_THROW(jyppx::liuding::experiment::validate_run_metadata(metadata));
  EXPECT_TRUE(jyppx::liuding::experiment::is_valid_run_transition(
      jyppx::liuding::experiment::RunStatus::running,
      jyppx::liuding::experiment::RunStatus::completed));
  EXPECT_FALSE(jyppx::liuding::experiment::is_valid_run_transition(
      jyppx::liuding::experiment::RunStatus::completed,
      jyppx::liuding::experiment::RunStatus::running));
  metadata.updated_at_unix_ms = 1U;
  EXPECT_THROW(jyppx::liuding::experiment::validate_run_metadata(metadata),
               jyppx::liuding::InvalidArgument);
}

}  // namespace
