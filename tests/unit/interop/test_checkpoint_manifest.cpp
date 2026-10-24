// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/capability.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

constexpr std::string_view kDigestA =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
constexpr std::string_view kDigestB =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

jyppx::liuding::BuildFingerprint MakeBuild() {
  return {.jyppx_version = "0.3.0",
          .git_commit = "0123456789abcdef",
          .compiler = "test-compiler 1",
          .cxx_standard = "202002",
          .os_arch = "test-x86_64",
          .build_type = "Release",
          .libtorch_version = "2.5.1",
          .libtorch_build = "cpu-cxx11-abi1",
          .accelerator_runtime = "cpu",
          .dependency_lock_digest = "unlocked"};
}

jyppx::liuding::interop::CheckpointParticipantEntry MakeParticipant(
    std::string participant_id, std::string field_name, std::string path,
    std::string digest = std::string(kDigestA), std::uint64_t size = 8U) {
  return {
      .state = {.participant_id = std::move(participant_id),
                .schema_version = 1U,
                .restore_level = jyppx::liuding::RestoreLevel::exact,
                .fields = {{.name = field_name,
                            .media_type = "application/octet-stream",
                            .max_bytes = 64U,
                            .required = true},
                           {.name = "optional",
                            .media_type = "application/octet-stream",
                            .max_bytes = 16U,
                            .required = false}}},
      .blobs = {{.field_name = std::move(field_name),
                 .relative_path = std::move(path),
                 .size_bytes = size,
                 .digest = {.algorithm =
                                jyppx::liuding::ArtifactDigestAlgorithm::sha256,
                            .hex = std::move(digest)}}},
  };
}

jyppx::liuding::interop::CheckpointArtifactManifest MakeManifest() {
  return {
      .schema_version =
          jyppx::liuding::interop::CheckpointArtifactManifest::format_version,
      .build = MakeBuild(),
      .participants = {MakeParticipant("worker.second", "weights",
                                       "blobs/second.bin",
                                       std::string(kDigestB), 12U),
                       MakeParticipant("worker.first", "state",
                                       "blobs/first.bin")},
  };
}

TEST(CheckpointManifestTest, DeterministicallySerializesAndRoundTrips) {
  const auto manifest = MakeManifest();
  const auto encoded =
      jyppx::liuding::interop::checkpoint_artifact_manifest_json(manifest);
  const auto first_position = encoded.find("worker.first");
  const auto second_position = encoded.find("worker.second");
  ASSERT_NE(first_position, std::string::npos);
  ASSERT_NE(second_position, std::string::npos);
  EXPECT_LT(first_position, second_position);
  EXPECT_EQ(encoded.find('\n'), std::string::npos);

  const auto decoded =
      jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(encoded);
  ASSERT_EQ(decoded.participants.size(), 2U);
  EXPECT_EQ(decoded.participants[0].state.participant_id, "worker.first");
  EXPECT_EQ(decoded.participants[0].blobs[0].field_name, "state");
  EXPECT_EQ(decoded.participants[1].blobs[0].size_bytes, 12U);
  EXPECT_EQ(jyppx::liuding::interop::checkpoint_artifact_manifest_json(decoded),
            encoded);
}

TEST(CheckpointManifestTest, AcceptsAbsentOptionalFieldsAndUtf8Reasons) {
  auto manifest = MakeManifest();
  auto& participant = manifest.participants.front();
  participant.state.restore_level = jyppx::liuding::RestoreLevel::bounded;
  participant.state.degradation_reasons = {"设备拓扑变化时只保证数值边界"};
  EXPECT_NO_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest));
  const auto encoded =
      jyppx::liuding::interop::checkpoint_artifact_manifest_json(manifest);
  EXPECT_EQ(encoded.find("best_effort"), std::string::npos);
  EXPECT_NE(encoded.find("bounded"), std::string::npos);
}

TEST(CheckpointManifestTest, RejectsUnknownDuplicateAndMalformedJson) {
  const auto encoded =
      jyppx::liuding::interop::checkpoint_artifact_manifest_json(
          MakeManifest());
  auto unknown = encoded;
  unknown.insert(unknown.size() - 1U, R"(,"unknown":true)");
  EXPECT_THROW(
      jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(unknown),
      jyppx::liuding::SerializationError);

  const std::string duplicate = R"({"schema_version":1,"schema_version":1})";
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   duplicate),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(
      jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json("{"),
      jyppx::liuding::SerializationError);
}

TEST(CheckpointManifestTest, RejectsInvalidUtf8BeforeParsing) {
  std::string invalid = R"({"schema_version":")";
  invalid.push_back(static_cast<char>(0xFF));
  invalid += R"("})";
  EXPECT_THROW(
      jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(invalid),
      jyppx::liuding::SerializationError);
}

TEST(CheckpointManifestTest, RejectsUnsupportedSchemaVersion) {
  auto manifest = MakeManifest();
  manifest.schema_version = 2U;
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::CompatibilityError);

  auto encoded = jyppx::liuding::interop::checkpoint_artifact_manifest_json(
      MakeManifest());
  const auto version = encoded.rfind(R"("schema_version":1)");
  ASSERT_NE(version, std::string::npos);
  encoded.replace(version, std::string(R"("schema_version":1)").size(),
                  R"("schema_version":2)");
  EXPECT_THROW(
      jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(encoded),
      jyppx::liuding::CompatibilityError);
}

TEST(CheckpointManifestTest, EnforcesParserResourceLimits) {
  const auto encoded =
      jyppx::liuding::interop::checkpoint_artifact_manifest_json(
          MakeManifest());
  auto limits = jyppx::liuding::interop::CheckpointManifestLimits{};
  limits.max_manifest_bytes = encoded.size() - 1U;
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   encoded, limits),
               jyppx::liuding::ResourceExhausted);

  limits = {};
  limits.max_depth = 1U;
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   encoded, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_nodes = 1U;
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   encoded, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_string_bytes = 3U;
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   encoded, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_total_string_bytes = 3U;
  EXPECT_THROW(jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(
                   encoded, limits),
               jyppx::liuding::ResourceExhausted);
}

TEST(CheckpointManifestTest, EnforcesParticipantBlobAndByteLimits) {
  const auto manifest = MakeManifest();
  auto limits = jyppx::liuding::interop::CheckpointManifestLimits{};
  limits.max_participants = 1U;
  EXPECT_THROW(jyppx::liuding::interop::validate_checkpoint_artifact_manifest(
                   manifest, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_blobs = 1U;
  EXPECT_THROW(jyppx::liuding::interop::validate_checkpoint_artifact_manifest(
                   manifest, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_total_blob_bytes = 19U;
  EXPECT_THROW(jyppx::liuding::interop::validate_checkpoint_artifact_manifest(
                   manifest, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_path_bytes = 4U;
  EXPECT_THROW(jyppx::liuding::interop::validate_checkpoint_artifact_manifest(
                   manifest, limits),
               jyppx::liuding::ResourceExhausted);
}

TEST(CheckpointManifestTest,
     RejectsMissingUndeclaredDuplicateAndOversizedBlobs) {
  auto manifest = MakeManifest();
  manifest.participants.front().blobs.clear();
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants.front().blobs.front().field_name = "undeclared";
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants.front().blobs.push_back(
      manifest.participants.front().blobs.front());
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants.front().blobs.front().size_bytes = 65U;
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::ResourceExhausted);
}

TEST(CheckpointManifestTest, RejectsUnsafeAndDuplicateArtifactPaths) {
  auto manifest = MakeManifest();
  manifest.participants.front().blobs.front().relative_path = "../escape.bin";
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants[1].blobs.front().relative_path =
      manifest.participants[0].blobs.front().relative_path;
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants.front().blobs.front().relative_path = "drive:C.bin";
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);
}

TEST(CheckpointManifestTest, RejectsInvalidDigestsAndDuplicateParticipants) {
  auto manifest = MakeManifest();
  manifest.participants.front().blobs.front().digest.hex = "ABC";
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants.front().blobs.front().digest.algorithm =
      static_cast<jyppx::liuding::ArtifactDigestAlgorithm>(99U);
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);

  manifest = MakeManifest();
  manifest.participants[1].state.participant_id =
      manifest.participants[0].state.participant_id;
  EXPECT_THROW(
      jyppx::liuding::interop::validate_checkpoint_artifact_manifest(manifest),
      jyppx::liuding::InvalidArgument);
}

TEST(CheckpointManifestTest, PublishesBoundedExperimentalCapability) {
  const auto capability =
      jyppx::liuding::interop::make_checkpoint_manifest_capability();
  EXPECT_EQ(capability.feature_id, "interop.checkpoint_manifest");
  EXPECT_EQ(capability.status, jyppx::liuding::CapabilityStatus::experimental);
  EXPECT_EQ(capability.source, jyppx::liuding::CapabilitySource::build);
  EXPECT_EQ(capability.schemas,
            std::vector<std::string>{"jyppx.checkpoint.artifact_manifest.v1"});
  EXPECT_NO_THROW(jyppx::liuding::validate_capability_descriptor(capability));

  auto limits = jyppx::liuding::interop::CheckpointManifestLimits{};
  limits.max_blobs = 0U;
  EXPECT_THROW(
      jyppx::liuding::interop::make_checkpoint_manifest_capability(limits),
      jyppx::liuding::InvalidArgument);
}

}  // namespace
