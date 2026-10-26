// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <span>
#include <string>
#include <variant>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;
using namespace jyppx::liuding::experiment;

class OutputDirectory final {
 public:
  OutputDirectory() {
    static std::atomic<std::uint64_t> next{0U};
    path = std::filesystem::temp_directory_path() /
           ("liuding-image-evidence-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(next.fetch_add(1U)));
    std::filesystem::create_directories(path);
  }
  ~OutputDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  std::filesystem::path path;
};

EncodedImage png() {
  PixelSpec spec;
  spec.layout = ImageLayout::hwc;
  spec.color = PixelColorModel::rgb;
  spec.alpha = PixelAlphaMode::none;
  spec.dtype = torch::kUInt8;
  spec.range = PixelValueRange::unsigned_full_scale;
  return encode_image(torch::full({2, 3, 3}, 17, torch::kUInt8), spec);
}

RunMetadata run() {
  return {.run_id = "run-1",
          .project_id = "project-1",
          .experiment_id = "experiment-1",
          .display_name = "image output evidence",
          .status = RunStatus::running,
          .created_at_unix_ms = 100U,
          .updated_at_unix_ms = 100U};
}

ArtifactDescriptor descriptor(std::size_t size) {
  return {.logical_name = "preview.png",
          .media_type = "image/png",
          .producer = "image-output-evidence-test",
          .expected_size_bytes = size};
}

TEST(ImageOutputEvidenceTest, PublishedBytesBecomeDurableArtifactAndEvent) {
  OutputDirectory directory;
  SQLiteRunStore store(directory.path / "store");
  store.start(run());
  ImageOutputSink sink;
  auto ticket = sink.submit(directory.path / "preview.png", png());
  ASSERT_EQ(ticket.disposition, ImageSinkDisposition::queued);
  const auto published = ticket.completion.get();
  sink.close();

  const auto reference = store.put_artifact_file(
      "run-1", "image-preview", RunArtifactKind::artifact, "image.preview",
      published.path, descriptor(published.bytes), 101U,
      ArtifactDigest{.hex = published.sha256});
  EXPECT_EQ(reference.artifact.size_bytes, published.bytes);
  EXPECT_EQ(reference.artifact.digest.hex, published.sha256);

  TelemetryEnvelope event;
  event.identity = {.stream_id = "image-output-1",
                    .event_id = "image-preview-1",
                    .idempotency_key = "image-preview-1",
                    .sequence = 1U};
  event.correlation.project_id = "project-1";
  event.correlation.run_id = "run-1";
  event.payload = TelemetryArtifactReference{.role = "image.preview",
                                             .artifact = reference.artifact};
  const auto events = std::span<const TelemetryEnvelope>(&event, 1U);
  const auto accepted = store.append("run-1", "image-preview-event", events);
  ASSERT_EQ(accepted.receipts.size(), 1U);
  EXPECT_FALSE(accepted.batch_replay);
  EXPECT_TRUE(
      store.append("run-1", "image-preview-event", events).batch_replay);

  SQLiteRunStore reopened(directory.path / "store");
  const auto page = reopened.events("run-1");
  ASSERT_EQ(page.events.size(), 1U);
  const auto* linked =
      std::get_if<TelemetryArtifactReference>(&page.events[0].envelope.payload);
  ASSERT_NE(linked, nullptr);
  EXPECT_EQ(linked->artifact.digest, reference.artifact.digest);
  EXPECT_EQ(reopened.references("run-1").size(), 2U);
  EXPECT_TRUE(reopened.integrity_check().ok());
  EXPECT_EQ(
      reopened.collect_garbage(std::chrono::seconds::zero()).removed_files, 0U);
}

TEST(ImageOutputEvidenceTest, ChangedPublishedBytesCannotBeRegistered) {
  OutputDirectory directory;
  SQLiteRunStore store(directory.path / "store");
  store.start(run());
  ImageOutputSink sink;
  auto ticket = sink.submit(directory.path / "preview.png", png());
  ASSERT_EQ(ticket.disposition, ImageSinkDisposition::queued);
  const auto published = ticket.completion.get();
  sink.close();

  std::ifstream input(published.path, std::ios::binary);
  std::string bytes{std::istreambuf_iterator<char>{input},
                    std::istreambuf_iterator<char>{}};
  ASSERT_FALSE(bytes.empty());
  bytes.back() ^= 1;
  std::ofstream output(published.path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  ASSERT_TRUE(output);

  EXPECT_THROW(store.put_artifact_file(
                   "run-1", "image-preview", RunArtifactKind::artifact,
                   "image.preview", published.path, descriptor(published.bytes),
                   101U, ArtifactDigest{.hex = published.sha256}),
               SerializationError);
  EXPECT_TRUE(store.references("run-1").empty());
  EXPECT_TRUE(store.events("run-1").events.empty());
}

TEST(ImageOutputEvidenceTest,
     FileRegistrationStreamsAcrossChunksAndHonorsBounds) {
  OutputDirectory directory;
  const auto path = directory.path / "large-output.bin";
  std::string content(2U * 64U * 1024U + 7U, 'a');
  content[64U * 1024U] = 'b';
  std::ofstream output(path, std::ios::binary);
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  output.close();
  ASSERT_TRUE(output);
  const ArtifactDigest digest{
      .hex = sha256_hex(std::as_bytes(std::span(content)))};
  const ArtifactDescriptor file_descriptor{
      .logical_name = "large-output.bin",
      .media_type = "application/octet-stream",
      .producer = "image-output-evidence-test",
      .expected_size_bytes = content.size()};

  SQLiteRunStore store(directory.path / "store");
  store.start(run());
  const auto reference = store.put_artifact_file(
      "run-1", "large-output", RunArtifactKind::artifact, "output.file", path,
      file_descriptor, 101U, digest);
  EXPECT_EQ(reference.artifact.digest, digest);
  EXPECT_EQ(reference.artifact.size_bytes, content.size());
  auto wrong_size = file_descriptor;
  wrong_size.expected_size_bytes = content.size() - 1U;
  EXPECT_THROW(
      store.put_artifact_file("run-1", "wrong-size", RunArtifactKind::artifact,
                              "output.file", path, wrong_size, 101U, digest),
      InvalidArgument);
  EXPECT_EQ(store.references("run-1").size(), 1U);

  SQLiteRunStoreOptions limited_options;
  limited_options.artifact_limits.max_artifact_bytes = 64U * 1024U;
  SQLiteRunStore limited(directory.path / "limited", limited_options);
  limited.start(run());
  EXPECT_THROW(limited.put_artifact_file(
                   "run-1", "large-output", RunArtifactKind::artifact,
                   "output.file", path, file_descriptor, 101U, digest),
               ResourceExhausted);
  EXPECT_TRUE(limited.references("run-1").empty());
}

}  // namespace
