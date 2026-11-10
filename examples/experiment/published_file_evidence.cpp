// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
class WorkDirectory final {
 public:
  explicit WorkDirectory(std::optional<std::filesystem::path> requested)
      : remove_(!requested),
        path(requested.value_or(std::filesystem::temp_directory_path() /
                                ("liuding-published-file-" +
                                 std::to_string(std::chrono::steady_clock::now()
                                                    .time_since_epoch()
                                                    .count())))) {
    if (!std::filesystem::create_directory(path))
      throw jyppx::liuding::InvalidArgument("Choose a new output directory");
  }
  ~WorkDirectory() {
    if (remove_) {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  }

 private:
  bool remove_;

 public:
  std::filesystem::path path;
};
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2) {
      std::cerr
          << "Usage: liuding-forge-example-published-file [NEW_DIRECTORY]\n";
      return 2;
    }
    const WorkDirectory work(argc == 2
                                 ? std::optional<std::filesystem::path>(argv[1])
                                 : std::nullopt);
    namespace forge = jyppx::liuding;
    namespace image = forge::data::image;
    namespace experiment = forge::experiment;
    {
      experiment::SQLiteRunStore runs(work.path / "runs");
      runs.start({.run_id = "example.run",
                  .project_id = "example.project",
                  .experiment_id = "example.output",
                  .display_name = "Published preview",
                  .status = experiment::RunStatus::running,
                  .created_at_unix_ms = 100U,
                  .updated_at_unix_ms = 100U});
      image::PixelSpec pixels;
      pixels.layout = image::ImageLayout::hwc;
      pixels.color = image::PixelColorModel::rgb;
      pixels.alpha = image::PixelAlphaMode::none;
      pixels.dtype = torch::kUInt8;
      pixels.range = image::PixelValueRange::unsigned_full_scale;
      auto encoded = image::encode_image(
          torch::full({16, 16, 3}, 96, torch::kUInt8), pixels);
      image::ImageOutputSink output({.max_pending_items = 2U,
                                     .max_pending_bytes = 64U << 10U,
                                     .max_item_bytes = 64U << 10U});
      auto ticket = output.submit_with_evidence(
          work.path / "preview.input.png", std::move(encoded),
          [&](const image::ImageSinkWriteResult &published) {
            experiment::PublishedFileEvidenceRequest request;
            request.run_id = "example.run";
            request.reference_id = "preview";
            request.path = published.path;
            request.descriptor = {.logical_name = "preview.png",
                                  .media_type = "image/png",
                                  .producer = "published-file-example",
                                  .expected_size_bytes = published.bytes};
            request.attached_at_unix_ms = 101U;
            request.expected_digest = {.hex = published.sha256};
            request.event_batch_key = "preview.event";
            request.event.identity = {.stream_id = "example.output",
                                      .event_id = "preview",
                                      .idempotency_key = "preview",
                                      .sequence = 1U};
            request.event.correlation.project_id = "example.project";
            request.event.correlation.run_id = "example.run";
            request.event.payload =
                forge::TelemetryArtifactReference{.role = "image.preview"};
            // Retain the complete intent and CAS content before publishing
            // the final preview. This temporary input can then disappear.
            runs.prepare_file_publication(
                {.publication_id = "preview.intent",
                 .destination = work.path / "preview.png",
                 .evidence = std::move(request)});
          });
      ticket.completion.get();
      ticket.evidence_completion.get();
      output.close();
    }
    std::filesystem::remove(work.path / "preview.input.png");
    experiment::SQLiteRunStore reopened(work.path / "runs");
    const auto ids = reopened.file_publications("example.run");
    if (ids != std::vector<std::string>{"preview.intent"})
      throw forge::IoError("Prepared preview intent was not discoverable");
    reopened.recover_file_publication("example.run", ids.front());
    const auto replay =
        reopened.recover_file_publication("example.run", ids.front());
    std::ifstream preview(replay.path, std::ios::binary);
    forge::Sha256 digest;
    std::array<std::byte, 4096U> buffer{};
    while (preview) {
      preview.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
      digest.update(std::span<const std::byte>(
          buffer.data(), static_cast<std::size_t>(preview.gcount())));
    }
    if (!replay.evidence.reference_replay || !preview.eof() ||
        forge::sha256_digest_hex(digest.finish()) !=
            replay.evidence.reference.artifact.digest.hex ||
        std::filesystem::file_size(replay.path) !=
            replay.evidence.reference.artifact.size_bytes ||
        !replay.evidence.event.batch_replay ||
        reopened.references("example.run").size() != 4U ||
        reopened.events("example.run").events.size() != 1U ||
        !reopened.integrity_check().ok())
      throw forge::IoError(
          "Published evidence replay did not retain its identity");
    std::cout << "Recorded preview and replayed one event: "
              << replay.evidence.reference.artifact.digest.hex << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
