// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef LIUDING_OUTPUT_NATIVE_FAULTS
bool configure_native_image_output(const std::filesystem::path& root,
                                   std::string_view selected);
nlohmann::json native_image_output_observations();
#endif

namespace {
namespace image = jyppx::liuding::data::image;
namespace experiment = jyppx::liuding::experiment;
using Json = nlohmann::json;

void report(const Json& value) { std::cout << value.dump() << std::endl; }

void pause_at(std::string_view selected, std::string_view boundary,
              const image::ImageSinkWriteResult& result) {
  if (selected != boundary) return;
  report({{"phase", "ready"},
          {"boundary", boundary},
          {"digest", result.sha256},
          {"bytes", result.bytes}});
  char release;
  if (!std::cin.get(release))
    throw std::runtime_error("missing parent release");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected root and boundary");
    const std::filesystem::path root(argv[1]);
    const std::string boundary(argv[2]);
    bool valid_boundary =
        boundary == "file" || boundary == "evidence" || boundary == "none";
#ifdef LIUDING_OUTPUT_NATIVE_FAULTS
    valid_boundary = configure_native_image_output(root, boundary);
#endif
    if (!root.is_absolute() || !std::filesystem::is_directory(root) ||
        !valid_boundary)
      throw std::runtime_error("invalid recovery child input");
    experiment::SQLiteRunStore runs(root / "store");
    runs.start({.run_id = "output.recovery",
                .project_id = "project.recovery",
                .experiment_id = "experiment.recovery",
                .display_name = "output recovery",
                .status = experiment::RunStatus::running,
                .created_at_unix_ms = 100,
                .updated_at_unix_ms = 100});
    image::PixelSpec pixels;
    pixels.layout = image::ImageLayout::hwc;
    pixels.color = image::PixelColorModel::rgb;
    pixels.alpha = image::PixelAlphaMode::none;
    pixels.dtype = torch::kUInt8;
    pixels.range = image::PixelValueRange::unsigned_full_scale;
    image::ImageOutputSinkOptions options;
    options.replay_identical_existing = true;
    if (boundary == "replace-writing") {
      options.replay_identical_existing = false;
      options.replace_existing = true;
    }
    auto budget = std::make_shared<image::ImageResourceBudget>();
    options.resource_budget = budget;
    image::ImageOutputSink sink(options);
    bool reference_replay = false, event_replay = false;
    auto ticket = sink.submit_with_evidence(
        root / "preview.png",
        image::encode_image(torch::full({16, 16, 3}, 17, torch::kUInt8),
                            pixels),
        [&](const image::ImageSinkWriteResult& published) {
          pause_at(boundary, "file", published);
          experiment::PublishedFileEvidenceRequest request;
          request.run_id = "output.recovery";
          request.reference_id = "preview";
          request.path = published.path;
          request.descriptor = {.logical_name = "preview.png",
                                .media_type = "image/png",
                                .producer = "native-output-recovery",
                                .expected_size_bytes = published.bytes};
          request.expected_digest = {.hex = published.sha256};
          request.attached_at_unix_ms = 101;
          request.event_batch_key = "preview.batch";
          request.event.identity = {.stream_id = "preview.stream",
                                    .event_id = "preview.event",
                                    .idempotency_key = "preview.key",
                                    .sequence = 1};
          request.event.correlation.project_id = "project.recovery";
          request.event.correlation.run_id = "output.recovery";
          request.event.payload =
              jyppx::liuding::TelemetryArtifactReference{.role = "preview"};
          const auto recorded =
              runs.record_published_file_evidence_atomic(request);
          reference_replay = recorded.reference_replay;
          event_replay = recorded.event.batch_replay;
          pause_at(boundary, "evidence", published);
        });
    const auto published = ticket.completion.get();
    ticket.evidence_completion.get();
    sink.close();
    Json completed = {
        {"phase", "completed"},
        {"digest", published.sha256},
        {"bytes", published.bytes},
        {"reference_replay", reference_replay},
        {"event_replay", event_replay},
        {"references", runs.references("output.recovery").size()},
        {"events", runs.events("output.recovery").events.size()},
        {"integrity_ok", runs.integrity_check().ok()},
        {"queue_bytes",
         budget->usage(image::ImageResourceKind::encoded_queue_bytes).current},
        {"spool_bytes",
         budget->usage(image::ImageResourceKind::disk_spool_bytes).current}};
#ifdef LIUDING_OUTPUT_NATIVE_FAULTS
    completed["native"] = native_image_output_observations();
#endif
    report(completed);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
