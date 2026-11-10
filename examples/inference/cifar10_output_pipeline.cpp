// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file cifar10_output_pipeline.cpp
 * @brief Keeps source admission through asynchronous preview evidence. */

#include "cifar10_model_support.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/visualization/render.hpp>

#include <torch/utils.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>

namespace {
using namespace cifar10_example;
using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kEncodedBound = 256U << 10U;
constexpr std::array<std::string_view, 14> kResourceNames{
    "encoded_bytes",
    "decoded_bytes",
    "pixels",
    "codec_scratch_bytes",
    "annotation_bytes",
    "index_bytes",
    "pinned_bytes",
    "gpu_bytes",
    "staged_batches",
    "render_queue",
    "encoded_queue_bytes",
    "disk_spool_bytes",
    "pipeline_inflight_batches",
    "pipeline_inflight_bytes"};

struct PressureObservation {
  std::mutex mutex;
  std::condition_variable changed;
  bool first_consumer_released = false;
  bool next_admission_started = false;
  std::atomic<std::size_t> attempts{0};
  std::atomic<std::size_t> acquired{0};
  std::atomic<std::uint64_t> admission_wait_ns{0};
  Json evidence;
};

image::EncodedImage Preview(
    const torch::Tensor& normalized,
    const std::shared_ptr<image::ImageResourceBudget>& budget,
    std::stop_token stop) {
  auto render_slot = budget->reserve(image::ImageResourceKind::render_queue, 1);
  // Reserve both the float conversion and RGB8 result before allocation.
  auto conversion = budget->reserve(image::ImageResourceKind::decoded_bytes,
                                    32U * 32U * 3U * (sizeof(float) + 1U));
  auto scaled = normalized.mul(255);
  scaled.round_().clamp_(0, 255);
  auto pixels = scaled.to(torch::kUInt8);
  auto spec = image::PixelSpec{};
  spec.layout = image::ImageLayout::chw;
  spec.color = image::PixelColorModel::rgb;
  spec.alpha = image::PixelAlphaMode::none;
  spec.dtype = torch::kUInt8;
  spec.range = image::PixelValueRange::unsigned_full_scale;
  forge::visualization::RenderOptions render;
  render.resource_budget = budget;
  render.max_pixels = 32U * 32U;
  render.max_output_bytes = 32U * 32U * 3U;
  const auto preview = forge::visualization::render_annotations(
      pixels, spec, {.image_path = "preview", .width = 32, .height = 32}, {},
      {}, {}, render, stop);
  image::ImageEncodeOptions encode;
  encode.resource_budget = budget;
  encode.limits.max_input_bytes = render.max_output_bytes;
  encode.limits.max_pixels = render.max_pixels;
  encode.limits.max_output_bytes = kEncodedBound;
  encode.limits.max_working_bytes = 4U << 20U;
  return image::encode_image(preview.pixels, preview.spec, encode, stop);
}

int Run(const fs::path& model_path, std::string_view expected_sha,
        const fs::path& validation_root, const fs::path& output) {
  Require(!fs::exists(output), "Use a new output directory");
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  forge::core::RngGuard seeded(kSeed, true);
  torch::NoGradGuard no_grad;
  forge::models::ResNetOptions architecture;
  architecture.num_classes = 10;
  architecture.input_channels = 3;
  auto model = forge::models::adapt(forge::models::resnet18(architecture));
  const auto imported = LoadModel(model, model_path, expected_sha);
  model.state().eval();
  const auto before = StateDigest(model);
  image::ImageResourceBudgetLimits limits;
  limits.max_encoded_bytes = limits.max_decoded_bytes =
      limits.max_codec_scratch_bytes = 256 * kMiB;
  limits.max_pinned_bytes = limits.max_gpu_bytes = 0;
  limits.max_pipeline_inflight_batches = 1;
  limits.max_pipeline_inflight_bytes = 64 * kMiB;
  limits.max_render_queue = 1;
  limits.max_encoded_queue_bytes = limits.max_disk_spool_bytes = kEncodedBound;
  const auto budget = std::make_shared<image::ImageResourceBudget>(limits);
  PressureObservation pressure;
  forge::training::metrics::Precision counts({.num_classes = 10});
  Json published = Json::array();
  fs::create_directories(output);
  std::ofstream rows(output / "batches.jsonl");
  Require(rows.good(), "Cannot create bounded batch evidence");
  forge::experiment::SQLiteRunStore store(output / "run-store");
  const auto timestamp = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  store.start({.run_id = "cifar10-output",
               .project_id = "cifar10",
               .experiment_id = "model-reload",
               .display_name = "CIFAR10 previews",
               .status = forge::experiment::RunStatus::running,
               .created_at_unix_ms = timestamp,
               .updated_at_unix_ms = timestamp});
  std::size_t batches = 0;
  std::size_t samples = 0;
  const auto started = Clock::now();
  {
    Images images(validation_root, budget);
    forge::data::DataPipelineOptions options;
    options.pipeline_id = "example.cifar10-output";
    options.dataset_version = "cifar10-validation-10000-rgb-float32-bilinear32";
    options.dataset_size = kSamples;
    options.batch_factory_id = "example.cifar10-output/v1";
    options.batch_size = kBatchSize;
    options.num_workers = 2;
    options.prefetch_batches = 2;
    const auto admission =
        image::make_image_batch_byte_admission_factory(budget, images.limits());
    options.admission_factory = [&pressure, admission](std::stop_token stop) {
      if (pressure.attempts.fetch_add(1) == 1) {
        std::scoped_lock lock(pressure.mutex);
        pressure.next_admission_started = true;
        pressure.changed.notify_all();
      }
      const auto start = Clock::now();
      auto lease = admission(stop);
      pressure.admission_wait_ns.fetch_add(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                               start)
              .count()));
      pressure.acquired.fetch_add(1);
      return lease;
    };
    options.stage_resource_factory =
        image::make_image_batch_stage_resource_factory(budget, images.limits());
    forge::data::DataPipeline pipeline(
        std::make_shared<forge::data::samplers::RandomSampler>(kSamples, kSeed),
        options,
        [&images](std::span<const std::size_t> ids,
                  const forge::data::DataWorkerContext&, std::stop_token stop) {
          return image::image_batch_value(images.batch(ids, stop));
        });
    image::ImageOutputSinkOptions sink_options;
    sink_options.max_pending_items = 1;
    sink_options.max_pending_bytes = sink_options.max_item_bytes =
        kEncodedBound;
    sink_options.resource_budget = budget;
    sink_options.wait_for_aggregate_queue = true;
    image::ImageOutputSink sink(sink_options);
    pipeline.set_epoch(0);
    std::vector<image::ImageSinkSubmission> tickets;
    while (auto envelope = pipeline.next()) {
      const auto sequence = envelope->sequence;
      {
        auto batch = image::image_batch_from_value(envelope->value);
        const auto logits =
            model.invoke({.positional = {batch.images}, .keyword = {}})
                .value.toTensor();
        counts.update(logits, batch.labels);
        const auto predictions = logits.argmax(1);
        const auto n = static_cast<std::size_t>(batch.images.size(0));
        Json row{{"sequence", sequence},
                 {"sample_ids", envelope->sample_ids},
                 {"input_sha256",
                  forge::sha256_hex({static_cast<const std::byte*>(
                                         batch.images.const_data_ptr()),
                                     batch.images.nbytes()})},
                 {"labels", Json::array()},
                 {"logits", Json::array()}};
        for (std::size_t i = 0; i < n; ++i) {
          row["labels"].push_back(
              batch.labels[static_cast<std::int64_t>(i)].item<std::int64_t>());
          Json scores = Json::array();
          for (std::int64_t c = 0; c < 10; ++c)
            scores.push_back(
                logits[static_cast<std::int64_t>(i)][c].item<float>());
          row["logits"].push_back(std::move(scores));
        }
        rows << row.dump() << '\n';
        Require(rows.good(), "Cannot save bounded batch evidence");
        const auto name =
            "batch-" + std::to_string(sequence) + "-label-" +
            std::to_string(batch.labels[0].item<std::int64_t>()) + "-pred-" +
            std::to_string(predictions[0].item<std::int64_t>()) + ".png";
        tickets.push_back(sink.produce_and_submit(
            output / name, kEncodedBound,
            [&](std::stop_token stop) {
              return Preview(batch.images[0], budget, stop);
            },
            {}, {},
            [&, sequence, lease = envelope->admission_lease](
                const image::ImageSinkWriteResult& file) {
              (void)lease;  // Keep source count and byte admission through
                            // evidence.
              if (sequence == 0) {
                std::unique_lock lock(pressure.mutex);
                Require(pressure.changed.wait_for(
                            lock, std::chrono::seconds(10),
                            [&] {
                              return pressure.first_consumer_released &&
                                     pressure.next_admission_started;
                            }),
                        "Timed out observing downstream-to-source pressure");
                Require(images.decoded_samples() == kBatchSize &&
                            pressure.acquired == 1,
                        "Source decoded another batch while first evidence "
                        "held admission");
                Require(
                    budget->usage(image::ImageResourceKind::
                                      pipeline_inflight_batches)
                                .current == 1 &&
                        budget->usage(image::ImageResourceKind::
                                          pipeline_inflight_bytes)
                                .current == images.limits().max_output_bytes,
                    "Evidence lost source admission");
                Require(budget->usage(image::ImageResourceKind::decoded_bytes)
                                .current == 0,
                        "Consumer pixels remained live during evidence wait");
                pressure.evidence = {
                    {"decoded_samples", images.decoded_samples()},
                    {"admission_attempts", pressure.attempts.load()},
                    {"acquired_batches", pressure.acquired.load()},
                    {"consumer_pixels_released", true},
                    {"next_batch_blocked", true}};
              }
              const auto id = "preview-" + std::to_string(sequence);
              forge::experiment::PublishedFileEvidenceRequest request;
              request.run_id = "cifar10-output";
              request.reference_id = id;
              request.path = file.path;
              request.descriptor = {.logical_name = id + ".png",
                                    .media_type = "image/png",
                                    .producer = "example.cifar10-output",
                                    .expected_size_bytes = file.bytes};
              request.attached_at_unix_ms = timestamp;
              request.expected_digest = {.hex = file.sha256};
              request.event_batch_key = id;
              request.event.identity = {.stream_id = "previews",
                                        .event_id = id,
                                        .idempotency_key = id,
                                        .sequence = sequence + 1};
              request.event.correlation.project_id = "cifar10";
              request.event.correlation.run_id = "cifar10-output";
              request.event.payload =
                  forge::TelemetryArtifactReference{.role = "image.preview"};
              const auto receipt =
                  store.record_published_file_evidence_atomic(request);
              Require(
                  !receipt.reference_replay && !receipt.event.batch_replay,
                  "Unexpected duplicate evidence in a new output directory");
            }));
        Require(
            tickets.back().disposition == image::ImageSinkDisposition::queued,
            "Preview was not admitted");
        samples += n;
      }
      envelope.reset();
      if (sequence == 0) {
        std::scoped_lock lock(pressure.mutex);
        pressure.first_consumer_released = true;
        pressure.changed.notify_all();
      }
      ++batches;
    }
    pipeline.close();
    sink.close();
    for (auto& ticket : tickets) {
      const auto file = ticket.completion.get();
      ticket.evidence_completion.get();
      published.push_back({{"path", file.path.filename().string()},
                           {"bytes", file.bytes},
                           {"sha256", file.sha256}});
    }
    const auto stats = sink.stats();
    Require(stats.completed == kBatches && stats.failed == 0 &&
                stats.pending_items == 0 && stats.pending_bytes == 0 &&
                stats.producing_items == 0,
            "Output did not drain");
    Require(images.decoded_samples() == kSamples,
            "Source count differs from complete validation");
  }
  const auto elapsed =
      std::chrono::duration<double>(Clock::now() - started).count();
  Require(samples == kSamples && batches == kBatches &&
              pressure.acquired == kBatches,
          "Incomplete bounded evaluation");
  Require(store.references("cifar10-output").size() == 2 * kBatches &&
              store.events("cifar10-output").events.size() == kBatches &&
              store.integrity_check().ok(),
          "Incomplete durable evidence");
  const auto after = StateDigest(model);
  Require(before == after, "Evaluation mutated model state");
  const auto& confusion = counts.confusion_matrix();
  std::int64_t correct = 0;
  for (std::size_t i = 0; i < confusion.size(); ++i) correct += confusion[i][i];
  const auto accuracy = static_cast<double>(correct) / kSamples;
  Json resources = Json::object();
  for (std::size_t i = 0; i < kResourceNames.size(); ++i) {
    const auto use = budget->usage(static_cast<image::ImageResourceKind>(i));
    Require(use.current == 0 && use.peak <= use.limit,
            "Image budget did not release completely");
    resources[kResourceNames[i]] = {
        {"current", use.current}, {"peak", use.peak}, {"limit", use.limit}};
  }
  const Json report{
      {"schema_id", "jyppx.examples.cifar10-output-pipeline"},
      {"schema_version", 1},
      {"device", "cpu"},
      {"model_sha256", expected_sha},
      {"build", Json::parse(forge::build_fingerprint_json(
                    forge::current_build_fingerprint()))},
      {"applied_tensors", imported.applied_tensors},
      {"training_updates", 0},
      {"state_sha256_before", before},
      {"state_sha256_after", after},
      {"forward_calls", batches},
      {"validation_samples", samples},
      {"accuracy", accuracy},
      {"minimum_accuracy", .65},
      {"quality_passed", accuracy >= .65},
      {"confusion", confusion},
      {"seconds", elapsed},
      {"admission_wait_seconds", pressure.admission_wait_ns.load() / 1e9},
      {"pressure_observation", pressure.evidence},
      {"resources", resources},
      {"published", published}};
  rows.close();
  Require(rows.good(), "Cannot save bounded batch evidence");
  std::ofstream saved(output / "report.json");
  saved << report.dump(2) << '\n';
  saved.close();
  Require(saved.good(), "Cannot save final report");
  std::cout << report.dump() << '\n';
  return accuracy >= .65 ? 0 : 4;
}

template <class Character>
int Main(int argc, Character* argv[]) {
  constexpr auto usage =
      "usage: liuding-forge-example-cifar10-output-pipeline "
      "MODEL_SAFETENSORS MODEL_SHA256 VALIDATION_ROOT NEW_OUTPUT_DIRECTORY\n";
  try {
    if (argc == 2 && fs::path(argv[1]) == fs::path("--help")) {
      std::cout << usage;
      return 0;
    }
    if (argc != 5) {
      std::cerr << usage;
      return 2;
    }
    std::string sha;
    for (const auto* c = argv[2]; *c != 0; ++c) {
      Require(sha.size() < 64 &&
                  ((*c >= '0' && *c <= '9') || (*c >= 'a' && *c <= 'f')),
              "Expected a lowercase SHA-256");
      sha.push_back(static_cast<char>(*c));
    }
    Require(sha.size() == 64, "Expected a complete model SHA-256");
    return Run(fs::path(argv[1]), sha, fs::path(argv[3]), fs::path(argv[4]));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return Main(argc, argv); }
#else
int main(int argc, char* argv[]) { return Main(argc, argv); }
#endif
