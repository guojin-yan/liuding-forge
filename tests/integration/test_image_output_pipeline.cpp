// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/visualization/render.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace {
using namespace jyppx::liuding;
namespace image = data::image;
using namespace std::chrono_literals;
constexpr std::size_t kBatchBytes = 1U << 20U;

struct Fixture {
  std::filesystem::path root;
  std::shared_ptr<image::ImageResourceBudget> budget;
  std::shared_ptr<image::ImageDataset> dataset;
  std::unique_ptr<data::DataPipeline> pipeline;
  std::atomic<std::size_t> decoded{0}, attempts{0}, acquired{0};
  std::promise<void> second_attempt;

  Fixture() {
    static std::atomic<unsigned> serial{0};
    root = std::filesystem::temp_directory_path() /
           ("liuding-output-pipeline-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(serial.fetch_add(1)));
    std::filesystem::create_directories(root / "images/class");
    image::PixelSpec spec;
    spec.layout = image::ImageLayout::chw;
    for (int i = 0; i < 4; ++i)
      image::write_image_file(
          root / "images/class" / (std::to_string(i) + ".png"),
          torch::full({3, 8, 8}, 17 + i, torch::kUInt8), spec);
    image::ImageResourceBudgetLimits limits;
    limits.max_pipeline_inflight_batches = 1;
    limits.max_pipeline_inflight_bytes = kBatchBytes;
    limits.max_pinned_bytes = limits.max_gpu_bytes = 0;
    budget = std::make_shared<image::ImageResourceBudget>(limits);
    image::ImageDatasetOptions read;
    read.resource_budget = budget;
    dataset = std::make_shared<image::ImageDataset>(
        root / "images", image::index_image_folder(root / "images"), read);
    image::ImageCollateOptions collate;
    collate.resource_budget = budget;
    collate.limits.max_samples = 2;
    collate.limits.max_output_bytes = kBatchBytes;
    data::DataPipelineOptions options;
    options.dataset_size = 4;
    options.dataset_version = "test-four-pngs";
    options.batch_factory_id = "resize-collate/v1";
    options.batch_size = 2;
    options.num_workers = options.prefetch_batches = 2;
    const auto admission =
        image::make_image_batch_byte_admission_factory(budget, collate.limits);
    options.admission_factory = [this, admission](std::stop_token stop) {
      if (attempts.fetch_add(1) == 1) second_attempt.set_value();
      auto result = admission(stop);
      acquired.fetch_add(1);
      return result;
    };
    options.stage_resource_factory =
        image::make_image_batch_stage_resource_factory(budget, collate.limits);
    pipeline = std::make_unique<data::DataPipeline>(
        std::make_shared<data::samplers::SequentialSampler>(4), options,
        [this, collate](std::span<const std::size_t> ids,
                        const data::DataWorkerContext&, std::stop_token stop) {
          std::vector<image::VisionSample> samples;
          std::vector<image::ImageBatchKey> keys;
          image::JointTransformOptions transform;
          transform.resource_budget = budget;
          const image::ImageGeometryStep resize{
              .kind = image::ImageGeometryKind::resize, .extent = {8, 8}};
          for (const auto id : ids) {
            const auto sample = dataset->get(id, stop);
            decoded.fetch_add(1);
            samples.push_back(image::transform_sample(
                sample, std::span(&resize, 1), transform, stop));
            keys.push_back({sample.image_path, 0, 0});
          }
          return image::image_batch_value(
              image::collate_images(samples, keys, collate, stop));
        });
    pipeline->set_epoch(0);
  }

  ~Fixture() {
    pipeline.reset();
    dataset.reset();
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
  }

  image::EncodedImage preview(const data::BatchEnvelope& envelope,
                              std::stop_token stop) {
    const auto batch = image::image_batch_from_value(envelope.value);
    // A real tensor consumer runs between collate and rendering.
    EXPECT_TRUE(
        torch::isfinite(batch.images.to(torch::kFloat32).mean()).item<bool>());
    const auto rendered = visualization::render_annotations(
        batch.images[0], batch.pixels,
        {.image_path = "preview", .width = 8, .height = 8}, {}, {}, {},
        {.resource_budget = budget}, stop);
    image::ImageEncodeOptions encode;
    encode.resource_budget = budget;
    return image::encode_image(rendered.pixels, rendered.spec, encode, stop);
  }

  void expect_released() {
    pipeline.reset();
    dataset.reset();
    for (unsigned i = 0; i < 14; ++i)
      EXPECT_EQ(budget->usage(static_cast<image::ImageResourceKind>(i)).current,
                0U)
          << i;
  }
};

TEST(ImageOutputPipelineTest,
     SlowEvidenceRetainsCountAndBytesBeforeNextDecode) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future();
  image::ImageOutputSink sink;
  auto ticket = sink.produce_and_submit(
      f.root / "preview.png", kBatchBytes,
      [&](std::stop_token stop) { return f.preview(*first, stop); }, {}, {},
      [&, lease = first->admission_lease](const image::ImageSinkWriteResult&) {
        (void)lease;
        entered.set_value();
        if (release_future.wait_for(10s) != std::future_status::ready)
          throw std::runtime_error("Test evidence gate timed out");
      });
  first.reset();
  auto next =
      std::async(std::launch::async, [&] { return f.pipeline->next(); });
  EXPECT_EQ(entered_future.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(f.second_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  EXPECT_EQ(next.wait_for(0s), std::future_status::timeout);
  EXPECT_EQ(f.decoded, 2U);
  EXPECT_EQ(f.acquired, 1U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_batches)
                .current,
            1U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .current,
            kBatchBytes);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::decoded_bytes).current,
            0U);
  release.set_value();
  ticket.evidence_completion.get();
  auto second = next.get();
  EXPECT_TRUE(second);
  EXPECT_EQ(f.decoded, 4U);
  second.reset();
  sink.close();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     CancellationWhileEvidenceOwnsAdmissionDoesNotDecode) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  auto lease = first->admission_lease;
  first.reset();
  EXPECT_EQ(f.second_attempt.get_future().wait_for(5s),
            std::future_status::ready);
  std::stop_source stop;
  auto next = std::async(std::launch::async,
                         [&] { return f.pipeline->next(stop.get_token()); });
  EXPECT_EQ(next.wait_for(0s), std::future_status::timeout);
  stop.request_stop();
  EXPECT_FALSE(next.get());
  f.pipeline->close();
  EXPECT_EQ(f.decoded, 2U);
  EXPECT_EQ(f.budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                .current,
            kBatchBytes);
  lease.reset();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     FailedEvidenceReleasesAdmissionAndPreservesPublishedFile) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  image::ImageOutputSink sink;
  auto ticket = sink.produce_and_submit(
      f.root / "preview.png", kBatchBytes,
      [&](std::stop_token stop) { return f.preview(*first, stop); }, {}, {},
      [lease = first->admission_lease](const image::ImageSinkWriteResult&) {
        (void)lease;
        throw std::runtime_error("Evidence rejected");
      });
  first.reset();
  EXPECT_NO_THROW(ticket.completion.get());
  EXPECT_THROW(ticket.evidence_completion.get(), std::runtime_error);
  EXPECT_THROW(sink.close(), std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_regular_file(f.root / "preview.png"));
  auto second = f.pipeline->next();
  EXPECT_TRUE(second);
  second.reset();
  f.expect_released();
}

TEST(ImageOutputPipelineTest,
     FailedProducerReleasesOutputReservationWithoutEvidence) {
  Fixture f;
  auto first = f.pipeline->next();
  ASSERT_TRUE(first);
  image::ImageOutputSink sink({.resource_budget = f.budget});
  bool recorded = false;
  EXPECT_THROW(sink.produce_and_submit(
                   f.root / "preview.png", kBatchBytes,
                   [](std::stop_token) -> image::EncodedImage {
                     throw std::runtime_error("Encoder rejected");
                   },
                   {}, {},
                   [&, lease = first->admission_lease](
                       const image::ImageSinkWriteResult&) {
                     (void)lease;
                     recorded = true;
                   }),
               std::runtime_error);
  first.reset();
  sink.close();
  EXPECT_FALSE(recorded);
  EXPECT_FALSE(std::filesystem::exists(f.root / "preview.png"));
  EXPECT_EQ(sink.stats().producing_items, 0U);
  f.expect_released();
}
}  // namespace
