// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#ifndef JYPPX_TEST_IMAGE_OUTPUT_PIPELINE_FIXTURE_HPP
#define JYPPX_TEST_IMAGE_OUTPUT_PIPELINE_FIXTURE_HPP

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

namespace image_output_pipeline_test {
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
  std::promise<void> blocked_attempt;

  explicit Fixture(std::size_t live_batches = 1, std::size_t samples = 4) {
    static std::atomic<unsigned> serial{0};
    root = std::filesystem::temp_directory_path() /
           ("liuding-output-pipeline-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(serial.fetch_add(1)));
    std::filesystem::create_directories(root / "images/class");
    image::PixelSpec spec;
    spec.layout = image::ImageLayout::chw;
    for (std::size_t i = 0; i < samples; ++i)
      image::write_image_file(
          root / "images/class" / (std::to_string(i) + ".png"),
          torch::full({3, 8, 8}, 17 + i, torch::kUInt8), spec);
    image::ImageResourceBudgetLimits limits;
    limits.max_pipeline_inflight_batches = live_batches == 1 ? 1 : 4;
    limits.max_pipeline_inflight_bytes = live_batches * kBatchBytes;
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
    options.dataset_size = samples;
    options.dataset_version = "test-pngs-" + std::to_string(samples);
    options.batch_factory_id = "resize-collate/v1";
    options.batch_size = 2;
    options.num_workers = options.prefetch_batches = live_batches + 1;
    const auto admission =
        image::make_image_batch_byte_admission_factory(budget, collate.limits);
    options.admission_factory = [this, admission, live_batches](
                                    std::stop_token stop) {
      if (attempts.fetch_add(1) == live_batches) blocked_attempt.set_value();
      auto result = admission(stop);
      acquired.fetch_add(1);
      return result;
    };
    options.stage_resource_factory =
        image::make_image_batch_stage_resource_factory(budget, collate.limits);
    pipeline = std::make_unique<data::DataPipeline>(
        std::make_shared<data::samplers::SequentialSampler>(samples), options,
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

}  // namespace image_output_pipeline_test

#endif
