// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_YOLO_ORIGINAL_DETAIL_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_YOLO_ORIGINAL_DETAIL_HPP
#include <jyppx/liuding/vision/tasks/data/opencv_preparation.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>

#include <array>
#include <deque>
#include <vector>

namespace jyppx::liuding::vision::tasks::detail {
struct OriginalYoloView {
  std::array<std::size_t, 4> indices{};
  YoloOpenCvGeometryPlan geometry;
};
struct OriginalYoloSample {
  OriginalYoloView primary, partner;
  bool mixup = false;
  YoloOpenCvColorPlan color;
  bool bgr = false;
};
using OriginalYoloBuffer = std::deque<std::size_t>;
OriginalYoloSample plan_original_yolo_sample(
    std::size_t primary, std::size_t population, std::size_t buffer_limit,
    OriginalYoloBuffer& buffer, const YoloDataOptions& options,
    std::uint64_t epoch, std::uint64_t occurrence);
double original_yolo_beta32(const data::image::SampleRandom& random,
                            std::uint64_t first_draw);
void check_original_yolo(const YoloDataOptions& options);
data::image::ImageBatch prepare_original_yolo_evaluation_batch(
    std::span<const data::image::VisionSample> samples,
    std::span<const data::image::ImageBatchKey> keys,
    const YoloDataOptions& options, bool defer_numeric, std::stop_token stop);
std::shared_ptr<data::DataPipeline> make_original_yolo_pipeline(
    std::shared_ptr<const data::image::ImageDataset> dataset,
    data::DataPipelineOptions pipeline, const YoloDataOptions& options,
    std::shared_ptr<data::samplers::Sampler> sampler);
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
