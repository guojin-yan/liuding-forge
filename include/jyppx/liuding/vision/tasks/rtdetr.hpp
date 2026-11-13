// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_RTDETR_HPP
#define JYPPX_LIUDING_VISION_TASKS_RTDETR_HPP
/** @file rtdetr.hpp @brief RT-DETR v1 matching, auxiliary and denoising losses. */
#include <jyppx/liuding/vision/tasks/results.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief VFL + 5*L1 + 2*GIoU with detached focal Hungarian cost 2/5/2.
 * Targets use canvas pixels; empty targets retain classification gradients.
 * Every decoder layer and selected encoder queries are supervised. Denoising
 * uses known positive/negative correspondence and group-normalized losses. */
TaskLoss rtdetr_loss(std::span<const Tensor> outputs,const Tensor& targets,
    std::array<std::int64_t,2> canvas,std::int64_t num_classes);
training::StepFunction make_rtdetr_step(std::int64_t num_classes);
/** @brief Sigmoid query/class top-k without NMS, with standard result filters. */
std::vector<ImageResult> decode_rtdetr(std::span<const Tensor> outputs,
    std::array<std::int64_t,2> canvas,const ResultOptions& options={});
training::BatchMapper make_rtdetr_image_mapper(const YoloDataOptions& data={});
}
#endif
