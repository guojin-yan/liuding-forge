// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_EXTENDED_HPP
#define JYPPX_LIUDING_VISION_TASKS_EXTENDED_HPP
/** @file extended.hpp @brief Typed field/embedding decoding and region criteria. */
#include <jyppx/liuding/vision/tasks/results.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>
namespace jyppx::liuding::vision::tasks {
std::vector<ImageResult> decode_extended_results(std::span<const Tensor> raw,
    std::array<std::int64_t,2> size,TaskKind task,const ResultOptions& options={},std::span<const double> thresholds={});
training::StepFunction make_fast_rcnn_step(std::int64_t classes);
training::StepFunction make_reid_step(double triplet_margin=.3,double triplet_weight=1.);
/** @brief Existing unified workflow dispatch for supplementary task evaluation. */
struct RecipeSpec;struct TaskBinding;
std::string calibrate_vision_multilabel(TaskBinding&,const data::image::ImageDataset&,
    const RecipeSpec&,std::string validation_identity,std::stop_token stop={});
std::string evaluate_extended_task(TaskBinding& binding,
    const data::image::ImageDataset& dataset,const RecipeSpec& recipe,
    std::string dataset_id,std::stop_token stop={});
}
#endif
