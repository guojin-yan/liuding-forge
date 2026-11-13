// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_TARGETS_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_TARGETS_HPP
/** @file targets.hpp @brief Rich training projections of canonical annotations.
 */
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Projects the same filtered object rows to box/mask/pose/OBB tensors.
 * Polygon masks use union of exterior rings and pixel-center scan conversion;
 * RLE uses canonical COCO column-major runs. Missing required task geometry is
 * rejected. OBB targets must describe rectangles, rather than arbitrary quads.
 * No file I/O, global RNG, source mutation or alternate annotation store.
 */
TaskTargets project_task_targets(const data::image::ImageBatch& batch,
                                 const YoloTaskOptions& task,
                                 const YoloDataOptions& data = {});
/** @brief Rich-target mapper returning boxes/masks/keypoints/oriented-box
 * tuple. */
training::TargetMapper make_task_target_mapper(
    const YoloTaskOptions& task, const YoloDataOptions& data = {});
/** @brief Existing ImageBatch labels as exactly one hard class per image. */
training::TargetMapper make_classification_target_mapper();
/** @brief Faster R-CNN invocation: normalized images, targets and valid
 * extents. The supplied image normalization must match the chosen backbone
 * recipe.
 */
training::BatchMapper make_rcnn_image_mapper(const YoloDataOptions& data = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_TARGETS_HPP
