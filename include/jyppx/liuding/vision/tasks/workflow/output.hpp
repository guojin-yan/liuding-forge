// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_OUTPUT_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_OUTPUT_HPP
/** @file output.hpp @brief Independent data previews and local prediction
 * artifacts. */
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>
namespace jyppx::liuding::vision::tasks {
struct PreviewOptions {
  std::size_t samples = 16;  ///< First N samples, at most 256.
  std::uint64_t epoch = 0;   ///< Isolated logical augmentation epoch.
  bool annotations = true;   ///< Requires optional vision_render.
};
/** @brief Uses its own pipeline, sample keys and counters; never advances
 * training. Writes original/augmented PNG pairs and a provenance/annotation
 * JSON manifest.
 */
std::string preview_vision(RecipeSpec recipe,
                           const std::filesystem::path& output,
                           const PreviewOptions& options = {},
                           std::stop_token stop = {});
struct PredictionOutputOptions {
  bool json = true, text = false, render = false, crops = false;
  std::size_t max_crops = 300;
};
/** @brief Saves one result with native JSON/TXT, optional PNG and box crops.
 * sample and result must describe the same original coordinates. Classification
 * produces probability records and no object crops. Output paths use a caller
 * supplied basename; category names are never used as filesystem paths.
 */
void save_prediction(const data::image::VisionSample& sample,
                     const ImageResult& result,
                     const std::filesystem::path& basename,
                     const PredictionOutputOptions& options = {},
                     std::stop_token stop = {});
/** @brief Converts canonical boxes/masks/polygons/pose to interoperable COCO
 * JSON. Four-corner OBB uses segmentation and explicit oriented_bbox extension.
 */
std::string coco_annotations(const data::image::AnnotationIndex& index,
                             std::size_t max_bytes = 64U << 20U);
}  // namespace jyppx::liuding::vision::tasks
#endif
