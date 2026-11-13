// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_RESULTS_HPP
#define JYPPX_LIUDING_VISION_TASKS_RESULTS_HPP
/** @file results.hpp @brief Shared typed task predictions and local result
 * formats. */
#include <jyppx/liuding/data_image/geometry_trace.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>

#include <filesystem>
namespace jyppx::liuding::vision::tasks {
/** @brief One image's owning CPU predictions, ordered by descending confidence.
 */
struct ImageResult {
  TaskKind task = TaskKind::detect;
  std::string source_id;
  data::image::ImageExtent extent;
  Tensor boxes;             ///< [N,4] xyxy, float32 CPU.
  Tensor scores;            ///< [N] float32 CPU.
  Tensor classes;           ///< [N] int64 CPU.
  Tensor anchor_indices;    ///< [N] original dense anchor / ROI row identity.
  Tensor masks;             ///< Optional [N,H,W] bool CPU at extent resolution.
  Tensor keypoints;         ///< Optional [N,K,3] xy/confidence, float32 CPU.
  Tensor oriented_corners;  ///< Optional [N,4,2], including anisotropic
                            ///< restoration.
  Tensor field;  ///< Semantic [H,W] int64 or depth/reconstruction [C,H,W] float32 CPU.
  Tensor field_valid;  ///< Optional bool [H,W], excludes inverse-trace padding.
  Tensor embedding;  ///< Optional normalized ReID [D] float32 CPU.
  std::string decode_revision = "native-vision-result/v1";
};
/** @brief Explicit prediction thresholds, filters and finite work bounds. */
struct ResultOptions {
  double confidence = .25;
  double nms_iou = .45;
  double mask_threshold = .5;
  std::int64_t max_candidates = 4096;
  std::int64_t max_detections = 300;
  bool class_agnostic = false;
  bool multi_label = false;
  std::vector<std::int64_t> classes;  ///< Empty selects all classes.
  std::int64_t max_mask_elements = 67108864;
};
/** @brief Stable descending softmax results with confidence and class filters.
 * Accepts finite [B,C] logits. Outputs own CPU tensors at the supplied extent.
 */
std::vector<ImageResult> decode_classification_results(
    const Tensor& logits, std::array<std::int64_t, 2> image_size,
    const ResultOptions& options = {});
/** @brief Decodes selected YOLO branch with true task geometry and bounded NMS.
 * YOLO26 uses one-to-one output and performs score TopK without NMS. Other
 * families use class-aware box or exact rotated IoU NMS. Masks are cropped
 * to predicted boxes before thresholding; keypoints retain their score.
 */
std::vector<ImageResult> decode_yolo_results(
    std::span<const Tensor> outputs, std::array<std::int64_t, 2> image_size,
    const YoloTaskOptions& task = {}, const ResultOptions& options = {});
/** @brief Common boxes/scores suppression for original YOLOv5 or other
 * decoders. */
std::vector<ImageResult> decode_detection_results(
    const YoloDecodedPrediction& decoded,
    std::array<std::int64_t, 2> image_size, const ResultOptions& options = {});
/** @brief Decodes nine-tensor Faster R-CNN raw output, excluding background. */
std::vector<ImageResult> decode_faster_rcnn_results(
    std::span<const Tensor> outputs, const ResultOptions& options = {});
/** @brief Restores boxes, masks, keypoints and corners through one source
 * trace. Synthetic-padding-only detections are dropped consistently across
 * fields. Masks use nearest sampling at original pixel centers, within explicit
 * bounds.
 */
ImageResult restore_result(const ImageResult& result,
                           const data::image::GeometryTrace& trace,
                           std::int64_t max_mask_elements = 67108864);
/** @brief Versioned JSON with complete task geometry and lossless mask RLE.
 * No files are opened; encoded bytes are bounded before return.
 */
std::string result_json(const ImageResult& result,
                        std::size_t max_bytes = 64U << 20U);
/** @brief Local JSON or versioned native TXT; creates no image outputs
 * implicitly. */
void write_result(const std::filesystem::path& path, const ImageResult& result,
                  bool json = true, std::size_t max_bytes = 64U << 20U);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_RESULTS_HPP
