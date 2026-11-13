// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_EXPORT_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_EXPORT_HPP
/** @file export.hpp @brief Fixed-profile native ONNX artifacts and offline
 * prediction. */
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
struct VisionExportOptions {
  std::int64_t batch_size =
      1;  ///< Fixed batch, width and height recorded in manifest.
  std::uint32_t opset = 17;
  std::uint64_t max_bytes = 1ULL << 30U;
  std::int64_t rcnn_proposals =
      300;  ///< Fixed padded proposals per image in RCNN graph.
  std::string weight_role = "raw";  ///< Explicit raw/ema input state identity.
  bool fuse =
      false;  ///< Fold supported Conv/BN blocks in the disposable eval clone.
};
/** @brief Exports a CPU clone of the bound state, leaving caller modes/device
 * intact. The optional onnx_export component is required. YOLO exposes raw task
 * outputs; RCNN lowers RPN selection, NMS and multiscale ROIAlign to standard
 * ONNX nodes. Output is a new directory with model.onnx and a complete
 * preprocessing/decoder manifest. Fixed profiles reject other shapes; no
 * dynamic-batch claim is made.
 */
std::string export_vision(const TaskBinding& binding, const RecipeSpec& recipe,
                          const std::filesystem::path& output,
                          const VisionExportOptions& options = {});
/** @brief Offline ORT execution followed by the same native task postprocessor.
 * Requires onnx_runtime. Every model digest/profile is checked before
 * invocation. Returned JSON records preprocessing, model and postprocessing
 * wall times.
 */
std::string predict_vision_bundle(
    const std::filesystem::path& bundle,
    std::span<const data::image::VisionSample> samples,
    std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_EXPORT_HPP
