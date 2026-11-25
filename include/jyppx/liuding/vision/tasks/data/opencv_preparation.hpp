// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_OPENCV_PREPARATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_OPENCV_PREPARATION_HPP
/** @file opencv_preparation.hpp @brief Explicit OpenCV box-only training
 * operations. */

#include <jyppx/liuding/data_image/joint_transform.hpp>

#include <array>
#include <span>

namespace jyppx::liuding::vision::tasks {
/** @brief Fixed values for T*S*R*P*C geometry on a square training canvas.
 * Inputs are resized to the longest side with ceil rounding and INTER_LINEAR.
 * Mosaic consumes exactly four samples and builds a twice-sized canvas; other
 * plans consume one sample and center-pad it. This plan does not select
 * partners or sample randomness. Box coordinates use float32 normalized source
 * values.
 */
struct YoloOpenCvGeometryPlan {
  std::uint32_t size = 640;
  bool mosaic = false;
  std::array<std::uint32_t, 2> mosaic_center{
      640, 640};  ///< x,y in [size/2,3*size/2].
  bool warp = true;
  bool perspective = false;  ///< Selects warpPerspective instead of warpAffine.
  double perspective_x = 0, perspective_y = 0;
  double rotation_degrees = 0, scale = 1;
  double shear_x_degrees = 0, shear_y_degrees = 0;
  double translate_x = .5,
         translate_y = .5;  ///< Fractions of output dimensions.
};
/** @brief Fixed HSV LUT gains, axis flips and optional complete-partner MixUp.
 * MixUp uses truncating uint8 arithmetic before HSV. The caller supplies an
 * already fully transformed partner and a coefficient; this operation neither
 * approximates a Beta distribution nor chooses a partner. RGB output remains
 * RGB.
 */
struct YoloOpenCvColorPlan {
  double mixup_ratio =
      1;             ///< [0,1]; a partner is required exactly when below1.
  bool hsv = false;  ///< Also apply HSV conversion when all sampled gains are
                     ///< identity.
  double hue_shift = 0;  ///< Fraction of the OpenCV180 hue circle, in [-1,1].
  double saturation_gain = 1, value_gain = 1;  ///< Multipliers in [0,2].
  bool horizontal_flip = false, vertical_flip = false;
};
/** @brief Reports whether the explicit OpenCV5.1 backend was built. */
bool yolo_opencv_available() noexcept;
/** @brief Applies fixed longest-side load, mosaic/letterbox and optional warp.
 * @param samples One sample, or exactly four when mosaic is set; CPU uint8 RGB.
 * @param plan Fixed geometry and box-candidate filtering (2px, .1 area,
 * aspect100).
 * @param options Existing allocation/geometry limits and aggregate budget.
 * @param stop Cancellation checked before allocation and between
 * stages/objects.
 * @return Owning CHW uint8 sample; rich targets reject, crowd/ignore are
 * excluded. Multiple sources or a warp retain object provenance without an
 * inverse trace.
 * @throws NotImplemented Backend was not explicitly enabled at build time.
 * @throws InvalidArgument Invalid plan, pixels, categories or non-box targets.
 * @throws ResourceExhausted Dimension, object, output or working budget
 * exceeded.
 * @throws Cancelled Cancellation was requested before publication.
 * @note Supplies fixed operations for custom native batch factories, not a
 * complete official trainer, partner-buffer sampler or recipe qualification.
 */
data::image::VisionSample prepare_yolo_opencv_geometry(
    std::span<const data::image::VisionSample> samples,
    const YoloOpenCvGeometryPlan& plan,
    const data::image::JointTransformOptions& options = {},
    std::stop_token stop = {});
/** @brief Applies MixUp, RGB HSV LUT and flips without modifying source
 * storage.
 * @param sample Prepared square CPU uint8 RGB box-only sample.
 * @param plan Fixed photometric/flip values.
 * @param partner Fully pre-transformed sample, required for ratio below1.
 * @param options Existing payload/object limits and aggregate budget.
 * @param stop Cooperative cancellation.
 * @return Independent CHW uint8 pixels, boxes and per-object source mappings.
 * @throws NotImplemented Backend was not enabled at build time.
 * @throws InvalidArgument Unsupported samples, partner or plan.
 * @throws ResourceExhausted Existing resource budget was exceeded.
 * @throws Cancelled Cancellation was requested before publication.
 */
data::image::VisionSample prepare_yolo_opencv_color(
    const data::image::VisionSample& sample, const YoloOpenCvColorPlan& plan,
    const data::image::VisionSample* partner = nullptr,
    const data::image::JointTransformOptions& options = {},
    std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif
