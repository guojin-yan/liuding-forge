// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_AUGMENTATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_AUGMENTATION_HPP
/** @file augmentation.hpp @brief Bounded native photo and joint geometry
 * augmentation. */

#include <jyppx/liuding/data_image/joint_transform.hpp>

namespace jyppx::liuding::vision::tasks {

struct MultiSourceOptions {
  double mosaic_probability = 0;
  double mixup_probability = 0;
  double cutmix_probability = 0;
  double copypaste_probability = 0;
  std::uint64_t total_epochs = 100;
  std::uint64_t close_epochs = 10;  ///< Closes all four multi-source nodes.
  double max_occlusion = .6;  ///< Maximum CopyPaste intersection/receiver area.
  std::size_t max_objects = 4096;
  std::uint64_t max_pixels = 16777216;
};
struct ProjectiveOptions {
  double degrees = 0;        ///< Symmetric rotation magnitude [0,180].
  double shear_degrees = 0;  ///< Independent x/y shear [0,45].
  double perspective = 0;    ///< Bottom-row projective coefficients [0,.001].
  double scale_gain = 0;     ///< Symmetric scale magnitude [0,.9].
  double translate_fraction = 0;  ///< Independent canvas-relative shift [0,.5].
  double min_box_area = 4;
  double min_visibility = .1;
  std::uint64_t max_pixels = 16777216;
  std::size_t max_objects = 4096;
};
/** @brief CPU uint8 RGB augmentation, applied in the order declared below.
 *
 * HSV uses floating RGB/HSV conversion, additive hue in turns and
 * multiplicative saturation/value. Each stage rounds half up to uint8.
 * This is a native profile, not an OpenCV pixel-equivalence claim.
 */
struct PhotometricOptions {
  double hue_shift = 0;        ///< Uniform shift in [-h,h], h in [0,0.5] turns.
  double saturation_gain = 0;  ///< Multiplier uniform in [1-g,1+g], g in [0,1].
  double value_gain = 0;       ///< Multiplier uniform in [1-g,1+g], g in [0,1].
  double grayscale_probability =
      0;                          ///< RGB BT.601 luma replicated to 3 channels.
  double blur_probability = 0;    ///< Separable Gaussian, replicated border.
  std::uint32_t blur_kernel = 3;  ///< Odd integer in [1,31].
  double blur_sigma = 1;  ///< Finite positive Gaussian sigma in pixel units.
  double median_blur_probability =
      0;  ///< Per-channel median, replicated border.
  double clahe_probability = 0;
  std::uint32_t clahe_tiles = 8;  ///< Square tile grid [1,32].
  double clahe_clip = 2;  ///< Histogram cap multiplier [1,64], BT.601 luma.
  std::uint32_t median_kernel = 3;  ///< Odd integer in [1,7].
  data::image::JointTransformLimits
      limits;  ///< Canvas, output and live payload bounds.
  std::uint64_t max_filter_operations =
      1000000000;  ///< Conservative tap visits.
  std::shared_ptr<data::image::ImageResourceBudget> resource_budget;
  ///< Fallback aggregate budget; retained input leases take precedence.
};

/** @brief Fixed-canvas isotropic scaling and translation, excluding rotation.
 */
struct ScaleTranslateOptions {
  double scale_gain = 0;  ///< Uniform factor [1-g,1+g], g in [0,0.9].
  double translate_fraction =
      0;  ///< Independent offsets within +/- f*canvas, f in [0,0.5].
};

/** @brief Validates and fingerprints photo semantics and logical limits.
 * @param options Explicit profile; runtime budget owner is excluded.
 * @return Versioned SHA-256; no random draws or pixel reads.
 * @throws InvalidArgument Invalid probability, magnitude, kernel or bound.
 */
std::string photometric_identity(const PhotometricOptions& options = {});

/** @brief Applies HSV, grayscale, Gaussian and median stages to input pixels.
 * @param sample CPU uint8 RGB CHW/HWC, with consistent dimensions/PixelSpec.
 * @param options Frozen semantics and finite work/payload bounds.
 * @param random Stateless stream with an identity binding this configuration.
 * @param stop Checked between rows and before publication.
 * @return Sample with new pageable pixels and leases; annotations, masks and
 * geometry are shared unchanged. With all strengths/probabilities zero, shares
 * pixels.
 * @throws InvalidArgument Invalid input metadata or profile.
 * @throws ResourceExhausted Logical resource or filter-work bound exceeded.
 * @throws Cancelled Cancellation requested before publication.
 * @note Does not mutate source pixels, consume global RNG or change paired
 * targets.
 */
data::image::VisionSample augment_photometric(
    const data::image::VisionSample& sample, const PhotometricOptions& options,
    const data::image::SampleRandom& random, std::stop_token stop = {});

/** @brief Validates and fingerprints random scale/translate semantics.
 * @param options Explicit bounded magnitudes.
 * @return Versioned SHA-256; execution geometry is bound separately.
 * @throws InvalidArgument Magnitudes are invalid.
 */
std::string scale_translate_identity(const ScaleTranslateOptions& options = {});

/** @brief Samples resize/crop/pad while retaining the original canvas extent.
 * @param sample Source with the existing joint geometry contract.
 * @param options Isotropic scale and independent translation magnitudes.
 * @param random Stable sample stream; draw slots 0/1/2 are scale/x/y.
 * @param geometry Existing interpolation, fill, filtering and resource policy.
 * @param stop Forwarded to the existing transform owner.
 * @return Jointly transformed sample with composed GeometryTrace. Translation
 * is rounded half away from zero and clamped to retain at least one pixel.
 * @throws InvalidArgument Invalid input or options.
 * @throws ResourceExhausted Existing joint transformation limits exceeded.
 * @throws Cancelled Cancellation requested.
 */
data::image::VisionSample augment_scale_translate(
    const data::image::VisionSample& sample,
    const ScaleTranslateOptions& options,
    const data::image::SampleRandom& random,
    const data::image::JointTransformOptions& geometry = {},
    std::stop_token stop = {});

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_AUGMENTATION_HPP
