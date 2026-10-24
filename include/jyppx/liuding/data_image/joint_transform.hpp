// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_JOINT_TRANSFORM_HPP
#define JYPPX_LIUDING_DATA_IMAGE_JOINT_TRANSFORM_HPP
/** @file joint_transform.hpp
 * @brief Joint CPU geometry for photos, discrete labels, boxes and pose.
 */

#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/data_image/sample_random.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Axis-aligned operation applied identically to all aligned targets. */
enum class ImageGeometryKind : std::uint8_t {
  resize,
  crop,
  pad,
  horizontal_flip,
  vertical_flip
};

/** @brief One explicit step; inactive fields must remain zero. */
struct ImageGeometryStep {
  ImageGeometryKind kind = ImageGeometryKind::resize;  ///< Operation selector.
  ImageExtent extent = {};  ///< Resize/crop output; zero for flips and padding.
  std::uint32_t left = 0;   ///< Crop origin or left padding.
  std::uint32_t top = 0;    ///< Crop origin or top padding.
  std::uint32_t right = 0;  ///< Right padding only.
  std::uint32_t bottom = 0;  ///< Bottom padding only.
};

/** @brief Photo resampling; labels always use integer nearest-exact sampling.
 */
enum class ImageInterpolation : std::uint8_t {
  nearest_exact,  ///< floor((output_index+0.5)*input_size/output_size).
  bilinear,       ///< LibTorch align_corners=false, antialias=true.
};

/** @brief A partially cropped quadrilateral cannot remain a four-corner OBB. */
enum class OrientedBoxCrop : std::uint8_t {
  to_polygon,  ///< Preserve its clipped region and mark the conversion.
  discard,     ///< Explicitly drop the partially cropped object.
};

/** @brief Per-call tensor payload and annotation limits, excluding input. */
struct JointTransformLimits {
  std::uint32_t max_dimension =
      65'536;  ///< Any input/intermediate/output side.
  std::uint64_t max_pixels = 268'435'456;  ///< Pixels in any single canvas.
  std::uint64_t max_output_bytes = 512ULL << 20U;  ///< All output tensors.
  std::uint64_t max_working_bytes = 1024ULL << 20U;
  ///< Conservative live tensor payload estimate, including interpolation.
  ///< Excludes allocator overhead, caller inputs and LibTorch internal
  ///< workspace.
  std::size_t max_objects = 1'000'000;          ///< Input annotation objects.
  std::size_t max_keypoints = 10'000'000;       ///< Total input keypoints.
  std::size_t max_polygon_points = 10'000'000;  ///< Input/output independently.
  std::size_t max_polygon_fragments =
      1'000'000;  ///< Input/output independently.
  std::size_t max_geometry_checks = 20'000'000;  ///< Validation/triangulation.
  std::size_t max_rle_runs = 10'000'000;  ///< Input/output independently.
  std::uint64_t max_mask_work_pixels = 1'000'000'000;
  ///< Total discrete-raster visits, including all instance RLE masks.
};

/** @brief Explicit aligned-target, interpolation, clipping and pose policies.
 */
struct JointTransformOptions {
  ImageInterpolation interpolation = ImageInterpolation::nearest_exact;
  ///< Applies to photos only; no color/range/alpha conversion is performed.
  double image_padding = 0;   ///< Finite scalar in the declared photo encoding.
  double paired_padding = 0;  ///< Separate paired-photo scalar encoding.
  std::optional<std::int64_t> mask_padding_id;
  ///< Required whenever a discrete mask is padded; never inferred as zero.
  double min_box_area = 0;  ///< Final clipped box area in output pixel units.
  double min_box_visibility = 0;  ///< Surviving source-box area fraction [0,1].
  double min_region_area = 0;  ///< Output polygon area or actual mask pixels.
  double min_region_visibility =
      0;  ///< Surviving source region fraction [0,1].
  OrientedBoxCrop oriented_box_crop = OrientedBoxCrop::to_polygon;
  bool drop_empty_pose =
      false;  ///< Drop pose objects with no remaining points.
  std::vector<std::vector<std::size_t>> horizontal_keypoint_permutations;
  ///< Per-category output-to-input permutation; explicit identity is allowed.
  std::vector<std::vector<std::size_t>> vertical_keypoint_permutations;
  ///< Required for each applied flip axis when keypoints are present.
  JointTransformLimits limits;  ///< Checked before tensor allocation.
};

/**
 * @brief Applies one shared geometry sequence without modifying input storage.
 * @param sample CPU CHW/HWC photo, aligned paired photo/mask and annotations.
 * @param steps At most 128 resize/crop/pad/flip steps in execution order.
 * @param options Explicit interpolation, padding, filtering and pose policy.
 * @param stop Cooperative cancellation between steps, rows and objects.
 * @return Owning contiguous CPU photos/mask, copied annotations and composed
 * trace.
 * @throws InvalidArgument Bad shape, policy, alignment, or unsupported target.
 * @throws ResourceExhausted Dimension, tensor, object or operation budget
 * exceeded.
 * @throws DeviceError A LibTorch operation failed for a backend reason.
 * @throws Cancelled Cancellation was requested before publication.
 *
 * Supports classification, boxes, keypoints, polygons, ordered convex OBBs,
 * COCO RLE and explicitly bound panoptic masks. Clipped concave rings may be
 * partitioned into nonoverlapping convex pieces; holes are not represented.
 * Region filters apply independently to each source exterior ring. RLE and
 * panoptic outputs have exact current pixel counts and tight boxes. Filtering
 * a panoptic segment replaces all its pixels with the declared void ID.
 * Source IDs, categories and flags,
 * including source-declared area, are retained. Box filtering is measured
 * against the original input box, then its clipped final output area. Missing
 * or cropped-away keypoints become (0,0,visibility=0). Paired images must have
 * identical extents; independently registered/resolution targets are rejected.
 * Results are pageable CPU tensors; final-batch pinning is a separate stage.
 */
VisionSample transform_sample(const VisionSample& sample,
                              std::span<const ImageGeometryStep> steps,
                              const JointTransformOptions& options = {},
                              std::stop_token stop = {});

/** @brief Frozen fixed-resize, random-crop and independent flip sampler. */
struct RandomGeometryOptions {
  std::optional<ImageExtent>
      resize;  ///< Optional fixed resize before cropping.
  std::optional<ImageExtent>
      crop;  ///< Uniform integer origin inside the canvas.
  double horizontal_flip_probability = 0;  ///< Finite probability in [0,1].
  double vertical_flip_probability = 0;    ///< Finite probability in [0,1].
};

/**
 * @brief Samples once, with stable draw slots for x/y/horizontal/vertical.
 * @param input Positive input canvas extent.
 * @param options Fixed resize, optional random crop and flip probabilities.
 * @param random Stateless sample stream; its transform ID must bind
 * configuration.
 * @return Explicit sequence reusable for every target and provenance.
 * @throws InvalidArgument Invalid probability, crop or extent.
 */
std::vector<ImageGeometryStep> sample_geometry(
    ImageExtent input, const RandomGeometryOptions& options,
    const SampleRandom& random);

/**
 * @brief Hashes all random/joint options, including resource and pose policies.
 * @param random Frozen sampler configuration.
 * @param joint Frozen execution configuration.
 * @return Versioned lowercase SHA-256 fingerprint for transform_id/factory ID.
 * @throws InvalidArgument Invalid sampler/execution configuration.
 * @note This binds configuration, not dataset contents or a pipeline
 * checkpoint. Bind dataset_version and DataPipeline::batch_factory_id
 * separately. Existing pipeline restore continues to require identical worker
 * topology.
 */
std::string joint_transform_identity(const RandomGeometryOptions& random,
                                     const JointTransformOptions& joint);

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_JOINT_TRANSFORM_HPP
