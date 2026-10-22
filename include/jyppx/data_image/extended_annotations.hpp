// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_IMAGE_EXTENDED_ANNOTATIONS_HPP
#define JYPPX_DATA_IMAGE_EXTENDED_ANNOTATIONS_HPP

/**
 * @file extended_annotations.hpp
 * @brief Built-in DOTA, Cityscapes, KITTI object and MOTChallenge readers.
 *
 * Readers consume UTF-8 document bytes and never open files. Source-specific
 * geometry, ignore markers, unknown-value sentinels and tracking metadata stay
 * in owned values alongside the common vision targets. Image decoding, camera
 * calibration, coordinate projection and label remapping are separate steps.
 */

#include <jyppx/data_image/annotations.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::data::image {

/** @brief Controls treatment of finite geometry outside the declared image. */
enum class AnnotationBounds : std::uint8_t {
  within_image,      ///< Reject coordinates outside [0,width] x [0,height].
  preserve_outside,  ///< Preserve finite coordinates without clipping them.
};

/** @brief Frozen class/size context and independent delimited-text budgets. */
struct TextAnnotationReadOptions {
  AnnotationLimits limits;  ///< Input, class, object, point and node limits.
  std::vector<VisionCategory> categories =
      {};                       ///< Required exact source classes.
  std::string image_path = {};  ///< Image path, or MOT sequence reference.
  std::uint32_t width = 0;      ///< Required original image/sequence width.
  std::uint32_t height = 0;     ///< Required original image/sequence height.
  AnnotationBounds bounds =
      AnnotationBounds::within_image;   ///< Clip-free policy.
  std::size_t max_line_bytes = 65'536;  ///< Maximum bytes in one physical line.
  std::size_t max_lines =
      5'000'002;  ///< Physical lines, including blanks/header.
};

/** @brief DOTA quadrilateral and its exact source difficulty value. */
struct DotaObjectAnnotation {
  VisionObject
      target;  ///< Ordered quadrilateral; difficult means difficulty != 0.
  std::int64_t difficulty = 0;  ///< Nonnegative original difficulty integer.
};

/** @brief DOTA 1.x image labels with optional standard header metadata. */
struct DotaAnnotationIndex {
  std::vector<VisionCategory> categories =
      {};                       ///< Frozen source class mapping.
  std::string image_path = {};  ///< Dataset-relative image reference.
  std::uint32_t width = 0;      ///< Declared image width.
  std::uint32_t height = 0;     ///< Declared image height.
  std::optional<std::string> image_source;  ///< Optional imagesource header.
  std::optional<double>
      ground_sample_distance;  ///< Positive gsd, absent for null.
  bool has_ground_sample_distance_header =
      false;  ///< Distinguishes absent/null.
  std::vector<DotaObjectAnnotation> objects = {};  ///< Source row order.
  std::string source_sha256 =
      {};  ///< Digest of the exact input document bytes.
};

/** @brief One Cityscapes polygon with source review and instance metadata. */
struct CityscapesObjectAnnotation {
  VisionObject
      target;  ///< Polygon in image coordinates; deleted objects are ignored.
  std::optional<std::int64_t> object_id;    ///< Optional annotation object ID.
  std::optional<std::int64_t> instance_id;  ///< Optional semantic instance ID.
  bool deleted =
      false;  ///< Source marks the object deleted; geometry may be empty.
  bool verified =
      false;  ///< Source review marker, retained without changing geometry.
  std::optional<std::string> user;  ///< Optional annotator identifier.
  std::optional<std::string> date;  ///< Optional source timestamp string.
};

/** @brief Cityscapes polygon JSON for one explicitly sized image. */
struct CityscapesAnnotationIndex {
  std::vector<VisionCategory> categories =
      {};  ///< Frozen source label mapping.
  std::string image_path =
      {};  ///< Caller-supplied dataset-relative image reference.
  std::uint32_t width = 0;   ///< Document image width.
  std::uint32_t height = 0;  ///< Document image height.
  std::vector<CityscapesObjectAnnotation> objects =
      {};                          ///< Source object order.
  std::string source_sha256 = {};  ///< Digest of the exact JSON bytes.
};

/** @brief Explicit KITTI object-label or detection-result column contract. */
enum class KittiObjectDialect : std::uint8_t {
  ground_truth,      ///< Exactly 15 whitespace-delimited columns.
  detection_result,  ///< Exactly 16 columns, including the detection score.
};

/** @brief KITTI camera-coordinate 3D values, including original sentinels. */
struct KittiBox3D {
  std::array<double, 3> dimensions_hwl =
      {};  ///< Height, width, length in metres.
  std::array<double, 3> location_xyz =
      {};  ///< Bottom-centre camera coordinates.
  double rotation_y =
      -10;  ///< Rotation about camera Y, or unknown sentinel -10.
  bool dimensions_known =
      false;                    ///< False only for the complete {-1,-1,-1} set.
  bool location_known = false;  ///< False only for {-1000,-1000,-1000}.
  bool rotation_known = false;  ///< False only when rotation_y is -10.
};

/** @brief One KITTI object preserving 2D, 3D and evaluation metadata. */
struct KittiObjectAnnotation {
  VisionObject target;     ///< Absolute 2D box; DontCare has ignore=true.
  double truncation = -1;  ///< Fraction [0,1], or KITTI unknown sentinel -1.
  std::int32_t occlusion =
      -1;              ///< -1 unknown, 0 visible, 1 partial, 2 full, 3 unknown.
  double alpha = -10;  ///< Observation angle [-pi,pi], or unknown sentinel -10.
  bool alpha_known = false;  ///< False only when alpha is -10.
  KittiBox3D
      box_3d;  ///< Source values; no calibration/projection is performed.
  std::optional<double>
      score;  ///< Unrestricted finite detection score, if present.
};

/** @brief Owned KITTI label/result records for one explicitly sized image. */
struct KittiAnnotationIndex {
  KittiObjectDialect dialect =
      KittiObjectDialect::ground_truth;  ///< Input contract.
  std::vector<VisionCategory> categories =
      {};                       ///< Exact mapping, including DontCare.
  std::string image_path = {};  ///< Dataset-relative image reference.
  std::uint32_t width = 0;      ///< Declared image width.
  std::uint32_t height = 0;     ///< Declared image height.
  std::vector<KittiObjectAnnotation> objects = {};  ///< Source row order.
  std::string source_sha256 =
      {};  ///< Digest of the exact input document bytes.
};

/** @brief Explicit MOTChallenge CSV schema; no column-based auto-detection. */
enum class MotChallengeDialect : std::uint8_t {
  ground_truth_9,         ///< MOT16/17: frame,id,box,mark,class,visibility.
  ground_truth_world_10,  ///< MOT15: frame,id,box,mark,world_x,world_y,world_z.
  detection_10,  ///< Detection rows; ID may be -1 and confidence is finite.
  tracking_result_10,  ///< Tracking results with positive per-frame unique IDs.
};

/** @brief Explicit source convention for MOT left/top coordinates. */
enum class MotCoordinates : std::uint8_t {
  one_based,   ///< Standard MOT: subtract one from left/top; preserve
               ///< width/height.
  zero_based,  ///< Converted data already uses continuous zero-based left/top.
};

/** @brief MOT sequence context; all frames share the declared dimensions. */
struct MotChallengeReadOptions {
  TextAnnotationReadOptions
      text;  ///< Size, sequence path, class mapping and limits.
  MotChallengeDialect dialect = MotChallengeDialect::ground_truth_9;
  ///< Explicit source column layout; it is never inferred from row values.
  MotCoordinates coordinates =
      MotCoordinates::one_based;  ///< Source left/top policy.
  std::optional<std::size_t>
      unclassified_category;  ///< Required for 10-column data.
  std::optional<std::int64_t>
      frame_count;  ///< Optional positive sequence length.
};

/** @brief MOT target plus identity, confidence and optional source metadata. */
struct MotObjectAnnotation {
  VisionObject target;  ///< Canonical box; ground-truth mark 0 has ignore=true.
  std::int64_t track_id =
      -1;  ///< Original identity, including detection sentinel -1.
  double confidence =
      0;  ///< Original mark/confidence; no implicit score threshold.
  std::optional<std::int64_t> source_class;  ///< Nine-column source class ID.
  std::optional<double> visibility;  ///< Nine-column visible fraction in [0,1].
  std::optional<std::array<double, 3>>
      world_xyz;  ///< Ten-column values, including -1.
};

/** @brief One observed MOT frame; empty/unlisted frames are not synthesized. */
struct MotFrameAnnotation {
  std::int64_t frame = 0;  ///< Positive source frame number, never renumbered.
  std::vector<MotObjectAnnotation> objects =
      {};  ///< Source row order in this frame.
};

/** @brief A single MOT sequence with frames in first-appearance order. */
struct MotChallengeAnnotationIndex {
  MotChallengeDialect dialect = MotChallengeDialect::ground_truth_9;
  ///< Source column layout retained by this index.
  std::vector<VisionCategory> categories =
      {};  ///< Frozen source class mapping.
  std::string sequence_path =
      {};                    ///< Caller-supplied dataset-relative reference.
  std::uint32_t width = 0;   ///< Shared frame width.
  std::uint32_t height = 0;  ///< Shared frame height.
  std::vector<MotFrameAnnotation> frames = {};  ///< Observed frames only.
  std::string source_sha256 =
      {};  ///< Digest of the exact input document bytes.
};

/**
 * @brief Parses DOTA 1.x per-image quadrilateral TXT without dropping
 * difficulty.
 * @param bytes UTF-8 labels with optional imagesource/gsd headers before
 * objects.
 * @param options Required image path, dimensions, frozen class map and budgets.
 * @param stop Cooperative cancellation checked between lines and fields.
 * @return Source-ordered simple quadrilaterals and original header metadata.
 * @throws InvalidArgument Invalid reader options.
 * @throws SerializationError Invalid rows, unknown labels or invalid geometry.
 * @throws ResourceExhausted Configured limits or allocation failure.
 * @throws Cancelled Parsing was cancelled before returning a result.
 * @note Both polygon winding orders and nonrectangular simple quads are
 * retained. Comments, result-score TXT and image dimensions inferred from files
 * are excluded.
 */
DotaAnnotationIndex parse_dota_1x_annotations(
    std::string_view bytes, const TextAnnotationReadOptions& options,
    std::stop_token stop = {});

/**
 * @brief Parses a Cityscapes polygon JSON document with explicit label mapping.
 * @param bytes UTF-8 JSON containing imgWidth/imgHeight and an objects array.
 * @param options Required image path, dimensions, categories and JSON budgets.
 * @param stop Cooperative cancellation checked between objects and points.
 * @return Polygon targets plus deleted/verified/user/date/instance metadata.
 * @throws InvalidArgument Invalid dimensions, categories or reader options.
 * @throws SerializationError Malformed JSON, unknown labels or invalid
 * polygons.
 * @throws ResourceExhausted Configured JSON/object/point limits or allocation
 * failure.
 * @throws Cancelled Parsing was cancelled before returning a result.
 * @note Deleted objects are retained with ignore=true and may have no polygon.
 * Polygon rasterization, imageData and label-id inference are intentionally
 * absent.
 */
CityscapesAnnotationIndex parse_cityscapes_polygon_annotations(
    std::string_view bytes, const TextAnnotationReadOptions& options,
    std::stop_token stop = {});

/**
 * @brief Parses the explicit KITTI object detection label/result schema.
 * @param bytes UTF-8 whitespace-delimited 15- or 16-column document.
 * @param dialect Exact ground-truth or scored-result contract.
 * @param options Required image path, dimensions and source classes including
 * DontCare.
 * @param stop Cooperative cancellation checked between lines and fields.
 * @return Owned 2D/3D labels, unknown sentinels and optional result scores.
 * @throws InvalidArgument Invalid options or dialect.
 * @throws SerializationError Invalid columns, values, classes or geometry.
 * @throws ResourceExhausted Configured limits or allocation failure.
 * @throws Cancelled Parsing was cancelled before returning a result.
 * @note Unknown values remain exact; the parser never reconstructs missing 3D
 * data. KITTI tracking, odometry, calibration and point clouds use different
 * schemas.
 */
KittiAnnotationIndex parse_kitti_object_annotations(
    std::string_view bytes, KittiObjectDialect dialect,
    const TextAnnotationReadOptions& options, std::stop_token stop = {});

/**
 * @brief Parses one explicitly sized MOTChallenge sequence CSV document.
 * @param bytes UTF-8 unquoted numeric CSV with the selected exact column count.
 * @param options Explicit dialect, coordinate policy, class context and limits.
 * @param stop Cooperative cancellation checked between lines and fields.
 * @return Source frame/track identities, confidence, visibility and world
 * metadata.
 * @throws InvalidArgument Invalid options, dialect or coordinate policy.
 * @throws SerializationError Invalid CSV, duplicates, values or class
 * references.
 * @throws ResourceExhausted Configured limits or allocation failure.
 * @throws Cancelled Parsing was cancelled before returning a result.
 * @note IDs must be unique per frame except repeated -1 detection sentinels.
 * Ten-column data requires an explicit unclassified_category; no class is
 * guessed. Sequence INI, per-frame paths and interpolation are not inferred by
 * this parser.
 */
MotChallengeAnnotationIndex parse_mot_challenge_annotations(
    std::string_view bytes, const MotChallengeReadOptions& options,
    std::stop_token stop = {});

}  // namespace jyppx::data::image

#endif  // JYPPX_DATA_IMAGE_EXTENDED_ANNOTATIONS_HPP
