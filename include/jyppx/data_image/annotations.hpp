// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_IMAGE_ANNOTATIONS_HPP
#define JYPPX_DATA_IMAGE_ANNOTATIONS_HPP

/**
 * @file annotations.hpp
 * @brief Defines image annotation values and built-in bounded format readers.
 *
 * Coordinates use absolute continuous pixel boundaries, with boxes stored as
 * xyxy in [0,width] x [0,height]. Readers preserve source IDs separately from
 * contiguous training IDs. These values own their storage and contain no
 * decoder, model, mutable global registry, or third-party parser objects.
 */

#include <jyppx/foundation/error.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::data::image {

/** @brief Selects an explicit image annotation dialect. */
enum class AnnotationFormat : std::uint8_t {
  coco_instances,     ///< COCO images/categories/instances JSON.
  coco_keypoints,     ///< COCO person or custom keypoint JSON.
  coco_panoptic,      ///< COCO segment table plus external ID image reference.
  yolo_detection,     ///< Class and normalized center/size TXT rows.
  yolo_segmentation,  ///< Class and normalized polygon TXT rows.
  yolo_pose,          ///< Class/box/keypoints TXT with an explicit shape.
  yolo_obb,           ///< Class and four normalized corners TXT rows.
  voc,                ///< Pascal VOC XML with an explicit coordinate policy.
  labelme,            ///< LabelMe JSON with external image references.
};

/** @brief Selects how Pascal VOC integer box coordinates are interpreted. */
enum class VocCoordinates : std::uint8_t {
  one_based_inclusive,  ///< [xmin-1,ymin-1,xmax,ymax] boundary conversion.
  zero_based_boundary,  ///< Coordinates already describe continuous boundaries.
};

/** @brief Stable category metadata shared by every record in an index. */
struct VisionCategory {
  std::int64_t source_id = 0;  ///< Original external category ID.
  std::string name = {};       ///< Nonempty UTF-8 class label.
  bool is_thing = true;  ///< False for semantic/panoptic stuff categories.
  std::vector<std::string> keypoint_names = {};  ///< Ordered pose point names.
  std::vector<std::array<std::size_t, 2>> skeleton =
      {};  ///< Zero-based point links.
};

/** @brief Absolute continuous image coordinate. */
struct ImagePoint {
  double x = 0;  ///< Horizontal coordinate in the declared image space.
  double y = 0;  ///< Vertical coordinate in the declared image space.
};

/** @brief Absolute xyxy rectangle with positive area. */
struct ImageBox {
  double x_min = 0;  ///< Left pixel boundary.
  double y_min = 0;  ///< Upper pixel boundary.
  double x_max = 0;  ///< Right pixel boundary.
  double y_max = 0;  ///< Lower pixel boundary.
};

/** @brief Point whose visibility semantics match COCO (0/1/2). */
struct ImageKeypoint {
  ImagePoint point;             ///< Position; ignored when visibility is zero.
  std::uint8_t visibility = 0;  ///< 0 absent, 1 labeled/occluded, 2 visible.
};

/** @brief Validated COCO column-major alternating background/foreground RLE. */
struct ImageMaskRle {
  std::uint32_t width = 0;            ///< Mask width in pixels.
  std::uint32_t height = 0;           ///< Mask height in pixels.
  std::vector<std::uint64_t> counts;  ///< Runs summing exactly to width*height.
};

/** @brief One object or segment in the common annotation representation. */
struct VisionObject {
  std::int64_t source_id =
      0;                     ///< Dataset-local original annotation/segment ID.
  std::size_t category = 0;  ///< Contiguous index in the category table.
  std::optional<ImageBox> box;  ///< Absolute axis-aligned extent when present.
  std::vector<std::vector<ImagePoint>> polygons;  ///< Separate exterior rings.
  std::optional<std::array<ImagePoint, 4>> oriented_box;  ///< Ordered corners.
  std::optional<ImageMaskRle> rle;       ///< Lossless mask runs when supplied.
  std::vector<ImageKeypoint> keypoints;  ///< Category-ordered keypoints.
  std::optional<double>
      area;  ///< Source-declared area, not an inferred bbox area.
  std::optional<std::int64_t> group_id;  ///< LabelMe shape grouping.
  bool crowd = false;                    ///< COCO crowd region marker.
  bool ignore = false;                   ///< Explicit ignore region marker.
  bool difficult = false;  ///< VOC difficult marker retained for evaluation.
  bool truncated = false;  ///< Source marks a truncated object.
};

/** @brief One image and its annotations; an empty object list is valid. */
struct VisionAnnotationRecord {
  std::int64_t source_image_id =
      0;                     ///< Original image ID or deterministic row ID.
  std::string image_path;    ///< Validated dataset-relative UTF-8 path.
  std::uint32_t width = 0;   ///< Width before augmentation/orientation.
  std::uint32_t height = 0;  ///< Height before augmentation/orientation.
  std::vector<VisionObject>
      objects;  ///< Ordered source objects, including crowd.
  std::optional<std::string>
      panoptic_mask_path;  ///< External segment ID image.
};

/** @brief Independently bounded annotation parser resources. */
struct AnnotationLimits {
  std::size_t max_input_bytes = 64U << 20U;  ///< Encoded document byte limit.
  std::size_t max_images = 1'000'000;        ///< Image record limit.
  std::size_t max_objects = 5'000'000;       ///< Total object limit.
  std::size_t max_categories = 65'536;       ///< Class table limit.
  std::size_t max_points = 10'000'000;       ///< Total polygon/keypoint limit.
  std::size_t max_rle_runs = 10'000'000;     ///< Total expanded RLE run limit.
  std::size_t max_rle_bytes = 16U
                              << 20U;  ///< One encoded COCO RLE string limit.
  std::size_t max_geometry_checks =
      20'000'000;                          ///< Polygon candidate-pair budget.
  std::size_t max_string_bytes = 16'384;   ///< Individual decoded string limit.
  std::size_t max_nodes = 20'000'000;      ///< JSON/XML structural node limit.
  std::uint32_t max_dimension = 65'536;    ///< Image/mask side limit.
  std::uint64_t max_pixels = 268'435'456;  ///< Image/mask pixel limit.
  std::uint32_t max_depth = 32;            ///< JSON/XML nesting depth limit.
};

/** @brief Input context for formats that omit image sizes or class tables. */
struct AnnotationReadOptions {
  AnnotationLimits
      limits;  ///< Logical limits; parser scratch also uses input bounds.
  std::vector<VisionCategory>
      categories;            ///< YOLO/VOC/LabelMe frozen class map.
  std::string image_path;    ///< Required for YOLO, fallback for LabelMe/VOC.
  std::uint32_t width = 0;   ///< Required for YOLO, optional source-size check.
  std::uint32_t height = 0;  ///< Required for YOLO, optional source-size check.
  std::size_t keypoint_count = 0;        ///< YOLO pose point count per object.
  std::uint8_t keypoint_dimensions = 3;  ///< YOLO pose x/y or x/y/visibility.
  VocCoordinates voc_coordinates = VocCoordinates::one_based_inclusive;
  ///< Source VOC coordinate convention before canonical conversion.
};

/** @brief Fully owned index that can be read concurrently without mutation. */
struct AnnotationIndex {
  AnnotationFormat format =
      AnnotationFormat::coco_instances;    ///< Source dialect.
  std::vector<VisionCategory> categories;  ///< Frozen continuous-ID mapping.
  std::vector<VisionAnnotationRecord>
      images;  ///< Source order, including empties.
  std::string
      source_sha256;  ///< Digest of the exact annotation document bytes.
};

/** @brief Data-only YOLO YAML specification; paths are not opened or executed.
 */
struct YoloDatasetSpec {
  std::string
      root;  ///< Declared path, resolved by the caller against a safe root.
  std::vector<std::string> train;  ///< Training directory/list declarations.
  std::vector<std::string>
      validation;  ///< Validation directory/list declarations.
  std::vector<std::string>
      test;  ///< Optional test directory/list declarations.
  std::vector<VisionCategory>
      categories;                        ///< Contiguous names indexed by class.
  std::size_t keypoint_count = 0;        ///< Zero for non-pose datasets.
  std::uint8_t keypoint_dimensions = 3;  ///< Two coordinates plus optional v.
  std::vector<std::size_t> flip_indices;  ///< Validated left/right permutation.
  bool has_download_instruction = false;  ///< Metadata only; never executed.
  std::string source_sha256;              ///< Digest of the YAML input.
};

/**
 * @brief Parses a safe, data-only YOLO dataset YAML document.
 * @param bytes UTF-8 YAML containing names and optional paths/pose metadata.
 * @param limits Byte, depth, string, node and category resource limits.
 * @param stop Cooperative cancellation checked at parser events.
 * @return Owned class/split/pose metadata without performing I/O or downloads.
 * @throws SerializationError Invalid YAML/schema, aliases, tags or duplicate
 * keys.
 * @throws InvalidArgument Invalid resource options.
 * @throws ResourceExhausted Limits or allocation failure.
 * @throws Cancelled Parsing was cooperatively cancelled.
 */
YoloDatasetSpec parse_yolo_dataset_yaml(std::string_view bytes,
                                        const AnnotationLimits& limits = {},
                                        std::stop_token stop = {});

/**
 * @brief Parses one document using the selected built-in format.
 * @param bytes UTF-8 JSON, XML, or TXT document; no I/O or network is
 * performed.
 * @param format Explicit source dialect; no shape-based auto-detection.
 * @param options Frozen class/size/coordinate context and parser limits.
 * @param stop Cooperative cancellation checked between parser events/objects.
 * @return An owned source-ordered image/target index with an input digest.
 * @throws InvalidArgument Invalid options, class map, or format value.
 * @throws SerializationError Malformed, duplicate, or inconsistent annotations.
 * @throws ResourceExhausted Configured resource limits or allocation failure.
 * @throws Cancelled Cancellation was requested before a commit.
 * @note Parsing is independent across calls; no global state is mutated.
 * Cancellation cannot interrupt an arbitrary third-party token scan mid-call.
 */
AnnotationIndex parse_annotations(std::string_view bytes,
                                  AnnotationFormat format,
                                  const AnnotationReadOptions& options = {},
                                  std::stop_token stop = {});

/**
 * @brief Returns the stable name of a supported dialect.
 * @param format Dialect to identify.
 * @return A static token, or "unknown" for an invalid enum value.
 */
std::string_view annotation_format_name(AnnotationFormat format) noexcept;

}  // namespace jyppx::data::image

#endif  // JYPPX_DATA_IMAGE_ANNOTATIONS_HPP
