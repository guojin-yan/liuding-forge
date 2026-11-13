// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_DATASET_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_DATASET_HPP
/** @file dataset.hpp @brief Composes detection dataset layouts into canonical
 * annotations. */

#include <jyppx/liuding/data_image/annotation_files.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>

#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::vision::tasks {

/** @brief External layout only; training never branches on this enum. */
enum class DetectionDatasetFormat { coco, voc, yolo, labelme };
/** @brief Selects one data-only YAML split. */
enum class DatasetSplit { train, validation, test };
/** @brief Missing files are distinct from an explicitly empty label document.
 */
enum class MissingLabelPolicy { reject, empty_negative };
/** @brief Label dialect within an external dataset layout. */
enum class DatasetAnnotationTask { detect, instance_segment, pose, obb };

/** @brief Explicit, local-only detection import configuration. */
struct DetectionDatasetOptions {
  DatasetAnnotationTask task = DatasetAnnotationTask::detect;
  std::size_t keypoint_count = 0;  ///< Zero uses explicit YOLO YAML shape.
  std::uint8_t keypoint_dimensions = 3;
  DetectionDatasetFormat format =
      DetectionDatasetFormat::coco;  ///< Source layout.
  std::filesystem::path source;  ///< COCO JSON, VOC XML directory or YOLO YAML.
  std::filesystem::path
      image_root;  ///< Required for COCO/VOC; optional YAML root override.
  DatasetSplit split = DatasetSplit::train;  ///< YOLO YAML split to assemble.
  std::vector<data::image::VisionCategory>
      categories;  ///< Required for VOC; optional frozen train map elsewhere.
  data::image::VocCoordinates voc_coordinates =
      data::image::VocCoordinates::one_based_inclusive;
  MissingLabelPolicy missing_labels =
      MissingLabelPolicy::reject;  ///< YOLO missing-TXT policy.
  data::image::AnnotationLimits
      limits;  ///< Shared aggregate image/object/document bounds.
  data::image::ImageDecodeOptions
      dimension_decode;  ///< Bounded decode used when TXT lacks image
                         ///< dimensions.
  std::size_t max_directory_entries =
      2000000;  ///< Aggregate filesystem scan bound.
  std::size_t max_directory_depth =
      32;  ///< Maximum recursive split traversal depth.
};

/** @brief Owns one canonical split and its resolved image root; no new target
 * schema. */
struct DetectionDatasetImport {
  std::vector<std::size_t> flip_indices;  ///< YOLO pose left/right permutation.
  std::filesystem::path
      image_root;  ///< Absolute local root for canonical relative paths.
  data::image::AnnotationIndex
      annotations;  ///< Shared downstream representation for every layout.
  std::string
      metadata_identity;  ///< SHA-256 of import configuration and annotation
                          ///< inventory; excludes image contents.
};

/** @brief Imports one explicit local split without rewriting the user's files.
 * @param options Layout, paths, frozen categories and finite resource limits.
 * @param stop Cancellation checked between entries, files and parser stages.
 * @return Canonical annotations ready for ImageDataset(root, annotations).
 * @throws InvalidArgument Invalid configuration, split or category mapping.
 * @throws IoError A required local file or directory cannot be read.
 * @throws SerializationError Malformed data, duplicate paths or class mismatch.
 * @throws ResourceExhausted Aggregate import or decoder limits are exceeded.
 * @throws Cancelled Cancellation was requested.
 * @note YOLO accepts directory and UTF-8 image-list splits, with the last
 * `images` path component mapped to `labels`. Image-list entries resolve
 * against the list's directory and must remain within image_root. VOC reads XML
 * files with a frozen class map. No network, downloads, conversion files or
 * global RNG.
 */
DetectionDatasetImport read_detection_dataset(
    const DetectionDatasetOptions& options, std::stop_token stop = {});

/** @brief Rejects train/validation images sharing a resolved canonical path.
 * @param training Imported training split with local image paths.
 * @param validation Imported validation split with local image paths.
 * @param stop Cancellation checked between path resolutions.
 * @throws InvalidArgument If the two selections contain a shared image path.
 * @throws IoError If a listed image cannot be resolved.
 * @note Compares paths, including link resolution and Windows ASCII casing;
 * does not compare image contents or identify distinct hard links.
 */
void require_disjoint_detection_splits(const DetectionDatasetImport& training,
                                       const DetectionDatasetImport& validation,
                                       std::stop_token stop = {});

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_DATASET_HPP
