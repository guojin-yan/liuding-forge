// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_DATASET_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_DATASET_HPP
/**
 * @file image_dataset.hpp
 * @brief Connects built-in image indexes and decoders to the map dataset API.
 *
 * Indexes are retained as immutable metadata and images are decoded on demand.
 * The source can therefore be read by the existing DataPipeline worker pool
 * without eagerly materializing the complete dataset in host memory.
 */

#include <jyppx/liuding/data/dataset.hpp>
#include <jyppx/liuding/data_image/annotations.hpp>
#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/data_image/manifests.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief One lazily decoded image and its owned or shared target metadata. */
struct VisionSample {
  Tensor image;  ///< Decoded owning input tensor.
  std::optional<Tensor>
      paired_target;                ///< Decoded target/mask when manifested.
  std::vector<std::size_t> labels;  ///< Contiguous classification IDs.
  std::string image_path;  ///< Dataset-relative UTF-8 path from the index.
  std::optional<std::string>
      target_path;  ///< Dataset-relative paired target path.
  std::shared_ptr<const std::vector<VisionCategory>> label_categories;
  ///< Immutable class map for classification labels.
  std::shared_ptr<const std::vector<VisionCategory>> categories;
  ///< Immutable object-annotation class map when annotations are attached.
  std::shared_ptr<const VisionAnnotationRecord> annotation;
  ///< Shared immutable object/mask/keypoint annotations for this image.
  std::uint32_t width = 0;   ///< Decoded input width in pixels.
  std::uint32_t height = 0;  ///< Decoded input height in pixels.
};

/** @brief Decode policies and alignment requirements for a dataset source. */
struct ImageDatasetOptions {
  ImageDecodeOptions
      image;  ///< Input image output, pinning and resource limits.
  ImageDecodeOptions paired_target;  ///< Target/mask decode policy.
  bool require_matching_target_size = true;
  ///< Reject a paired target whose width or height differs from the input.
};

/**
 * @brief Map-style source backed by a manifest/ImageFolder and built-in
 * decoder.
 *
 * The immutable index can be read concurrently. Files are reopened per sample;
 * construction does not decode image contents. Every indexed path is checked
 * again before use, and symlink/reparse-point components are rejected. The
 * caller must keep the dataset tree stable while reads are in progress; this
 * check is not an OS-level sandbox against concurrent filesystem replacement.
 */
class ImageDataset final : public Dataset<VisionSample> {
 public:
  /**
   * @brief Creates an image source from a parsed manifest or ImageFolder index.
   * @param root Existing dataset root containing all indexed files.
   * @param index Immutable classification or paired-file index.
   * @param options Decode policies for input and optional paired targets.
   * @throws InvalidArgument Empty root or an unsupported manifest task.
   * @throws IoError The root is inaccessible or is not a regular directory.
   */
  ImageDataset(std::filesystem::path root, ManifestIndex index,
               ImageDatasetOptions options = {});

  /**
   * @brief Creates a source with an object-annotation index joined by image
   * path.
   * @param root Existing dataset root containing all indexed image files.
   * @param index Manifest or ImageFolder index defining sample order.
   * @param annotations Built-in annotation reader result joined by exact path.
   * @param options Decode policies for input and optional paired targets.
   * @throws InvalidArgument Invalid indexes or task combinations.
   * @throws IoError The root is inaccessible or is not a regular directory.
   * @throws SerializationError Annotation paths do not map uniquely to samples.
   */
  ImageDataset(std::filesystem::path root, ManifestIndex index,
               AnnotationIndex annotations, ImageDatasetOptions options = {});

  /** @brief Returns one decoded image sample.
   * @param index Zero-based index less than size().
   * @return Owned pixels and immutable indexed labels/annotations.
   */
  VisionSample get(std::size_t index) const override;

  /**
   * @brief Returns one sample with cooperative cancellation for worker use.
   * @param index Zero-based index less than size().
   * @param stop Cancellation request checked before file access and decode.
   * @return Decoded image and indexed labels/annotations.
   * @throws InvalidArgument Index is outside this dataset.
   * @throws IoError File is missing, changes during read, or traverses a link.
   * @throws SerializationError Image/target contents or dimensions are invalid.
   * @throws ResourceExhausted Configured decoder limits are exceeded.
   * @throws DeviceError Pinned allocation or LibTorch operation failed.
   * @throws Cancelled Cancellation was requested before publication.
   */
  VisionSample get(std::size_t index, std::stop_token stop) const;

  /** @brief Returns the immutable number of indexed samples.
   * @return Number of manifest records, without decoding any image.
   */
  std::size_t size() const override;

  /** @brief Returns the immutable source index for provenance and inspection.
   * @return Index retained for this dataset's lifetime.
   */
  const ManifestIndex& index() const noexcept;

 private:
  /** @brief Adopts validated source metadata shared by public constructors.
   * @param root Existing local dataset root.
   * @param index Immutable source order and class mapping.
   * @param annotations Optional shared object-annotation metadata.
   * @param options Input and paired-target decode policies.
   */
  ImageDataset(std::filesystem::path root, ManifestIndex index,
               std::shared_ptr<const AnnotationIndex> annotations,
               ImageDatasetOptions options);

  std::filesystem::path root_;  ///< Canonical validated local source root.
  ManifestIndex index_;         ///< Immutable sample order and source metadata.
  ImageDatasetOptions options_;  ///< Per-sample decode and alignment policies.
  std::shared_ptr<const AnnotationIndex>
      annotations_;  ///< Shared object index.
  std::shared_ptr<const std::vector<VisionCategory>> label_categories_;
  ///< Shared classification class map retained by returned samples.
  std::shared_ptr<const std::vector<VisionCategory>> categories_;
  ///< Aliased object class map owned by annotations_.
  std::vector<std::shared_ptr<const VisionAnnotationRecord>>
      sample_annotations_;
  ///< Optional object records joined to the exact manifest order.
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_DATASET_HPP