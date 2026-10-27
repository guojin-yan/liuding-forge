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
#include <jyppx/liuding/data_image/geometry_trace.hpp>
#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/data_image/manifests.hpp>
#include <jyppx/liuding/data_image/mask_decoder.hpp>
#include <jyppx/liuding/data_image/panoptic.hpp>

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
  std::shared_ptr<ImageResourceLeaseSet> image_resource_leases;
  ///< Keeps the decoded input charged while this sample retains it.
  Tensor image;  ///< Owning decoded pixels or a shared source Tensor handle.
  std::shared_ptr<ImageResourceLeaseSet> paired_target_resource_leases;
  ///< Keeps a decoded paired image charged while this sample retains it.
  std::optional<Tensor>
      paired_target;  ///< Decoded paired image when manifested.
  std::optional<DecodedMask>
      mask_target;  ///< Explicit discrete labels, never photo pixels.
  std::vector<std::size_t> labels;  ///< Contiguous classification IDs.
  std::string image_path;  ///< Dataset-relative path or Tensor source ID.
  std::optional<std::string>
      target_path;  ///< Dataset-relative paired target path.
  std::shared_ptr<const std::vector<VisionCategory>> label_categories;
  ///< Immutable class map for classification labels.
  std::shared_ptr<const std::vector<VisionCategory>> categories;
  ///< Immutable object-annotation class map when annotations are attached.
  std::shared_ptr<const VisionAnnotationRecord> annotation;
  ///< Shared immutable object/mask/keypoint annotations for this image.
  std::uint32_t width = 0;   ///< Current sample canvas width in pixels.
  std::uint32_t height = 0;  ///< Current sample canvas height in pixels.
  PixelSpec
      pixels;  ///< Explicit image channel, dtype, layout and range semantics.
  std::optional<PixelSpec>
      paired_target_pixels;  ///< Separate paired-image semantics.
  std::optional<GeometryTrace>
      geometry;  ///< Decoder-to-current coordinates; identity on read.
};

/** @brief Decode policies and alignment requirements for a dataset source. */
struct ImageDatasetOptions {
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Shared default for all image and mask decoder options in this source.
  ImageDecodeOptions
      image;  ///< Input image output, pinning and resource limits.
  ImageDecodeOptions paired_target;  ///< Paired-image decode policy.
  std::optional<MaskDecodeOptions> mask_target;
  ///< When present, paired manifest targets use the lossless label reader
  ///< instead.
  std::optional<MaskDecodeOptions> panoptic_target;
  ///< Opt-in COCO panoptic references; requires panoptic_rgb and explicit void.
  PanopticBindOptions panoptic_validation;
  ///< Verify segment IDs, source area/box and resource budgets on every read.
  bool require_matching_target_size = true;
  ///< Reject mismatched paired images. Discrete masks always require an exact
  ///< size match.
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

/** @brief One already-decoded image and its explicit classification metadata.
 */
struct TensorImageRecord {
  Tensor image;      ///< CPU/CUDA storage retained by the dataset and returned
                     ///< samples.
  PixelSpec pixels;  ///< Layout, dtype, color and value range; never inferred.
  std::string id;    ///< Stable source ID, not a filesystem path.
  std::vector<std::size_t> labels;  ///< Indices into the dataset category map.
  std::optional<Tensor> paired_target;  ///< Same-device paired image storage.
  std::optional<PixelSpec> paired_target_pixels;  ///< Independent semantics.
  std::optional<DecodedMask> mask_target;  ///< Shared CPU int64 class IDs.
  std::optional<std::string> target_id;    ///< Required for either target kind.
  std::optional<GeometryTrace> geometry;   ///< Source-to-current mapping when
                                           ///< pixels were pretransformed.
};

/** @brief Limits checked against each caller-owned Tensor before publication.
 */
struct TensorImageDatasetLimits {
  std::size_t max_records = 1'000'000;
  std::uint32_t max_dimension = 65'536;
  std::uint64_t max_pixels = 268'435'456;
  std::uint64_t max_sample_bytes = 1ULL << 30U;  ///< All logical pixel fields.
  std::size_t max_id_bytes = 4096;
  std::size_t max_labels_per_sample = 4096;
};

/**
 * @brief Map-style source for already-decoded CPU or CUDA image Tensors.
 *
 * The dataset retains Tensor handles and returns shared handles without
 * decoding, copying or transferring pixels. The caller must not mutate the
 * backing storage while the dataset or returned samples are in use. A record
 * may carry one same-device paired image or one CPU int64 class-index mask;
 * both require an explicit target ID and exact spatial alignment. Views may
 * retain storage larger than their logical payload. Existing source storage is
 * caller-owned and is not charged to ImageResourceBudget; downstream transforms
 * and collation account for their own new allocations. An optional geometry
 * trace must end at the supplied Tensor canvas; otherwise get() starts an
 * identity trace there.
 */
class TensorImageDataset final : public Dataset<VisionSample> {
 public:
  /**
   * @param records Finite source order; Tensor handles remain valid after the
   * caller releases its handles.
   * @param categories Classification map required for non-empty labels.
   * @param limits Per-source and per-image logical limits.
   * @throws InvalidArgument Invalid limits, pixels, IDs or label indices.
   * @throws ResourceExhausted A source or image exceeds configured limits.
   */
  explicit TensorImageDataset(std::vector<TensorImageRecord> records,
                              std::vector<VisionCategory> categories = {},
                              TensorImageDatasetLimits limits = {});

  /** @brief Returns a shared Tensor handle and copied bounded metadata. */
  VisionSample get(std::size_t index) const override;
  /** @brief Returns one sample with cooperative cancellation. */
  VisionSample get(std::size_t index, std::stop_token stop) const;
  /** @brief Returns the immutable number of source records. */
  std::size_t size() const override;

 private:
  std::vector<TensorImageRecord> records_;
  std::shared_ptr<const std::vector<VisionCategory>> categories_;
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_DATASET_HPP
