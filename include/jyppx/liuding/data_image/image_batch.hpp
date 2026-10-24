// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_HPP
/** @file image_batch.hpp
 * @brief Bounded image collation and structured pipeline values.
 */

#include <jyppx/liuding/data/batch_memory.hpp>
#include <jyppx/liuding/data/batch_schema.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>

#include <span>

namespace jyppx::liuding::data::image {

/** @brief Explicit image shape policy; no implicit resizing or bucketing. */
enum class ImageBatchLayout : std::uint8_t { stack, pad, ragged };

/** @brief Stable sample identity; duplicates can use distinct occurrences. */
struct ImageBatchKey {
  std::string id;
  std::uint64_t occurrence = 0;
  std::uint64_t view = 0;
};

/** @brief Discrete source metadata without retaining individual mask storage.
 */
struct ImageBatchMaskInfo {
  MaskEncoding encoding = MaskEncoding::class_index;
  std::uint8_t source_bit_depth = 0;
  bool source_interlaced = false;
  std::vector<MaskPaletteEntry> palette;
  std::optional<std::int64_t> ignore_id;
  std::optional<std::int64_t> void_id;
};

/** @brief Host metadata retained through pinning and device transfer. */
struct ImageBatchItem {
  ImageBatchKey key;
  ImageExtent valid_extent;  ///< Top-left extent before collation padding.
  std::string image_path;
  std::optional<std::string> target_path;
  std::shared_ptr<const std::vector<VisionCategory>> label_categories;
  std::shared_ptr<const std::vector<VisionCategory>> categories;
  std::shared_ptr<const VisionAnnotationRecord> annotation;
  std::optional<GeometryTrace> geometry;
  std::optional<ImageBatchMaskInfo> mask;
};

/** @brief Per-call tensor payload and host metadata limits, excluding input. */
struct ImageBatchLimits {
  std::size_t max_samples = 4096;
  std::uint32_t max_dimension = 65'536;
  std::uint64_t max_pixels = 268'435'456;  ///< Sum of output image pixels.
  std::uint64_t max_output_bytes = 1ULL << 30U;
  std::uint64_t max_working_bytes = 2ULL << 30U;
  std::size_t max_metadata_nodes = 65'536;
  std::size_t max_metadata_bytes = 64U << 20U;
  std::size_t max_string_bytes = 16'384;
  std::size_t max_geometry_checks = 20'000'000;
};

/** @brief CPU final-batch pack policy; memory staging is a separate call. */
struct ImageCollateOptions {
  ImageBatchLayout layout = ImageBatchLayout::stack;
  std::optional<ImageExtent> padded_extent;  ///< pad only; otherwise max H/W.
  std::uint32_t size_divisor = 1;  ///< pad only; round canvas sides upward.
  double image_padding = 0;
  double paired_padding = 0;
  std::optional<std::int64_t> mask_padding_id;  ///< Explicit whenever padded.
  ImageBatchLimits limits;
};

/**
 * @brief Owning packed NCHW or ragged CHW images plus complete host targets.
 *
 * PixelSpec describes each CHW image; the outer batch axes are determined by
 * layout. Exactly one of images/image_list is used. The same rule applies to
 * paired/mask/valid fields. valid marks collation padding only, independently
 * of real-content bounds in GeometryTrace. labels is packed int64, with host
 * label_offsets of length N+1. Annotation coordinates use each output canvas.
 */
struct ImageBatch {
  ImageBatchLayout layout = ImageBatchLayout::stack;
  PixelSpec pixels;
  std::optional<PixelSpec> paired_pixels;
  Tensor images;
  std::vector<Tensor> image_list;
  std::optional<Tensor> paired_images;
  std::vector<Tensor> paired_image_list;
  std::optional<Tensor> masks;
  std::vector<Tensor> mask_list;
  Tensor valid;
  std::vector<Tensor> valid_list;
  Tensor labels;
  std::vector<std::int64_t> label_offsets;
  std::vector<ImageBatchItem> items;
  data::BatchMemoryMetadata memory;
};

/**
 * @brief Packs CPU samples into final stack, padded or ragged owning tensors.
 * @param samples Nonempty aligned photos/targets without autograd.
 * @param keys One explicit stable identity per sample, in the same order.
 * @param options Shape, fill and independent resource policies.
 * @param stop Cooperative cancellation before publication.
 * @return Pageable CPU batch; input storage and annotations are unchanged.
 * @throws InvalidArgument Invalid pixels, shape policy, targets or identity.
 * @throws ResourceExhausted Tensor or host metadata budget exceeded.
 * @throws SerializationError Panoptic source table and pixels disagree.
 * @throws DeviceError LibTorch backend failure.
 * @throws Cancelled Cancellation requested before publication.
 *
 * Mixed CHW/HWC input is accepted with matching nonlayout pixel semantics.
 * Target presence is uniform: all paired photos, all masks, or neither.
 * Annotations may be missing or valid empty records. pad adds right/bottom
 * pixels only, composes traces and losslessly extends RLE canvas runs.
 */
ImageBatch collate_images(std::span<const VisionSample> samples,
                          std::span<const ImageBatchKey> keys,
                          const ImageCollateOptions& options = {},
                          std::stop_token stop = {});

/** @brief Encodes all images and host metadata as a versioned IValue tree.
 * @note Host annotations use scalar/list/tuple values, so memory staging
 * transfers only actual tensor leaves. Unsigned counters retain all 64 bits.
 */
c10::IValue image_batch_value(const ImageBatch& batch,
                              const ImageBatchLimits& limits = {});

/** @brief Decodes a structured batch without tensor copies or host sync.
 * @note Validates shape and metadata, not pixel values. Input tensor contents
 * must come from a trusted producer; source panoptic binding occurs on CPU.
 */
ImageBatch image_batch_from_value(const c10::IValue& value,
                                  const ImageBatchLimits& limits = {});

/** @brief Creates the full versioned structural BatchSchema for this batch.
 * @param batch Exemplar supplying layout, pixel and optional-target contracts.
 * @param device Consumer tensor device, e.g. pipeline memory target.
 * @return Dynamic N/H/W with typed image/mask/label tensors and host metadata.
 */
data::BatchSchema image_batch_schema(const ImageBatch& batch, Device device);

/** @brief Pins the final packed tensors on a producer; never transfers.
 * @param batch Valid owning CPU batch.
 * @param options Existing data memory-stage policy and bounds.
 * @return Batch with pinned tensor leaves and observable memory metadata.
 */
ImageBatch prepare_image_batch(const ImageBatch& batch,
                               const data::BatchMemoryOptions& options,
                               const ImageBatchLimits& limits = {});

/** @brief Transfers a prepared batch on the calling consumer's current stream.
 * @note Uses existing two-phase data memory contract, without hidden streams.
 */
ImageBatch transfer_image_batch(ImageBatch prepared,
                                const data::BatchMemoryOptions& options,
                                const ImageBatchLimits& limits = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_HPP
