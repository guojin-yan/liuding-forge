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
#include <jyppx/liuding/data/batch_transform.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>

#include <span>
#include <utility>

namespace jyppx::liuding::data::image {

/** @brief Explicit image shape policy; no implicit resizing or bucketing. */
enum class ImageBatchLayout : std::uint8_t {
  stack,  ///< Equal-size CHW images packed into one NCHW tensor.
  pad,    ///< Right/bottom padding to a common NCHW canvas with validity masks.
  ragged  ///< Independently sized CHW tensors retained in source order.
};

/** @brief Stable sample identity; duplicates can use distinct occurrences. */
struct ImageBatchKey {
  std::string
      id;  ///< Stable nonempty dataset-local identity, at most 4096 bytes.
  std::uint64_t occurrence = 0;  ///< Repetition of this sample in its epoch.
  std::uint64_t view = 0;        ///< Independent augmentation-view coordinate.
};

/** @brief Discrete source metadata without retaining individual mask storage.
 */
struct ImageBatchMaskInfo {
  MaskEncoding encoding = MaskEncoding::class_index;  ///< Source ID encoding.
  std::uint8_t source_bit_depth = 0;  ///< Original file's declared bit depth.
  bool source_interlaced = false;     ///< Original PNG interlace flag.
  std::vector<MaskPaletteEntry> palette;  ///< Source palette for palette_index.
  std::optional<std::int64_t>
      ignore_id;  ///< Explicit ignored label, if defined.
  std::optional<std::int64_t>
      void_id;  ///< Explicit panoptic void ID, if defined.
};

/** @brief Host metadata retained through pinning and device transfer. */
struct ImageBatchItem {
  ImageBatchKey key;  ///< Stable identity used for sampling and provenance.
  ImageExtent valid_extent;  ///< Top-left extent before collation padding.
  std::string image_path;  ///< Dataset-relative image path or Tensor source ID.
  std::optional<std::string> target_path;  ///< Paired target path or source ID.
  std::shared_ptr<const std::vector<VisionCategory>> label_categories;
  ///< Shared immutable classification category map.
  std::shared_ptr<const std::vector<VisionCategory>> categories;
  ///< Shared immutable object-annotation category map.
  std::shared_ptr<const VisionAnnotationRecord> annotation;
  ///< Shared immutable annotation in the current output canvas.
  std::optional<GeometryTrace>
      geometry;  ///< Source-to-canvas coordinate trace.
  std::optional<ImageBatchMaskInfo>
      mask;  ///< Discrete mask's source semantics.
};

/** @brief Per-call tensor payload and host metadata limits, excluding input. */
struct ImageBatchLimits {
  std::size_t max_samples = 4096;        ///< Positive batch-item count cap.
  std::uint32_t max_dimension = 65'536;  ///< Positive cap for any canvas side.
  std::uint64_t max_pixels = 268'435'456;  ///< Sum of output image pixels.
  std::uint64_t max_output_bytes = 1ULL << 30U;  ///< All output tensor payload.
  std::uint64_t max_working_bytes = 2ULL << 30U;  ///< Conservative working cap.
  std::size_t max_metadata_nodes = 65'536;  ///< Recursive host-metadata nodes.
  std::size_t max_metadata_bytes = 64U
                                   << 20U;  ///< Logical host metadata bytes.
  std::size_t max_string_bytes = 16'384;    ///< Bytes in any metadata string.
  std::size_t max_geometry_checks = 20'000'000;  ///< Polygon validation budget.
};

/** @brief CPU final-batch pack policy; memory staging is a separate call. */
struct ImageCollateOptions {
  ImageBatchLayout layout = ImageBatchLayout::stack;  ///< Output shape policy.
  std::optional<ImageExtent> padded_extent;  ///< pad only; otherwise max H/W.
  std::uint32_t size_divisor = 1;  ///< pad only; round canvas sides upward.
  double image_padding =
      0;  ///< Main-photo fill in its declared value encoding.
  double paired_padding = 0;  ///< Independent paired-photo fill value.
  std::optional<std::int64_t> mask_padding_id;  ///< Explicit whenever padded.
  ImageBatchLimits limits;  ///< Positive per-call tensor and host limits.
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Optional aggregate accounting retained with the output batch.
};

/**
 * @brief Owning packed NCHW or ragged CHW images plus complete host targets.
 *
 * PixelSpec describes each CHW image; the outer batch axes are determined by
 * layout. Exactly one of images/image_list is used. The same rule applies to
 * paired/mask/valid fields. valid marks collation padding only, independently
 * of real-content bounds in GeometryTrace. labels is packed int64, with host
 * label_offsets of length N+1. Annotation coordinates use each output canvas.
 * Copying shares Tensor storage, immutable annotation owners and lease sets;
 * it does not clone pixels. Callers synchronize mutation of any shared Tensor
 * storage. A default-constructed value must be populated and validated before
 * use as a batch.
 */
struct ImageBatch {
  /** @brief Creates an empty value, not yet a structurally valid batch. */
  ImageBatch() = default;
  /** @brief Copies metadata and shares Tensor storage and resource leases. */
  ImageBatch(const ImageBatch&) = default;
  /** @brief Moves metadata, Tensor handles and lease ownership. */
  ImageBatch(ImageBatch&&) = default;
  /** @brief Swaps owners before the replaced batch releases its storage.
   * @param a First batch value.
   * @param b Second batch value; may refer to the same object.
   */
  friend void swap(ImageBatch& a, ImageBatch& b) {
    using std::swap;
    swap(a.resource_lease_sets, b.resource_lease_sets);
    swap(a.layout, b.layout);
    swap(a.pixels, b.pixels);
    swap(a.paired_pixels, b.paired_pixels);
    swap(a.images, b.images);
    swap(a.image_list, b.image_list);
    swap(a.paired_images, b.paired_images);
    swap(a.paired_image_list, b.paired_image_list);
    swap(a.masks, b.masks);
    swap(a.mask_list, b.mask_list);
    swap(a.valid, b.valid);
    swap(a.valid_list, b.valid_list);
    swap(a.labels, b.labels);
    swap(a.label_offsets, b.label_offsets);
    swap(a.items, b.items);
    swap(a.memory, b.memory);
  }

  /** @brief Replaces this value while keeping old leases alive through swap.
   * @param other Copied or moved replacement batch value.
   * @return This batch after replacement.
   */
  ImageBatch& operator=(ImageBatch other) {
    swap(*this, other);
    return *this;
  }

  /**
   * @brief Aggregate reservations outlive every tensor and host field below.
   *
   * Copies share each lease set. Runtime leases are not serialized and are
   * recreated only by a stage receiving a resource budget.
   */
  std::vector<std::shared_ptr<ImageResourceLeaseSet>> resource_lease_sets;
  ImageBatchLayout layout = ImageBatchLayout::stack;  ///< Packed/ragged policy.
  PixelSpec pixels;  ///< Per-image CHW semantics of the main photo field.
  std::optional<PixelSpec> paired_pixels;  ///< Optional paired-photo semantics.
  Tensor images;  ///< Main NCHW photos for stack/pad; undefined for ragged.
  std::vector<Tensor> image_list;  ///< Main CHW photos only for ragged layout.
  std::optional<Tensor> paired_images;  ///< Optional packed NCHW paired photos.
  std::vector<Tensor>
      paired_image_list;        ///< Optional ragged CHW paired photos.
  std::optional<Tensor> masks;  ///< Optional packed int64 [N,H,W] discrete IDs.
  std::vector<Tensor> mask_list;  ///< Optional ragged int64 [H,W] discrete IDs.
  Tensor valid;  ///< Packed bool [N,H,W]; true inside each valid_extent.
  std::vector<Tensor> valid_list;  ///< Ragged bool [H,W] validity fields.
  Tensor labels;  ///< Flattened int64 category IDs on the batch device.
  std::vector<std::int64_t> label_offsets;  ///< N+1 host offsets into labels.
  std::vector<ImageBatchItem> items;  ///< Host metadata in Tensor sample order.
  data::BatchMemoryMetadata memory;   ///< Observable two-phase staging state.
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
 * @param batch Structurally valid batch with trusted Tensor contents.
 * @param limits Positive structural, Tensor-payload and host-metadata caps.
 * @return Versioned value sharing Tensor storage and copying host metadata;
 * runtime leases and memory-stage state are not serialized.
 * @throws InvalidArgument Invalid batch structure or metadata.
 * @throws ResourceExhausted Tensor or metadata bounds are exceeded.
 * @note Host annotations use scalar/list/tuple values, so memory staging
 * transfers only actual tensor leaves. Unsigned counters retain all 64 bits.
 */
c10::IValue image_batch_value(const ImageBatch& batch,
                              const ImageBatchLimits& limits = {});

/** @brief Decodes a structured batch without tensor copies or host sync.
 * @param value Versioned ImageBatch tree from a trusted pixel producer.
 * @param limits Positive structural, Tensor-payload and host-metadata caps.
 * @return Batch sharing Tensor storage, with reconstructed host metadata and
 * no runtime leases or memory-stage state.
 * @throws InvalidArgument Invalid schema version, structure or metadata.
 * @throws ResourceExhausted Tensor or metadata bounds are exceeded.
 * @note Validates shape and metadata, not pixel values. Input tensor contents
 * must come from a trusted producer; source panoptic binding occurs on CPU.
 */
ImageBatch image_batch_from_value(const c10::IValue& value,
                                  const ImageBatchLimits& limits = {});

/**
 * @brief Reserves image resources for an encoded ImageBatch value.
 *
 * The returned opaque owner is suitable for
 * `data::BatchResourceLeaseFactory`; it keeps all aggregate leases alive until
 * the pipeline result is released. The runtime resource state is never
 * serialized into the ImageBatch value.
 * @param value Versioned ImageBatch value to validate and account for.
 * @param budget Shared aggregate budget; absent returns an empty owner.
 * @param limits Positive structural, Tensor-payload and host-metadata caps.
 * @param stop Cancellation checked before aggregate resource reservation.
 * @return Opaque shared lease collection; pixel storage is not copied.
 * @throws InvalidArgument Invalid batch structure or metadata.
 * @throws ResourceExhausted Per-call limits or aggregate capacity exceeded.
 * @throws Cancelled Cancellation requested before reservation completes.
 */
std::shared_ptr<void> image_batch_resource_lease(
    const c10::IValue& value, std::shared_ptr<ImageResourceBudget> budget,
    const ImageBatchLimits& limits = {}, std::stop_token stop = {});

/** @brief Creates a pipeline callback for ImageBatch aggregate accounting.
 * @param budget Shared budget retained by the callback; absent disables it.
 * @param limits Copy of the per-batch validation and payload limits.
 * @return Callback that validates a value and retains aggregate leases.
 * @note Invocation propagates image_batch_resource_lease() failures.
 */
data::BatchResourceLeaseFactory make_image_batch_resource_lease_factory(
    std::shared_ptr<ImageResourceBudget> budget, ImageBatchLimits limits = {});

/** @brief Waits for a bounded in-flight slot before a pipeline builds a batch.
 * @param budget Shared budget retained by the callback; absent disables it.
 * @return Admission callback reserving one pipeline_inflight_batches slot.
 * @note Invocation may block until capacity is released, fail immediately
 * when the category limit is zero, or throw Cancelled on its stop token.
 */
data::BatchAdmissionFactory make_image_batch_admission_factory(
    std::shared_ptr<ImageResourceBudget> budget);

/**
 * @brief Admits a batch against both count and its validated tensor-byte cap.
 * @note Pair with the resource lease factory using the same limits so an
 * output above `max_output_bytes` is rejected before delivery.
 * @param budget Shared budget retained by the callback; absent disables it.
 * @param limits Positive max_output_bytes reserved for every admitted batch.
 * @return Callback atomically reserving one in-flight slot and the byte cap.
 * @throws InvalidArgument max_output_bytes is zero.
 * @note Invocation propagates reserve_wait_all() errors or cancellation.
 */
data::BatchAdmissionFactory make_image_batch_byte_admission_factory(
    std::shared_ptr<ImageResourceBudget> budget, ImageBatchLimits limits);

/** @brief Creates the full versioned structural BatchSchema for this batch.
 * @param batch Exemplar supplying layout, pixel and optional-target contracts.
 * @param device Consumer tensor device, e.g. pipeline memory target.
 * @return Dynamic N/H/W with typed image/mask/label tensors and host metadata.
 */
data::BatchSchema image_batch_schema(const ImageBatch& batch, Device device);

/** @brief Pins the final packed tensors on a producer; never transfers.
 * @param batch Valid owning CPU batch.
 * @param options Existing data memory-stage policy and bounds.
 * @param limits Positive per-batch structural and logical payload caps.
 * @param resource_budget Optional shared accounting retained with the result.
 * @return Batch with pinned tensor leaves and observable memory metadata.
 * @throws InvalidArgument Invalid batch or memory-stage settings.
 * @throws ResourceExhausted Per-call limits or aggregate capacity exceeded.
 * @throws DeviceError LibTorch cannot pin requested CPU tensors.
 */
ImageBatch prepare_image_batch(
    const ImageBatch& batch, const data::BatchMemoryOptions& options,
    const ImageBatchLimits& limits = {},
    std::shared_ptr<ImageResourceBudget> resource_budget = {});

/** @brief Transfers a prepared batch on the calling consumer's current stream.
 * @param prepared Batch returned by prepare_image_batch(), with memory state.
 * @param options Same immutable memory-stage policy used for preparation.
 * @param limits Positive per-batch structural and logical payload caps.
 * @param resource_budget Optional shared accounting retained with the result.
 * @return Consumer-ready batch with finalized transfer metadata.
 * @throws CompatibilityError Preparation metadata and options differ.
 * @throws InvalidArgument Invalid prepared batch or memory-stage settings.
 * @throws ResourceExhausted Per-call limits or aggregate capacity exceeded.
 * @throws DeviceError LibTorch cannot perform the requested transfer.
 * @note Uses existing two-phase data memory contract, without hidden streams.
 */
ImageBatch transfer_image_batch(
    ImageBatch prepared, const data::BatchMemoryOptions& options,
    const ImageBatchLimits& limits = {},
    std::shared_ptr<ImageResourceBudget> resource_budget = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_HPP
