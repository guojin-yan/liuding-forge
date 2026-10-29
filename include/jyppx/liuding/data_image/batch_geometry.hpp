// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
#define JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
/** @file batch_geometry.hpp
 * @brief Per-sample geometry for structured image batches.
 */

#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>

#include <string_view>

namespace jyppx::liuding::data::image {

/** @brief Joint geometry policy and aggregate batch resource limits. */
struct ImageBatchGeometryOptions {
  JointTransformOptions joint;  ///< Per-sample pixel and annotation policy.
  ImageBatchLimits limits;      ///< Aggregate output and metadata limits.
  std::optional<ImageExtent> output_padded_extent;
  ///< Explicit pad-layout canvas; absent uses the largest transformed sample.
  std::uint32_t output_size_divisor = 1;  ///< Positive pad-canvas side divisor.
  /// Authorizes one bounded CUDA-to-host segment summary per panoptic sample.
  bool allow_cuda_panoptic_summary_transfer = false;
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Optional shared accounting for new output and working allocations.
};

/**
 * @brief Samples geometry independently from each stable batch sample key.
 * @param batch Batch metadata supplies sample ID, occurrence, view and extent.
 * @param dataset_id Frozen source/configuration identity.
 * @param random Fixed resize, optional random crop and flip probabilities.
 * @param options Joint, aggregate and output-canvas policies bound to identity.
 * @param seed Experiment seed, independent of global LibTorch RNG.
 * @param epoch Explicit logical epoch selected by the coordinator.
 * @return One deterministic step sequence per item, independent of worker
 * count and batch ordering.
 * @throws InvalidArgument Invalid batch, sample identity or geometry policy.
 * @throws ResourceExhausted Batch validation or bounded sampling exceeds
 * limits.
 */
std::vector<std::vector<ImageGeometryStep>> sample_image_batch_geometry(
    const ImageBatch& batch, std::string_view dataset_id,
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options = {}, std::uint64_t seed = 0,
    std::uint64_t epoch = 0);

/** @brief Hashes random, joint and aggregate geometry policies.
 * @param random Frozen sampler configuration.
 * @param options Joint limits, output canvas and explicit transfer policy.
 * @return Versioned lowercase SHA-256; excludes the runtime budget instance.
 * @throws InvalidArgument Invalid sampler, joint or output policy.
 */
std::string image_batch_geometry_identity(
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options = {});

/**
 * @brief Applies per-sample geometry while preserving stack/pad/ragged layout.
 * @param batch Structurally valid CPU or CUDA batch from a trusted producer.
 * @param sample_steps One explicit sequence per batch item.
 * @param options Joint policy and aggregate per-call limits.
 * @param stop Cooperative cancellation before publication.
 * @return New batch on the input device; input tensors and metadata remain
 * unchanged.
 * @throws InvalidArgument Invalid geometry, schema, or implicit CUDA panoptic
 * transfer.
 * @throws ResourceExhausted Per-sample or aggregate tensor/metadata budget.
 * @throws DeviceError LibTorch operation failed.
 * @throws Cancelled Cancellation requested before publication.
 *
 * Right/bottom collation padding is removed before each sample's sequence is
 * applied, then the configured layout is rebuilt. The valid tensor must match
 * that rectangular padding contract. Discrete masks are transformed as int64
 * IDs with nearest-exact sampling. CPU and CUDA pixel leaves stay on their
 * existing device and caller-current stream. CUDA maps boxes, polygons, OBBs,
 * keypoints and RLE from host metadata without reading device pixels.
 * Panoptic CUDA mapping requires allow_cuda_panoptic_summary_transfer: it
 * computes bounded exact segment statistics on the calling stream, then
 * synchronizes one compact summary per sample to update host metadata. It
 * never transfers image or mask pixels. For pad layout, an output extent and
 * size divisor may be specified; their resolved canvas is part of the
 * transform identity.
 */
ImageBatch transform_image_batch_geometry(
    const ImageBatch& batch,
    std::span<const std::vector<ImageGeometryStep>> sample_steps,
    const ImageBatchGeometryOptions& options = {}, std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
