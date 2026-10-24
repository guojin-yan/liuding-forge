// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
#define JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
/** @file batch_geometry.hpp
 * @brief Per-sample CPU geometry for structured image batches.
 */

#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>

#include <string_view>

namespace jyppx::liuding::data::image {

/** @brief Joint CPU geometry policy and aggregate batch resource limits. */
struct ImageBatchGeometryOptions {
  JointTransformOptions joint;
  ImageBatchLimits limits;
  std::optional<ImageExtent> output_padded_extent;
  std::uint32_t output_size_divisor = 1;
};

/**
 * @brief Samples geometry independently from each stable batch sample key.
 * @param batch Batch metadata supplies sample ID, occurrence, view and extent.
 * @param dataset_id Frozen source/configuration identity.
 * @param random Fixed resize, optional random crop and flip probabilities.
 * @param joint Pixel, annotation and per-sample resource policies.
 * @param limits Aggregate batch resource limits included in the identity.
 * @param seed Experiment seed; epoch is an explicit logical coordinate.
 * @return One deterministic step sequence per item, independent of worker
 * count and batch ordering.
 */
std::vector<std::vector<ImageGeometryStep>> sample_image_batch_geometry(
    const ImageBatch& batch, std::string_view dataset_id,
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options = {}, std::uint64_t seed = 0,
    std::uint64_t epoch = 0);

/** @brief SHA-256 identity of random, joint and aggregate geometry policies. */
std::string image_batch_geometry_identity(
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options = {});

/**
 * @brief Applies per-sample geometry while preserving stack/pad/ragged layout.
 * @param batch Structurally valid CPU batch from a trusted producer.
 * @param sample_steps One explicit sequence per batch item.
 * @param options Joint policy and aggregate per-call limits.
 * @param stop Cooperative cancellation before publication.
 * @return New pageable CPU batch; input tensors and metadata remain unchanged.
 * @throws InvalidArgument Invalid geometry, schema, or non-CPU batch.
 * @throws ResourceExhausted Per-sample or aggregate tensor/metadata budget.
 * @throws DeviceError LibTorch operation failed.
 * @throws Cancelled Cancellation requested before publication.
 *
 * Right/bottom collation padding is removed before each sample's sequence is
 * applied, then the configured layout is rebuilt. The valid tensor must match
 * that rectangular padding contract. Discrete masks are transformed as int64
 * IDs with nearest-exact sampling. CUDA tensors are rejected explicitly; no
 * hidden device transfer or synchronization is performed. For pad layout, an
 * output extent and size divisor may be specified; their resolved canvas is
 * part of the transform identity.
 */
ImageBatch transform_image_batch_geometry(
    const ImageBatch& batch,
    std::span<const std::vector<ImageGeometryStep>> sample_steps,
    const ImageBatchGeometryOptions& options = {}, std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_BATCH_GEOMETRY_HPP
