// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_BATCH_TRANSFORM_HPP
#define JYPPX_LIUDING_DATA_IMAGE_BATCH_TRANSFORM_HPP
/** @file batch_transform.hpp
 * @brief Device-preserving numeric transforms for structured image batches.
 */

#include <jyppx/liuding/data_image/image_batch.hpp>

namespace jyppx::liuding::data::image {

/** @brief Explicit intensity interpretation during floating conversion. */
enum class BatchPixelScale : std::uint8_t {
  preserve,  ///< Keep numeric values; unsigned encodings become unbounded.
  to_unit,   ///< Divide uint8/uint16 by 255/65535; accept existing unit floats.
};

/** @brief One photo-field transform, shared by all samples in the batch. */
struct BatchPixelTransform {
  Dtype dtype = torch::kFloat32;  ///< float16, bfloat16, float32 or float64.
  BatchPixelScale scale = BatchPixelScale::to_unit;
  bool swap_red_blue = false;  ///< RGB <-> BGR, preserving optional alpha.
  std::vector<double> mean;    ///< Empty disables normalization; otherwise C.
  std::vector<double> std;     ///< Same length as mean; finite and positive.
};

/** @brief Independent input/paired-photo policy and per-call limits. */
struct ImageBatchTransformOptions {
  BatchPixelTransform image;
  std::optional<BatchPixelTransform>
      paired;  ///< Absent preserves paired pixels.
  ImageBatchLimits limits;
  std::shared_ptr<ImageResourceBudget> resource_budget;
};

/**
 * @brief Converts/scales/normalizes an NCHW or ragged batch on CPU or CUDA.
 * @param batch Structurally valid batch produced by a trusted pixel producer.
 * @param options Explicit dtype, range, channel and normalization policy.
 * @param stop Cancellation checked between fields/ragged images and
 * publication.
 * @return Independent transformed photos; other tensors/host targets are shared
 * unchanged. Memory-stage metadata is cleared because dtype/bytes can change.
 * @throws InvalidArgument Invalid policies or ambiguous unbounded-to-unit
 * input.
 * @throws ResourceExhausted Output or conservative working payload exceeded.
 * @throws DeviceError Backend operation failed; there is no CPU fallback.
 * @throws Cancelled Cancellation requested before publication.
 *
 * Packed batches use whole NCHW operations, never a per-sample host loop.
 * Arithmetic uses float32, or float64 for float64 output, before the final
 * cast. Mean/std follow the output channel order. Alpha is preserved during
 * scaling and channel exchange; normalization of alpha-bearing photos is
 * rejected. Pixel values are trusted to follow PixelSpec; no pixel scan or
 * device-to-host synchronization occurs. Padding pixels receive the same
 * transform as photos. CUDA uses the caller's current stream without creating
 * streams or events. This API does not resize, transform geometry or install a
 * pipeline hook.
 */
ImageBatch transform_image_batch(const ImageBatch& batch,
                                 const ImageBatchTransformOptions& options = {},
                                 std::stop_token stop = {});

/** @brief SHA-256 of the versioned collate and optional numeric policies.
 * @param collate Shape, fill and resource policy to bind to batch_factory_id.
 * @param numeric Optional numeric policy; presence differs from omission.
 * @return Canonical v1 digest, including all limits and signed-zero folding.
 * @note Source pixels and source identities are separately bound by the full
 * batch schema and dataset_version. Source-dependent validation occurs on use.
 */
std::string image_batch_identity(
    const ImageCollateOptions& collate,
    const std::optional<ImageBatchTransformOptions>& numeric = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_BATCH_TRANSFORM_HPP
