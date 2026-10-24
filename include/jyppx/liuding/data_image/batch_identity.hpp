// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_BATCH_IDENTITY_HPP
#define JYPPX_LIUDING_DATA_IMAGE_BATCH_IDENTITY_HPP
/**
 * @file batch_identity.hpp
 * @brief Canonical SHA-256 identity for validated CPU image batches.
 */

#include <jyppx/liuding/data_image/image_batch.hpp>

#include <stop_token>
#include <string>

namespace jyppx::liuding::data::image {

/**
 * @brief Hashes batch pixels, target tensors, pixel semantics and metadata.
 * @param batch A valid CPU batch; CUDA inputs are rejected without transfer.
 * @param limits Validation and metadata construction limits.
 * @param stop Cooperative cancellation checked between fields and data chunks.
 * @return Lowercase SHA-256 for the versioned canonical batch payload.
 * @throws InvalidArgument Invalid batch or a non-CPU tensor.
 * @throws ResourceExhausted Batch limits or allocation bounds are exceeded.
 * @throws DeviceError A LibTorch CPU operation fails.
 * @throws Cancelled Cancellation was requested before the digest was returned.
 *
 * The identity includes stable sample keys, source paths, labels, annotations,
 * geometry, masks and all tensor values. It identifies the current batch
 * payload, not compressed source files, and is not an authenticity signature.
 */
std::string image_batch_content_identity(const ImageBatch& batch,
                                         const ImageBatchLimits& limits = {},
                                         std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_BATCH_IDENTITY_HPP
