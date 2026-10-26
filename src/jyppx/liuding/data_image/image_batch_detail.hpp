// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_BATCH_DETAIL_HPP
#define JYPPX_LIUDING_DATA_IMAGE_BATCH_DETAIL_HPP

#include <jyppx/liuding/data_image/image_batch.hpp>

namespace jyppx::liuding::data::image::detail {
[[noreturn]] void InvalidBatch(const char* operation);
[[noreturn]] void BatchLimit(const char* operation);
void CheckBatchLimits(const ImageBatchLimits& limits);
void CheckImageBatch(const ImageBatch& batch, const ImageBatchLimits& limits,
                     std::stop_token stop = {});
// Host-only preflight, also usable before allocating the pixel tensors.
std::size_t CheckBatchMetadata(const ImageBatch& batch,
                               std::span<const ImageExtent> canvases,
                               const ImageBatchLimits& limits,
                               bool source_panoptic = false,
                               std::stop_token stop = {});
std::uint64_t BatchTensorBytes(const ImageBatch& batch);
std::shared_ptr<ImageResourceLeaseSet> ReserveBatchTotals(
    const std::shared_ptr<ImageResourceBudget>& budget,
    std::uint64_t tensor_bytes, std::uint64_t pixel_count,
    std::uint64_t metadata_bytes, std::uint64_t index_bytes,
    std::uint64_t pinned_bytes, std::uint64_t gpu_bytes,
    std::stop_token stop = {});
std::shared_ptr<ImageResourceLeaseSet> ReserveBatchResources(
    const ImageBatch& batch, const std::shared_ptr<ImageResourceBudget>& budget,
    const ImageBatchLimits& limits, std::stop_token stop = {});
bool SamePixels(const PixelSpec& a, const PixelSpec& b);
void CheckImageFill(const PixelSpec& spec, double value);
}  // namespace jyppx::liuding::data::image::detail

#endif  // JYPPX_LIUDING_DATA_IMAGE_BATCH_DETAIL_HPP
