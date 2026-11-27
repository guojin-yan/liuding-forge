// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_PRIVATE_ROI_GPU_HPP
#define JYPPX_LIUDING_VISION_PRIVATE_ROI_GPU_HPP
#include <jyppx/liuding/vision/ops/roi.hpp>
namespace jyppx::liuding::vision_ops::detail {
// Rows have already passed the public metadata, finite and extent checks.
// An undefined result asks the caller to retain its existing ATen path.
Tensor try_roi_align_gpu(const Tensor& features, const Tensor& rows,
                         const RoiAlignOptions& options);
}  // namespace jyppx::liuding::vision_ops::detail
#endif
