// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_PRIVATE_ROI_GPU_LAUNCH_HPP
#define JYPPX_LIUDING_VISION_PRIVATE_ROI_GPU_LAUNCH_HPP
#include <cstdint>
namespace jyppx::liuding::vision_ops::detail {
struct RoiGpuShape {
  std::int64_t count, channels, height, width, pooled_h, pooled_w, ratio;
  std::int64_t sn, sc, sh, sw;
};
bool vision_gpu_supported();
int launch_roi_gpu(const void* input, const double* rows,
                   const std::int64_t* batches, void* output, RoiGpuShape shape,
                   bool fp64, bool backward, void* stream);
}  // namespace jyppx::liuding::vision_ops::detail
#endif
