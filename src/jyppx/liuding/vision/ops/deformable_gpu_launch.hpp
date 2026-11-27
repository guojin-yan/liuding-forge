// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <cstdint>
namespace jyppx::liuding::vision_ops::detail {
inline constexpr int kDeformGpuMaxLevels = 16;
struct DeformGpuShape {
  std::int64_t batch, tokens, heads, channels, queries, levels, points;
  std::int64_t vs[4], ls[6], ws[5], spatial[kDeformGpuMaxLevels * 3];
};
int launch_deform_gpu_forward(const void*, const void*, const void*, void*,
                              DeformGpuShape, bool, void*);
int launch_deform_gpu_backward(const void*, const void*, const void*,
                               const void*, void*, void*, void*, DeformGpuShape,
                               bool, void*);
}  // namespace jyppx::liuding::vision_ops::detail
