// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once
#include <jyppx/liuding/vision/ops/deformable.hpp>
namespace jyppx::liuding::vision_ops::detail {
// Called only after the public shape and sampled-temporary budget checks.
// An undefined result selects the ATen path without changing its contract.
Tensor try_deform_gpu(const Tensor& value,
                      std::span<const std::array<std::int64_t, 2>> shapes,
                      const Tensor& locations, const Tensor& weights);
}  // namespace jyppx::liuding::vision_ops::detail
