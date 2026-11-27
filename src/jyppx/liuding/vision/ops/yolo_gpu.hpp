// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_PRIVATE_YOLO_GPU_HPP
#define JYPPX_LIUDING_VISION_PRIVATE_YOLO_GPU_HPP
#include <jyppx/liuding/core/types.hpp>

#include <array>
#include <optional>
namespace jyppx::liuding::vision_ops::detail {
// Internal loss route only: image/class indices and group bounds were checked
// by the task, and all floating inputs are decoder FP32 values. No gradients.
// An empty result selects the original ATen math and postprocessing together.
std::optional<std::array<Tensor, 3>> try_yolo_pair_gpu(
    const Tensor& gt, const Tensor& candidates, const Tensor& centers,
    const Tensor& boxes, const Tensor& scores, const Tensor& images,
    const Tensor& classes, const Tensor& valid, double alpha, double beta);
// Only after try_yolo_pair_gpu succeeded. positive is a fresh private tensor;
// outputs are fresh zeros, indices validated, all tensors on the current GPU.
// ATen top-k choices and their existing tie behavior remain unchanged.
void finish_yolo_assignment_gpu(Tensor positive, const Tensor& overlaps,
                                const Tensor& alignment, const Tensor& gt,
                                const Tensor& classes, const Tensor& images,
                                const Tensor& rows, Tensor& boxes,
                                Tensor& scores, Tensor& foreground,
                                Tensor& target_ids, std::int64_t topk,
                                std::int64_t topk2);
}  // namespace jyppx::liuding::vision_ops::detail
#endif
