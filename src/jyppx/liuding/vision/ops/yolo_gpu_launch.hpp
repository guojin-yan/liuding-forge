// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_PRIVATE_YOLO_GPU_LAUNCH_HPP
#define JYPPX_LIUDING_VISION_PRIVATE_YOLO_GPU_LAUNCH_HPP
#include <cstdint>
namespace jyppx::liuding::vision_ops::detail {

struct YoloPairGpuShape {
  std::int64_t size, count, anchors;
  std::int64_t gt[3], candidates[3], centers[2], boxes[3], scores[3];
  std::int64_t images, classes[2], valid[2];
  float alpha, beta;
};
int launch_yolo_pair_gpu(const float* gt, const float* candidates,
                         const float* centers, const float* boxes,
                         const float* scores, const std::int64_t* images,
                         const std::int64_t* classes, const bool* valid,
                         bool* inside, float* overlaps, float* alignment,
                         YoloPairGpuShape shape, void* stream);
}  // namespace jyppx::liuding::vision_ops::detail

namespace jyppx::liuding::vision_ops::detail {

struct YoloPostGpuShape {
  std::int64_t size, count, anchors;
  std::int64_t gt[3], classes[2], images, rows[2];
  std::int64_t boxes[3], scores[3], foreground[2], target_ids[2];
};
int launch_yolo_conflict_gpu(bool* positive, const float* overlaps,
                             const float* alignment, float* masked_alignment,
                             std::int64_t size, std::int64_t count,
                             std::int64_t anchors, void* stream);
int launch_yolo_mask_gpu(const bool* positive, const float* overlaps,
                         const float* alignment, float* masked_alignment,
                         float* masked_overlap, std::int64_t size,
                         std::int64_t count, std::int64_t anchors,
                         void* stream);
int launch_yolo_finish_gpu(const bool* positive, const float* masked_alignment,
                           const float* best_alignment,
                           const float* best_overlap, const float* gt,
                           const std::int64_t* classes,
                           const std::int64_t* images, const std::int64_t* rows,
                           float* boxes, float* scores, bool* foreground,
                           std::int64_t* target_ids, YoloPostGpuShape shape,
                           void* stream);
}  // namespace jyppx::liuding::vision_ops::detail

#endif
