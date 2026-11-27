// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo_gpu.hpp"

#include "roi_gpu_launch.hpp"
#include "yolo_gpu_launch.hpp"
#ifdef JYPPX_VISION_GPU_HIP
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/torch.h>

#include <algorithm>

namespace jyppx::liuding::vision_ops::detail {
std::optional<std::array<Tensor, 3>> try_yolo_pair_gpu(
    const Tensor& gt, const Tensor& candidates, const Tensor& centers,
    const Tensor& boxes, const Tensor& scores, const Tensor& images,
    const Tensor& classes, const Tensor& valid, double alpha, double beta) {
  if (!gt.is_cuda() || torch::jit::tracer::isTracing()) return {};
  c10::cuda::CUDAGuard guard(gt.device());
  if (!jyppx::liuding::vision_ops::detail::vision_gpu_supported()) return {};
  for (const auto* value : {&gt, &candidates, &centers, &boxes, &scores})
    if (value->scalar_type() != torch::kFloat32 || !value->is_cuda() ||
        value->device() != gt.device() || value->layout() != torch::kStrided ||
        value->is_neg() || value->is_conj() || value->_is_zerotensor())
      return {};
  // Keep the qualified FP32 scalar exponents; other recipes retain ATen.
  if (alpha != .5 || beta != 6.) return {};
  YoloPairGpuShape s{};
  s.size = gt.size(0);
  s.count = gt.size(1);
  s.anchors = boxes.size(1);
  for (int k = 0; k < 3; ++k) {
    s.gt[k] = gt.stride(k);
    s.candidates[k] = candidates.stride(k);
    s.boxes[k] = boxes.stride(k);
    s.scores[k] = scores.stride(k);
  }
  for (int k = 0; k < 2; ++k) {
    s.centers[k] = centers.stride(k);
    s.classes[k] = classes.stride(k);
    s.valid[k] = valid.stride(k);
  }
  s.images = images.stride(0);
  s.alpha = float(alpha);
  s.beta = float(beta);
  auto inside = torch::empty({s.size, s.count, s.anchors},
                             gt.options().dtype(torch::kBool));
  auto overlaps = torch::empty({s.size, s.count, s.anchors}, gt.options());
  auto alignment = torch::empty_like(overlaps);
  auto status = launch_yolo_pair_gpu(
      gt.const_data_ptr<float>(), candidates.const_data_ptr<float>(),
      centers.const_data_ptr<float>(), boxes.const_data_ptr<float>(),
      scores.const_data_ptr<float>(), images.const_data_ptr<std::int64_t>(),
      classes.const_data_ptr<std::int64_t>(), valid.const_data_ptr<bool>(),
      inside.mutable_data_ptr<bool>(), overlaps.mutable_data_ptr<float>(),
      alignment.mutable_data_ptr<float>(), s,
      c10::cuda::getCurrentCUDAStream(gt.get_device()).stream());
  TORCH_CHECK(status == 0, "Native YOLO pair math launch failed: ", status);
  return std::array<Tensor, 3>{inside, overlaps, alignment};
}
}  // namespace jyppx::liuding::vision_ops::detail

namespace jyppx::liuding::vision_ops::detail {
void finish_yolo_assignment_gpu(
    torch::Tensor positive, const torch::Tensor& overlaps,
    const torch::Tensor& alignment, const torch::Tensor& gt,
    const torch::Tensor& classes, const torch::Tensor& images,
    const torch::Tensor& rows, torch::Tensor& boxes, torch::Tensor& scores,
    torch::Tensor& foreground, torch::Tensor& target_ids, std::int64_t topk,
    std::int64_t topk2) {
  c10::cuda::CUDAGuard guard(gt.device());
  const auto size = positive.size(0), count = positive.size(1),
             anchors = positive.size(2);
  void* stream = c10::cuda::getCurrentCUDAStream(gt.get_device()).stream();
  auto masked = torch::empty_like(alignment);
  auto status = launch_yolo_conflict_gpu(
      positive.mutable_data_ptr<bool>(), overlaps.const_data_ptr<float>(),
      alignment.const_data_ptr<float>(), masked.mutable_data_ptr<float>(), size,
      count, anchors, stream);
  TORCH_CHECK(status == 0, "Native YOLO conflict launch failed: ", status);
  if (topk2 > 0 && topk2 < topk) {
    const auto secondary =
        std::get<1>(masked.topk((std::min)(topk2, anchors), 2));
    positive = positive & torch::zeros_like(positive).scatter_(2, secondary, 1);
  }
  auto masked_overlap = torch::empty_like(overlaps);
  status = launch_yolo_mask_gpu(
      positive.const_data_ptr<bool>(), overlaps.const_data_ptr<float>(),
      alignment.const_data_ptr<float>(), masked.mutable_data_ptr<float>(),
      masked_overlap.mutable_data_ptr<float>(), size, count, anchors, stream);
  TORCH_CHECK(status == 0, "Native YOLO pair mask launch failed: ", status);
  const auto best_alignment = std::get<0>(masked.max(2, true));
  const auto best_overlap = std::get<0>(masked_overlap.max(2, true));
  YoloPostGpuShape s{};
  s.size = size;
  s.count = count;
  s.anchors = anchors;
  for (int k = 0; k < 3; ++k) {
    s.gt[k] = gt.stride(k);
    s.boxes[k] = boxes.stride(k);
    s.scores[k] = scores.stride(k);
  }
  for (int k = 0; k < 2; ++k) {
    s.classes[k] = classes.stride(k);
    s.rows[k] = rows.stride(k);
    s.foreground[k] = foreground.stride(k);
    s.target_ids[k] = target_ids.stride(k);
  }
  s.images = images.stride(0);
  status = launch_yolo_finish_gpu(
      positive.const_data_ptr<bool>(), masked.const_data_ptr<float>(),
      best_alignment.const_data_ptr<float>(),
      best_overlap.const_data_ptr<float>(), gt.const_data_ptr<float>(),
      classes.const_data_ptr<std::int64_t>(),
      images.const_data_ptr<std::int64_t>(),
      rows.const_data_ptr<std::int64_t>(), boxes.mutable_data_ptr<float>(),
      scores.mutable_data_ptr<float>(), foreground.mutable_data_ptr<bool>(),
      target_ids.mutable_data_ptr<std::int64_t>(), s, stream);
  TORCH_CHECK(status == 0,
              "Native YOLO assignment finish launch failed: ", status);
}
}  // namespace jyppx::liuding::vision_ops::detail
