// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo_gpu.hpp"
#include "yolo_gpu_launch.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <ATen/ops/_neg_view.h>
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
#include <gtest/gtest.h>
#include <limits>
#include <vector>
namespace jyppx::liuding::vision_ops::tests {
namespace {
namespace t = jyppx::liuding::vision::tasks;
namespace gpu = jyppx::liuding::vision_ops::detail;
class NativeYoloGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::set_num_threads(2);
    ASSERT_TRUE(torch::cuda::is_available());
  }
};
void check(const Tensor& a, const Tensor& b) {
  ASSERT_EQ(a.sizes(), b.sizes());
  ASSERT_EQ(a.scalar_type(), b.scalar_type());
  if (!a.is_floating_point()) {
    EXPECT_TRUE(torch::equal(a, b));
    return;
  }
  ASSERT_TRUE(torch::equal(torch::isnan(a), torch::isnan(b)));
  ASSERT_TRUE(torch::equal(torch::isinf(a), torch::isinf(b)));
  EXPECT_TRUE(torch::equal(a.masked_select(torch::isinf(a)),
                           b.masked_select(torch::isinf(b))));
  auto finite = torch::isfinite(a);
  EXPECT_TRUE(torch::allclose(a.masked_select(finite), b.masked_select(finite),
                              2e-4, 2e-5));
}
// Independent ATen oracle: keep each materialized FP32 boundary; the native
// implementation combines these into a single device launch.
Tensor OracleCiou(const Tensor& left, const Tensor& right) {
  constexpr double eps = 1e-7;
  auto l = left.chunk(4, -1), r = right.chunk(4, -1);
  auto lw = l[2] - l[0], lh = l[3] - l[1] + eps, rw = r[2] - r[0],
       rh = r[3] - r[1] + eps;
  auto intersection =
      (torch::minimum(l[2], r[2]) - torch::maximum(l[0], r[0])).clamp_min(0) *
      (torch::minimum(l[3], r[3]) - torch::maximum(l[1], r[1])).clamp_min(0);
  auto iou = intersection / (lw * lh + rw * rh - intersection + eps);
  auto cw = torch::maximum(l[2], r[2]) - torch::minimum(l[0], r[0]);
  auto ch = torch::maximum(l[3], r[3]) - torch::minimum(l[1], r[1]);
  auto diagonal = cw.square() + ch.square() + eps;
  auto center = ((r[0] + r[2] - l[0] - l[2]).square() +
                 (r[1] + r[3] - l[1] - l[3]).square()) /
                4;
  auto aspect = (4 / (std::acos(-1.) * std::acos(-1.))) *
                (torch::atan(rw / rh) - torch::atan(lw / lh)).square();
  auto alpha = aspect / (1 - iou + aspect + eps);
  return (iou - (center / diagonal + aspect * alpha)).squeeze(-1);
}
void staged_audit(const t::YoloDecodedPrediction& p,
                  const torch::Tensor& targets,
                  const t::YoloDetectionOptions& o) {
  if (targets.size(0) == 0) return;
  using I = int64_t;
  const I batch = p.boxes.size(0), anchors = p.boxes.size(1);
  auto cpu = targets.to(torch::kCPU, torch::kDouble).contiguous();
  auto data = cpu.accessor<double, 2>();
  std::vector<std::vector<I>> rr(batch);
  for (I i = 0; i < cpu.size(0); ++i) rr[I(data[i][0])].push_back(i);
  std::vector<I> im;
  for (I b = 0; b < batch; ++b)
    if (!rr[b].empty()) im.push_back(b);
  std::stable_sort(im.begin(), im.end(),
                   [&](auto x, auto y) { return rr[x].size() > rr[y].size(); });
  const I size = im.size(), count = rr[im[0]].size();
  std::vector<I> packed(size * count), lengths;
  for (I i = 0; i < size; ++i) {
    lengths.push_back(rr[im[i]].size());
    std::copy(rr[im[i]].begin(), rr[im[i]].end(), packed.begin() + i * count);
  }
  auto image_index = torch::tensor(im, torch::kInt64).cuda();
  auto row_ids =
      torch::tensor(packed, torch::kInt64).view({size, count}).cuda();
  auto valid =
      torch::arange(
          count,
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
          .unsqueeze(0) <
      torch::tensor(lengths, torch::kInt64).cuda().unsqueeze(1);
  auto selected =
      targets.index_select(0, row_ids.flatten()).view({size, count, 6});
  auto gt = selected.slice(2, 2, 6).to(torch::kFloat32),
       classes = selected.select(2, 1).to(torch::kInt64);
  auto candidates = gt, centers = p.anchor_points * p.strides;
  if (o.assignment_minimum_side > 0) {
    auto center = (gt.slice(2, 0, 2) + gt.slice(2, 2, 4)) * .5,
         side = gt.slice(2, 2, 4) - gt.slice(2, 0, 2);
    auto expanded =
        o.assignment_small_side_threshold > 0
            ? torch::where(side < o.assignment_small_side_threshold,
                           torch::full_like(side, o.assignment_minimum_side),
                           side)
            : side.clamp_min(o.assignment_minimum_side);
    auto half = expanded * .5;
    candidates = torch::cat({center - half, center + half}, 2);
  }
  auto inside =
      std::get<0>(torch::cat({centers.view({1, 1, anchors, 2}) -
                                  candidates.slice(2, 0, 2).unsqueeze(2),
                              candidates.slice(2, 2, 4).unsqueeze(2) -
                                  centers.view({1, 1, anchors, 2})},
                             -1)
                      .min(-1)) > 1e-9;
  inside = inside & valid.unsqueeze(-1);
  auto overlap = OracleCiou(gt.unsqueeze(2),
                            p.boxes.index_select(0, image_index).unsqueeze(1))
                     .clamp_min(0) *
                 inside;
  auto confidence =
      p.class_scores.index_select(0, image_index)
          .transpose(1, 2)
          .gather(1, classes.unsqueeze(-1).expand({size, count, anchors}));
  auto alignment = confidence.pow(o.assignment_alpha) *
                   overlap.pow(o.assignment_beta) * inside;
  auto native = gpu::try_yolo_pair_gpu(
      gt, candidates, centers, p.boxes, p.class_scores, image_index, classes,
      valid, o.assignment_alpha, o.assignment_beta);
  if (o.assignment_alpha == .5 && o.assignment_beta == 6.) {
    TORCH_CHECK(native, "Expected native pair path");
    check(inside, (*native)[0]);
    check(overlap, (*native)[1]);
    check(alignment, (*native)[2]);
  } else
    TORCH_CHECK(!native, "Unsupported exponent must fall back");
  // Both routes receive the SAME observed top-k decisions, including ties.
  // This proves the fused primitives; it does not qualify the separately
  // retained end-to-end extreme-tie failures or turn them into passes.
  auto top =
      std::get<1>(alignment.topk((std::min)(o.assignment_topk, anchors), 2));
  auto positive = torch::zeros_like(inside).scatter_(2, top, 1) & inside;
  auto candidate_positive = positive.clone(),
       masked = torch::empty_like(alignment);
  auto stream = c10::cuda::getCurrentCUDAStream(0).stream();
  auto status = gpu::launch_yolo_conflict_gpu(
      candidate_positive.mutable_data_ptr<bool>(),
      overlap.const_data_ptr<float>(), alignment.const_data_ptr<float>(),
      masked.mutable_data_ptr<float>(), size, count, anchors, stream);
  TORCH_CHECK(status == 0, "Resolve launch");
  auto conflicts = positive.to(torch::kInt64).sum(1) > 1;
  auto best_gt = std::get<1>(overlap.max(1));
  positive = torch::where(
      conflicts.unsqueeze(1),
      torch::zeros_like(positive).scatter_(1, best_gt.unsqueeze(1), 1),
      positive);
  check(positive, candidate_positive);
  check(alignment * positive, masked);
  if (o.assignment_topk2 > 0 && o.assignment_topk2 < o.assignment_topk) {
    auto secondary =
        std::get<1>((alignment * positive)
                        .topk((std::min)(o.assignment_topk2, anchors), 2));
    auto retained = torch::zeros_like(positive).scatter_(2, secondary, 1);
    positive = positive & retained;
    candidate_positive = candidate_positive & retained;
  }
  auto masked_overlap = torch::empty_like(overlap);
  status = gpu::launch_yolo_mask_gpu(
      candidate_positive.const_data_ptr<bool>(),
      overlap.const_data_ptr<float>(), alignment.const_data_ptr<float>(),
      masked.mutable_data_ptr<float>(),
      masked_overlap.mutable_data_ptr<float>(), size, count, anchors, stream);
  TORCH_CHECK(status == 0, "Mask launch");
  auto positive_alignment = alignment * positive;
  check(positive_alignment, masked);
  check(overlap * positive, masked_overlap);
  auto best_alignment = std::get<0>(positive_alignment.max(2, true));
  auto best_overlap = std::get<0>((overlap * positive).max(2, true));
  auto boxes = torch::zeros_like(p.boxes),
       scores = torch::zeros_like(p.class_scores);
  auto foreground =
      torch::zeros({batch, anchors}, p.boxes.options().dtype(torch::kBool));
  auto ids =
      torch::full({batch, anchors}, -1, p.boxes.options().dtype(torch::kInt64));
  auto new_boxes = torch::zeros_like(boxes),
       new_scores = torch::zeros_like(scores);
  auto new_foreground = torch::zeros_like(foreground),
       new_ids = torch::full_like(ids, -1);
  gpu::YoloPostGpuShape s{};
  s.size = size;
  s.count = count;
  s.anchors = anchors;
  for (int k = 0; k < 3; ++k) {
    s.gt[k] = gt.stride(k);
    s.boxes[k] = new_boxes.stride(k);
    s.scores[k] = new_scores.stride(k);
  }
  for (int k = 0; k < 2; ++k) {
    s.classes[k] = classes.stride(k);
    s.rows[k] = row_ids.stride(k);
    s.foreground[k] = new_foreground.stride(k);
    s.target_ids[k] = new_ids.stride(k);
  }
  s.images = image_index.stride(0);
  status = gpu::launch_yolo_finish_gpu(
      candidate_positive.const_data_ptr<bool>(), masked.const_data_ptr<float>(),
      best_alignment.const_data_ptr<float>(),
      best_overlap.const_data_ptr<float>(), gt.const_data_ptr<float>(),
      classes.const_data_ptr<I>(), image_index.const_data_ptr<I>(),
      row_ids.const_data_ptr<I>(), new_boxes.mutable_data_ptr<float>(),
      new_scores.mutable_data_ptr<float>(),
      new_foreground.mutable_data_ptr<bool>(), new_ids.mutable_data_ptr<I>(), s,
      stream);
  TORCH_CHECK(status == 0, "Finish launch");
  auto mask = positive.any(1),
       assigned = std::get<1>(positive.to(torch::kInt64).max(1));
  auto quality = std::get<0>(
      (positive_alignment * best_overlap / (best_alignment + 1e-9)).max(1));
  auto labels = classes.gather(1, assigned),
       anchor_ids = torch::arange(anchors, ids.options()).unsqueeze(0);
  scores.index_put_({image_index.unsqueeze(1), anchor_ids, labels}, quality);
  boxes.index_copy_(
      0, image_index,
      gt.gather(1, assigned.unsqueeze(-1).expand({size, anchors, 4})));
  foreground.index_copy_(0, image_index, mask);
  ids.index_copy_(0, image_index,
                  row_ids.gather(1, assigned).masked_fill(~mask, -1));
  check(boxes, new_boxes);
  check(scores, new_scores);
  check(foreground, new_foreground);
  check(ids, new_ids);
}

// Share the observed ATen top-k decisions between oracles. Indices of tied
// values are not stable across ATen invocations, even without this native path.
// This test proves the fused math, not end-to-end repeatability of those ties.
TEST_F(NativeYoloGpu, PairAndPostMathMatchAtenAcrossViewsStreamsAndExtremes) {
  torch::manual_seed(20261127);
  torch::NoGradGuard guard;
  for (int test = 0; test < 144; ++test) {
    SCOPED_TRACE(test);
    const int canvas = test >= 140 ? 640 : 32 * (1 + test % 4),
              batch = 1 + test % 4;
    t::YoloDetectionOptions o;
    o.num_classes = 5;
    o.reg_max = test % 3 == 0 ? 1 : 16;
    o.assignment_topk = std::vector<int>{1, 3, 8, 10, 100}[test % 5];
    o.assignment_topk2 = test % 4 == 0   ? 0
                         : test % 4 == 1 ? 1
                         : test % 4 == 2
                             ? o.assignment_topk
                             : (std::max)(int(o.assignment_topk) - 1, 1);
    o.assignment_minimum_side = test % 3 == 0 ? 0 : 20;
    o.assignment_small_side_threshold = test % 3 == 2 ? 10 : 0;
    o.max_assignment_pairs = 16777216;
    if (test % 11 == 0) o.assignment_alpha = 1.;
    if (test % 13 == 0) o.assignment_beta = 3.;
    int anchors = 0;
    for (auto s : o.strides) anchors += (canvas / s) * (canvas / s);
    auto raw = torch::randn({batch, 4 * o.reg_max + o.num_classes, anchors},
                            torch::TensorOptions().device(torch::kCUDA));
    if (o.reg_max == 1) raw.slice(1, 0, 4).add_(2);
    if (test % 9 == 0) raw.zero_();
    if (test % 9 == 1) raw.mul_(1000);
    if (test % 9 == 2) {
      raw.index_put_({0, 0, 0}, std::numeric_limits<float>::quiet_NaN());
    }
    if (test % 9 == 3)
      raw.index_put_({0, 1, 1}, std::numeric_limits<float>::infinity());
    if (test % 9 == 4)
      raw.index_put_({0, 2, 2}, -std::numeric_limits<float>::infinity());
    if (test % 3 == 1) raw = raw.to(torch::kHalf);
    if (test % 3 == 2) raw = raw.to(torch::kBFloat16);
    if (test % 4 == 1) raw = raw.transpose(1, 2).contiguous().transpose(1, 2);
    if (test % 4 == 2) raw = torch::stack({raw, raw}, -1).select(-1, 1);
    if (test % 4 == 3) raw = raw.slice(0, 0, 1).expand({batch, -1, -1});
    std::vector<float> packed;
    for (int b = 0; b < batch; ++b)
      for (int row = 0; row < (test % 12 == 0 ? 0 : ((b * 7 + test) % 23 + 1));
           ++row) {
        const float left = test % 7 == 0 ? 0 : float(row % 8),
                    top = test % 7 == 0 ? 0 : float(row % 6);
        packed.insert(packed.end(), {float(b), float(row % 5), left, top,
                                     float(canvas), float(canvas)});
      }
    auto targets = torch::tensor(packed).view({-1, 6}).cuda();
    if (test % 3 == 1) targets = targets.to(torch::kFloat64);
    if (test % 3 == 2) targets = targets.to(torch::kHalf);
    if (test % 4 == 1)
      targets = targets.transpose(0, 1).contiguous().transpose(0, 1);
    if (test % 4 == 2 && targets.size(0) > 0) targets = targets.flip(0);
    torch::cuda::synchronize();
    auto stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);
    auto p = t::decode_yolo_detection(raw, {canvas, canvas}, o);
    if (test % 5 == 1)
      p.boxes = p.boxes.transpose(1, 2).contiguous().transpose(1, 2);
    if (test % 5 == 2)
      p.class_scores =
          p.class_scores.transpose(1, 2).contiguous().transpose(1, 2);
    staged_audit(p, targets, o);
    stream.synchronize();
  }
}
TEST_F(NativeYoloGpu, UnsupportedTypesViewsExponentsAndTracingSelectAten) {
  torch::NoGradGuard guard;
  auto options =
      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto gt = torch::tensor({0.f, 0.f, 32.f, 32.f}, options).view({1, 1, 4});
  auto centers = torch::tensor({16.f, 16.f}, options).view({1, 2});
  auto boxes = gt.clone(), scores = torch::full({1, 1, 2}, .5, options);
  auto images = torch::zeros({1}, options.dtype(torch::kInt64));
  auto classes = torch::zeros({1, 1}, options.dtype(torch::kInt64));
  auto valid = torch::ones({1, 1}, options.dtype(torch::kBool));
  auto apply = [&](const Tensor& a, const Tensor& c, double alpha = .5,
                   double beta = 6.) {
    return gpu::try_yolo_pair_gpu(a, a, centers, boxes, c, images, classes,
                                  valid, alpha, beta);
  };
  ASSERT_TRUE(apply(gt, scores));
  for (auto dtype : {torch::kFloat64, torch::kFloat16, torch::kBFloat16})
    EXPECT_FALSE(apply(gt.to(dtype), scores));
  EXPECT_FALSE(apply(gt.cpu(), scores));
  EXPECT_FALSE(apply(gt, at::_neg_view(scores)));
  EXPECT_FALSE(apply(gt, scores, 1., 6.));
  EXPECT_FALSE(apply(gt, scores, .5, 3.));
  auto before = torch::jit::tracer::getTracingState();
  torch::jit::tracer::setTracingState(
      std::make_shared<torch::jit::tracer::TracingState>());
  try {
    EXPECT_FALSE(apply(gt, scores));
  } catch (...) {
    torch::jit::tracer::setTracingState(before);
    throw;
  }
  torch::jit::tracer::setTracingState(before);
}
TEST_F(NativeYoloGpu, LossRejectsInvalidTargetsAndBoundsBeforeNativeIndexing) {
  auto options = torch::TensorOptions().device(torch::kCUDA);
  auto raw = torch::zeros({1, 6, 21}, options).requires_grad_(true);
  auto gt =
      torch::tensor({0.f, 0.f, 1.f, 1.f, 30.f, 30.f}, options).view({1, 6});
  t::YoloDetectionOptions o;
  o.num_classes = 2;
  o.reg_max = 1;
  for (int column = 0; column < 6; ++column) {
    auto invalid = gt.clone();
    invalid.index_put_({0, column}, column < 2 ? -.5f : 40.f);
    EXPECT_THROW(t::yolo_detection_loss(raw, invalid, {32, 32}, o),
                 InvalidArgument);
  }
  auto limited = o;
  limited.max_targets = 1;
  EXPECT_THROW(
      t::yolo_detection_loss(raw, gt.repeat({2, 1}), {32, 32}, limited),
      ResourceExhausted);
  limited = o;
  limited.max_assignment_pairs = 20;
  EXPECT_THROW(t::yolo_detection_loss(raw, gt, {32, 32}, limited),
               ResourceExhausted);
}
}  // namespace
}  // namespace jyppx::liuding::vision_ops::tests
