// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "training/class_weights.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>

#include <torch/nn/functional/loss.h>

#include <algorithm>
#include <cmath>
#include <map>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
void Validate(const OriginalV5TaskOptions& o) {
  detail::check_class_weights(o.class_weights, o.num_classes);
  if (o.num_classes < 1 || o.num_classes > 10000 ||
      (o.strides != std::vector<I>{8, 16, 32} &&
       o.strides != std::vector<I>{8, 16, 32, 64}) ||
      o.anchors.size() != o.strides.size() ||
      o.objectness_balance.size() != o.strides.size() || o.max_targets < 1 ||
      o.max_prediction_elements < 1 ||
      !std::isfinite(o.anchor_ratio_threshold) ||
      o.anchor_ratio_threshold <= 1 || !std::isfinite(o.label_smoothing) ||
      o.label_smoothing < 0 || o.label_smoothing >= 1 ||
      !std::isfinite(o.focal_alpha) || o.focal_alpha < 0 || o.focal_alpha > 1)
    throw InvalidArgument(
        "Invalid original YOLOv5 task layout or matching recipe");
  for (auto v : {o.box_gain, o.objectness_gain, o.class_gain, o.focal_gamma})
    if (!std::isfinite(v) || v < 0)
      throw InvalidArgument("Invalid original YOLOv5 loss gain");
  for (const auto& level : o.anchors)
    for (auto v : level)
      if (!std::isfinite(v) || v <= 0 || v > 65536)
        throw InvalidArgument("Invalid original YOLOv5 pixel anchor");
  for (auto v : o.objectness_balance)
    if (!std::isfinite(v) || v < 0)
      throw InvalidArgument("Invalid objectness level balance");
}
void Raw(const Tensor& raw, std::array<I, 2> size,
         const OriginalV5TaskOptions& o) {
  Validate(o);
  if (!raw.defined() || raw.dim() != 3 || !raw.is_floating_point() ||
      raw.size(0) < 1 || raw.size(2) != o.num_classes + 5 || size[0] < 1 ||
      size[1] < 1 || size[0] > 65536 || size[1] > 65536 ||
      size[0] % o.strides.back() || size[1] % o.strides.back())
    throw InvalidArgument(
        "Original YOLOv5 requires packed [B,A,C+5] raw predictions and an "
        "aligned canvas");
  I anchors = 0;
  for (auto stride : o.strides)
    anchors += 3 * (size[0] / stride) * (size[1] / stride);
  if (anchors != raw.size(1))
    throw InvalidArgument("Original YOLOv5 raw anchor count mismatch");
  if (raw.numel() > o.max_prediction_elements)
    throw ResourceExhausted("Original YOLOv5 raw element bound exceeded");
}
Tensor Bce(const Tensor& p, const Tensor& t, const OriginalV5TaskOptions& o,
           bool classes = false) {
  auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
      p, t,
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
          .reduction(torch::kNone));
  if (o.focal_gamma > 0) {
    auto probability = p.sigmoid();
    auto pt = t * probability + (1 - t) * (1 - probability);
    loss = loss * (t * o.focal_alpha + (1 - t) * (1 - o.focal_alpha)) *
           (1 - pt).pow(o.focal_gamma);
  }
  if (classes && !o.class_weights.empty())
    loss = loss * detail::class_weight_tensor(o.class_weights, p);
  return loss.mean();
}
Tensor Ciou(const Tensor& a, const Tensor& b) {
  auto amin = a.slice(1, 0, 2) - a.slice(1, 2, 4) * .5,
       amax = a.slice(1, 0, 2) + a.slice(1, 2, 4) * .5;
  auto bmin = b.slice(1, 0, 2) - b.slice(1, 2, 4) * .5,
       bmax = b.slice(1, 0, 2) + b.slice(1, 2, 4) * .5;
  auto intersection = (torch::minimum(amax, bmax) - torch::maximum(amin, bmin))
                          .clamp_min(0)
                          .prod(1);
  auto iou = intersection / (a.slice(1, 2, 4).prod(1) +
                             b.slice(1, 2, 4).prod(1) - intersection + 1e-7);
  auto diagonal = (torch::maximum(amax, bmax) - torch::minimum(amin, bmin))
                      .square()
                      .sum(1) +
                  1e-7;
  auto center = (a.slice(1, 0, 2) - b.slice(1, 0, 2)).square().sum(1);
  auto v = (4 / std::pow(std::acos(-1.), 2)) *
           (torch::atan(a.select(1, 2) / (a.select(1, 3) + 1e-7)) -
            torch::atan(b.select(1, 2) / (b.select(1, 3) + 1e-7)))
               .square();
  auto alpha = (v / (1 - iou + v + 1e-7)).detach();
  return iou - center / diagonal - alpha * v;
}
}  // namespace
YoloDecodedPrediction decode_original_v5(const Tensor& raw,
                                         std::array<I, 2> size,
                                         const OriginalV5TaskOptions& o) {
  Raw(raw, size, o);
  auto values = raw.to(torch::kFloat32);
  std::vector<Tensor> boxes, scores, points, strides;
  I offset = 0;
  for (std::size_t l = 0; l < o.strides.size(); ++l) {
    const I stride = o.strides[l], h = size[0] / stride, w = size[1] / stride,
            count = 3 * h * w;
    auto p = values.slice(1, offset, offset + count)
                 .view({raw.size(0), 3, h * w, -1});
    offset += count;
    auto xs = torch::arange(w, values.options()).view({1, w}).expand({h, w});
    auto ys = torch::arange(h, values.options()).view({h, 1}).expand({h, w});
    auto grid =
        torch::stack({xs, ys}, -1).reshape({1, h * w, 2}).expand({3, h * w, 2});
    std::vector<float> av;
    for (auto v : o.anchors[l]) av.push_back(static_cast<float>(v));
    auto anchors = torch::tensor(av, values.options()).view({1, 3, 1, 2});
    auto center = (p.slice(3, 0, 2).sigmoid() * 2 - .5 + grid) * stride;
    auto wh = (p.slice(3, 2, 4).sigmoid() * 2).square() * anchors;
    boxes.push_back(torch::cat({center - wh * .5, center + wh * .5}, -1)
                        .reshape({raw.size(0), count, 4}));
    scores.push_back((p.slice(3, 5).sigmoid() * p.slice(3, 4, 5).sigmoid())
                         .reshape({raw.size(0), count, o.num_classes}));
    points.push_back((grid + .5).reshape({count, 2}));
    strides.push_back(torch::full({count, 1}, stride, values.options()));
  }
  return {torch::cat(boxes, 1), torch::cat(scores, 1), torch::cat(points),
          torch::cat(strides)};
}
TaskLoss original_v5_loss(const Tensor& raw, const Tensor& targets,
                          std::array<I, 2> size,
                          const OriginalV5TaskOptions& o) {
  Raw(raw, size, o);
  if (!targets.defined() || targets.dim() != 2 || targets.size(1) != 6 ||
      !targets.is_floating_point() || targets.device() != raw.device() ||
      targets.size(0) > o.max_targets ||
      !torch::isfinite(targets).all().item<bool>())
    throw InvalidArgument("Invalid original YOLOv5 target rows");
  auto host = targets.detach().to(torch::kCPU, torch::kFloat64).contiguous();
  auto gt = host.accessor<double, 2>();
  for (I i = 0; i < targets.size(0); ++i)
    if (gt[i][0] != std::floor(gt[i][0]) || gt[i][1] != std::floor(gt[i][1]) ||
        gt[i][0] < 0 || gt[i][0] >= raw.size(0) || gt[i][1] < 0 ||
        gt[i][1] >= o.num_classes || gt[i][2] < 0 || gt[i][3] < 0 ||
        gt[i][4] > size[1] || gt[i][5] > size[0] || gt[i][4] <= gt[i][2] ||
        gt[i][5] <= gt[i][3])
      throw InvalidArgument("Invalid original YOLOv5 batch/class/box values");
  auto values = raw.to(torch::kFloat32);
  auto zero = values.sum() * 0, box_loss = zero, obj_loss = zero,
       cls_loss = zero;
  I offset = 0;
  for (std::size_t l = 0; l < o.strides.size(); ++l) {
    const I stride = o.strides[l], h = size[0] / stride, w = size[1] / stride,
            count = 3 * h * w;
    auto p = values.slice(1, offset, offset + count);
    offset += count;
    std::vector<I> flat, labels;
    std::vector<float> desired, anchor_sizes;
    for (I anchor = 0; anchor < 3; ++anchor)
      for (I row = 0; row < targets.size(0); ++row) {
        const double aw = o.anchors[l][anchor * 2] / stride,
                     ah = o.anchors[l][anchor * 2 + 1] / stride;
        const double x = (gt[row][2] + gt[row][4]) * .5 / stride,
                     y = (gt[row][3] + gt[row][5]) * .5 / stride;
        const double tw = (gt[row][4] - gt[row][2]) / stride,
                     th = (gt[row][5] - gt[row][3]) / stride;
        if ((std::max)({tw / aw, aw / tw, th / ah, ah / th}) >=
            o.anchor_ratio_threshold)
          continue;
        const std::array<std::array<double, 2>, 5> offsets{
            {{0, 0}, {.5, 0}, {0, .5}, {-.5, 0}, {0, -.5}}};
        for (std::size_t k = 0; k < offsets.size(); ++k) {
          if ((k == 1 && !(x - std::floor(x) < .5 && x > 1)) ||
              (k == 2 && !(y - std::floor(y) < .5 && y > 1)) ||
              (k == 3 && !((w - x) - std::floor(w - x) < .5 && w - x > 1)) ||
              (k == 4 && !((h - y) - std::floor(h - y) < .5 && h - y > 1)))
            continue;
          const I gx = std::clamp(static_cast<I>(x - offsets[k][0]), I{0},
                                  w - 1),
                  gy = std::clamp(static_cast<I>(y - offsets[k][1]), I{0},
                                  h - 1);
          flat.push_back(static_cast<I>(gt[row][0]) * count + anchor * h * w +
                         gy * w + gx);
          labels.push_back(static_cast<I>(gt[row][1]));
          for (auto v : {x - gx, y - gy, tw, th})
            desired.push_back(static_cast<float>(v));
          anchor_sizes.push_back(static_cast<float>(aw));
          anchor_sizes.push_back(static_cast<float>(ah));
        }
      }
    auto objectness = torch::zeros({raw.size(0) * count}, values.options());
    if (!flat.empty()) {
      auto indices =
          torch::tensor(flat, torch::TensorOptions().dtype(torch::kInt64))
              .to(raw.device());
      auto selected =
          p.reshape({-1, 5 + o.num_classes}).index_select(0, indices);
      auto anchors =
          torch::tensor(anchor_sizes, values.options()).view({-1, 2});
      auto predicted = torch::cat(
          {selected.slice(1, 0, 2).sigmoid() * 2 - .5,
           (selected.slice(1, 2, 4).sigmoid() * 2).square() * anchors},
          1);
      auto expected = torch::tensor(desired, values.options()).view({-1, 4});
      auto iou = Ciou(predicted, expected);
      box_loss = box_loss + (1 - iou).mean();
      // A canonical max resolves repeated grid assignments before device
      // scatter.
      auto quality = iou.detach().clamp_min(0).to(torch::kCPU).contiguous();
      std::map<I, float> unique;
      for (std::size_t i = 0; i < flat.size(); ++i)
        unique[flat[i]] = (std::max)(unique[flat[i]],
                                     quality[static_cast<I>(i)].item<float>());
      std::vector<I> cells;
      std::vector<float> confidences;
      for (auto [cell, score] : unique) {
        cells.push_back(cell);
        confidences.push_back(score);
      }
      objectness.index_put_(
          {torch::tensor(cells, torch::kInt64).to(raw.device())},
          torch::tensor(confidences, values.options()));
      if (o.num_classes > 1) {
        auto target =
            torch::full_like(selected.slice(1, 5), o.label_smoothing * .5);
        target.scatter_(
            1,
            torch::tensor(labels, torch::kInt64).to(raw.device()).unsqueeze(1),
            1 - o.label_smoothing * .5);
        cls_loss = cls_loss + Bce(selected.slice(1, 5), target, o, true);
      }
    }
    obj_loss = obj_loss + Bce(p.select(2, 4).reshape({-1}), objectness, o) *
                              o.objectness_balance[l];
  }
  return {(box_loss * o.box_gain + obj_loss * o.objectness_gain +
           cls_loss * o.class_gain) *
              raw.size(0),
          {{"box", box_loss},
           {"objectness", obj_loss},
           {"classification", cls_loss}}};
}
training::StepFunction make_original_v5_step(const OriginalV5TaskOptions& o) {
  Validate(o);
  return [o](const training::StepInput& input) {
    if (!input.target || !input.target->isTensor() ||
        input.invocation.positional.empty())
      throw InvalidArgument("Original YOLOv5 needs images and box targets");
    auto images = input.invocation.positional[0].toTensor();
    auto prediction = input.model.invoke(input.invocation);
    auto policy = o;
    if (input.context.phase == training::StepPhase::validation)
      policy.class_weights.clear();
    auto loss =
        original_v5_loss(prediction.value.toTensor(), input.target->toTensor(),
                         {images.size(2), images.size(3)}, policy);
    training::StepOutput result;
    result.loss = input.context.phase == training::StepPhase::validation
                      ? loss.total / images.size(0)
                      : loss.total;
    result.prediction = std::move(prediction);
    result.sample_count = static_cast<std::uint64_t>(images.size(0));
    for (auto& [name, value] : loss.components)
      result.metrics["loss." + name] = value.detach().item<double>();
    return result;
  };
}
}  // namespace jyppx::liuding::vision::tasks
