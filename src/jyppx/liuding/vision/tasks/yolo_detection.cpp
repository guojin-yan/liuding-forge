// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "training/class_weights.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <torch/nn/functional/loss.h>

#include <ATen/autocast_mode.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace jyppx::liuding::vision::tasks {
namespace {

constexpr double kEpsilon = 1e-9;

void ValidateOptions(const YoloDetectionOptions& options) {
  detail::check_class_weights(options.class_weights, options.num_classes);
  if (options.num_classes < 1 || options.num_classes > 10000 ||
      options.reg_max < 1 || options.reg_max > 32 ||
      options.assignment_topk <= 0 || options.max_targets <= 0 ||
      options.assignment_topk2 < 0 ||
      options.assignment_topk2 > options.assignment_topk ||
      !std::isfinite(options.assignment_minimum_side) ||
      options.assignment_minimum_side < 0 ||
      options.assignment_minimum_side > 65536 ||
      options.max_assignment_pairs <= 0 ||
      options.max_prediction_elements <= 0 || options.strides.empty() ||
      options.strides.size() > 5) {
    throw InvalidArgument("Invalid detection head, bounds or stride recipe");
  }
  std::int64_t previous = 0;
  for (auto stride : options.strides) {
    if ((stride != 4 && stride != 8 && stride != 16 && stride != 32 &&
         stride != 64) ||
        (previous && stride != previous * 2))
      throw InvalidArgument(
          "Detection strides must be consecutive ascending pyramid levels");
    previous = stride;
  }
  for (const auto value :
       {options.assignment_alpha, options.assignment_beta, options.box_gain,
        options.class_gain, options.dfl_gain}) {
    if (!std::isfinite(value) || value < 0.0) {
      throw InvalidArgument(
          "Detection exponents and loss gains must be finite and nonnegative");
    }
  }
  if (options.assignment_beta == 0.0) {
    throw InvalidArgument("Detection overlap exponent must be positive");
  }
}

void ValidateRaw(const Tensor& raw, std::array<std::int64_t, 2> size,
                 const YoloDetectionOptions& options) {
  ValidateOptions(options);
  if (!raw.defined() || raw.layout() != torch::kStrided || raw.dim() != 3 ||
      !raw.is_floating_point() || (!raw.is_cpu() && !raw.is_cuda()) ||
      raw.size(0) <= 0 ||
      raw.size(1) != options.num_classes + 4 * options.reg_max ||
      size[0] <= 0 || size[1] <= 0 || size[0] > 65536 || size[1] > 65536 ||
      size[0] % options.strides.back() != 0 ||
      size[1] % options.strides.back() != 0) {
    throw InvalidArgument(
        "Detection requires packed floating predictions and a stride-32 "
        "canvas");
  }
  std::int64_t anchors = 0;
  for (const auto stride : options.strides) {
    anchors += (size[0] / stride) * (size[1] / stride);
  }
  if (raw.size(2) != anchors) {
    throw InvalidArgument(
        "Detection raw anchor count does not match the input canvas");
  }
  if (raw.numel() > options.max_prediction_elements) {
    throw ResourceExhausted("Detection raw prediction element bound exceeded");
  }
}

std::pair<Tensor, Tensor> AnchorGrid(const Tensor& raw,
                                     std::array<std::int64_t, 2> size,
                                     const YoloDetectionOptions& options) {
  auto tensor_options =
      raw.options().dtype(torch::kFloat32).requires_grad(false);
  std::vector<Tensor> points, strides;
  for (const auto stride : options.strides) {
    const auto height = size[0] / stride, width = size[1] / stride;
    auto xs = (torch::arange(width, tensor_options) + 0.5)
                  .view({1, width})
                  .expand({height, width});
    auto ys = (torch::arange(height, tensor_options) + 0.5)
                  .view({height, 1})
                  .expand({height, width});
    points.push_back(torch::stack({xs, ys}, -1).reshape({-1, 2}));
    strides.push_back(torch::full({height * width, 1}, stride, tensor_options));
  }
  return {torch::cat(points, 0), torch::cat(strides, 0)};
}

Tensor CompleteIou(const Tensor& left, const Tensor& right) {
  // FP32 order and stabilization follow Ultralytics 8.4.177
  // utils/metrics.py::bbox_iou; see MODEL_PROVENANCE.md for the pinned source.
  constexpr double epsilon = 1e-7;
  const auto l = left.chunk(4, -1), r = right.chunk(4, -1);
  const auto lw = l[2] - l[0], lh = l[3] - l[1] + epsilon;
  const auto rw = r[2] - r[0], rh = r[3] - r[1] + epsilon;
  const auto intersection =
      (torch::minimum(l[2], r[2]) - torch::maximum(l[0], r[0])).clamp_min(0) *
      (torch::minimum(l[3], r[3]) - torch::maximum(l[1], r[1])).clamp_min(0);
  const auto iou = intersection / (lw * lh + rw * rh - intersection + epsilon);
  const auto cw = torch::maximum(l[2], r[2]) - torch::minimum(l[0], r[0]);
  const auto ch = torch::maximum(l[3], r[3]) - torch::minimum(l[1], r[1]);
  const auto diagonal = cw.square() + ch.square() + epsilon;
  const auto center_distance = ((r[0] + r[2] - l[0] - l[2]).square() +
                                (r[1] + r[3] - l[1] - l[3]).square()) /
                               4;
  const auto aspect = (4 / (std::acos(-1.0) * std::acos(-1.0))) *
                      (torch::atan(rw / rh) - torch::atan(lw / lh)).square();
  const auto alpha = (aspect / (1 - iou + aspect + epsilon)).detach();
  return (iou - (center_distance / diagonal + aspect * alpha)).squeeze(-1);
}

void ValidateTargets(const Tensor& targets, const Tensor& raw,
                     std::array<std::int64_t, 2> size,
                     const YoloDetectionOptions& options) {
  if (!targets.defined() || targets.layout() != torch::kStrided ||
      targets.dim() != 2 || targets.size(1) != 6 ||
      !targets.is_floating_point() || targets.device() != raw.device()) {
    throw InvalidArgument(
        "Detection targets must be floating [T,6] on the prediction device");
  }
  if (targets.size(0) > options.max_targets) {
    throw ResourceExhausted("Detection target row bound exceeded");
  }
  if (targets.size(0) == 0) {
    return;
  }
  const auto indices = targets.slice(1, 0, 2);
  const auto valid =
      torch::isfinite(targets).all() & (indices == indices.round()).all() &
      (targets.select(1, 0) >= 0).all() &
      (targets.select(1, 0) < raw.size(0)).all() &
      (targets.select(1, 1) >= 0).all() &
      (targets.select(1, 1) < options.num_classes).all() &
      (targets.select(1, 2) >= 0).all() & (targets.select(1, 3) >= 0).all() &
      (targets.select(1, 4) > targets.select(1, 2)).all() &
      (targets.select(1, 5) > targets.select(1, 3)).all() &
      (targets.select(1, 4) <= size[1]).all() &
      (targets.select(1, 5) <= size[0]).all();
  if (!valid.item<bool>()) {
    throw InvalidArgument(
        "Detection targets contain invalid indices or canvas boxes");
  }
}

YoloAssignment Assign(const YoloDecodedPrediction& prediction,
                      const Tensor& targets,
                      const YoloDetectionOptions& options) {
  const torch::NoGradGuard no_grad;
  const auto batch = prediction.boxes.size(0),
             anchors = prediction.boxes.size(1);
  auto boxes = torch::zeros_like(prediction.boxes);
  auto scores = torch::zeros_like(prediction.class_scores);
  auto foreground = torch::zeros(
      {batch, anchors}, prediction.boxes.options().dtype(torch::kBool));
  auto target_indices = torch::full(
      {batch, anchors}, -1, prediction.boxes.options().dtype(torch::kInt64));
  const auto centers = prediction.anchor_points * prediction.strides;
  for (std::int64_t image = 0; image < batch; ++image) {
    auto target_rows = (targets.select(1, 0) == image).nonzero().flatten();
    auto selected = targets.index_select(0, target_rows);
    const auto count = selected.size(0);
    if (count == 0) {
      continue;
    }
    if (count > options.max_assignment_pairs / anchors) {
      throw ResourceExhausted("Detection assignment GT*anchor bound exceeded");
    }
    const auto gt = selected.slice(1, 2, 6).to(torch::kFloat32);
    const auto classes = selected.select(1, 1).to(torch::kInt64);
    auto candidates = gt;
    if (options.assignment_minimum_side > 0) {
      const auto center = (gt.slice(1, 0, 2) + gt.slice(1, 2, 4)) * .5;
      const auto half = (gt.slice(1, 2, 4) - gt.slice(1, 0, 2))
                            .clamp_min(options.assignment_minimum_side) *
                        .5;
      candidates = torch::cat({center - half, center + half}, 1);
    }
    const auto distances = torch::cat(
        {centers.unsqueeze(0) - candidates.slice(1, 0, 2).unsqueeze(1),
         candidates.slice(1, 2, 4).unsqueeze(1) - centers.unsqueeze(0)},
        -1);
    const auto inside = std::get<0>(distances.min(-1)) > kEpsilon;
    const auto overlaps =
        CompleteIou(gt.unsqueeze(1),
                    prediction.boxes[image].detach().unsqueeze(0))
            .clamp_min(0);
    const auto confidence = prediction.class_scores[image]
                                .detach()
                                .index_select(1, classes)
                                .transpose(0, 1);
    const auto alignment = confidence.pow(options.assignment_alpha) *
                           overlaps.pow(options.assignment_beta) * inside;
    const auto k = (std::min)(options.assignment_topk, anchors);
    const auto top_indices = std::get<1>(alignment.topk(k, 1));
    // A small positive alignment still belongs to the candidate set. The
    // epsilon stabilizes score normalization; it is not a foreground cutoff.
    auto positive =
        torch::zeros_like(inside).scatter_(1, top_indices, 1) & inside;
    auto conflicts = positive.to(torch::kInt64).sum(0) > 1;
    const auto best_gt = std::get<1>(overlaps.max(0));
    auto single_best =
        torch::zeros_like(positive).scatter_(0, best_gt.unsqueeze(0), 1);
    positive = torch::where(conflicts.unsqueeze(0), single_best, positive);
    if (options.assignment_topk2 > 0 &&
        options.assignment_topk2 < options.assignment_topk) {
      const auto secondary = std::get<1>(
          (alignment * positive)
              .topk((std::min)(options.assignment_topk2, anchors), 1));
      positive =
          positive & torch::zeros_like(positive).scatter_(1, secondary, 1);
    }
    auto mask = positive.any(0);
    auto assigned = std::get<1>(positive.to(torch::kInt64).max(0));
    auto positive_alignment = alignment * positive;
    auto best_alignment = std::get<0>(positive_alignment.max(1, true));
    auto best_overlap = std::get<0>((overlaps * positive).max(1, true));
    auto quality = std::get<0>(
        (positive_alignment * best_overlap / (best_alignment + kEpsilon))
            .max(0));
    const auto positives = mask.nonzero().flatten();
    const auto labels =
        classes.index_select(0, assigned).index_select(0, positives);
    scores[image].index_put_({positives, labels},
                             quality.index_select(0, positives));
    boxes[image].copy_(gt.index_select(0, assigned));
    foreground[image].copy_(mask);
    target_indices[image].index_put_(
        {positives},
        target_rows.index_select(0, assigned.index_select(0, positives)));
  }
  return {boxes, scores, foreground, target_indices};
}

}  // namespace

YoloAssignment assign_yolo_targets(const YoloDecodedPrediction& prediction,
                                   const Tensor& targets,
                                   const YoloDetectionOptions& options) {
  ValidateOptions(options);
  if (!prediction.boxes.defined() || prediction.boxes.dim() != 3 ||
      !targets.defined() || targets.dim() != 2 || targets.size(1) != 6 ||
      !targets.is_floating_point() ||
      targets.device() != prediction.boxes.device() ||
      targets.size(0) > options.max_targets ||
      !torch::isfinite(targets).all().item<bool>())
    throw InvalidArgument("Invalid YOLO assignment predictions/targets");
  return Assign(prediction, targets, options);
}

YoloDecodedPrediction decode_yolo_detection(
    const Tensor& raw, std::array<std::int64_t, 2> image_size,
    const YoloDetectionOptions& options) {
  ValidateRaw(raw, image_size, options);
  // The public decoder returns FP32 even inside a caller's autocast scope.
  const c10::impl::ExcludeDispatchKeyGuard no_autocast(
      at::autocast::get_autocast_dispatch_key_from_device_type(
          raw.device().type()));
  auto values = raw.to(torch::kFloat32).transpose(1, 2);
  auto [anchors, strides] = AnchorGrid(raw, image_size, options);
  const auto bins =
      torch::arange(options.reg_max,
                    raw.options().dtype(torch::kFloat32).requires_grad(false));
  const auto distances =
      options.reg_max == 1
          ? values.slice(2, 0, 4)
          : values.slice(2, 0, options.reg_max * 4)
                .reshape({raw.size(0), raw.size(2), 4, options.reg_max})
                .softmax(-1)
                .matmul(bins);
  auto boxes = torch::cat({anchors.unsqueeze(0) - distances.slice(2, 0, 2),
                           anchors.unsqueeze(0) + distances.slice(2, 2, 4)},
                          2) *
               strides;
  return {boxes, values.slice(2, options.reg_max * 4).sigmoid(), anchors,
          strides};
}

YoloDetectionLoss yolo_detection_loss(const Tensor& raw, const Tensor& targets,
                                      std::array<std::int64_t, 2> image_size,
                                      const YoloDetectionOptions& options) {
  ValidateRaw(raw, image_size, options);
  // Share one cast across all criterion branches. Casting each branch back
  // to FP16 separately rounds its adjoint before the branches accumulate.
  const auto fp32 = raw.to(torch::kFloat32);
  auto prediction = decode_yolo_detection(fp32, image_size, options);
  ValidateTargets(targets, raw, image_size, options);
  auto assigned = Assign(prediction, targets, options);
  auto values = fp32.transpose(1, 2);
  const auto denominator = assigned.scores.sum().clamp_min(1.0);
  auto classification =
      torch::nn::functional::binary_cross_entropy_with_logits(
          values.slice(2, options.reg_max * 4), assigned.scores,
          torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
              .weight(
                  detail::class_weight_tensor(options.class_weights, values))
              .reduction(torch::kSum)) /
      denominator;
  auto box_loss = values.sum() * 0.0;
  auto dfl_loss = values.sum() * 0.0;
  const auto mask = assigned.foreground;
  const auto count = mask.sum().item<std::int64_t>();
  if (count > 0) {
    const auto weight = assigned.scores.sum(-1).index({mask});
    // Stabilization is defined in feature-grid coordinates for regression;
    // assignment above still compares boxes in canvas coordinates.
    box_loss =
        ((1.0 -
          CompleteIou((prediction.boxes / prediction.strides).index({mask}),
                      (assigned.boxes / prediction.strides).index({mask}))) *
         weight)
            .sum() /
        denominator;
    if (options.reg_max > 1) {
      const auto anchor_grid = prediction.anchor_points.unsqueeze(0).expand(
          {raw.size(0), raw.size(2), 2});
      auto target_boxes = assigned.boxes / prediction.strides;
      const auto target_distance =
          torch::cat({anchor_grid - target_boxes.slice(2, 0, 2),
                      target_boxes.slice(2, 2, 4) - anchor_grid},
                     2)
              .index({mask})
              .clamp(0, options.reg_max - 1.01);
      const auto lower = target_distance.floor().to(torch::kInt64);
      const auto upper = lower + 1;
      const auto upper_weight = target_distance - lower.to(torch::kFloat32);
      const auto log_probs = values.slice(2, 0, options.reg_max * 4)
                                 .index({mask})
                                 .reshape({count, 4, options.reg_max})
                                 .log_softmax(-1);
      const auto lower_loss =
          -log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1);
      const auto upper_loss =
          -log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1);
      dfl_loss =
          ((lower_loss * (1.0 - upper_weight) + upper_loss * upper_weight)
               .mean(-1) *
           weight)
              .sum() /
          denominator;
    } else {
      auto anchors = prediction.anchor_points.unsqueeze(0).expand(
          {raw.size(0), raw.size(2), 2});
      auto target_grid = assigned.boxes / prediction.strides;
      auto desired = torch::cat({anchors - target_grid.slice(2, 0, 2),
                                 target_grid.slice(2, 2, 4) - anchors},
                                2);
      auto normalization =
          torch::tensor(std::vector<float>{static_cast<float>(image_size[1]),
                                           static_cast<float>(image_size[0]),
                                           static_cast<float>(image_size[1]),
                                           static_cast<float>(image_size[0])},
                        raw.options().dtype(torch::kFloat32));
      // Preserve the independent normalization of the two operands in
      // Ultralytics BboxLoss's direct-regression L1 path.
      auto error = (values.slice(2, 0, 4) * prediction.strides / normalization -
                    desired * prediction.strides / normalization)
                       .abs();
      dfl_loss = (error.index({mask}).mean(-1) * weight).sum() / denominator;
    }
  }
  auto total =
      (box_loss * options.box_gain + classification * options.class_gain +
       dfl_loss * options.dfl_gain) *
      raw.size(0);
  return {total, box_loss, classification, dfl_loss, count};
}

training::StepFunction make_yolo_detection_step(
    const YoloDetectionOptions& options) {
  ValidateOptions(options);
  return [options](const training::StepInput& input) {
    if (input.invocation.positional.size() != 1 ||
        !input.invocation.positional.front().isTensor() ||
        !input.target.has_value() || !input.target->isTensor()) {
      throw InvalidArgument(
          "YOLO detection step requires images and a [T,6] tensor target");
    }
    const auto images = input.invocation.positional.front().toTensor();
    if (images.dim() != 4) {
      throw InvalidArgument("YOLO step images must be NCHW");
    }
    auto result = input.model.invoke(input.invocation);
    if (!result.value.isTensor()) {
      throw InvalidArgument("YOLO model must return raw tensor predictions");
    }
    auto policy = options;
    if (input.context.phase == training::StepPhase::validation)
      policy.class_weights.clear();
    auto loss =
        yolo_detection_loss(result.value.toTensor(), input.target->toTensor(),
                            {images.size(2), images.size(3)}, policy);
    training::StepOutput output;
    // Training retains the criterion's batch scaling for backward. Validation
    // reports one per-image observation, which Trainer weights by sample count.
    output.loss = input.context.phase == training::StepPhase::validation
                      ? loss.total / static_cast<double>(images.size(0))
                      : loss.total;
    output.prediction = std::move(result);
    output.sample_count = static_cast<std::uint64_t>(images.size(0));
    output.metrics = {
        {"loss.box", loss.box.detach().item<double>()},
        {"loss.classification", loss.classification.detach().item<double>()},
        {"loss.distribution", loss.distribution.detach().item<double>()},
        {"assignment.foreground",
         static_cast<double>(loss.foreground_anchors)}};
    return output;
  };
}

}  // namespace jyppx::liuding::vision::tasks
