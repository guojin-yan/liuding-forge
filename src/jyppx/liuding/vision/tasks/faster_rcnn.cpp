// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "training/class_weights.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <jyppx/liuding/vision/tasks/faster_rcnn.hpp>

#include <torch/nn/functional/loss.h>

#include <algorithm>
#include <cmath>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
Tensor Sample(const Tensor& mask, I maximum, bool random) {
  auto ids = mask.nonzero().flatten();
  if (ids.size(0) <= maximum) return ids;
  if (!random) return ids.slice(0, 0, maximum);
  return ids.index_select(
      0, torch::randperm(ids.size(0), ids.options()).slice(0, 0, maximum));
}
Tensor Smooth(const Tensor& error, double beta) {
  auto absolute = error.abs();
  return beta == 0 ? absolute
                   : torch::where(absolute < beta, .5 * error.square() / beta,
                                  absolute - .5 * beta);
}
void Validate(const FasterRcnnTaskOptions& o) {
  detail::check_class_weights(o.class_weights, o.num_classes);
  if (o.num_classes < 1 || o.num_classes > 10000 || o.rpn_batch_size < 1 ||
      o.rpn_batch_size > 4096 || o.max_assignment_pairs < 1 ||
      o.max_targets < 1 || !std::isfinite(o.rpn_positive_fraction) ||
      o.rpn_positive_fraction < 0 || o.rpn_positive_fraction > 1 ||
      !std::isfinite(o.rpn_foreground_iou) ||
      !std::isfinite(o.rpn_background_iou) || o.rpn_background_iou < 0 ||
      o.rpn_foreground_iou > 1 ||
      o.rpn_foreground_iou <= o.rpn_background_iou ||
      !std::isfinite(o.smooth_l1_beta) || o.smooth_l1_beta < 0)
    throw InvalidArgument("Invalid Faster R-CNN loss/sampling recipe");
}
}  // namespace
TaskLoss faster_rcnn_loss(std::span<const Tensor> x, const Tensor& targets,
                          const FasterRcnnTaskOptions& o, bool training) {
  Validate(o);
  if (x.size() != 9 || !targets.defined() || targets.dim() != 2 ||
      targets.size(1) != 6 || !targets.is_floating_point() ||
      targets.size(0) > o.max_targets)
    throw InvalidArgument(
        "Faster R-CNN loss expects nine raw tensors and [T,6] targets");
  const auto logits = x[0].to(torch::kFloat32),
             deltas = x[1].to(torch::kFloat32),
             anchors = x[2].to(torch::kFloat32);
  const auto proposals = x[3].to(torch::kFloat32),
             classes = x[4].to(torch::kFloat32),
             boxes = x[5].to(torch::kFloat32);
  if (logits.dim() != 2 || deltas.dim() != 3 || anchors.dim() != 2 ||
      anchors.size(1) != 4 || deltas.size(2) != 4 ||
      deltas.size(0) != logits.size(0) || deltas.size(1) != logits.size(1) ||
      anchors.size(0) != logits.size(1) || classes.dim() != 2 ||
      classes.size(1) != o.num_classes + 1 || proposals.dim() != 2 ||
      proposals.size(1) != 5 || classes.size(0) != proposals.size(0) ||
      boxes.dim() != 2 || boxes.size(0) != classes.size(0) ||
      boxes.size(1) != 4 * (o.num_classes + 1) ||
      targets.device() != logits.device() ||
      !torch::isfinite(targets).all().item<bool>())
    throw InvalidArgument("Invalid Faster R-CNN raw shape/device contract");
  if (targets.numel() &&
      (!(targets.slice(1, 0, 2) == targets.slice(1, 0, 2).round())
            .all()
            .item<bool>() ||
       targets.select(1, 0).min().item<double>() < 0 ||
       targets.select(1, 0).max().item<double>() >= logits.size(0) ||
       targets.select(1, 1).min().item<double>() < 0 ||
       targets.select(1, 1).max().item<double>() >= o.num_classes ||
       !(targets.slice(1, 4, 6) > targets.slice(1, 2, 4)).all().item<bool>()))
    throw InvalidArgument("Invalid Faster R-CNN target identities/extents");
  auto zero =
      logits.sum() * 0 + deltas.sum() * 0 + classes.sum() * 0 + boxes.sum() * 0;
  auto rpn_objectness = zero, rpn_box = zero;
  I sampled_count = 0;
  for (I b = 0; b < logits.size(0); ++b) {
    auto gt = targets.index({targets.select(1, 0) == b})
                  .slice(1, 2, 6)
                  .to(torch::kFloat32);
    auto labels =
        torch::zeros({anchors.size(0)}, logits.options().dtype(torch::kInt64));
    auto matched = torch::zeros_like(anchors);
    {
      torch::NoGradGuard guard;
      if (gt.size(0)) {
        auto iou = vision_ops::pairwise_box_iou(
            anchors, gt, {.max_pairs = o.max_assignment_pairs});
        auto best = iou.max(1);
        labels.fill_(-1);
        labels.index_put_({std::get<0>(best) < o.rpn_background_iou}, 0);
        labels.index_put_({std::get<0>(best) >= o.rpn_foreground_iou}, 1);
        // Preserve ties for each GT's best anchor, including zero-IoU ties.
        auto low_quality = (iou == std::get<0>(iou.max(0, true))).any(1);
        labels.index_put_({low_quality}, 1);
        matched = gt.index_select(0, std::get<1>(best));
      }
    }
    auto pos = Sample(
        labels == 1, static_cast<I>(o.rpn_batch_size * o.rpn_positive_fraction),
        training);
    auto neg = Sample(labels == 0, o.rpn_batch_size - pos.size(0), training);
    auto all = torch::cat({pos, neg});
    sampled_count += all.size(0);
    if (all.numel())
      rpn_objectness =
          rpn_objectness +
          torch::nn::functional::binary_cross_entropy_with_logits(
              logits[b].index_select(0, all),
              labels.index_select(0, all).to(torch::kFloat32),
              torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                  .reduction(torch::kSum));
    if (pos.numel()) {
      auto expected = vision_ops::encode_box_deltas(
          anchors.index_select(0, pos), matched.index_select(0, pos));
      rpn_box = rpn_box + Smooth(deltas[b].index_select(0, pos) - expected,
                                 o.smooth_l1_beta)
                              .sum();
    }
  }
  rpn_objectness = rpn_objectness / (std::max)(I{1}, sampled_count);
  rpn_box = rpn_box / (std::max)(I{1}, sampled_count);
  auto labels = x[6].to(torch::kInt64).clone(),
       matched = x[7].to(torch::kFloat32).clone();
  if (!training) {
    torch::NoGradGuard guard;
    labels = torch::zeros({proposals.size(0)},
                          logits.options().dtype(torch::kInt64));
    matched = torch::zeros({proposals.size(0), 4}, logits.options());
    for (I b = 0; b < logits.size(0); ++b) {
      auto rows = (proposals.select(1, 0) == b).nonzero().flatten();
      auto gt = targets.index({targets.select(1, 0) == b});
      if (!rows.numel() || !gt.size(0)) continue;
      auto iou = vision_ops::pairwise_box_iou(
          proposals.index_select(0, rows).slice(1, 1, 5),
          gt.slice(1, 2, 6).to(torch::kFloat32),
          {.max_pairs = o.max_assignment_pairs});
      auto best = iou.max(1);
      auto ids = std::get<1>(best);
      auto chosen = gt.select(1, 1).to(torch::kInt64).index_select(0, ids) + 1;
      labels.index_copy_(0, rows,
                         torch::where(std::get<0>(best) >= .5, chosen,
                                      torch::zeros_like(chosen)));
      matched.index_copy_(
          0, rows, gt.slice(1, 2, 6).to(torch::kFloat32).index_select(0, ids));
    }
  }
  auto roi_class = zero, roi_box = zero;
  if (labels.numel()) {
    if (labels.min().item<I>() < 0 || labels.max().item<I>() > o.num_classes)
      throw InvalidArgument(
          "Training ROI labels were not matched by the detector");
    Tensor weights;
    if (training && !o.class_weights.empty())
      weights =
          torch::cat({torch::ones({1}, classes.options()),
                      detail::class_weight_tensor(o.class_weights, classes)});
    roi_class = torch::nn::functional::cross_entropy(
        classes, labels,
        torch::nn::functional::CrossEntropyFuncOptions().weight(weights));
    auto positive = (labels > 0).nonzero().flatten();
    if (positive.numel()) {
      auto expected = vision_ops::encode_box_deltas(
          proposals.index_select(0, positive).slice(1, 1, 5),
          matched.index_select(0, positive), {10, 10, 5, 5});
      auto predicted = boxes.view({-1, o.num_classes + 1, 4})
                           .index({positive, labels.index_select(0, positive)});
      roi_box =
          Smooth(predicted - expected, o.smooth_l1_beta).sum() / labels.size(0);
    }
  }
  return {rpn_objectness + rpn_box + roi_class + roi_box,
          {{"rpn.objectness", rpn_objectness},
           {"rpn.box", rpn_box},
           {"roi.classification", roi_class},
           {"roi.box", roi_box}}};
}
training::StepFunction make_faster_rcnn_step(const FasterRcnnTaskOptions& o) {
  Validate(o);
  return [o](const training::StepInput& input) {
    if (input.invocation.positional.size() != 3)
      throw InvalidArgument(
          "Faster R-CNN invocation requires images/targets/image_sizes");
    auto prediction = input.model.invoke(input.invocation);
    if (!prediction.value.isTuple())
      throw InvalidArgument("Faster R-CNN raw output must be a tuple");
    std::vector<Tensor> raw;
    for (const auto& x : prediction.value.toTupleRef().elements())
      raw.push_back(x.toTensor());
    auto loss =
        faster_rcnn_loss(raw, input.invocation.positional[1].toTensor(), o,
                         input.context.phase == training::StepPhase::training);
    training::StepOutput result;
    result.loss = loss.total;
    result.prediction = std::move(prediction);
    result.sample_count = static_cast<std::uint64_t>(raw[0].size(0));
    for (auto& [name, value] : loss.components)
      result.metrics["loss." + name] = value.detach().item<double>();
    return result;
  };
}
}  // namespace jyppx::liuding::vision::tasks
