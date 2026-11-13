// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "training/class_weights.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/rotated.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>

#include <torch/nn/functional/loss.h>
#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <cmath>
#include <tuple>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
constexpr double eps = 1e-7;
Tensor Bce(const Tensor& logits, const Tensor& targets) {
  return torch::nn::functional::binary_cross_entropy_with_logits(
      logits, targets,
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
          .reduction(torch::kNone));
}
std::vector<Tensor> Outputs(const c10::IValue& value) {
  if (value.isTensor()) return {value.toTensor()};
  if (!value.isTuple())
    throw InvalidArgument(
        "Task model must return a tensor or fixed tensor tuple");
  std::vector<Tensor> result;
  for (const auto& element : value.toTupleRef().elements()) {
    if (!element.isTensor())
      throw InvalidArgument("Task raw tuple contains a non-tensor value");
    result.push_back(element.toTensor());
  }
  return result;
}
TaskTargets Targets(const c10::IValue& value) {
  if (value.isTensor()) return {.boxes = value.toTensor()};
  auto tensors = Outputs(value);
  if (tensors.size() != 4)
    throw InvalidArgument(
        "Rich targets require boxes/masks/keypoints/oriented_boxes tuple");
  return {tensors[0], tensors[1], tensors[2], tensors[3]};
}
void Validate(const YoloTaskOptions& o) {
  detail::check_class_weights(o.detection.class_weights,
                              o.detection.num_classes);
  if (o.task == TaskKind::classify)
    throw InvalidArgument("Use classification_loss for classification logits");
  if (o.mask_channels < 1 || o.mask_channels > 256 || o.keypoints < 1 ||
      o.keypoints > 256 ||
      (o.keypoint_dimensions != 2 && o.keypoint_dimensions != 3) ||
      o.max_mask_elements < 1 || o.schedule_epochs < 1 ||
      (o.generation26 && (!o.end_to_end || o.detection.reg_max != 1)))
    throw InvalidArgument(
        "Invalid task head or generation26 dual-branch recipe");
  for (auto gain :
       {o.mask_gain, o.pose_gain, o.visibility_gain, o.semantic_gain,
        o.flow_gain, o.one_to_many_initial, o.one_to_many_final})
    if (!std::isfinite(gain) || gain < 0)
      throw InvalidArgument("Task loss gains must be finite and nonnegative");
  if (!o.keypoint_sigmas.empty() &&
      o.keypoint_sigmas.size() != static_cast<std::size_t>(o.keypoints))
    throw InvalidArgument("Pose sigma count differs from keypoint count");
  for (auto sigma : o.keypoint_sigmas)
    if (!std::isfinite(sigma) || sigma <= 0)
      throw InvalidArgument("Pose sigmas must be positive finite");
}
Tensor Distances(const Tensor& raw, const YoloDetectionOptions& d) {
  auto values =
      raw.to(torch::kFloat32).transpose(1, 2).slice(2, 0, 4 * d.reg_max);
  if (d.reg_max == 1) return values;
  return (values.reshape({raw.size(0), raw.size(2), 4, d.reg_max}).softmax(-1) *
          torch::arange(d.reg_max, values.options()))
      .sum(-1);
}
Tensor Rotated(const Tensor& raw, const YoloDecodedPrediction& decoded,
               const YoloTaskOptions& o) {
  auto distances = Distances(raw, o.detection);
  auto angle = raw.select(1, 4 * o.detection.reg_max + o.detection.num_classes)
                   .to(torch::kFloat32);
  if (!o.generation26) angle = (angle.sigmoid() - .25) * std::acos(-1.);
  auto half_offset = (distances.slice(2, 2, 4) - distances.slice(2, 0, 2)) * .5;
  auto dx = half_offset.select(2, 0), dy = half_offset.select(2, 1);
  auto rotated_offset = torch::stack({dx * angle.cos() - dy * angle.sin(),
                                      dx * angle.sin() + dy * angle.cos()},
                                     -1);
  auto center = (decoded.anchor_points + rotated_offset) * decoded.strides;
  auto wh =
      (distances.slice(2, 0, 2) + distances.slice(2, 2, 4)) * decoded.strides;
  return torch::cat({center, wh, angle.unsqueeze(-1)}, -1);
}
// Rotated matching uses the target's rotated interior and probabilistic
// overlap; no axis-aligned IoU is substituted into rotated assignment or
// regression.
YoloAssignment RotatedAssignment(const Tensor& rotated,
                                 const YoloDecodedPrediction& p,
                                 const TaskTargets& t,
                                 const YoloDetectionOptions& d) {
  torch::NoGradGuard guard;
  const I batch = rotated.size(0), anchors = rotated.size(1);
  auto boxes = torch::zeros_like(p.boxes),
       scores = torch::zeros_like(p.class_scores);
  auto fg =
      torch::zeros({batch, anchors}, rotated.options().dtype(torch::kBool));
  auto ids =
      torch::full({batch, anchors}, -1, rotated.options().dtype(torch::kInt64));
  const auto centers = p.anchor_points * p.strides;
  for (I b = 0; b < batch; ++b) {
    auto rows = (t.boxes.select(1, 0) == b).nonzero().flatten();
    if (!rows.numel()) continue;
    if (rows.size(0) > d.max_assignment_pairs / anchors)
      throw ResourceExhausted("Rotated target/anchor matching bound exceeded");
    auto gt = t.oriented_boxes.index_select(0, rows).to(torch::kFloat32);
    auto classes = t.boxes.select(1, 1).index_select(0, rows).to(torch::kInt64);
    auto delta = centers.unsqueeze(0) - gt.slice(1, 0, 2).unsqueeze(1);
    auto angle = gt.select(1, 4).unsqueeze(1);
    auto localx =
        delta.select(2, 0) * angle.cos() + delta.select(2, 1) * angle.sin();
    auto localy =
        -delta.select(2, 0) * angle.sin() + delta.select(2, 1) * angle.cos();
    auto inside = (localx.abs() < gt.select(1, 2).unsqueeze(1) * .5) &
                  (localy.abs() < gt.select(1, 3).unsqueeze(1) * .5);
    auto overlap = vision_ops::probabilistic_box_iou(
                       gt.unsqueeze(1), rotated[b].detach().unsqueeze(0))
                       .clamp_min(0);
    auto score =
        p.class_scores[b].detach().index_select(1, classes).transpose(0, 1);
    auto alignment =
        score.pow(d.assignment_alpha) * overlap.pow(d.assignment_beta) * inside;
    auto top =
        std::get<1>(alignment.topk((std::min)(d.assignment_topk, anchors), 1));
    auto pos = torch::zeros_like(inside).scatter_(1, top, 1) & inside;
    auto conflicts = pos.sum(0) > 1;
    auto best = std::get<1>(overlap.max(0));
    pos = torch::where(conflicts.unsqueeze(0),
                       torch::zeros_like(pos).scatter_(0, best.unsqueeze(0), 1),
                       pos);
    auto selected = pos.any(0),
         matched = std::get<1>(pos.to(torch::kInt64).max(0));
    auto positive_alignment = alignment * pos;
    auto quality = std::get<0>(
        (positive_alignment * std::get<0>((overlap * pos).max(1, true)) /
         (std::get<0>(positive_alignment.max(1, true)) + eps))
            .max(0));
    auto anchors_selected = selected.nonzero().flatten();
    scores[b].index_put_(
        {anchors_selected,
         classes.index_select(0, matched).index_select(0, anchors_selected)},
        quality.index_select(0, anchors_selected));
    ids[b].index_put_(
        {anchors_selected},
        rows.index_select(0, matched.index_select(0, anchors_selected)));
    fg[b].copy_(selected);
  }
  return {boxes, scores, fg, ids};
}
TaskLoss Branch(const Tensor& raw, const Tensor& proto, const Tensor& semantic,
                const TaskTargets& t, std::array<I, 2> size,
                const YoloTaskOptions& o, const PoseLogProbability& flow,
                bool detached_proto) {
  const I base = 4 * o.detection.reg_max + o.detection.num_classes;
  const I extra =
      o.task == TaskKind::instance_segment ? o.mask_channels
      : o.task == TaskKind::pose
          ? o.keypoints * (o.keypoint_dimensions + (o.generation26 ? 2 : 0))
      : o.task == TaskKind::obb ? 1
                                : 0;
  if (!raw.defined() || raw.dim() != 3 || raw.size(1) != base + extra)
    throw InvalidArgument("Raw YOLO head width differs from task definition");
  auto head = raw.slice(1, 0, base);
  auto prediction = decode_yolo_detection(head, size, o.detection);
  TaskLoss result;
  auto zero = raw.to(torch::kFloat32).sum() * 0;
  YoloAssignment assigned;
  Tensor rotated;
  if (o.task == TaskKind::obb) {
    if (!t.oriented_boxes.defined() || t.oriented_boxes.dim() != 2 ||
        t.oriented_boxes.size(0) != t.boxes.size(0) ||
        t.oriented_boxes.size(1) != 5 ||
        t.oriented_boxes.device() != raw.device() ||
        !torch::isfinite(t.oriented_boxes).all().item<bool>() ||
        !(t.oriented_boxes.slice(1, 2, 4) > 0).all().item<bool>())
      throw InvalidArgument(
          "OBB targets require finite positive [T,5] rotated boxes");
    rotated = Rotated(raw, prediction, o);
    assigned = RotatedAssignment(rotated, prediction, t, o.detection);
    auto norm = assigned.scores.sum().clamp_min(1.);
    auto class_loss = Bce(head.transpose(1, 2)
                              .slice(2, 4 * o.detection.reg_max)
                              .to(torch::kFloat32),
                          assigned.scores);
    if (!o.detection.class_weights.empty())
      class_loss = class_loss *
                   detail::class_weight_tensor(o.detection.class_weights, head);
    result.components["classification"] = class_loss.sum() / norm;
    result.components["box"] = zero;
    result.components["distribution"] = zero;
    auto mask = assigned.foreground;
    if (mask.any().item<bool>()) {
      auto target = t.oriented_boxes
                        .index_select(0, assigned.target_indices.index({mask}))
                        .to(torch::kFloat32);
      auto weights = assigned.scores.sum(-1).index({mask});
      result.components["box"] = ((1 - vision_ops::probabilistic_box_iou(
                                           rotated.index({mask}), target)) *
                                  weights)
                                     .sum() /
                                 norm;
      auto anchors = (prediction.anchor_points * prediction.strides)
                         .unsqueeze(0)
                         .expand({raw.size(0), raw.size(2), 2})
                         .index({mask});
      auto delta = target.slice(1, 0, 2) - anchors;
      auto angle = target.select(1, 4);
      auto offset = torch::stack(
          {delta.select(1, 0) * angle.cos() + delta.select(1, 1) * angle.sin(),
           -delta.select(1, 0) * angle.sin() +
               delta.select(1, 1) * angle.cos()},
          1);
      auto stride = prediction.strides.unsqueeze(0)
                        .expand({raw.size(0), raw.size(2), 1})
                        .index({mask});
      auto desired = torch::cat({target.slice(1, 2, 4) * .5 - offset,
                                 target.slice(1, 2, 4) * .5 + offset},
                                1) /
                     stride;
      auto logits = head.transpose(1, 2)
                        .slice(2, 0, 4 * o.detection.reg_max)
                        .index({mask});
      Tensor distance_loss;
      if (o.detection.reg_max > 1) {
        desired = desired.clamp(0, o.detection.reg_max - 1.01);
        auto low = desired.floor().to(torch::kInt64), high = low + 1;
        auto upper = desired - low.to(torch::kFloat32);
        auto logp = logits.reshape({-1, 4, o.detection.reg_max})
                        .to(torch::kFloat32)
                        .log_softmax(-1);
        distance_loss =
            (-logp.gather(-1, low.unsqueeze(-1)).squeeze(-1) * (1 - upper) -
             logp.gather(-1, high.unsqueeze(-1)).squeeze(-1) * upper)
                .mean(-1);
      } else {
        auto canvas = torch::tensor(
            std::vector<float>{
                static_cast<float>(size[1]), static_cast<float>(size[0]),
                static_cast<float>(size[1]), static_cast<float>(size[0])},
            logits.options());
        distance_loss = ((logits - desired) * stride / canvas).abs().mean(-1);
      }
      result.components["distribution"] =
          (distance_loss * weights).sum() / norm;
    }
  } else {
    auto detection = yolo_detection_loss(head, t.boxes, size, o.detection);
    result.components = {{"box", detection.box},
                         {"classification", detection.classification},
                         {"distribution", detection.distribution}};
    assigned = assign_yolo_targets(prediction, t.boxes, o.detection);
  }
  auto positive = assigned.foreground.nonzero();
  const auto norm = assigned.scores.sum().clamp_min(1.);
  if (o.task == TaskKind::instance_segment) {
    if (!proto.defined() || proto.dim() != 4 || proto.size(0) != raw.size(0) ||
        proto.size(1) != o.mask_channels || proto.device() != raw.device() ||
        !t.masks.defined() || t.masks.dim() != 3 ||
        t.masks.size(0) != t.boxes.size(0) || t.masks.device() != raw.device())
      throw InvalidArgument(
          "Instance segmentation requires [B,M,H,W] prototypes and [T,H,W] "
          "instance masks");
    const I h = proto.size(2), w = proto.size(3);
    if (proto.numel() > o.max_mask_elements ||
        t.masks.numel() > o.max_mask_elements ||
        positive.size(0) > o.max_mask_elements / h / w)
      throw ResourceExhausted("Instance mask/prototype work bound exceeded");
    auto prototypes = detached_proto ? proto.detach() : proto;
    auto mask_loss = proto.sum() * 0;
    for (I i = 0; i < positive.size(0); ++i) {
      const I b = positive[i][0].item<I>(), a = positive[i][1].item<I>(),
              row = assigned.target_indices[b][a].item<I>();
      auto coefficients = raw[b]
                              .slice(0, base, base + o.mask_channels)
                              .select(1, a)
                              .to(torch::kFloat32);
      auto logits = torch::matmul(coefficients,
                                  prototypes[b].to(torch::kFloat32).flatten(1))
                        .view({h, w});
      auto target = torch::nn::functional::interpolate(
                        t.masks[row]
                            .to(torch::kFloat32)
                            .view({1, 1, t.masks.size(1), t.masks.size(2)}),
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<I>{h, w})
                            .mode(torch::kNearest))
                        .view({h, w});
      auto box = t.boxes[row].slice(0, 2, 6).to(torch::kFloat32);
      auto xs = torch::arange(w, logits.options()).view({1, -1}),
           ys = torch::arange(h, logits.options()).view({-1, 1});
      auto inside = (xs >= box[0] * w / size[1]) & (xs < box[2] * w / size[1]) &
                    (ys >= box[1] * h / size[0]) & (ys < box[3] * h / size[0]);
      auto area = inside.sum().clamp_min(1).to(torch::kFloat32);
      mask_loss = mask_loss + (Bce(logits, target) * inside).sum() / area *
                                  assigned.scores[b][a].sum();
    }
    result.components["mask"] = mask_loss / norm;
    if (o.generation26 && semantic.defined() && !detached_proto) {
      auto semantic_target = torch::zeros_like(semantic);
      for (I row = 0; row < t.boxes.size(0); ++row) {
        const I b = t.boxes[row][0].item<I>(), c = t.boxes[row][1].item<I>();
        auto mask =
            torch::nn::functional::interpolate(
                t.masks[row]
                    .to(torch::kFloat32)
                    .view({1, 1, t.masks.size(1), t.masks.size(2)}),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<I>{semantic.size(2), semantic.size(3)})
                    .mode(torch::kNearest))
                .view({semantic.size(2), semantic.size(3)});
        semantic_target[b][c].copy_(
            torch::maximum(semantic_target[b][c], mask));
      }
      result.components["semantic"] =
          Bce(semantic.to(torch::kFloat32), semantic_target).mean();
    }
  } else if (o.task == TaskKind::pose) {
    if (!t.keypoints.defined() || t.keypoints.dim() != 3 ||
        t.keypoints.size(0) != t.boxes.size(0) ||
        t.keypoints.size(1) != o.keypoints || t.keypoints.size(2) != 3 ||
        t.keypoints.device() != raw.device() ||
        !torch::isfinite(t.keypoints).all().item<bool>())
      throw InvalidArgument(
          "Pose targets require finite [T,K,3] xy/visibility");
    auto pose_loss = zero, visibility = zero, rle = zero;
    auto sigmas = o.keypoint_sigmas;
    if (sigmas.empty())
      sigmas.assign(static_cast<std::size_t>(o.keypoints), 1. / o.keypoints);
    auto sigma_tensor =
        torch::tensor(sigmas, raw.options().dtype(torch::kFloat32));
    for (I i = 0; i < positive.size(0); ++i) {
      const I b = positive[i][0].item<I>(), a = positive[i][1].item<I>(),
              row = assigned.target_indices[b][a].item<I>();
      auto kpt = raw[b]
                     .slice(0, base, base + o.keypoints * o.keypoint_dimensions)
                     .select(1, a)
                     .view({o.keypoints, o.keypoint_dimensions})
                     .to(torch::kFloat32);
      auto stride = prediction.strides[a];
      auto xy = (kpt.slice(1, 0, 2) * (o.generation26 ? 1. : 2.) +
                 prediction.anchor_points[a] - (o.generation26 ? 0. : .5)) *
                stride;
      auto target = t.keypoints[row].to(torch::kFloat32);
      auto visible = target.select(1, 2) > 0;
      auto weight = assigned.scores[b][a].sum();
      auto box = t.boxes[row];
      auto area = ((box[4] - box[2]) * (box[5] - box[3])).clamp_min(eps);
      auto distance = (xy - target.slice(1, 0, 2)).square().sum(1);
      auto oks =
          1 - torch::exp(-distance / (8 * sigma_tensor.square() * area + eps));
      pose_loss = pose_loss +
                  (oks * visible).sum() / visible.sum().clamp_min(1) * weight;
      if (o.keypoint_dimensions == 3)
        visibility =
            visibility +
            Bce(kpt.select(1, 2), visible.to(torch::kFloat32)).mean() * weight;
      if (o.generation26) {
        if (!flow)
          throw InvalidArgument(
              "YOLO26 pose training requires its model-owned flow binding");
        auto sigma = raw[b]
                         .slice(0, base + o.keypoints * o.keypoint_dimensions)
                         .select(1, a)
                         .view({o.keypoints, 2})
                         .to(torch::kFloat32)
                         .sigmoid()
                         .clamp_min(eps);
        auto residual = (xy - target.slice(1, 0, 2)) / stride / sigma;
        auto log_probability = flow(residual);
        auto likelihood = sigma.log() - log_probability.unsqueeze(1) +
                          (2 * sigma).log() + residual.abs();
        rle = rle + (likelihood * visible.unsqueeze(1)).sum() /
                        visible.sum().clamp_min(1) * weight;
      }
    }
    result.components["pose"] = pose_loss / norm;
    result.components["visibility"] = visibility / norm;
    if (o.generation26) result.components["flow"] = rle / norm;
  }
  result.total = zero;
  for (const auto& [name, value] : result.components) {
    const double gain = name == "box"              ? o.detection.box_gain
                        : name == "classification" ? o.detection.class_gain
                        : name == "distribution"   ? o.detection.dfl_gain
                        : name == "mask"           ? o.mask_gain
                        : name == "pose"           ? o.pose_gain
                        : name == "visibility"     ? o.visibility_gain
                        : name == "flow"           ? o.flow_gain
                                                   : o.semantic_gain;
    result.total = result.total + value * gain;
  }
  result.total = result.total * raw.size(0);
  return result;
}
}  // namespace
TaskLoss yolo_task_loss(std::span<const Tensor> outputs,
                        const TaskTargets& targets, std::array<I, 2> size,
                        const YoloTaskOptions& o, std::uint64_t epoch,
                        PoseLogProbability pose_flow) {
  Validate(o);
  const std::size_t branch_count = o.end_to_end ? 2 : 1;
  const std::size_t expected =
      branch_count +
      (o.task == TaskKind::instance_segment ? (o.generation26 ? 2 : 1) : 0);
  if (outputs.size() != expected)
    throw InvalidArgument(
        "Raw output tuple differs from declared task profile");
  if (!targets.boxes.defined() || targets.boxes.dim() != 2 ||
      targets.boxes.size(1) != 6 || !targets.boxes.is_floating_point() ||
      targets.boxes.device() != outputs[0].device() ||
      targets.boxes.size(0) > o.detection.max_targets ||
      !torch::isfinite(targets.boxes).all().item<bool>())
    throw InvalidArgument(
        "Task boxes require bounded finite [T,6] on prediction device");
  if (targets.boxes.numel()) {
    const auto& t = targets.boxes;
    if (!(t.slice(1, 0, 2) == t.slice(1, 0, 2).round()).all().item<bool>() ||
        t.select(1, 0).min().item<double>() < 0 ||
        t.select(1, 0).max().item<double>() >= outputs[0].size(0) ||
        t.select(1, 1).min().item<double>() < 0 ||
        t.select(1, 1).max().item<double>() >= o.detection.num_classes ||
        !(t.slice(1, 4, 6) > t.slice(1, 2, 4)).all().item<bool>() ||
        !(t.slice(1, 2, 4) >= 0).all().item<bool>() ||
        !(t.select(1, 4) <= size[1]).all().item<bool>() ||
        !(t.select(1, 5) <= size[0]).all().item<bool>())
      throw InvalidArgument(
          "Invalid task target indices or canvas coordinates");
  }
  Tensor proto, semantic;
  if (o.task == TaskKind::instance_segment) {
    proto = outputs[branch_count];
    if (o.generation26) semantic = outputs[branch_count + 1];
  }
  auto result =
      Branch(outputs[0], proto, semantic, targets, size, o, pose_flow, false);
  if (o.end_to_end) {
    auto single = o;
    single.detection.assignment_topk = 1;
    auto one = Branch(outputs[1], proto, semantic, targets, size, single,
                      pose_flow, true);
    const double fraction =
        (std::min)(1., static_cast<double>(epoch) / o.schedule_epochs);
    const double weight =
        o.one_to_many_initial +
        (o.one_to_many_final - o.one_to_many_initial) * fraction;
    result.total = result.total * weight + one.total;
    auto components = result.components;
    result.components.clear();
    for (auto& [name, value] : components)
      result.components["one_to_many." + name] = value;
    for (auto& [name, value] : one.components)
      result.components["one_to_one." + name] = value;
  }
  return result;
}
Tensor classification_loss(const Tensor& logits, const Tensor& target,
                           double smoothing, const Tensor& weights) {
  if (!logits.defined() || logits.dim() != 2 || !logits.is_floating_point() ||
      logits.size(0) < 1 || logits.size(1) < 1 || !target.defined() ||
      target.device() != logits.device() || !std::isfinite(smoothing) ||
      smoothing < 0 || smoothing >= 1)
    throw InvalidArgument("Invalid classification logits/targets/smoothing");
  const auto values = logits.to(torch::kFloat32);
  Tensor probabilities;
  if (target.dim() == 1 && target.scalar_type() == torch::kInt64 &&
      target.size(0) == logits.size(0)) {
    if (target.min().item<I>() < 0 || target.max().item<I>() >= logits.size(1))
      throw InvalidArgument("Classification class index out of range");
    probabilities =
        torch::zeros_like(values).scatter_(1, target.unsqueeze(1), 1.);
  } else if (target.sizes() == logits.sizes() && target.is_floating_point()) {
    probabilities = target.to(torch::kFloat32);
    if (!torch::isfinite(probabilities).all().item<bool>() ||
        !(probabilities >= 0).all().item<bool>() ||
        !(probabilities.sum(1) - 1).abs().le(1e-5).all().item<bool>())
      throw InvalidArgument(
          "Soft class labels must be probability distributions");
  } else
    throw InvalidArgument(
        "Classification targets must be hard [B] int64 or soft [B,C]");
  probabilities = probabilities * (1 - smoothing) + smoothing / logits.size(1);
  if (weights.defined()) {
    if (weights.dim() != 1 || weights.size(0) != logits.size(1) ||
        weights.device() != logits.device() ||
        !torch::isfinite(weights).all().item<bool>() ||
        !(weights >= 0).all().item<bool>())
      throw InvalidArgument("Invalid fixed classification class weights");
    probabilities = probabilities * weights;
  }
  return -(probabilities * values.log_softmax(1)).sum(1).mean();
}
training::StepFunction make_yolo_task_step(const YoloTaskOptions& options,
                                           PoseLogProbability flow) {
  Validate(options);
  return [options, flow = std::move(flow)](const training::StepInput& input) {
    if (!input.target || input.invocation.positional.empty())
      throw InvalidArgument("YOLO task requires images and task targets");
    auto images = input.invocation.positional[0].toTensor();
    auto prediction = input.model.invoke(input.invocation);
    auto outputs = Outputs(prediction.value);
    auto policy = options;
    if (input.context.phase == training::StepPhase::validation)
      policy.detection.class_weights.clear();
    auto loss = yolo_task_loss(outputs, Targets(*input.target),
                               {images.size(2), images.size(3)}, policy,
                               input.context.epoch, flow);
    training::StepOutput result;
    result.loss = input.context.phase == training::StepPhase::validation
                      ? loss.total / images.size(0)
                      : loss.total;
    result.sample_count = static_cast<std::uint64_t>(images.size(0));
    result.prediction = std::move(prediction);
    for (const auto& [name, value] : loss.components)
      result.metrics["loss." + name] = value.detach().item<double>();
    return result;
  };
}
training::StepFunction make_classification_step(double smoothing,
                                                const Tensor& weights) {
  auto fixed = weights.defined() ? weights.detach().clone() : Tensor{};
  return [smoothing, fixed](const training::StepInput& input) {
    if (!input.target || !input.target->isTensor())
      throw InvalidArgument("Classification requires tensor labels");
    auto prediction = input.model.invoke(input.invocation);
    auto logits = Outputs(prediction.value);
    if (logits.size() != 1)
      throw InvalidArgument("Classification must return one logits tensor");
    training::StepOutput result;
    result.loss = classification_loss(logits[0], input.target->toTensor(),
                                      smoothing, fixed);
    result.sample_count = static_cast<std::uint64_t>(logits[0].size(0));
    result.prediction = std::move(prediction);
    return result;
  };
}
}  // namespace jyppx::liuding::vision::tasks
