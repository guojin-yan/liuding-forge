// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../ops/target_grouping.hpp"

#include <jyppx/liuding/vision/ops/assignment.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/deformable.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <jyppx/liuding/vision/tasks/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/nn/functional/upsampling.h>
#include <torch/torch.h>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
Tensor Xyxy(const Tensor& x) {
  return torch::cat({x.slice(-1, 0, 2) - x.slice(-1, 2, 4) * .5,
                     x.slice(-1, 0, 2) + x.slice(-1, 2, 4) * .5},
                    -1);
}
Tensor GIoU(const Tensor& a, const Tensor& b) {
  auto aa = Xyxy(a), bb = Xyxy(b);
  auto lo = torch::maximum(aa.slice(1, 0, 2).unsqueeze(1),
                           bb.slice(1, 0, 2).unsqueeze(0)),
       hi = torch::minimum(aa.slice(1, 2, 4).unsqueeze(1),
                           bb.slice(1, 2, 4).unsqueeze(0));
  auto inter = (hi - lo).clamp_min(0).prod(-1);
  auto area = a.slice(1, 2, 4).prod(-1).unsqueeze(1) +
              b.slice(1, 2, 4).prod(-1).unsqueeze(0) - inter;
  auto clo = torch::minimum(aa.slice(1, 0, 2).unsqueeze(1),
                            bb.slice(1, 0, 2).unsqueeze(0)),
       chi = torch::maximum(aa.slice(1, 2, 4).unsqueeze(1),
                            bb.slice(1, 2, 4).unsqueeze(0));
  auto cover = (chi - clo).clamp_min(0).prod(-1).clamp_min(1e-7);
  return inter / area.clamp_min(1e-7) - (cover - area) / cover;
}
struct Loss {
  Tensor classification, box, giou;
};
Loss One(const Tensor& raw, const Tensor& prediction,
         const std::vector<Tensor>& labels, const std::vector<Tensor>& boxes,
         double denominator, const Tensor& mapping = {},
         const Tensor& positive = {}, bool include_padding = false) {
  auto logits = raw.to(torch::kFloat32), pred = prediction.to(torch::kFloat32);
  const auto queries = logits.size(1);
  std::vector<I> offsets;
  I total = 0;
  for (const auto& label : labels) {
    offsets.push_back(total);
    total += label.numel();
  }
  Tensor row, column;
  {
    torch::NoGradGuard guard;
    if (mapping.defined()) {
      // One dynamic selection for the batch, instead of one per image.
      row = (positive & (mapping >= 0)).flatten().nonzero().flatten();
      auto offset = torch::tensor(offsets, torch::kInt64).to(pred.device());
      column = mapping.flatten().index_select(0, row) +
               offset.index_select(0, torch::floor_divide(row, queries));
    } else {
      std::vector<Tensor> costs;
      for (I b = 0; b < logits.size(0); ++b) {
        if (!labels[b].numel()) continue;
        if (queries && labels[b].numel() > 1048576 / queries)
          throw ResourceExhausted("Assignment pair limit exceeded");
        auto p = logits[b].sigmoid();
        auto neg = .75 * p.square() * (-(1 - p + 1e-8).log()),
             pos = .25 * (1 - p).square() * (-(p + 1e-8).log());
        costs.push_back(((pos - neg).index_select(1, labels[b]) * 2 +
                         torch::cdist(pred[b], boxes[b], 1) * 5 -
                         GIoU(pred[b], boxes[b]) * 2)
                            .flatten());
      }
      std::vector<Tensor> rows, columns;
      if (!costs.empty()) {
        // Matching remains independently bounded per image. Transfer all costs
        // once, then let the existing CPU solver inspect contiguous views.
        auto snapshot = torch::cat(costs).to(torch::kCPU, torch::kFloat64);
        I cursor = 0;
        for (I b = 0; b < logits.size(0); ++b) {
          if (!labels[b].numel()) continue;
          const auto pairs = queries * labels[b].numel();
          auto match = vision_ops::linear_assignment(
              snapshot.narrow(0, cursor, pairs)
                  .view({queries, labels[b].numel()}),
              {.max_pairs = 1048576, .max_work = 200000000});
          cursor += pairs;
          rows.push_back(match.rows + b * queries);
          columns.push_back(match.columns + offsets[b]);
        }
      }
      auto empty = torch::empty({0}, torch::kInt64);
      row = (rows.empty() ? empty : torch::cat(rows)).to(pred.device());
      column =
          (columns.empty() ? empty : torch::cat(columns)).to(pred.device());
    }
  }
  logits = logits.flatten(0, 1);
  pred = pred.flatten(0, 1);
  auto y = torch::zeros_like(logits), score = torch::zeros_like(y);
  auto l1 = pred.sum() * 0, giou = pred.sum() * 0;
  if (row.numel()) {
    auto pb = pred.index_select(0, row),
         gt = torch::cat(boxes).index_select(0, column),
         ids = torch::cat(labels).index_select(0, column);
    auto pb_xyxy = Xyxy(pb), gt_xyxy = Xyxy(gt);
    auto iou = vision_ops::aligned_box_iou(pb_xyxy.detach(), gt_xyxy);
    y.index_put_({row, ids}, 1);
    score.index_put_({row, ids}, iou);
    l1 = l1 + (pb - gt).abs().sum();
    giou = giou +
           (1 - vision_ops::aligned_generalized_iou(pb_xyxy, gt_xyxy)).sum();
  }
  auto weight = (.75 * logits.sigmoid().square() * (1 - y) + score).detach();
  auto loss =
      torch::nn::functional::binary_cross_entropy_with_logits(
          logits, score,
          torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
              .reduction(torch::kNone)) *
      weight;
  if (mapping.defined() && !include_padding)
    loss = loss * (mapping >= 0).flatten().unsqueeze(1);
  return {loss.sum() / denominator, l1 / denominator, giou / denominator};
}
}  // namespace
TaskLoss rtdetr_loss(std::span<const Tensor> x, const Tensor& targets,
                     std::array<I, 2> canvas, I classes, bool include_padding) {
  if (x.size() != 8 || x[0].dim() != 4 || x[1].dim() != 4 ||
      x[0].size(3) != classes || x[1].size(3) != 4 || targets.dim() != 2 ||
      targets.size(1) != 6 || canvas[0] < 1 || canvas[1] < 1)
    throw InvalidArgument("Invalid RT-DETR loss contract");
  for (const auto& value : x)
    if (value.device() != x[0].device() ||
        !torch::isfinite(value).all().item<bool>())
      throw InvalidArgument("RT-DETR outputs must be finite on one device");
  std::vector<Tensor> labels, boxes;
  auto scale = torch::tensor(
      {static_cast<float>(canvas[1]), static_cast<float>(canvas[0]),
       static_cast<float>(canvas[1]), static_cast<float>(canvas[0])},
      targets.options());
  for (const auto& gt :
       vision::detail::group_targets_by_image(targets, x[0].size(1))) {
    labels.push_back(gt.select(1, 1).to(torch::kInt64));
    auto xyxy = gt.slice(1, 2, 6) / scale;
    boxes.push_back(
        torch::cat({(xyxy.slice(1, 0, 2) + xyxy.slice(1, 2, 4)) * .5,
                    xyxy.slice(1, 2, 4) - xyxy.slice(1, 0, 2)},
                   1));
  }
  double denominator = static_cast<double>((std::max)(I{1}, targets.size(0)));
  auto cls = x[0].sum() * 0, box = x[1].sum() * 0, giou = box;
  for (I l = 0; l <= x[0].size(0); ++l) {
    auto loss = l == x[0].size(0)
                    ? One(x[2], x[3], labels, boxes, denominator)
                    : One(x[0][l], x[1][l], labels, boxes, denominator);
    cls = cls + loss.classification;
    box = box + loss.box;
    giou = giou + loss.giou;
  }
  if (x[6].size(1)) {
    double groups = static_cast<double>(x[7].sum().item<I>()) /
                    (std::max)(I{1}, targets.size(0));
    for (I l = 0; l < x[4].size(0); ++l) {
      auto loss = One(x[4][l], x[5][l], labels, boxes,
                      denominator * (std::max)(1., groups), x[6], x[7],
                      include_padding);
      cls = cls + loss.classification;
      box = box + loss.box;
      giou = giou + loss.giou;
    }
  }
  return {cls + box * 5 + giou * 2,
          {{"vfl", cls}, {"l1", box}, {"giou", giou}}};
}
training::StepFunction make_rtdetr_step(I classes, bool include_padding,
                                        std::vector<I> sizes,
                                        std::uint64_t seed) {
  if (sizes.size() > 256)
    throw InvalidArgument("Too many RT-DETR training sizes");
  for (auto size : sizes)
    if (size < 32 || size > 2048 || size % 32)
      throw InvalidArgument("Invalid RT-DETR training size");
  return [classes, include_padding, sizes = std::move(sizes),
          seed](const training::StepInput& in) {
    if (!in.target || !in.target->isTensor())
      throw InvalidArgument("RT-DETR needs box targets");
    auto image = in.invocation.positional[0].toTensor();
    auto invocation = in.invocation;
    auto target = in.target->toTensor();
    if (!sizes.empty() && in.context.phase == training::StepPhase::training) {
      data::image::SampleRandom random(
          {.dataset_id = "rtdetr.model-multiscale/v1",
           .sample_id = std::to_string(in.context.batch_sequence),
           .transform_id = "nearest/v1",
           .seed = seed,
           .epoch = in.context.epoch});
      const auto size = sizes[random.bounded(sizes.size(), 0)];
      const auto height = image.size(2), width = image.size(3);
      image = torch::nn::functional::interpolate(
          image, torch::nn::functional::InterpolateFuncOptions()
                     .size(std::vector<I>{size, size})
                     .mode(torch::kNearest));
      const auto scale = torch::tensor(
          {1.f, 1.f, static_cast<float>(size) / width,
           static_cast<float>(size) / height, static_cast<float>(size) / width,
           static_cast<float>(size) / height},
          target.options());
      target = target * scale;
      invocation.positional[0] = image;
      invocation.positional[1] = target;
    }
    auto prediction = in.model.invoke(invocation);
    auto loss =
        rtdetr_loss(raw_tensors(prediction.value), target,
                    {image.size(2), image.size(3)}, classes, include_padding);
    training::StepOutput out;
    out.loss = loss.total;
    out.prediction = std::move(prediction);
    out.sample_count = image.size(0);
    return out;
  };
}
training::BatchMapper make_rtdetr_image_mapper(const YoloDataOptions& data) {
  return [data](const data::BatchEnvelope& e) {
    auto b = data::image::image_batch_from_value(e.value);
    return nn::Invocation{{b.images, build_yolo_targets(b, data)}, {}};
  };
}
std::vector<ImageResult> decode_rtdetr(std::span<const Tensor> x,
                                       std::array<I, 2> canvas,
                                       const ResultOptions& o) {
  if (x.size() != 8 || o.max_detections < 1 || o.max_candidates < 1 ||
      o.confidence < 0 || o.confidence > 1)
    throw InvalidArgument("Invalid RT-DETR result contract");
  torch::NoGradGuard guard;
  auto logits = x[0][-1].to(torch::kFloat32),
       box = Xyxy(x[1][-1].to(torch::kFloat32));
  auto scale = torch::tensor(
      {static_cast<float>(canvas[1]), static_cast<float>(canvas[0]),
       static_cast<float>(canvas[1]), static_cast<float>(canvas[0])},
      box.options());
  box = box * scale;
  std::vector<ImageResult> out;
  for (I b = 0; b < logits.size(0); ++b) {
    auto scores = logits[b].sigmoid();
    if (!o.classes.empty()) {
      auto keep =
          torch::zeros({scores.size(1)}, scores.options().dtype(torch::kBool));
      for (auto c : o.classes) {
        if (c < 0 || c >= scores.size(1))
          throw InvalidArgument("RT-DETR result class filter invalid");
        keep[c] = true;
      }
      scores = scores.masked_fill(~keep.unsqueeze(0), -1);
    }
    Tensor ids, values;
    if (o.multi_label) {
      auto top =
          scores.flatten().topk((std::min)(scores.numel(), o.max_detections));
      values = std::get<0>(top);
      ids = std::get<1>(top);
    } else {
      auto best = scores.max(1);
      auto top =
          std::get<0>(best).topk((std::min)(scores.size(0), o.max_detections));
      values = std::get<0>(top);
      auto query = std::get<1>(top);
      ids = query * scores.size(1) + std::get<1>(best).index_select(0, query);
    }
    auto keep = (values >= o.confidence).nonzero().flatten();
    values = values.index_select(0, keep);
    ids = ids.index_select(0, keep);
    auto query = torch::floor_divide(ids, scores.size(1));
    auto selected = box[b].index_select(0, query);
    if (o.clip_boxes) {
      selected.select(1, 0).clamp_(0, canvas[1]);
      selected.select(1, 2).clamp_(0, canvas[1]);
      selected.select(1, 1).clamp_(0, canvas[0]);
      selected.select(1, 3).clamp_(0, canvas[0]);
    }
    ImageResult r;
    r.extent = {static_cast<std::uint32_t>(canvas[1]),
                static_cast<std::uint32_t>(canvas[0])};
    r.boxes = selected.cpu();
    r.scores = values.cpu();
    r.classes = ids.remainder(scores.size(1)).cpu();
    r.anchor_indices = query.cpu();
    r.decode_revision = "native-rtdetr-topk/v1";
    out.push_back(std::move(r));
  }
  return out;
}
}  // namespace jyppx::liuding::vision::tasks
