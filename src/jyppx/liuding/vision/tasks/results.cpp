// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/ops/roi.hpp>
#include <jyppx/liuding/vision/ops/rotated.hpp>
#include <jyppx/liuding/vision/tasks/results.hpp>

#include <torch/nn/functional/upsampling.h>
#include <torch/nn/functional/vision.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
void Validate(const ResultOptions& o) {
  for (auto v : {o.confidence, o.nms_iou, o.mask_threshold})
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw InvalidArgument("Prediction thresholds must be within [0,1]");
  if (o.max_candidates < 1 || o.max_candidates > 65536 ||
      o.max_detections < 1 || o.max_detections > o.max_candidates ||
      o.max_mask_elements < 1)
    throw InvalidArgument("Invalid prediction candidate/mask bound");
  for (auto c : o.classes)
    if (c < 0)
      throw InvalidArgument("Prediction class filter must be nonnegative");
}
Tensor Cpu(const Tensor& value) {
  return value.detach().to(torch::kCPU, torch::kFloat32).contiguous();
}
ImageResult Select(ImageResult r, const Tensor& ids) {
  for (auto* field : {&r.boxes, &r.scores, &r.classes, &r.anchor_indices,
                      &r.masks, &r.keypoints, &r.oriented_corners})
    if (field->defined()) *field = field->index_select(0, ids);
  return r;
}
Tensor Keep(const Tensor& boxes, const Tensor& scores, const Tensor& classes,
            const ResultOptions& o, bool end_to_end,
            const Tensor& rotated = {}) {
  std::map<I, std::vector<I>> groups;
  auto labels = classes.accessor<I, 1>();
  for (I i = 0; i < boxes.size(0); ++i)
    if ((o.classes.empty() || std::find(o.classes.begin(), o.classes.end(),
                                        labels[i]) != o.classes.end()) &&
        boxes[i][2].item<float>() > boxes[i][0].item<float>() &&
        boxes[i][3].item<float>() > boxes[i][1].item<float>() &&
        (!rotated.defined() ||
         (torch::isfinite(rotated[i]).all().item<bool>() &&
          rotated[i][2].item<float>() > 0 && rotated[i][3].item<float>() > 0)))
      groups[o.class_agnostic ? 0 : labels[i]].push_back(i);
  std::vector<I> kept;
  for (const auto& [category, rows] : groups) {
    static_cast<void>(category);
    auto ids = torch::tensor(rows, torch::kInt64);
    auto selected =
        end_to_end ? torch::arange(static_cast<I>(rows.size()), torch::kInt64)
        : rotated.defined()
            ? vision_ops::rotated_nms(
                  rotated.index_select(0, ids), scores.index_select(0, ids),
                  {o.nms_iou, o.max_candidates, o.max_detections})
            : vision_ops::nms(boxes.index_select(0, ids),
                              scores.index_select(0, ids),
                              {o.nms_iou, o.max_candidates, o.max_detections});
    for (I i = 0; i < selected.numel(); ++i)
      kept.push_back(rows[static_cast<std::size_t>(selected[i].item<I>())]);
  }
  std::stable_sort(kept.begin(), kept.end(), [&](I a, I b) {
    return scores[a].item<float>() > scores[b].item<float>();
  });
  if (kept.size() > static_cast<std::size_t>(o.max_detections))
    kept.resize(static_cast<std::size_t>(o.max_detections));
  return torch::tensor(kept, torch::kInt64);
}
ImageResult Candidates(const Tensor& boxes, const Tensor& probabilities,
                       std::array<I, 2> size, const ResultOptions& o,
                       bool end_to_end = false, const Tensor& rotated = {}) {
  Tensor anchors, labels, scores;
  if (o.multi_label) {
    auto pairs = (probabilities >= o.confidence).nonzero();
    anchors = pairs.select(1, 0);
    labels = pairs.select(1, 1);
    scores = probabilities.index({anchors, labels});
  } else {
    auto best = probabilities.max(1);
    auto all_scores = std::get<0>(best);
    auto all_labels = std::get<1>(best);
    anchors = (all_scores >= o.confidence).nonzero().flatten();
    scores = all_scores.index_select(0, anchors);
    labels = all_labels.index_select(0, anchors);
  }
  auto order = torch::argsort(scores, true, 0, true)
                   .slice(0, 0, (std::min)(scores.size(0), o.max_candidates));
  anchors = anchors.index_select(0, order);
  scores = Cpu(scores.index_select(0, order));
  labels = labels.index_select(0, order).to(torch::kCPU).contiguous();
  auto selected = Cpu(boxes.index_select(0, anchors));
  selected.select(1, 0).clamp_(0, size[1]);
  selected.select(1, 2).clamp_(0, size[1]);
  selected.select(1, 1).clamp_(0, size[0]);
  selected.select(1, 3).clamp_(0, size[0]);
  if (!torch::isfinite(selected).all().item<bool>() ||
      !torch::isfinite(scores).all().item<bool>())
    throw InvalidArgument("Nonfinite task prediction");
  auto rboxes =
      rotated.defined() ? Cpu(rotated.index_select(0, anchors)) : Tensor{};
  auto keep = Keep(selected, scores, labels, o, end_to_end, rboxes);
  ImageResult r;
  r.extent = {static_cast<std::uint32_t>(size[1]),
              static_cast<std::uint32_t>(size[0])};
  r.boxes = selected.index_select(0, keep);
  r.scores = scores.index_select(0, keep);
  r.classes = labels.index_select(0, keep);
  r.anchor_indices = anchors.to(torch::kCPU).index_select(0, keep);
  if (rboxes.defined()) {
    rboxes = rboxes.index_select(0, keep);
    const auto signs = torch::tensor(std::vector<float>{-.5f, -.5f, .5f, -.5f,
                                                        .5f, .5f, -.5f, .5f})
                           .view({1, 4, 2});
    auto local = rboxes.slice(1, 2, 4).unsqueeze(1) * signs;
    auto angle = rboxes.select(1, 4).unsqueeze(1), x = local.select(2, 0),
         y = local.select(2, 1);
    r.oriented_corners = torch::stack({x * angle.cos() - y * angle.sin(),
                                       x * angle.sin() + y * angle.cos()},
                                      2) +
                         rboxes.slice(1, 0, 2).unsqueeze(1);
  }
  return r;
}
}  // namespace
std::vector<ImageResult> decode_classification_results(const Tensor& logits,
                                                       std::array<I, 2> size,
                                                       const ResultOptions& o) {
  Validate(o);
  torch::NoGradGuard guard;
  if (!logits.defined() || logits.dim() != 2 || logits.size(0) < 1 ||
      logits.size(1) < 1 || !logits.is_floating_point() ||
      !torch::isfinite(logits).all().item<bool>() || size[0] < 1 ||
      size[1] < 1 || size[0] > UINT32_MAX || size[1] > UINT32_MAX)
    throw InvalidArgument(
        "Classification decoding requires finite [B,C] logits and a valid "
        "extent");
  for (auto c : o.classes)
    if (c >= logits.size(1))
      throw InvalidArgument("Classification class filter exceeds vocabulary");
  auto probabilities = Cpu(logits).softmax(1);
  std::vector<ImageResult> results;
  for (I i = 0; i < logits.size(0); ++i) {
    auto scores = probabilities[i];
    auto keep = scores >= o.confidence;
    if (!o.classes.empty()) {
      auto selected = torch::zeros_like(keep);
      for (auto c : o.classes) selected.index_put_({c}, true);
      keep = keep & selected;
    }
    auto order = torch::argsort(scores, true, 0, true);
    order = order.index({keep.index_select(0, order)});
    order = order.slice(0, 0, (std::min)(order.numel(), o.max_detections));
    ImageResult item;
    item.task = TaskKind::classify;
    item.extent = {static_cast<std::uint32_t>(size[1]),
                   static_cast<std::uint32_t>(size[0])};
    item.boxes = torch::empty({0, 4});
    item.scores = scores.index_select(0, order);
    item.classes = order;
    item.anchor_indices = torch::empty({0}, torch::kInt64);
    results.push_back(std::move(item));
  }
  return results;
}
std::vector<ImageResult> decode_detection_results(
    const YoloDecodedPrediction& p, std::array<I, 2> size,
    const ResultOptions& o) {
  Validate(o);
  torch::NoGradGuard guard;
  std::vector<ImageResult> result;
  for (I b = 0; b < p.boxes.size(0); ++b)
    result.push_back(Candidates(p.boxes[b], p.class_scores[b], size, o));
  return result;
}
std::vector<ImageResult> decode_yolo_results(std::span<const Tensor> outputs,
                                             std::array<I, 2> size,
                                             const YoloTaskOptions& task,
                                             const ResultOptions& o) {
  Validate(o);
  torch::NoGradGuard guard;
  if (task.task == TaskKind::classify)
    throw InvalidArgument("Classification logits use top-k class prediction");
  const std::size_t branch = task.end_to_end ? 1 : 0;
  if (outputs.size() <= branch)
    throw InvalidArgument("Task result raw tuple lacks selected branch");
  const auto& raw = outputs[branch];
  const I base = task.detection.reg_max * 4 + task.detection.num_classes;
  auto p = decode_yolo_detection(raw.slice(1, 0, base), size, task.detection);
  Tensor rotated;
  if (task.task == TaskKind::obb) {
    auto v = raw.transpose(1, 2).to(torch::kFloat32);
    auto distance = v.slice(2, 0, 4 * task.detection.reg_max);
    if (task.detection.reg_max > 1)
      distance =
          (distance
               .reshape({raw.size(0), raw.size(2), 4, task.detection.reg_max})
               .softmax(-1) *
           torch::arange(task.detection.reg_max, v.options()))
              .sum(-1);
    auto angle = v.select(2, base);
    if (!task.generation26) angle = (angle.sigmoid() - .25) * std::acos(-1.);
    auto delta = (distance.slice(2, 2, 4) - distance.slice(2, 0, 2)) * .5;
    auto dx = delta.select(2, 0), dy = delta.select(2, 1);
    auto center =
        (p.anchor_points + torch::stack({dx * angle.cos() - dy * angle.sin(),
                                         dx * angle.sin() + dy * angle.cos()},
                                        2)) *
        p.strides;
    auto wh = (distance.slice(2, 2, 4) + distance.slice(2, 0, 2)) * p.strides;
    rotated = torch::cat({center, wh, angle.unsqueeze(-1)}, 2);
    // Exact enclosing xyxy from rotated extents is used for canvas filtering
    // only.
    auto half = torch::stack({wh.select(2, 0) * angle.cos().abs() +
                                  wh.select(2, 1) * angle.sin().abs(),
                              wh.select(2, 0) * angle.sin().abs() +
                                  wh.select(2, 1) * angle.cos().abs()},
                             2) *
                .5;
    p.boxes = torch::cat({center - half, center + half}, 2);
  }
  std::vector<ImageResult> result;
  for (I b = 0; b < raw.size(0); ++b) {
    auto r = Candidates(p.boxes[b], p.class_scores[b], size, o, task.end_to_end,
                        rotated.defined() ? rotated[b] : Tensor{});
    r.task = task.task;
    auto ids = r.anchor_indices.to(raw.device());
    if (task.task == TaskKind::instance_segment) {
      const auto index = task.end_to_end ? 2U : 1U;
      if (outputs.size() <= index)
        throw InvalidArgument("Segmentation output lacks prototypes");
      const auto& proto = outputs[index];
      const I n = ids.numel();
      if (n > o.max_mask_elements / size[0] / size[1])
        throw ResourceExhausted("Decoded instance mask bound exceeded");
      auto coefficients = raw[b]
                              .slice(0, base, base + task.mask_channels)
                              .index_select(1, ids)
                              .transpose(0, 1)
                              .to(torch::kFloat32);
      auto masks =
          torch::matmul(coefficients, proto[b].to(torch::kFloat32).flatten(1))
              .view({n, 1, proto.size(2), proto.size(3)});
      if (n)
        masks = torch::nn::functional::interpolate(
            masks, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<I>{size[0], size[1]})
                       .mode(torch::kBilinear)
                       .align_corners(false));
      else
        masks = torch::empty({0, 1, size[0], size[1]}, raw.options());
      r.masks = Cpu(masks.squeeze(1)).sigmoid() >= o.mask_threshold;
      auto xs = torch::arange(size[1]).view({1, 1, -1}),
           ys = torch::arange(size[0]).view({1, -1, 1});
      r.masks = r.masks & (xs >= r.boxes.select(1, 0).view({-1, 1, 1})) &
                (xs < r.boxes.select(1, 2).view({-1, 1, 1})) &
                (ys >= r.boxes.select(1, 1).view({-1, 1, 1})) &
                (ys < r.boxes.select(1, 3).view({-1, 1, 1}));
    } else if (task.task == TaskKind::pose) {
      auto k =
          raw[b]
              .slice(0, base, base + task.keypoints * task.keypoint_dimensions)
              .index_select(1, ids)
              .transpose(0, 1)
              .reshape({-1, task.keypoints, task.keypoint_dimensions})
              .to(torch::kFloat32);
      auto anchor = p.anchor_points.index_select(0, ids).unsqueeze(1),
           stride = p.strides.index_select(0, ids).unsqueeze(1);
      auto xy = (k.slice(2, 0, 2) * (task.generation26 ? 1. : 2.) + anchor -
                 (task.generation26 ? 0. : .5)) *
                stride;
      auto visibility = task.keypoint_dimensions == 3
                            ? k.slice(2, 2, 3).sigmoid()
                            : torch::ones_like(k.slice(2, 0, 1));
      r.keypoints = Cpu(torch::cat({xy, visibility}, 2));
    }
    result.push_back(std::move(r));
  }
  return result;
}
std::vector<ImageResult> decode_faster_rcnn_results(std::span<const Tensor> x,
                                                    const ResultOptions& o) {
  Validate(o);
  torch::NoGradGuard guard;
  if (x.size() != 9)
    throw InvalidArgument("Faster R-CNN results require nine raw tensors");
  auto proposals = x[3], logits = x[4], deltas = x[5];
  const I classes = logits.size(1) - 1;
  auto decoded = vision_ops::decode_box_deltas(proposals.slice(1, 1, 5), deltas,
                                               {10, 10, 5, 5})
                     .view({-1, classes + 1, 4});
  auto probabilities = logits.to(torch::kFloat32).softmax(1).slice(1, 1);
  std::vector<ImageResult> results;
  for (I b = 0; b < x[0].size(0); ++b) {
    auto rows = (proposals.select(1, 0) == b).nonzero().flatten();
    auto probabilities_b = probabilities.index_select(0, rows);
    auto pairs = (probabilities_b >= o.confidence).nonzero();
    auto local_rows = pairs.select(1, 0), labels = pairs.select(1, 1);
    auto scores = probabilities_b.index({local_rows, labels});
    auto order = torch::argsort(scores, true, 0, true)
                     .slice(0, 0, (std::min)(scores.size(0), o.max_candidates));
    local_rows = local_rows.index_select(0, order);
    labels = labels.index_select(0, order);
    auto original_rows = rows.index_select(0, local_rows);
    auto boxes = Cpu(decoded.index({original_rows, labels + 1}));
    scores = Cpu(scores.index_select(0, order));
    labels = labels.to(torch::kCPU).contiguous();
    std::array<I, 2> size{x[8][b][0].item<I>(), x[8][b][1].item<I>()};
    boxes.select(1, 0).clamp_(0, size[1]);
    boxes.select(1, 2).clamp_(0, size[1]);
    boxes.select(1, 1).clamp_(0, size[0]);
    boxes.select(1, 3).clamp_(0, size[0]);
    if (!torch::isfinite(boxes).all().item<bool>() ||
        !torch::isfinite(scores).all().item<bool>())
      throw InvalidArgument("Nonfinite Faster R-CNN result");
    auto keep = Keep(boxes, scores, labels, o, false);
    ImageResult r;
    r.extent = {static_cast<std::uint32_t>(size[1]),
                static_cast<std::uint32_t>(size[0])};
    r.boxes = boxes.index_select(0, keep);
    r.scores = scores.index_select(0, keep);
    r.classes = labels.index_select(0, keep);
    r.anchor_indices = original_rows.to(torch::kCPU).index_select(0, keep);
    results.push_back(std::move(r));
  }
  return results;
}
ImageResult restore_result(const ImageResult& input,
                           const data::image::GeometryTrace& trace,
                           I max_elements) {
  torch::NoGradGuard guard;
  if (input.extent.width != trace.output_extent().width ||
      input.extent.height != trace.output_extent().height)
    throw InvalidArgument("Result canvas differs from geometry trace");
  auto r = input;
  if(static_cast<unsigned>(input.task)>4) {
    r.extent=trace.source_extent();
    if(!input.field.defined())return r;
    const auto m=trace.matrix();I h=r.extent.height,w=r.extent.width;
    I channels=input.field.dim()==2?1:input.field.size(0);
    if(max_elements<1||h>max_elements/w/channels)throw ResourceExhausted("Restored field bound exceeded");
    auto xs=(torch::arange(w,torch::kFloat32)+.5)*m[0]+m[2];
    auto ys=(torch::arange(h,torch::kFloat32)+.5)*m[4]+m[5];
    auto xx=xs.view({1,w}).expand({h,w}),yy=ys.view({h,1}).expand({h,w});
    auto grid=torch::stack({xx*(2./input.extent.width)-1,yy*(2./input.extent.height)-1},2).unsqueeze(0);
    auto valid=(xx>=0)&(xx<input.extent.width)&(yy>=0)&(yy<input.extent.height);
    auto values=input.field.to(torch::kFloat32).reshape({1,channels,input.extent.height,input.extent.width});
    auto sampling=torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false);
    if(input.task==TaskKind::semantic_segment) sampling.mode(torch::kNearest);
    r.field=torch::nn::functional::grid_sample(values,grid,sampling).squeeze(0);
    if(input.field.dim()==2)r.field=r.field.squeeze(0).to(torch::kInt64);
    r.field_valid=valid;return r;
  }
  std::vector<I> keep;
  std::vector<float> coordinates;
  for (I i = 0; i < input.boxes.size(0); ++i) {
    auto box = trace.inverse_box(
        {input.boxes[i][0].item<double>(), input.boxes[i][1].item<double>(),
         input.boxes[i][2].item<double>(), input.boxes[i][3].item<double>()});
    if (!box) continue;
    keep.push_back(i);
    for (auto v : {box->x_min, box->y_min, box->x_max, box->y_max})
      coordinates.push_back(static_cast<float>(v));
  }
  r = Select(r, torch::tensor(keep, torch::kInt64));
  r.boxes = torch::tensor(coordinates).view({-1, 4});
  r.extent = trace.source_extent();
  const auto matrix = trace.matrix();
  if (r.oriented_corners.defined()) {
    r.oriented_corners = r.oriented_corners.clone();
    r.oriented_corners.select(2, 0).sub_(matrix[2]).div_(matrix[0]);
    r.oriented_corners.select(2, 1).sub_(matrix[5]).div_(matrix[4]);
  }
  if (r.keypoints.defined()) {
    r.keypoints = r.keypoints.clone();
    for (I i = 0; i < r.keypoints.size(0); ++i)
      for (I k = 0; k < r.keypoints.size(1); ++k) {
        auto point = trace.inverse_point({r.keypoints[i][k][0].item<double>(),
                                          r.keypoints[i][k][1].item<double>()});
        if (point) {
          r.keypoints[i][k][0] = point->x;
          r.keypoints[i][k][1] = point->y;
        } else
          r.keypoints[i][k].zero_();
      }
  }
  if (r.masks.defined()) {
    const I h = r.extent.height, w = r.extent.width, n = r.masks.size(0);
    if (max_elements < 1 || n > max_elements / h / w)
      throw ResourceExhausted("Restored mask element bound exceeded");
    auto xs = (torch::arange(w, torch::kFloat32) + .5) * matrix[0] + matrix[2];
    auto ys = (torch::arange(h, torch::kFloat32) + .5) * matrix[4] + matrix[5];
    auto xx = xs.view({1, -1}).expand({h, w}),
         yy = ys.view({-1, 1}).expand({h, w});
    auto grid = torch::stack({xx * (2. / input.extent.width) - 1,
                              yy * (2. / input.extent.height) - 1},
                             2)
                    .unsqueeze(0)
                    .expand({n, h, w, 2});
    if (n)
      r.masks = torch::nn::functional::grid_sample(
                    r.masks.to(torch::kFloat32).unsqueeze(1), grid,
                    torch::nn::functional::GridSampleFuncOptions()
                        .mode(torch::kNearest)
                        .padding_mode(torch::kZeros)
                        .align_corners(false))
                    .squeeze(1) > .5;
    else
      r.masks = torch::empty({0, h, w}, torch::kBool);
  }
  return r;
}
std::string result_json(const ImageResult& r, std::size_t max_bytes) {
  using Json = nlohmann::json;
  Json root = {{"schema", "vision.result/v1"},
               {"source_id", r.source_id},
               {"width", r.extent.width},
               {"height", r.extent.height},
               {"task", static_cast<int>(r.task)},
               {"decode_revision", r.decode_revision},
               {"instances", Json::array()}};
  root["task_name"]=task_name(r.task);
  if(r.task==TaskKind::classify || r.task==TaskKind::classify_multilabel) {
    root["probabilities"]=Json::array();for(I i=0;i<r.scores.numel();++i)root["probabilities"].push_back({{"class",r.classes[i].item<I>()},{"score",r.scores[i].item<double>()}});
  }
  auto tensor_json=[&](const Tensor& value) {
    Json tensor={{"shape",value.sizes().vec()},{"values",Json::array()}};
    if(static_cast<std::uint64_t>(value.numel())>max_bytes/24)throw ResourceExhausted("Dense result JSON element bound exceeded");
    auto flat=value.to(torch::kCPU).reshape({-1});
    for(I i=0;i<flat.numel();++i) {
      if(value.scalar_type()==torch::kBool)tensor["values"].push_back(flat[i].item<bool>());
      else if(value.is_floating_point())tensor["values"].push_back(flat[i].item<double>());
      else tensor["values"].push_back(flat[i].item<I>());
    }
    return tensor;
  };
  if(r.field.defined())root["field"]=tensor_json(r.field);
  if(r.field_valid.defined())root["field_valid"]=tensor_json(r.field_valid);
  if(r.embedding.defined())root["embedding"]=tensor_json(r.embedding);
  for (I row = 0; row < r.boxes.size(0); ++row) {
    Json instance = {{"class", r.classes[row].item<I>()},
                     {"score", r.scores[row].item<double>()},
                     {"box", Json::array()}};
    for (I j = 0; j < 4; ++j)
      instance["box"].push_back(r.boxes[row][j].item<double>());
    if (r.keypoints.defined()) {
      instance["keypoints"] = Json::array();
      for (I k = 0; k < r.keypoints.size(1); ++k)
        instance["keypoints"].push_back(
            {r.keypoints[row][k][0].item<double>(),
             r.keypoints[row][k][1].item<double>(),
             r.keypoints[row][k][2].item<double>()});
    }
    if (r.oriented_corners.defined()) {
      instance["corners"] = Json::array();
      for (I k = 0; k < 4; ++k)
        instance["corners"].push_back(
            {r.oriented_corners[row][k][0].item<double>(),
             r.oriented_corners[row][k][1].item<double>()});
    }
    if (r.masks.defined()) {
      std::vector<std::uint64_t> runs;
      std::uint64_t length = 0;
      bool foreground = false;
      auto mask = r.masks[row].to(torch::kBool).contiguous();
      auto pixels = mask.accessor<bool, 2>();
      for (I x = 0; x < mask.size(1); ++x)
        for (I y = 0; y < mask.size(0); ++y) {
          if (pixels[y][x] == foreground)
            ++length;
          else {
            runs.push_back(length);
            length = 1;
            foreground = !foreground;
          }
        }
      runs.push_back(length);
      instance["mask"] = {{"size", {mask.size(0), mask.size(1)}},
                          {"counts", runs},
                          {"order", "column_major"}};
    }
    root["instances"].push_back(std::move(instance));
  }
  auto bytes = root.dump();
  if (bytes.size() > max_bytes)
    throw ResourceExhausted("Encoded task result byte bound exceeded");
  return bytes;
}
void write_result(const std::filesystem::path& path, const ImageResult& r,
                  bool json, std::size_t max_bytes) {
  const auto bytes = result_json(r, max_bytes);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!json)
    stream << "# liuding vision.result/v1; one complete image JSON record per "
              "line\n";
  stream << bytes << '\n';
  stream.close();
  if (!stream.good())
    throw IoError("Cannot write local task prediction result");
}
}  // namespace jyppx::liuding::vision::tasks
