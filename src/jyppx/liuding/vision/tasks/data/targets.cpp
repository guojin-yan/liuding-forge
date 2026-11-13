// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>

#include <algorithm>
#include <cmath>
namespace jyppx::liuding::vision::tasks {
namespace {
using I = std::int64_t;
using namespace data::image;
bool Include(const VisionObject& object, const YoloDataOptions& o) {
  return !object.ignore && !(o.exclude_crowd && object.crowd) &&
         !(o.exclude_difficult && object.difficult);
}
void Rasterize(const VisionObject& object, Tensor mask) {
  const I h = mask.size(0), w = mask.size(1);
  auto pixels = mask.accessor<float, 2>();
  if (object.rle) {
    if (object.rle->width != w || object.rle->height != h)
      throw InvalidArgument("Instance RLE extent differs from task canvas");
    std::uint64_t cursor = 0;
    bool foreground = false;
    for (auto length : object.rle->counts) {
      if (length > static_cast<std::uint64_t>(h * w) - cursor)
        throw InvalidArgument("Instance RLE exceeds canvas");
      if (foreground)
        for (std::uint64_t n = 0; n < length; ++n) {
          const auto index = cursor + n;
          pixels[index % h][index / h] = 1.f;
        }
      cursor += length;
      foreground = !foreground;
    }
    if (cursor != static_cast<std::uint64_t>(h * w))
      throw InvalidArgument("Instance RLE does not cover canvas");
    return;
  }
  if (object.polygons.empty())
    throw InvalidArgument(
        "Instance segmentation object lacks a polygon or mask");
  for (const auto& polygon : object.polygons) {
    if (polygon.size() < 3)
      throw InvalidArgument("Instance polygon has fewer than three points");
    double low = h, high = 0;
    for (const auto& p : polygon) {
      low = (std::min)(low, p.y);
      high = (std::max)(high, p.y);
    }
    const I start = (std::max)(I{0}, static_cast<I>(std::ceil(low - .5)));
    const I end = (std::min)(h, static_cast<I>(std::ceil(high - .5)));
    for (I y = start; y < end; ++y) {
      std::vector<double> crossings;
      const double scan = y + .5;
      for (std::size_t j = 0; j < polygon.size(); ++j) {
        const auto a = polygon[j], b = polygon[(j + 1) % polygon.size()];
        if ((a.y <= scan && b.y > scan) || (b.y <= scan && a.y > scan))
          crossings.push_back(a.x + (scan - a.y) * (b.x - a.x) / (b.y - a.y));
      }
      std::sort(crossings.begin(), crossings.end());
      for (std::size_t j = 0; j + 1 < crossings.size(); j += 2) {
        const I x1 =
            (std::max)(I{0}, static_cast<I>(std::ceil(crossings[j] - .5)));
        const I x2 =
            (std::min)(w, static_cast<I>(std::ceil(crossings[j + 1] - .5)));
        for (I x = x1; x < x2; ++x) pixels[y][x] = 1.f;
      }
    }
  }
}
}  // namespace
TaskTargets project_task_targets(const ImageBatch& batch,
                                 const YoloTaskOptions& task,
                                 const YoloDataOptions& data_options) {
  TaskTargets result;
  result.boxes = build_yolo_targets(batch, data_options);
  const I rows = result.boxes.size(0), h = batch.images.size(2),
          w = batch.images.size(3);
  const auto host =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  Tensor masks, keypoints, obb;
  if (task.task == TaskKind::instance_segment) {
    if (rows > task.max_mask_elements / h / w)
      throw ResourceExhausted(
          "Task instance-mask projection element bound exceeded");
    masks = torch::zeros({rows, h, w}, host);
  }
  if (task.task == TaskKind::pose)
    keypoints = torch::zeros({rows, task.keypoints, 3}, host);
  if (task.task == TaskKind::obb) obb = torch::zeros({rows, 5}, host);
  I row = 0;
  for (const auto& item : batch.items) {
    if (!item.annotation) continue;
    for (const auto& object : item.annotation->objects) {
      if (!Include(object, data_options)) continue;
      if (!object.box)
        throw InvalidArgument("Task target requires a canonical object box");
      if (row >= rows)
        throw InvalidArgument(
            "Rich target row projection differs from box projection");
      if (masks.defined()) Rasterize(object, masks[row]);
      if (keypoints.defined()) {
        if (object.keypoints.size() != static_cast<std::size_t>(task.keypoints))
          throw InvalidArgument(
              "Object keypoint count differs from task contract");
        auto point_row = keypoints[row];
        auto points = point_row.accessor<float, 2>();
        for (I k = 0; k < task.keypoints; ++k) {
          points[k][0] = static_cast<float>(object.keypoints[k].point.x);
          points[k][1] = static_cast<float>(object.keypoints[k].point.y);
          points[k][2] = object.keypoints[k].visibility;
        }
      }
      if (obb.defined()) {
        if (!object.oriented_box)
          throw InvalidArgument(
              "OBB object lacks its canonical four-corner geometry");
        const auto& q = *object.oriented_box;
        const double ax = q[1].x - q[0].x, ay = q[1].y - q[0].y,
                     bx = q[2].x - q[1].x, by = q[2].y - q[1].y;
        const double width = std::hypot(ax, ay), height = std::hypot(bx, by);
        if (width <= 0 || height <= 0 ||
            std::abs(ax * bx + ay * by) > 1e-4 * width * height ||
            std::hypot(q[0].x + q[2].x - q[1].x - q[3].x,
                       q[0].y + q[2].y - q[1].y - q[3].y) >
                1e-4 * (width + height))
          throw InvalidArgument("OBB target quadrilateral is not a rectangle");
        auto box_row = obb[row];
        auto values = box_row.accessor<float, 1>();
        values[0] = static_cast<float>((q[0].x + q[2].x) * .5);
        values[1] = static_cast<float>((q[0].y + q[2].y) * .5);
        values[2] = static_cast<float>(width);
        values[3] = static_cast<float>(height);
        values[4] = static_cast<float>(std::atan2(ay, ax));
      }
      ++row;
    }
  }
  if (row != rows)
    throw InvalidArgument("Rich target projection lost object row identity");
  const auto device = batch.images.device();
  result.masks = masks.defined() ? masks.to(device)
                                 : torch::empty({0}, batch.images.options());
  result.keypoints = keypoints.defined()
                         ? keypoints.to(device)
                         : torch::empty({0}, batch.images.options());
  result.oriented_boxes = obb.defined()
                              ? obb.to(device)
                              : torch::empty({0}, batch.images.options());
  return result;
}
training::TargetMapper make_task_target_mapper(
    const YoloTaskOptions& task, const YoloDataOptions& data_options) {
  return
      [task, data_options](
          const data::BatchEnvelope& envelope) -> std::optional<c10::IValue> {
        const auto& batch = data::image::image_batch_from_value(envelope.value);
        auto t = project_task_targets(batch, task, data_options);
        return c10::IValue(c10::ivalue::Tuple::create(
            {t.boxes, t.masks, t.keypoints, t.oriented_boxes}));
      };
}
training::TargetMapper make_classification_target_mapper() {
  return [](const data::BatchEnvelope& envelope) -> std::optional<c10::IValue> {
    const auto& batch = data::image::image_batch_from_value(envelope.value);
    if (batch.label_offsets.size() != batch.items.size() + 1 ||
        batch.labels.numel() != static_cast<I>(batch.items.size()))
      throw InvalidArgument(
          "Classification expects one label per ImageBatch item");
    for (std::size_t i = 0; i < batch.items.size(); ++i)
      if (batch.label_offsets[i + 1] - batch.label_offsets[i] != 1)
        throw InvalidArgument(
            "Classification requires single-label source items");
    return batch.labels;
  };
}
training::BatchMapper make_rcnn_image_mapper(
    const YoloDataOptions& data_options) {
  return [data_options](const data::BatchEnvelope& envelope) {
    const auto& batch = data::image::image_batch_from_value(envelope.value);
    auto targets = build_yolo_targets(batch, data_options);
    std::vector<float> extents;
    for (const auto& item : batch.items) {
      extents.push_back(static_cast<float>(item.valid_extent.height));
      extents.push_back(static_cast<float>(item.valid_extent.width));
    }
    auto sizes = torch::tensor(extents, batch.images.options())
                     .view({static_cast<I>(batch.items.size()), 2});
    return nn::Invocation{{batch.images, targets, sizes}, {}};
  };
}
}  // namespace jyppx::liuding::vision::tasks
