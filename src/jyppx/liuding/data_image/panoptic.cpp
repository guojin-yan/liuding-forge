// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "panoptic_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>

namespace jyppx::liuding::data::image {
namespace {
ErrorDetails Details(const char* operation) {
  return {.component = "data.image.panoptic", .operation = operation};
}

[[noreturn]] void Invalid(const char* operation) {
  throw InvalidArgument("Invalid panoptic binding", Details(operation));
}

[[noreturn]] void Bad(const char* operation) {
  throw SerializationError("Panoptic mask/table mismatch", Details(operation));
}

[[noreturn]] void Limit(const char* operation) {
  throw ResourceExhausted("Panoptic binding budget exceeded",
                          Details(operation));
}

void Stop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Panoptic binding cancelled", Details("bind.cancel"));
}

bool SameBox(const ImageBox& a, const ImageBox& b) {
  return a.x_min == b.x_min && a.y_min == b.y_min && a.x_max == b.x_max &&
         a.y_max == b.y_max;
}

struct Segment {
  std::uint64_t pixels = 0;
  ImageBox box;
};
}  // namespace

std::uint64_t detail::PanopticRecordBytes(
    const VisionAnnotationRecord& record) {
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (record.objects.size() >
      (maximum - sizeof(VisionAnnotationRecord)) / sizeof(VisionObject))
    Limit("bind.output_bytes");
  std::uint64_t bytes = sizeof(VisionAnnotationRecord) +
                        record.objects.size() * sizeof(VisionObject);
  const auto add_path = [&](const std::string& path) {
    if (path.size() >= maximum - bytes) Limit("bind.output_bytes");
    bytes += path.size() + 1U;
  };
  add_path(record.image_path);
  if (record.panoptic_mask_path) add_path(*record.panoptic_mask_path);
  return bytes;
}

detail::PanopticBinding detail::BindPanopticMask(
    const VisionAnnotationRecord& record, const DecodedMask& mask,
    std::size_t category_count, const PanopticBindOptions& options,
    const std::shared_ptr<ImageResourceBudget>& budget, std::stop_token stop) {
  Stop(stop);
  const auto& limits = options.limits;
  if (!limits.max_dimension || !limits.max_pixels || !options.max_working_bytes)
    Invalid("bind.limits");
  if (!record.panoptic_mask_path || record.panoptic_mask_path->empty())
    Invalid("bind.source_reference");
  if (record.panoptic_mask_path->size() > limits.max_string_bytes ||
      record.image_path.size() > limits.max_string_bytes ||
      record.objects.size() > limits.max_objects ||
      category_count > limits.max_categories)
    Limit("bind.metadata");
  if (!record.width || !record.height || record.width != mask.width ||
      record.height != mask.height)
    Bad("bind.dimensions");
  if (record.width > limits.max_dimension ||
      record.height > limits.max_dimension ||
      static_cast<std::uint64_t>(record.width) * record.height >
          limits.max_pixels)
    Limit("bind.pixels");
  if (mask.encoding != MaskEncoding::panoptic_rgb || !mask.void_id ||
      !mask.ids.defined() || !mask.ids.device().is_cpu() ||
      mask.ids.scalar_type() != torch::kInt64 || mask.ids.dim() != 2 ||
      mask.ids.layout() != torch::kStrided || mask.ids.is_neg() ||
      mask.ids.size(0) != record.height || mask.ids.size(1) != record.width)
    Invalid("bind.mask");
  if (record.objects.size() > options.max_working_bytes / 128U)
    Limit("bind.lookup_bytes");
  ImageResourceLease scratch_lease;
  if (budget)
    scratch_lease = budget->reserve(ImageResourceKind::annotation_bytes,
                                    record.objects.size() * 128U);
  try {
    std::unordered_map<std::int64_t, std::size_t> lookup;
    lookup.reserve(record.objects.size());
    std::vector<Segment> segments(record.objects.size());
    for (std::size_t i = 0; i < record.objects.size(); ++i) {
      Stop(stop);
      const auto& object = record.objects[i];
      if (object.source_id <= 0 || object.source_id > 0xffffff ||
          object.source_id == *mask.void_id ||
          object.source_id == mask.ignore_id ||
          object.category >= category_count ||
          !lookup.emplace(object.source_id, i).second)
        Bad("bind.segment_identity");
      if (!object.polygons.empty() || object.oriented_box || object.rle ||
          !object.keypoints.empty())
        Bad("bind.mixed_geometry");
      if (object.area && (!std::isfinite(*object.area) || *object.area < 0))
        Bad("bind.invalid_source_area");
    }
    const auto pixels = mask.ids.accessor<std::int64_t, 2>();
    for (std::uint32_t y = 0; y < record.height; ++y) {
      Stop(stop);
      for (std::uint32_t x = 0; x < record.width; ++x) {
        const auto id = pixels[y][x];
        if (id == *mask.void_id || id == mask.ignore_id) continue;
        const auto found = lookup.find(id);
        if (found == lookup.end()) Bad("bind.unknown_segment");
        auto& segment = segments[found->second];
        const double left = x, top = y, right = static_cast<double>(x) + 1,
                     bottom = static_cast<double>(y) + 1;
        if (segment.pixels++ == 0)
          segment.box = {left, top, right, bottom};
        else
          segment.box = {std::min(segment.box.x_min, left),
                         std::min(segment.box.y_min, top),
                         std::max(segment.box.x_max, right),
                         std::max(segment.box.y_max, bottom)};
      }
    }
    ImageResourceLease output_lease;
    if (budget)
      output_lease = budget->reserve(ImageResourceKind::annotation_bytes,
                                     PanopticRecordBytes(record));
    auto result = record;
    for (std::size_t i = 0; i < segments.size(); ++i) {
      Stop(stop);
      const auto& segment = segments[i];
      const auto& source = record.objects[i];
      if (!segment.pixels) Bad("bind.missing_segment");
      if (record.panoptic_mask_materialized) {
        if (source.mask_pixel_area != segment.pixels || !source.box)
          Bad("bind.stale_pixel_area");
      } else if (options.verify_declared_area && source.area &&
                 (!std::isfinite(*source.area) ||
                  *source.area != static_cast<double>(segment.pixels)))
        Bad("bind.source_area");
      if ((record.panoptic_mask_materialized || options.verify_declared_box) &&
          source.box && !SameBox(*source.box, segment.box))
        Bad("bind.box");
      result.objects[i].box = segment.box;
      result.objects[i].mask_pixel_area = segment.pixels;
    }
    result.panoptic_mask_materialized = true;
    Stop(stop);
    return {std::move(output_lease), std::move(result)};
  } catch (const std::bad_alloc&) {
    Limit("bind.allocate");
  }
}

VisionAnnotationRecord bind_panoptic_mask(const VisionAnnotationRecord& record,
                                          const DecodedMask& mask,
                                          std::size_t category_count,
                                          const PanopticBindOptions& options,
                                          std::stop_token stop) {
  auto bound =
      detail::BindPanopticMask(record, mask, category_count, options, {}, stop);
  return std::move(bound.record);
}
}  // namespace jyppx::liuding::data::image
