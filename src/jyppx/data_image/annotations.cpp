// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/data_image/annotations.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <new>
#include <span>

namespace jyppx::data::image {
namespace detail {

void ValidateCategories(const std::vector<VisionCategory>& categories,
                        const AnnotationLimits& limits) {
  if (categories.size() > limits.max_categories) Resource("categories.count");
  std::unordered_set<std::int64_t> ids;
  std::unordered_set<std::string> names;
  std::size_t points = 0;
  for (const auto& category : categories) {
    Text(category.name, limits);
    if (!ids.insert(category.source_id).second ||
        !names.insert(category.name).second)
      Malformed("categories.unique");
    Add(points, category.keypoint_names.size(), limits.max_points,
        "category.keypoints");
    std::unordered_set<std::string> keypoints;
    for (const auto& point : category.keypoint_names) {
      Text(point, limits);
      if (!keypoints.insert(point).second) Malformed("keypoints.unique");
    }
    Add(points, category.skeleton.size(), limits.max_points, "skeleton.count");
    for (const auto& edge : category.skeleton)
      if (edge[0] >= category.keypoint_names.size() ||
          edge[1] >= category.keypoint_names.size() || edge[0] == edge[1])
        Malformed("skeleton.indices");
  }
}

Json ParseJson(std::string_view bytes, ParseState& state) {
  std::size_t nodes = 0;
  struct ObjectKeys {
    std::unordered_set<std::string> seen;
    std::string field;
    int depth = 0;
  };
  std::vector<ObjectKeys> keys;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    CheckStop(state.stop);
    if (depth < 0 ||
        static_cast<std::uint32_t>(depth) > state.options.limits.max_depth)
      Resource("json.depth");
    Add(nodes, 1U, state.options.limits.max_nodes, "json.nodes");
    if (event == Json::parse_event_t::object_start)
      keys.push_back({{}, {}, depth});
    if (event == Json::parse_event_t::key) {
      const auto& key = value.get_ref<const std::string&>();
      Text(key, state.options.limits);
      if (keys.empty() || !keys.back().seen.insert(key).second)
        Malformed("json.duplicate_key");
      keys.back().field = key;
    }
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    if (event == Json::parse_event_t::value && value.is_string()) {
      const auto& text = value.get_ref<const std::string&>();
      const bool rle = !keys.empty() && keys.back().field == "counts" &&
                       depth == keys.back().depth + 1;
      const auto maximum = rle ? state.options.limits.max_rle_bytes
                               : state.options.limits.max_string_bytes;
      if (text.size() > maximum) Resource("json.string_bytes");
    }
    return true;
  };
  auto result = Json::parse(bytes.begin(), bytes.end(), callback, true, false);
  if (!result.is_object()) Malformed("json.root_object");
  return result;
}

}  // namespace detail

std::string_view annotation_format_name(AnnotationFormat format) noexcept {
  switch (format) {
    case AnnotationFormat::coco_instances:
      return "coco.instances";
    case AnnotationFormat::coco_keypoints:
      return "coco.keypoints";
    case AnnotationFormat::coco_panoptic:
      return "coco.panoptic";
    case AnnotationFormat::yolo_detection:
      return "yolo.detect";
    case AnnotationFormat::yolo_segmentation:
      return "yolo.segment";
    case AnnotationFormat::yolo_pose:
      return "yolo.pose";
    case AnnotationFormat::yolo_obb:
      return "yolo.obb";
    case AnnotationFormat::voc:
      return "voc";
    case AnnotationFormat::labelme:
      return "labelme";
  }
  return "unknown";
}

AnnotationIndex parse_annotations(std::string_view bytes,
                                  AnnotationFormat format,
                                  const AnnotationReadOptions& options,
                                  std::stop_token stop) {
  try {
    detail::CheckStop(stop);
    const auto& l = options.limits;
    if (annotation_format_name(format) == "unknown" ||
        l.max_input_bytes == 0U || l.max_input_bytes > (1ULL << 30U) ||
        l.max_images == 0U || l.max_categories == 0U ||
        l.max_string_bytes == 0U || l.max_nodes == 0U || l.max_depth == 0U ||
        l.max_depth > 128U || l.max_pixels == 0U || l.max_dimension == 0U ||
        l.max_pixels > (1ULL << 40U) ||
        (options.width == 0U) != (options.height == 0U))
      throw InvalidArgument("Invalid annotation reader options",
                            detail::Details("parse.options"));
    if (bytes.size() > l.max_input_bytes) detail::Resource("input.bytes");
    if (!detail::Utf8(bytes)) detail::Malformed("input.utf8");
    detail::ValidateCategories(options.categories, l);
    detail::ParseState state{options, stop};
    if (format != AnnotationFormat::coco_instances &&
        format != AnnotationFormat::coco_keypoints &&
        format != AnnotationFormat::coco_panoptic) {
      for (const auto& category : options.categories) {
        state.Points(category.keypoint_names.size());
        state.Points(category.skeleton.size());
      }
    }
    AnnotationIndex result;
    switch (format) {
      case AnnotationFormat::coco_instances:
      case AnnotationFormat::coco_keypoints:
      case AnnotationFormat::coco_panoptic:
        result =
            detail::ReadCoco(detail::ParseJson(bytes, state), format, state);
        break;
      case AnnotationFormat::labelme:
        result = detail::ReadLabelMe(detail::ParseJson(bytes, state), state);
        break;
      case AnnotationFormat::yolo_detection:
      case AnnotationFormat::yolo_segmentation:
      case AnnotationFormat::yolo_pose:
      case AnnotationFormat::yolo_obb:
        result = detail::ReadYolo(bytes, format, state);
        break;
      case AnnotationFormat::voc:
        result = detail::ReadVoc(bytes, state);
        break;
    }
    detail::CheckStop(stop);
    result.format = format;
    result.source_sha256 = sha256_hex(std::as_bytes(std::span(bytes)));
    detail::CheckStop(stop);
    return result;
  } catch (const nlohmann::json::exception&) {
    // Parser diagnostics can contain label text/paths; retain the error class
    // without echoing untrusted payloads into the stable public diagnostic.
    detail::Malformed("json.parse");
  } catch (const std::bad_alloc&) {
    detail::Resource("parse.allocation");
  } catch (const std::length_error&) {
    detail::Resource("parse.container");
  }
}

}  // namespace jyppx::data::image
