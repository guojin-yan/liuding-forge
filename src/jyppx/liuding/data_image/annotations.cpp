// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/liuding/data_image/annotations.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <new>
#include <span>

namespace jyppx::liuding::data::image {
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

namespace {
// The callback DOM parser scans the parent array after each completed object.
// Build through the public SAX interface so large COCO arrays stay linear,
// while retaining per-event cancellation, budgets and duplicate-key checks.
class BoundedJsonSax final : public nlohmann::json_sax<Json> {
  struct Frame {
    Json value;
    std::unordered_set<std::string> seen;
    std::string field;
    std::size_t depth = 0;
  };

 public:
  explicit BoundedJsonSax(ParseState& state) : state_(state) {}

  bool null() override { return Value(nullptr); }
  bool boolean(bool value) override { return Value(value); }
  bool number_integer(number_integer_t value) override { return Value(value); }
  bool number_unsigned(number_unsigned_t value) override {
    return Value(value);
  }
  bool number_float(number_float_t value, const string_t&) override {
    return Value(value);
  }
  bool string(string_t& value) override {
    Node(frames_.size());
    const auto owner =
        std::find_if(frames_.rbegin(), frames_.rend(),
                     [](const Frame& f) { return f.value.is_object(); });
    const bool rle = owner != frames_.rend() && owner->field == "counts" &&
                     frames_.size() == owner->depth + 1;
    const auto maximum = rle ? state_.options.limits.max_rle_bytes
                             : state_.options.limits.max_string_bytes;
    if (value.size() > maximum) Resource("json.string_bytes");
    Attach(Json(std::move(value)));
    return true;
  }
  bool binary(binary_t&) override { Malformed("json.parse"); }
  bool start_object(std::size_t) override { return Start(true); }
  bool start_array(std::size_t) override { return Start(false); }
  bool key(string_t& value) override {
    Node(frames_.size());
    Text(value, state_.options.limits);
    if (frames_.empty() || !frames_.back().value.is_object() ||
        !frames_.back().seen.insert(value).second)
      Malformed("json.duplicate_key");
    frames_.back().field = std::move(value);
    return true;
  }
  bool end_object() override { return End(); }
  bool end_array() override { return End(); }
  bool parse_error(std::size_t, const std::string&,
                   const Json::exception&) override {
    Malformed("json.parse");
  }
  Json take() {
    if (!root_.is_object()) Malformed("json.root_object");
    return std::move(root_);
  }

 private:
  void Node(std::size_t depth) {
    CheckStop(state_.stop);
    if (depth > state_.options.limits.max_depth) Resource("json.depth");
    Add(nodes_, 1U, state_.options.limits.max_nodes, "json.nodes");
  }
  void Attach(Json value) {
    if (frames_.empty())
      root_ = std::move(value);
    else if (frames_.back().value.is_array())
      frames_.back().value.push_back(std::move(value));
    else
      frames_.back().value[frames_.back().field] = std::move(value);
  }
  template <class T>
  bool Value(T value) {
    Node(frames_.size());
    Attach(Json(value));
    return true;
  }
  bool Start(bool object) {
    Node(frames_.size());
    frames_.push_back(
        {object ? Json::object() : Json::array(), {}, {}, frames_.size()});
    return true;
  }
  bool End() {
    if (frames_.empty()) Malformed("json.parse");
    Node(frames_.size() - 1);
    auto value = std::move(frames_.back().value);
    frames_.pop_back();
    Attach(std::move(value));
    return true;
  }
  ParseState& state_;
  std::size_t nodes_ = 0;
  std::vector<Frame> frames_;
  Json root_;
};
}  // namespace

Json ParseJson(std::string_view bytes, ParseState& state) {
  BoundedJsonSax sax(state);
  if (!Json::sax_parse(bytes.begin(), bytes.end(), &sax)) {
    Malformed("json.parse");
  }
  return sax.take();
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
    if (options.coco_detection_boxes &&
        format != AnnotationFormat::coco_instances)
      throw InvalidArgument("COCO box-only policy requires instances format",
                            detail::Details("parse.options"));
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

}  // namespace jyppx::liuding::data::image
