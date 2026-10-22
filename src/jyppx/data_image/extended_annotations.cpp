// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/data_image/extended_annotations.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <new>
#include <numbers>
#include <span>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jyppx::data::image {
namespace {

bool Space(char value) {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() && Space(value.front())) value.remove_prefix(1U);
  while (!value.empty() && Space(value.back())) value.remove_suffix(1U);
  return value;
}

template <class Function>
auto Guarded(Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const nlohmann::json::exception&) {
    detail::Malformed("json.parse");
  } catch (const std::bad_alloc&) {
    detail::Resource("text.allocation");
  } catch (const std::length_error&) {
    detail::Resource("text.container");
  }
}

void ValidateOptions(std::string_view bytes,
                     const TextAnnotationReadOptions& options,
                     std::stop_token stop) {
  detail::CheckStop(stop);
  const auto& limits = options.limits;
  if (limits.max_input_bytes == 0U || limits.max_input_bytes > (1ULL << 30U) ||
      limits.max_images == 0U || limits.max_categories == 0U ||
      limits.max_string_bytes == 0U || limits.max_nodes == 0U ||
      limits.max_dimension == 0U || limits.max_pixels == 0U ||
      limits.max_pixels > (1ULL << 40U) || options.max_line_bytes == 0U ||
      options.max_line_bytes > (1ULL << 30U) || options.max_lines == 0U ||
      options.width == 0U || options.height == 0U ||
      options.categories.empty() ||
      (options.bounds != AnnotationBounds::within_image &&
       options.bounds != AnnotationBounds::preserve_outside))
    throw InvalidArgument("Invalid extended annotation reader options",
                          detail::Details("text.options"));
  if (bytes.size() > limits.max_input_bytes) detail::Resource("input.bytes");
  if (!detail::Utf8(bytes)) detail::Malformed("input.utf8");
  detail::Dimensions(options.width, options.height, limits);
  detail::Path(options.image_path, limits);
  detail::ValidateCategories(options.categories, limits);
  detail::CheckStop(stop);
}

struct TextState {
  AnnotationReadOptions common;
  detail::ParseState parse;
  const TextAnnotationReadOptions& options;
  std::size_t lines = 0;

  TextState(const TextAnnotationReadOptions& input, std::stop_token stop)
      : common{}, parse{common, stop}, options(input) {
    common.limits = input.limits;
    for (const auto& category : input.categories) {
      parse.Points(category.keypoint_names.size());
      parse.Points(category.skeleton.size());
    }
  }

  std::string_view NextLine(std::string_view& bytes) {
    detail::CheckStop(parse.stop);
    detail::Add(lines, 1U, options.max_lines, "text.lines");
    const auto end = bytes.find('\n');
    const auto line = bytes.substr(0U, end);
    if (line.size() > options.max_line_bytes)
      detail::Resource("text.line_bytes");
    bytes = end == std::string_view::npos ? std::string_view{}
                                          : bytes.substr(end + 1U);
    return Trim(line);
  }

  void Field(std::string_view field) {
    detail::CheckStop(parse.stop);
    detail::Add(parse.text_nodes, 1U, options.limits.max_nodes, "text.fields");
    if (field.size() > options.limits.max_string_bytes)
      detail::Resource("text.field_bytes");
    if (field.empty()) detail::Malformed("text.empty_field");
  }
};

std::vector<std::string_view> WhitespaceFields(std::string_view line,
                                               std::size_t maximum,
                                               TextState& state) {
  std::vector<std::string_view> fields;
  while (!(line = Trim(line)).empty()) {
    const auto end = std::find_if(line.begin(), line.end(), Space);
    const auto length = static_cast<std::size_t>(end - line.begin());
    const auto field = line.substr(0U, length);
    state.Field(field);
    if (fields.size() == maximum) detail::Malformed("text.field_count");
    fields.push_back(field);
    line.remove_prefix(length);
  }
  return fields;
}

std::vector<std::string_view> CsvFields(std::string_view line,
                                        std::size_t maximum, TextState& state) {
  std::vector<std::string_view> fields;
  while (true) {
    const auto comma = line.find(',');
    const auto field = Trim(line.substr(0U, comma));
    state.Field(field);
    if (fields.size() == maximum) detail::Malformed("csv.field_count");
    fields.push_back(field);
    if (comma == std::string_view::npos) break;
    line.remove_prefix(comma + 1U);
  }
  return fields;
}

double Numeric(std::string_view value) {
  double number = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                      number, std::chars_format::general);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size() || !std::isfinite(number))
    detail::Malformed("text.finite_number");
  return number;
}

std::int64_t Integral(std::string_view value) {
  std::int64_t number = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), number);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size())
    detail::Malformed("text.integer");
  return number;
}

using CategoryNames = std::unordered_map<std::string_view, std::size_t>;

CategoryNames Names(const std::vector<VisionCategory>& categories) {
  CategoryNames names;
  names.reserve(categories.size());
  for (std::size_t i = 0; i < categories.size(); ++i)
    names.emplace(categories[i].name, i);
  return names;
}

std::size_t Category(std::string_view name, const CategoryNames& categories) {
  const auto found = categories.find(name);
  if (found == categories.end()) detail::Malformed("text.unknown_category");
  return found->second;
}

std::int64_t RowIdentity(std::size_t row) {
  if (row >
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()))
    detail::Resource("text.row_identity");
  return static_cast<std::int64_t>(row);
}

void PointBounds(const ImagePoint& point,
                 const TextAnnotationReadOptions& options) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y))
    detail::Malformed("point.finite");
  if (options.bounds == AnnotationBounds::within_image &&
      (point.x < 0 || point.y < 0 || point.x > options.width ||
       point.y > options.height))
    detail::Malformed("point.coordinates");
}

void BoxBounds(const ImageBox& box, const TextAnnotationReadOptions& options) {
  PointBounds({box.x_min, box.y_min}, options);
  PointBounds({box.x_max, box.y_max}, options);
  if (box.x_max <= box.x_min || box.y_max <= box.y_min ||
      !std::isfinite(box.x_max - box.x_min) ||
      !std::isfinite(box.y_max - box.y_min))
    detail::Malformed("box.positive_finite_extent");
}

std::string Digest(std::string_view bytes, std::stop_token stop) {
  detail::CheckStop(stop);
  const auto digest = sha256_hex(std::as_bytes(std::span(bytes)));
  detail::CheckStop(stop);
  return digest;
}

bool KnownAngle(double value, std::string_view operation) {
  if (value == -10) return false;
  if (value < -std::numbers::pi_v<double> || value > std::numbers::pi_v<double>)
    detail::Malformed(operation);
  return true;
}

bool AllEqual(const std::array<double, 3>& values, double expected) {
  return std::all_of(values.begin(), values.end(),
                     [expected](double value) { return value == expected; });
}

struct MotFrameState {
  std::size_t index = 0;
  std::unordered_set<std::int64_t> identities;
};

}  // namespace

DotaAnnotationIndex parse_dota_1x_annotations(
    std::string_view bytes, const TextAnnotationReadOptions& options,
    std::stop_token stop) {
  return Guarded([&] {
    ValidateOptions(bytes, options, stop);
    const auto original = bytes;
    TextState state(options, stop);
    const auto names = Names(options.categories);
    DotaAnnotationIndex result;
    result.categories = options.categories;
    result.image_path = options.image_path;
    result.width = options.width;
    result.height = options.height;
    bool saw_object = false;
    while (!bytes.empty()) {
      const auto line = state.NextLine(bytes);
      if (line.empty()) continue;
      if (line.starts_with("imagesource:") || line.starts_with("gsd:")) {
        if (saw_object) detail::Malformed("dota.header_position");
        const auto colon = line.find(':');
        const auto value = Trim(line.substr(colon + 1U));
        state.Field(value);
        detail::Text(value, options.limits);
        if (line.starts_with("imagesource:")) {
          if (result.image_source) detail::Malformed("dota.duplicate_header");
          result.image_source = std::string(value);
        } else {
          if (result.has_ground_sample_distance_header)
            detail::Malformed("dota.duplicate_header");
          result.has_ground_sample_distance_header = true;
          if (value != "null") {
            const auto gsd = Numeric(value);
            if (gsd <= 0) detail::Malformed("dota.gsd");
            result.ground_sample_distance = gsd;
          }
        }
        continue;
      }
      saw_object = true;
      const auto fields = WhitespaceFields(line, 10U, state);
      if (fields.size() != 10U) detail::Malformed("dota.field_count");
      state.parse.Object();
      state.parse.Points(4U);
      DotaObjectAnnotation object;
      object.target.source_id = RowIdentity(result.objects.size());
      object.target.category = Category(fields[8], names);
      std::array<ImagePoint, 4> quad{};
      for (std::size_t i = 0; i < quad.size(); ++i) {
        quad[i] = {Numeric(fields[2U * i]), Numeric(fields[2U * i + 1U])};
        PointBounds(quad[i], options);
      }
      for (std::size_t i = 1U; i < quad.size(); ++i)
        if (quad[i].x == quad[0].x && quad[i].y == quad[0].y)
          detail::Malformed("dota.repeated_vertex");
      detail::ValidatePolygon(quad, state.parse);
      object.target.oriented_box = quad;
      object.difficulty = Integral(fields[9]);
      if (object.difficulty < 0) detail::Malformed("dota.difficulty");
      object.target.difficult = object.difficulty != 0;
      result.objects.push_back(std::move(object));
    }
    result.source_sha256 = Digest(original, stop);
    return result;
  });
}

KittiAnnotationIndex parse_kitti_object_annotations(
    std::string_view bytes, KittiObjectDialect dialect,
    const TextAnnotationReadOptions& options, std::stop_token stop) {
  return Guarded([&] {
    ValidateOptions(bytes, options, stop);
    if (dialect != KittiObjectDialect::ground_truth &&
        dialect != KittiObjectDialect::detection_result)
      throw InvalidArgument("Unknown KITTI object dialect",
                            detail::Details("kitti.dialect"));
    const auto original = bytes;
    TextState state(options, stop);
    const auto names = Names(options.categories);
    KittiAnnotationIndex result;
    result.dialect = dialect;
    result.categories = options.categories;
    result.image_path = options.image_path;
    result.width = options.width;
    result.height = options.height;
    const std::size_t expected =
        dialect == KittiObjectDialect::ground_truth ? 15U : 16U;
    while (!bytes.empty()) {
      const auto line = state.NextLine(bytes);
      if (line.empty()) continue;
      const auto fields = WhitespaceFields(line, expected, state);
      if (fields.size() != expected) detail::Malformed("kitti.field_count");
      state.parse.Object();
      KittiObjectAnnotation object;
      object.target.source_id = RowIdentity(result.objects.size());
      object.target.category = Category(fields[0], names);
      object.target.ignore = fields[0] == "DontCare";
      object.truncation = Numeric(fields[1]);
      if (object.truncation != -1 &&
          (object.truncation < 0 || object.truncation > 1))
        detail::Malformed("kitti.truncation");
      object.target.truncated = object.truncation > 0;
      const auto occlusion = Integral(fields[2]);
      if (occlusion < -1 || occlusion > 3) detail::Malformed("kitti.occlusion");
      object.occlusion = static_cast<std::int32_t>(occlusion);
      object.alpha = Numeric(fields[3]);
      object.alpha_known = KnownAngle(object.alpha, "kitti.alpha");
      object.target.box = ImageBox{Numeric(fields[4]), Numeric(fields[5]),
                                   Numeric(fields[6]), Numeric(fields[7])};
      BoxBounds(*object.target.box, options);
      for (std::size_t i = 0; i < 3U; ++i) {
        object.box_3d.dimensions_hwl[i] = Numeric(fields[8U + i]);
        object.box_3d.location_xyz[i] = Numeric(fields[11U + i]);
      }
      object.box_3d.dimensions_known =
          !AllEqual(object.box_3d.dimensions_hwl, -1);
      if (object.box_3d.dimensions_known &&
          std::any_of(object.box_3d.dimensions_hwl.begin(),
                      object.box_3d.dimensions_hwl.end(),
                      [](double value) { return value <= 0; }))
        detail::Malformed("kitti.dimensions");
      object.box_3d.location_known =
          !AllEqual(object.box_3d.location_xyz, -1000);
      if (object.box_3d.location_known &&
          std::any_of(object.box_3d.location_xyz.begin(),
                      object.box_3d.location_xyz.end(),
                      [](double value) { return value == -1000; }))
        detail::Malformed("kitti.partial_unknown_location");
      object.box_3d.rotation_y = Numeric(fields[14]);
      object.box_3d.rotation_known =
          KnownAngle(object.box_3d.rotation_y, "kitti.rotation_y");
      if (dialect == KittiObjectDialect::detection_result)
        object.score = Numeric(fields[15]);
      result.objects.push_back(std::move(object));
    }
    result.source_sha256 = Digest(original, stop);
    return result;
  });
}

MotChallengeAnnotationIndex parse_mot_challenge_annotations(
    std::string_view bytes, const MotChallengeReadOptions& options,
    std::stop_token stop) {
  return Guarded([&] {
    ValidateOptions(bytes, options.text, stop);
    const bool class_labels =
        options.dialect == MotChallengeDialect::ground_truth_9;
    const bool ground_truth =
        class_labels ||
        options.dialect == MotChallengeDialect::ground_truth_world_10;
    if ((options.dialect != MotChallengeDialect::ground_truth_9 &&
         options.dialect != MotChallengeDialect::ground_truth_world_10 &&
         options.dialect != MotChallengeDialect::detection_10 &&
         options.dialect != MotChallengeDialect::tracking_result_10) ||
        (options.coordinates != MotCoordinates::one_based &&
         options.coordinates != MotCoordinates::zero_based) ||
        (options.frame_count && *options.frame_count <= 0) ||
        (!class_labels && !options.unclassified_category) ||
        (options.unclassified_category &&
         *options.unclassified_category >= options.text.categories.size()))
      throw InvalidArgument("Invalid MOTChallenge reader options",
                            detail::Details("mot.options"));
    const auto original = bytes;
    TextState state(options.text, stop);
    std::unordered_map<std::int64_t, std::size_t> classes;
    for (std::size_t i = 0; i < options.text.categories.size(); ++i)
      classes.emplace(options.text.categories[i].source_id, i);
    std::unordered_map<std::int64_t, MotFrameState> frames;
    MotChallengeAnnotationIndex result;
    result.dialect = options.dialect;
    result.categories = options.text.categories;
    result.sequence_path = options.text.image_path;
    result.width = options.text.width;
    result.height = options.text.height;
    const double offset =
        options.coordinates == MotCoordinates::one_based ? 1.0 : 0.0;
    const std::size_t expected = class_labels ? 9U : 10U;
    std::size_t row = 0;
    while (!bytes.empty()) {
      const auto line = state.NextLine(bytes);
      if (line.empty()) continue;
      const auto fields = CsvFields(line, expected, state);
      if (fields.size() != expected) detail::Malformed("mot.field_count");
      state.parse.Object();
      const auto frame = Integral(fields[0]);
      if (frame <= 0 || (options.frame_count && frame > *options.frame_count))
        detail::Malformed("mot.frame");
      MotObjectAnnotation object;
      object.target.source_id = RowIdentity(row++);
      object.track_id = Integral(fields[1]);
      if (object.track_id <= 0 &&
          (options.dialect != MotChallengeDialect::detection_10 ||
           object.track_id != -1))
        detail::Malformed("mot.track_identity");
      const auto x = Numeric(fields[2]) - offset;
      const auto y = Numeric(fields[3]) - offset;
      const auto width = Numeric(fields[4]);
      const auto height = Numeric(fields[5]);
      if (width <= 0 || height <= 0) detail::Malformed("mot.box_extent");
      object.target.box = ImageBox{x, y, x + width, y + height};
      BoxBounds(*object.target.box, options.text);
      object.confidence = Numeric(fields[6]);
      if (ground_truth && object.confidence != 0 && object.confidence != 1)
        detail::Malformed("mot.ground_truth_mark");
      object.target.ignore = ground_truth && object.confidence == 0;
      if (class_labels) {
        const auto source_class = Integral(fields[7]);
        const auto found = classes.find(source_class);
        if (source_class <= 0 || found == classes.end())
          detail::Malformed("mot.source_class");
        object.source_class = source_class;
        object.target.category = found->second;
        object.visibility = Numeric(fields[8]);
        if (*object.visibility < 0 || *object.visibility > 1)
          detail::Malformed("mot.visibility");
      } else {
        object.target.category = *options.unclassified_category;
        object.world_xyz = std::array<double, 3>{
            Numeric(fields[7]), Numeric(fields[8]), Numeric(fields[9])};
      }
      auto found = frames.find(frame);
      if (found == frames.end()) {
        if (result.frames.size() >= options.text.limits.max_images)
          detail::Resource("mot.frames");
        found = frames.emplace(frame, MotFrameState{result.frames.size(), {}})
                    .first;
        result.frames.push_back(MotFrameAnnotation{frame, {}});
      }
      if (object.track_id > 0 &&
          !found->second.identities.insert(object.track_id).second)
        detail::Malformed("mot.duplicate_frame_identity");
      result.frames[found->second.index].objects.push_back(std::move(object));
    }
    result.source_sha256 = Digest(original, stop);
    return result;
  });
}

CityscapesAnnotationIndex parse_cityscapes_polygon_annotations(
    std::string_view bytes, const TextAnnotationReadOptions& options,
    std::stop_token stop) {
  return Guarded([&] {
    ValidateOptions(bytes, options, stop);
    if (options.limits.max_depth == 0U || options.limits.max_depth > 128U)
      throw InvalidArgument("Invalid Cityscapes JSON depth limit",
                            detail::Details("cityscapes.options"));
    const auto original = bytes;
    AnnotationReadOptions json_options;
    json_options.limits = options.limits;
    detail::ParseState state{json_options, stop};
    const auto document = detail::ParseJson(bytes, state);
    const auto require_number = [&](const detail::Json& object,
                                    std::string_view name) -> std::int64_t {
      const auto found = object.find(std::string(name));
      if (found == object.end()) detail::Malformed("cityscapes.required_field");
      return detail::Integer(*found);
    };
    const auto optional_id =
        [&](const detail::Json& object,
            std::string_view name) -> std::optional<std::int64_t> {
      const auto found = object.find(std::string(name));
      if (found == object.end() || found->is_null()) return std::nullopt;
      return detail::Integer(*found);
    };
    const auto optional_text =
        [&](const detail::Json& object,
            std::string_view name) -> std::optional<std::string> {
      const auto found = object.find(std::string(name));
      if (found == object.end() || found->is_null()) return std::nullopt;
      if (!found->is_string()) detail::Malformed("cityscapes.metadata");
      const auto& text = found->get_ref<const std::string&>();
      if (!text.empty()) detail::Text(text, options.limits);
      return text;
    };
    const auto width = require_number(document, "imgWidth");
    const auto height = require_number(document, "imgHeight");
    if (width <= 0 || height <= 0 ||
        static_cast<std::uint64_t>(width) != options.width ||
        static_cast<std::uint64_t>(height) != options.height)
      detail::Malformed("cityscapes.image_dimensions");
    const auto objects = document.find("objects");
    if (objects == document.end() || !objects->is_array())
      detail::Malformed("cityscapes.objects");
    const auto names = Names(options.categories);
    CityscapesAnnotationIndex result;
    result.categories = options.categories;
    result.image_path = options.image_path;
    result.width = options.width;
    result.height = options.height;
    std::unordered_set<std::int64_t> identities;
    for (const auto& value : *objects) {
      detail::CheckStop(stop);
      if (!value.is_object()) detail::Malformed("cityscapes.object");
      state.Object();
      const auto label = value.find("label");
      if (label == value.end()) detail::Malformed("cityscapes.label");
      const auto label_name = detail::String(*label, options.limits);
      const auto category = Category(label_name, names);
      CityscapesObjectAnnotation object;
      object.target.category = category;
      object.object_id = optional_id(value, "id");
      if (object.object_id && (*object.object_id < 0 ||
                               !identities.insert(*object.object_id).second))
        detail::Malformed("cityscapes.object_identity");
      // Cityscapes polygon JSON does not require IDs on every object. A row ID
      // remains unique even when an optional source ID equals another row
      // index.
      object.target.source_id = RowIdentity(result.objects.size());
      object.instance_id = optional_id(value, "instanceId");
      if (object.instance_id && *object.instance_id < 0)
        detail::Malformed("cityscapes.instance_identity");
      object.user = optional_text(value, "user");
      object.date = optional_text(value, "date");
      object.deleted = detail::Flag(value, "deleted");
      object.verified = detail::Flag(value, "verified");
      object.target.ignore = object.deleted;
      const auto polygon = value.find("polygon");
      if (polygon == value.end() || !polygon->is_array())
        detail::Malformed("cityscapes.polygon");
      if (object.deleted && polygon->empty()) {
        result.objects.push_back(std::move(object));
        continue;
      }
      if (polygon->size() > options.limits.max_points)
        detail::Resource("cityscapes.points");
      state.Points(polygon->size());
      std::vector<ImagePoint> ring;
      ring.reserve(polygon->size());
      for (const auto& point : *polygon) {
        if (!point.is_array() || point.size() != 2U)
          detail::Malformed("cityscapes.point");
        ImagePoint parsed{detail::Number(point[0]), detail::Number(point[1])};
        PointBounds(parsed, options);
        ring.push_back(parsed);
      }
      detail::ValidatePolygon(ring, state);
      object.target.polygons.push_back(std::move(ring));
      result.objects.push_back(std::move(object));
    }
    result.source_sha256 = Digest(original, stop);
    return result;
  });
}

}  // namespace jyppx::data::image
