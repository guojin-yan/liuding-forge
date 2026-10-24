// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_IMAGE_ANNOTATION_DETAIL_HPP
#define JYPPX_DATA_IMAGE_ANNOTATION_DETAIL_HPP

#include <jyppx/liuding/data_image/annotations.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jyppx::liuding::data::image::detail {

using Json = nlohmann::json;

inline ErrorDetails Details(std::string_view operation) {
  return {
      .component = "data.image.annotations",
      .operation = std::string(operation),
      .expected = "the selected dialect, class map and bounded image space",
      .actual = "annotation contract not satisfied",
      .hint =
          "Validate the format, dimensions, categories and resource limits"};
}

[[noreturn]] inline void Malformed(std::string_view operation) {
  throw SerializationError("Invalid image annotations", Details(operation));
}

[[noreturn]] inline void Resource(std::string_view operation) {
  throw ResourceExhausted("Image annotation limit exceeded",
                          Details(operation));
}

inline void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Annotation parsing cancelled", Details("parse.cancel"));
}

inline void Add(std::size_t& counter, std::size_t count, std::size_t maximum,
                std::string_view field) {
  if (counter > maximum || count > maximum - counter) Resource(field);
  counter += count;
}

// Reject overlong encodings, surrogates, and code points beyond Unicode.
inline bool Utf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i]);
    if (first < 0x80U) {
      if (first == 0U) return false;
      ++i;
      continue;
    }
    const std::size_t n = first >= 0xc2U && first <= 0xdfU   ? 2U
                          : first >= 0xe0U && first <= 0xefU ? 3U
                          : first >= 0xf0U && first <= 0xf4U ? 4U
                                                             : 0U;
    if (n == 0U || n > text.size() - i) return false;
    const auto second = static_cast<unsigned char>(text[i + 1U]);
    if ((first == 0xe0U && second < 0xa0U) ||
        (first == 0xedU && second >= 0xa0U) ||
        (first == 0xf0U && second < 0x90U) ||
        (first == 0xf4U && second >= 0x90U))
      return false;
    for (std::size_t j = 1; j < n; ++j)
      if ((static_cast<unsigned char>(text[i + j]) & 0xc0U) != 0x80U)
        return false;
    i += n;
  }
  return true;
}

inline void Text(std::string_view value, const AnnotationLimits& limits) {
  if (value.size() > limits.max_string_bytes) Resource("string.bytes");
  if (value.empty() || !Utf8(value) ||
      std::any_of(value.begin(), value.end(),
                  [](unsigned char c) { return c < 0x20U || c == 0x7fU; }))
    Malformed("string.utf8");
}

inline void Path(std::string_view value, const AnnotationLimits& limits) {
  Text(value, limits);
  if (value.front() == '/' || value.front() == '\\' ||
      value.find(':') != std::string_view::npos ||
      value.find('\\') != std::string_view::npos)
    Malformed("path.relative");
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto end = value.find('/', begin);
    const auto part =
        value.substr(begin, end == std::string_view::npos ? value.size() - begin
                                                          : end - begin);
    if (part.empty() || part == "." || part == "..")
      Malformed("path.component");
    if (end == std::string_view::npos) break;
    begin = end + 1U;
  }
}

inline void Dimensions(std::uint32_t width, std::uint32_t height,
                       const AnnotationLimits& limits) {
  if (width == 0U || height == 0U) Malformed("image.dimensions");
  if (width > limits.max_dimension || height > limits.max_dimension ||
      static_cast<std::uint64_t>(width) * height > limits.max_pixels)
    Resource("image.pixels");
}

inline void Point(const ImagePoint& point,
                  const VisionAnnotationRecord& image) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 ||
      point.y < 0 || point.x > image.width || point.y > image.height)
    Malformed("point.coordinates");
}

inline void Box(const ImageBox& box, const VisionAnnotationRecord& image) {
  Point({box.x_min, box.y_min}, image);
  Point({box.x_max, box.y_max}, image);
  if (box.x_max <= box.x_min || box.y_max <= box.y_min)
    Malformed("box.positive_area");
}

inline std::int64_t Integer(const Json& value) {
  if (!value.is_number_integer()) Malformed("json.integer");
  if (value.is_number_unsigned() &&
      value.get<std::uint64_t>() >
          static_cast<std::uint64_t>(
              (std::numeric_limits<std::int64_t>::max)()))
    Malformed("json.integer_range");
  return value.get<std::int64_t>();
}

inline double Number(const Json& value) {
  if (!value.is_number()) Malformed("json.number");
  const auto result = value.get<double>();
  if (!std::isfinite(result)) Malformed("json.finite");
  return result;
}

inline std::string String(const Json& value, const AnnotationLimits& limits) {
  if (!value.is_string()) Malformed("json.string");
  const auto& result = value.get_ref<const std::string&>();
  Text(result, limits);
  return result;
}

inline bool Flag(const Json& object, std::string_view key) {
  const auto it = object.find(std::string(key));
  if (it == object.end()) return false;
  if (it->is_boolean()) return it->get<bool>();
  const auto v = Integer(*it);
  if (v != 0 && v != 1) Malformed("json.flag");
  return v == 1;
}

inline std::uint32_t Dimension(const Json& value) {
  const auto number = Integer(value);
  if (number <= 0 || static_cast<std::uint64_t>(number) >
                         (std::numeric_limits<std::uint32_t>::max)())
    Malformed("image.dimension");
  return static_cast<std::uint32_t>(number);
}

inline void Array(const Json& value) {
  if (!value.is_array()) Malformed("json.array");
}

struct ParseState {
  const AnnotationReadOptions& options;
  std::stop_token stop;
  std::size_t objects = 0;
  std::size_t points = 0;
  std::size_t runs = 0;
  std::size_t geometry_checks = 0;
  std::size_t text_nodes = 0;

  void Object() {
    CheckStop(stop);
    Add(objects, 1U, options.limits.max_objects, "objects.count");
  }
  void Points(std::size_t count) {
    Add(points, count, options.limits.max_points, "points.count");
  }
};

void ValidatePolygon(std::span<const ImagePoint> points, ParseState& state);
void ValidateConvexQuad(const std::array<ImagePoint, 4>& quad);
std::string ReadAnnotationBytes(const std::filesystem::path& path,
                                std::size_t maximum, std::stop_token stop);

Json ParseJson(std::string_view bytes, ParseState& state);
AnnotationIndex ReadCoco(const Json& document, AnnotationFormat format,
                         ParseState& state);
AnnotationIndex ReadLabelMe(const Json& document, ParseState& state);
AnnotationIndex ReadYolo(std::string_view bytes, AnnotationFormat format,
                         ParseState& state);
AnnotationIndex ReadVoc(std::string_view bytes, ParseState& state);
void ValidateCategories(const std::vector<VisionCategory>& categories,
                        const AnnotationLimits& limits);

}  // namespace jyppx::liuding::data::image::detail

#endif  // JYPPX_DATA_IMAGE_ANNOTATION_DETAIL_HPP
