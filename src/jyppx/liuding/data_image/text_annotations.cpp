// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <charconv>
#include <system_error>
#include <tinyxml2.h>

namespace jyppx::liuding::data::image::detail {
namespace {

bool Space(char value) {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() && Space(value.front())) value.remove_prefix(1U);
  while (!value.empty() && Space(value.back())) value.remove_suffix(1U);
  return value;
}

double Numeric(std::string_view value) {
  value = Trim(value);
  double result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(),
                                      result, std::chars_format::general);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size() || !std::isfinite(result))
    Malformed("text.number");
  return result;
}

std::vector<std::string_view> Tokens(std::string_view line, ParseState& state) {
  std::vector<std::string_view> tokens;
  while (!(line = Trim(line)).empty()) {
    CheckStop(state.stop);
    Add(state.text_nodes, 1U, state.options.limits.max_nodes, "text.fields");
    const auto it = std::find_if(line.begin(), line.end(), Space);
    const auto size = static_cast<std::size_t>(it - line.begin());
    tokens.push_back(line.substr(0U, size));
    line.remove_prefix(size);
  }
  return tokens;
}

ImageBox YoloBox(const std::vector<std::string_view>& fields,
                 const VisionAnnotationRecord& image) {
  const auto x = Numeric(fields[1]), y = Numeric(fields[2]);
  const auto w = Numeric(fields[3]), h = Numeric(fields[4]);
  if (x < 0 || x > 1 || y < 0 || y > 1 || w <= 0 || w > 1 || h <= 0 || h > 1)
    Malformed("yolo.normalized_box");
  ImageBox box{(x - w / 2) * image.width, (y - h / 2) * image.height,
               (x + w / 2) * image.width, (y + h / 2) * image.height};
  // Allow only rounding at mathematically exact normalized boundaries.
  const double tolerance = 1e-9;
  if (box.x_min < -tolerance || box.y_min < -tolerance ||
      box.x_max > image.width + tolerance ||
      box.y_max > image.height + tolerance)
    Malformed("yolo.box_bounds");
  box.x_min = (std::max)(0.0, box.x_min);
  box.y_min = (std::max)(0.0, box.y_min);
  box.x_max = (std::min)(static_cast<double>(image.width), box.x_max);
  box.y_max = (std::min)(static_cast<double>(image.height), box.y_max);
  Box(box, image);
  return box;
}

void ConvexQuad(const std::array<ImagePoint, 4>& quad) {
  double sign = 0;
  for (std::size_t i = 0; i < 4U; ++i) {
    const auto& a = quad[i];
    const auto& b = quad[(i + 1U) % 4U];
    const auto& c = quad[(i + 2U) % 4U];
    const double cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (!std::isfinite(cross) || cross == 0 || (sign != 0 && cross * sign <= 0))
      Malformed("yolo.obb_convex_order");
    sign = cross;
  }
}

const tinyxml2::XMLElement* Element(const tinyxml2::XMLNode& node,
                                    const char* name, bool required = true) {
  const auto* element = node.FirstChildElement(name);
  if ((!element && required) || (element && element->NextSiblingElement(name)))
    Malformed("voc.element");
  return element;
}

std::string XmlText(const tinyxml2::XMLNode& node, const char* name,
                    const AnnotationLimits& limits) {
  const auto* element = Element(node, name);
  std::string result;
  for (const auto* child = element->FirstChild(); child;
       child = child->NextSibling()) {
    if (const auto* text = child->ToText()) {
      const std::string_view part(text->Value());
      if (part.size() > limits.max_string_bytes - result.size())
        Resource("voc.text_bytes");
      result.append(part);
    } else if (!child->ToComment()) {
      Malformed("voc.text");
    }
  }
  const auto trimmed = Trim(result);
  Text(trimmed, limits);
  return std::string(trimmed);
}

bool XmlFlag(const tinyxml2::XMLNode& node, const char* name,
             const AnnotationLimits& limits) {
  if (!Element(node, name, false)) return false;
  const double value = Numeric(XmlText(node, name, limits));
  if (value != 0 && value != 1) Malformed("voc.flag");
  return value == 1;
}

class XmlGuard final : public tinyxml2::XMLVisitor {
 public:
  explicit XmlGuard(ParseState& state) : state_(state) {}
  bool VisitEnter(const tinyxml2::XMLElement& element,
                  const tinyxml2::XMLAttribute* attribute) override {
    CheckStop(state_.stop);
    Add(nodes_, 1U, state_.options.limits.max_nodes, "xml.nodes");
    if (++depth_ > state_.options.limits.max_depth) Resource("xml.depth");
    Text(element.Name(), state_.options.limits);
    for (const auto* attr = attribute; attr; attr = attr->Next()) {
      Add(nodes_, 1U, state_.options.limits.max_nodes, "xml.attributes");
      Text(attr->Name(), state_.options.limits);
      if (std::string_view(attr->Value()).size() >
          state_.options.limits.max_string_bytes)
        Resource("xml.attribute_bytes");
    }
    return true;
  }
  bool VisitExit(const tinyxml2::XMLElement&) override {
    --depth_;
    return true;
  }
  bool Visit(const tinyxml2::XMLText& text) override {
    CheckStop(state_.stop);
    Add(nodes_, 1U, state_.options.limits.max_nodes, "xml.nodes");
    if (std::string_view(text.Value()).size() >
        state_.options.limits.max_string_bytes)
      Resource("xml.text_bytes");
    return true;
  }
  bool Visit(const tinyxml2::XMLUnknown&) override {
    // TinyXML2 does not resolve external entities. Reject DTD/other unknown
    // declarations as well, so an alternate parser cannot change that policy.
    Malformed("xml.dtd_or_unknown_declaration");
  }
  bool Visit(const tinyxml2::XMLComment& node) override {
    Metadata(node.Value());
    return true;
  }
  bool Visit(const tinyxml2::XMLDeclaration& node) override {
    Metadata(node.Value());
    return true;
  }

 private:
  void Metadata(std::string_view value) {
    CheckStop(state_.stop);
    Add(nodes_, 1U, state_.options.limits.max_nodes, "xml.nodes");
    if (value.size() > state_.options.limits.max_string_bytes)
      Resource("xml.metadata_bytes");
  }
  ParseState& state_;
  std::size_t nodes_ = 0;
  std::uint32_t depth_ = 0;
};

}  // namespace

AnnotationIndex ReadYolo(std::string_view bytes, AnnotationFormat format,
                         ParseState& state) {
  const auto& options = state.options;
  const auto& limits = options.limits;
  if (options.categories.empty() || options.width == 0U ||
      (format == AnnotationFormat::yolo_pose &&
       (options.keypoint_count == 0U ||
        options.keypoint_count > limits.max_points ||
        (options.keypoint_dimensions != 2U &&
         options.keypoint_dimensions != 3U) ||
        options.keypoint_count >
            (limits.max_nodes - (std::min)(limits.max_nodes, std::size_t{5U})) /
                options.keypoint_dimensions)))
    throw InvalidArgument(
        "YOLO requires dimensions, class names and pose shape",
        Details("yolo.options"));
  AnnotationIndex result;
  result.categories = options.categories;
  VisionAnnotationRecord image;
  image.image_path = options.image_path;
  Path(image.image_path, limits);
  image.width = options.width;
  image.height = options.height;
  Dimensions(image.width, image.height, limits);
  while (!bytes.empty()) {
    CheckStop(state.stop);
    const auto newline = bytes.find('\n');
    const auto line = Trim(bytes.substr(0U, newline));
    bytes = newline == std::string_view::npos ? std::string_view{}
                                              : bytes.substr(newline + 1U);
    if (line.empty()) continue;
    state.Object();
    const auto fields = Tokens(line, state);
    if (fields.empty()) Malformed("yolo.row");
    std::int64_t category = -1;
    const auto parsed = std::from_chars(
        fields[0].data(), fields[0].data() + fields[0].size(), category);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != fields[0].data() + fields[0].size() || category < 0 ||
        static_cast<std::uint64_t>(category) >= result.categories.size())
      Malformed("yolo.category_index");
    VisionObject object;
    object.source_id = static_cast<std::int64_t>(image.objects.size());
    object.category = static_cast<std::size_t>(category);
    if (format == AnnotationFormat::yolo_detection ||
        format == AnnotationFormat::yolo_pose) {
      const auto expected =
          format == AnnotationFormat::yolo_detection
              ? 5U
              : 5U + options.keypoint_count * options.keypoint_dimensions;
      if (fields.size() != expected) Malformed("yolo.field_count");
      object.box = YoloBox(fields, image);
      if (format == AnnotationFormat::yolo_pose) {
        const auto& names = result.categories[object.category].keypoint_names;
        if (!names.empty() && names.size() != options.keypoint_count)
          Malformed("yolo.keypoint_category_shape");
        state.Points(options.keypoint_count);
        for (std::size_t i = 5U; i < fields.size();
             i += options.keypoint_dimensions) {
          const auto x = Numeric(fields[i]), y = Numeric(fields[i + 1U]);
          const auto v =
              options.keypoint_dimensions == 3U ? Numeric(fields[i + 2U]) : 2.0;
          if (v != 0 && v != 1 && v != 2) Malformed("yolo.visibility");
          ImageKeypoint point{{x * image.width, y * image.height},
                              static_cast<std::uint8_t>(v)};
          if (v != 0) Point(point.point, image);
          object.keypoints.push_back(point);
        }
      }
    } else {
      if ((format == AnnotationFormat::yolo_obb && fields.size() != 9U) ||
          fields.size() < 7U || fields.size() % 2U != 1U)
        Malformed("yolo.polygon_field_count");
      state.Points((fields.size() - 1U) / 2U);
      std::vector<ImagePoint> ring;
      for (std::size_t i = 1U; i < fields.size(); i += 2U) {
        ImagePoint point{Numeric(fields[i]) * image.width,
                         Numeric(fields[i + 1U]) * image.height};
        Point(point, image);
        ring.push_back(point);
      }
      if (format == AnnotationFormat::yolo_obb) {
        object.oriented_box =
            std::array<ImagePoint, 4>{ring[0], ring[1], ring[2], ring[3]};
        ConvexQuad(*object.oriented_box);
      } else {
        ValidatePolygon(ring, state);
        object.polygons.push_back(std::move(ring));
      }
    }
    image.objects.push_back(std::move(object));
  }
  result.images.push_back(std::move(image));
  return result;
}

AnnotationIndex ReadVoc(std::string_view bytes, ParseState& state) {
  const auto& options = state.options;
  const auto& limits = options.limits;
  if (options.voc_coordinates != VocCoordinates::one_based_inclusive &&
      options.voc_coordinates != VocCoordinates::zero_based_boundary)
    throw InvalidArgument("Unknown VOC coordinate convention",
                          Details("voc.options"));
  tinyxml2::XMLDocument document;
  if (document.Parse(bytes.data(), bytes.size()) != tinyxml2::XML_SUCCESS)
    Malformed("voc.xml_parse");
  XmlGuard guard(state);
  document.Accept(&guard);
  const auto* root = Element(document, "annotation");
  if (root != document.RootElement() || root->NextSiblingElement())
    Malformed("voc.root");
  const auto* size = Element(*root, "size");
  const double width = Numeric(XmlText(*size, "width", limits));
  const double height = Numeric(XmlText(*size, "height", limits));
  if (width <= 0 || height <= 0 || width > limits.max_dimension ||
      height > limits.max_dimension || std::floor(width) != width ||
      std::floor(height) != height)
    Malformed("voc.dimensions");
  VisionAnnotationRecord image;
  image.width = static_cast<std::uint32_t>(width);
  image.height = static_cast<std::uint32_t>(height);
  Dimensions(image.width, image.height, limits);
  if (options.width != 0U &&
      (image.width != options.width || image.height != options.height))
    Malformed("voc.image_dimensions");
  image.image_path = options.image_path.empty()
                         ? XmlText(*root, "filename", limits)
                         : options.image_path;
  Path(image.image_path, limits);
  AnnotationIndex result;
  result.categories = options.categories;
  for (const auto* source = root->FirstChildElement("object"); source;
       source = source->NextSiblingElement("object")) {
    state.Object();
    const auto name = XmlText(*source, "name", limits);
    auto category = std::find_if(
        result.categories.begin(), result.categories.end(),
        [&](const VisionCategory& value) { return value.name == name; });
    if (category == result.categories.end()) {
      if (!options.categories.empty()) Malformed("voc.unknown_class");
      if (result.categories.size() >= limits.max_categories)
        Resource("categories.count");
      result.categories.push_back(
          {static_cast<std::int64_t>(result.categories.size()), name});
      category = result.categories.end() - 1;
    }
    VisionObject object;
    object.source_id = static_cast<std::int64_t>(image.objects.size());
    object.category =
        static_cast<std::size_t>(category - result.categories.begin());
    object.difficult = XmlFlag(*source, "difficult", limits);
    object.truncated = XmlFlag(*source, "truncated", limits);
    const auto* box = Element(*source, "bndbox");
    const auto offset =
        options.voc_coordinates == VocCoordinates::one_based_inclusive ? 1.0
                                                                       : 0.0;
    const auto xmin = Numeric(XmlText(*box, "xmin", limits));
    const auto ymin = Numeric(XmlText(*box, "ymin", limits));
    const auto xmax = Numeric(XmlText(*box, "xmax", limits));
    const auto ymax = Numeric(XmlText(*box, "ymax", limits));
    if (options.voc_coordinates == VocCoordinates::one_based_inclusive &&
        (std::floor(xmin) != xmin || std::floor(ymin) != ymin ||
         std::floor(xmax) != xmax || std::floor(ymax) != ymax || xmin > xmax ||
         ymin > ymax))
      Malformed("voc.inclusive_coordinates");
    object.box = ImageBox{xmin - offset, ymin - offset, xmax, ymax};
    Box(*object.box, image);
    image.objects.push_back(std::move(object));
  }
  result.images.push_back(std::move(image));
  return result;
}

}  // namespace jyppx::liuding::data::image::detail
