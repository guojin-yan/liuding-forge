// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace jyppx::data::image::detail {
namespace {

VisionCategory Category(const Json& source, ParseState& state) {
  VisionCategory category;
  category.source_id = Integer(source.at("id"));
  category.name = String(source.at("name"), state.options.limits);
  if (source.contains("isthing")) category.is_thing = Flag(source, "isthing");
  if (source.contains("keypoints")) {
    const auto& points = source.at("keypoints");
    Array(points);
    state.Points(points.size());
    for (const auto& point : points)
      category.keypoint_names.push_back(String(point, state.options.limits));
  }
  if (source.contains("skeleton")) {
    const auto& skeleton = source.at("skeleton");
    Array(skeleton);
    state.Points(skeleton.size());
    for (const auto& edge : skeleton) {
      Array(edge);
      if (edge.size() != 2U) Malformed("coco.skeleton");
      const auto a = Integer(edge[0]), b = Integer(edge[1]);
      if (a <= 0 || b <= 0 ||
          static_cast<std::uint64_t>(a) > category.keypoint_names.size() ||
          static_cast<std::uint64_t>(b) > category.keypoint_names.size())
        Malformed("coco.skeleton_index");
      category.skeleton.push_back(
          {static_cast<std::size_t>(a - 1), static_cast<std::size_t>(b - 1)});
    }
  }
  return category;
}

ImageBox XywhBox(const Json& value, const VisionAnnotationRecord& image) {
  Array(value);
  if (value.size() != 4U) Malformed("coco.bbox");
  const auto x = Number(value[0]), y = Number(value[1]);
  const auto width = Number(value[2]), height = Number(value[3]);
  ImageBox box{x, y, x + width, y + height};
  Box(box, image);
  return box;
}

ImageMaskRle ReadRle(const Json& value, const VisionAnnotationRecord& image,
                     ParseState& state) {
  ImageMaskRle result;
  const auto& size = value.at("size");
  Array(size);
  if (size.size() != 2U) Malformed("coco.rle_size");
  result.height = Dimension(size[0]);
  result.width = Dimension(size[1]);
  Dimensions(result.width, result.height, state.options.limits);
  if (result.width != image.width || result.height != image.height)
    Malformed("coco.rle_image_size");
  const std::uint64_t pixels =
      static_cast<std::uint64_t>(result.width) * result.height;
  std::uint64_t total = 0;
  const auto append = [&](std::uint64_t count) {
    Add(state.runs, 1U, state.options.limits.max_rle_runs, "rle.runs");
    if (count > pixels - total) Malformed("coco.rle_sum");
    total += count;
    result.counts.push_back(count);
  };
  const auto& counts = value.at("counts");
  if (counts.is_array()) {
    for (const auto& count : counts) {
      CheckStop(state.stop);
      const auto n = Integer(count);
      if (n < 0) Malformed("coco.rle_negative");
      append(static_cast<std::uint64_t>(n));
    }
  } else if (counts.is_string()) {
    // COCO compressed RLE is a signed 5-bit variable-length integer stream.
    // From the fourth run onward, values are deltas to the run two back.
    const auto& text = counts.get_ref<const std::string&>();
    if (text.size() > state.options.limits.max_rle_bytes)
      Resource("coco.rle_bytes");
    std::size_t position = 0;
    while (position < text.size()) {
      CheckStop(state.stop);
      std::uint64_t bits = 0;
      unsigned shift = 0;
      unsigned code = 0;
      do {
        if (position >= text.size() || shift >= 60U)
          Malformed("coco.rle_varint");
        const auto byte = static_cast<unsigned char>(text[position++]);
        if (byte < 48U || byte > 111U) Malformed("coco.rle_character");
        code = byte - 48U;
        bits |= static_cast<std::uint64_t>(code & 31U) << shift;
        shift += 5U;
      } while ((code & 32U) != 0U);
      const std::int64_t delta =
          (code & 16U) != 0U ? static_cast<std::int64_t>(bits) -
                                   static_cast<std::int64_t>(1ULL << shift)
                             : static_cast<std::int64_t>(bits);
      const auto base = result.counts.size() > 2U
                            ? static_cast<std::int64_t>(
                                  result.counts[result.counts.size() - 2U])
                            : 0;
      // Delta values outside the image size are unnecessary for valid RLE;
      // checking before addition avoids signed overflow on hostile streams.
      if (delta < -base || delta > static_cast<std::int64_t>(pixels) - base)
        Malformed("coco.rle_delta");
      append(static_cast<std::uint64_t>(base + delta));
    }
  } else {
    Malformed("coco.rle_counts");
  }
  if (result.counts.empty() || total != pixels) Malformed("coco.rle_sum");
  return result;
}

void Segmentation(const Json& value, VisionObject& object,
                  const VisionAnnotationRecord& image, ParseState& state) {
  if (value.is_object()) {
    object.rle = ReadRle(value, image, state);
    return;
  }
  Array(value);
  for (const auto& ring : value) {
    CheckStop(state.stop);
    Array(ring);
    if (ring.size() < 6U || ring.size() % 2U != 0U) Malformed("coco.polygon");
    state.Points(ring.size() / 2U);
    std::vector<ImagePoint> points;
    points.reserve(ring.size() / 2U);
    for (std::size_t i = 0; i < ring.size(); i += 2U) {
      ImagePoint point{Number(ring[i]), Number(ring[i + 1U])};
      Point(point, image);
      points.push_back(point);
    }
    ValidatePolygon(points, state);
    object.polygons.push_back(std::move(points));
  }
}

}  // namespace

AnnotationIndex ReadCoco(const Json& document, AnnotationFormat format,
                         ParseState& state) {
  const auto& limits = state.options.limits;
  AnnotationIndex result;
  const auto& categories = document.at("categories");
  const auto& images = document.at("images");
  const auto& annotations = document.at("annotations");
  Array(categories);
  Array(images);
  Array(annotations);
  if (categories.size() > limits.max_categories) Resource("categories.count");
  if (images.size() > limits.max_images) Resource("images.count");
  if (format != AnnotationFormat::coco_panoptic &&
      annotations.size() > limits.max_objects)
    Resource("annotations.count");
  std::unordered_map<std::int64_t, std::size_t> category_ids;
  for (const auto& source : categories) {
    CheckStop(state.stop);
    auto category = Category(source, state);
    if (!category_ids.emplace(category.source_id, result.categories.size())
             .second)
      Malformed("coco.category_id");
    result.categories.push_back(std::move(category));
  }
  ValidateCategories(result.categories, limits);
  if (!state.options.categories.empty() &&
      state.options.categories.size() != result.categories.size())
    Malformed("coco.class_map_size");
  for (std::size_t i = 0; i < state.options.categories.size(); ++i) {
    if (state.options.categories[i].source_id !=
            result.categories[i].source_id ||
        state.options.categories[i].name != result.categories[i].name)
      Malformed("coco.class_map");
  }
  std::unordered_map<std::int64_t, std::size_t> image_ids;
  std::unordered_set<std::string> paths;
  for (const auto& source : images) {
    CheckStop(state.stop);
    VisionAnnotationRecord record;
    record.source_image_id = Integer(source.at("id"));
    record.image_path = String(source.at("file_name"), limits);
    Path(record.image_path, limits);
    record.width = Dimension(source.at("width"));
    record.height = Dimension(source.at("height"));
    Dimensions(record.width, record.height, limits);
    if (state.options.width != 0U && (record.width != state.options.width ||
                                      record.height != state.options.height))
      Malformed("coco.image_dimensions");
    if (!image_ids.emplace(record.source_image_id, result.images.size())
             .second ||
        !paths.insert(record.image_path).second)
      Malformed("coco.image_identity");
    result.images.push_back(std::move(record));
  }
  std::unordered_set<std::int64_t> object_ids;
  std::unordered_set<std::int64_t> panoptic_images;
  for (const auto& annotation : annotations) {
    CheckStop(state.stop);
    const auto image_id = Integer(annotation.at("image_id"));
    const auto image_it = image_ids.find(image_id);
    if (image_it == image_ids.end()) Malformed("coco.image_reference");
    auto& image = result.images[image_it->second];
    const bool panoptic = format == AnnotationFormat::coco_panoptic;
    if (panoptic) {
      if (!panoptic_images.insert(image_id).second)
        Malformed("coco.panoptic_duplicate_image");
      image.panoptic_mask_path = String(annotation.at("file_name"), limits);
      Path(*image.panoptic_mask_path, limits);
    }
    std::unordered_set<std::int64_t> segment_ids;
    const auto read_object = [&](const Json& source) {
      state.Object();
      VisionObject object;
      object.source_id = Integer(source.at("id"));
      if (panoptic) {
        if (object.source_id <= 0 || object.source_id > 0xffffff ||
            !segment_ids.insert(object.source_id).second)
          Malformed("coco.panoptic_segment_id");
      } else if (!object_ids.insert(object.source_id).second) {
        Malformed("coco.annotation_id");
      }
      const auto category =
          category_ids.find(Integer(source.at("category_id")));
      if (category == category_ids.end()) Malformed("coco.category_reference");
      object.category = category->second;
      object.crowd = Flag(source, "iscrowd");
      object.ignore = Flag(source, "ignore");
      if (source.contains("bbox"))
        object.box = XywhBox(source.at("bbox"), image);
      if (source.contains("area")) {
        object.area = Number(source.at("area"));
        if (*object.area < 0 ||
            *object.area > static_cast<double>(image.width) * image.height)
          Malformed("coco.area");
      }
      if (source.contains("segmentation"))
        Segmentation(source.at("segmentation"), object, image, state);
      if (format == AnnotationFormat::coco_keypoints) {
        const auto& points = source.at("keypoints");
        Array(points);
        const auto expected =
            result.categories[object.category].keypoint_names.size();
        if (expected == 0U || points.size() != expected * 3U)
          Malformed("coco.keypoint_shape");
        state.Points(expected);
        std::size_t labeled = 0;
        for (std::size_t i = 0; i < points.size(); i += 3U) {
          const auto visibility = Integer(points[i + 2U]);
          if (visibility < 0 || visibility > 2) Malformed("coco.visibility");
          ImageKeypoint point{{Number(points[i]), Number(points[i + 1U])},
                              static_cast<std::uint8_t>(visibility)};
          if (visibility != 0) {
            Point(point.point, image);
            ++labeled;
          }
          object.keypoints.push_back(point);
        }
        if (source.contains("num_keypoints") &&
            Integer(source.at("num_keypoints")) !=
                static_cast<std::int64_t>(labeled))
          Malformed("coco.num_keypoints");
      }
      if (!object.box && object.polygons.empty() && !object.rle &&
          object.keypoints.empty() && !panoptic)
        Malformed("coco.object_geometry");
      image.objects.push_back(std::move(object));
    };
    if (panoptic) {
      const auto& objects = annotation.at("segments_info");
      Array(objects);
      for (const auto& source : objects) read_object(source);
    } else {
      read_object(annotation);
    }
  }
  return result;
}

AnnotationIndex ReadLabelMe(const Json& document, ParseState& state) {
  AnnotationIndex result;
  const auto& options = state.options;
  const auto& limits = options.limits;
  VisionAnnotationRecord image;
  image.width = Dimension(document.at("imageWidth"));
  image.height = Dimension(document.at("imageHeight"));
  Dimensions(image.width, image.height, limits);
  if (options.width != 0U &&
      (image.width != options.width || image.height != options.height))
    Malformed("labelme.image_dimensions");
  image.image_path = options.image_path.empty()
                         ? String(document.at("imagePath"), limits)
                         : options.image_path;
  Path(image.image_path, limits);
  if (document.contains("imageData") && !document.at("imageData").is_null())
    Malformed("labelme.embedded_image_not_enabled");
  result.categories = options.categories;
  const auto& shapes = document.at("shapes");
  Array(shapes);
  for (const auto& shape : shapes) {
    state.Object();
    const auto label = String(shape.at("label"), limits);
    auto category = std::find_if(
        result.categories.begin(), result.categories.end(),
        [&](const VisionCategory& value) { return value.name == label; });
    if (category == result.categories.end()) {
      if (!options.categories.empty()) Malformed("labelme.unknown_class");
      if (result.categories.size() >= limits.max_categories)
        Resource("categories.count");
      result.categories.push_back(
          {static_cast<std::int64_t>(result.categories.size()), label});
      category = result.categories.end() - 1;
    }
    VisionObject object;
    object.source_id = static_cast<std::int64_t>(image.objects.size());
    object.category =
        static_cast<std::size_t>(category - result.categories.begin());
    if (shape.contains("group_id") && !shape.at("group_id").is_null())
      object.group_id = Integer(shape.at("group_id"));
    const auto type = String(shape.at("shape_type"), limits);
    const auto& points = shape.at("points");
    Array(points);
    state.Points(points.size());
    std::vector<ImagePoint> ring;
    for (const auto& value : points) {
      Array(value);
      if (value.size() != 2U) Malformed("labelme.point");
      ImagePoint point{Number(value[0]), Number(value[1])};
      Point(point, image);
      ring.push_back(point);
    }
    if (type == "rectangle") {
      if (ring.size() != 2U) Malformed("labelme.rectangle");
      object.box = ImageBox{
          (std::min)(ring[0].x, ring[1].x), (std::min)(ring[0].y, ring[1].y),
          (std::max)(ring[0].x, ring[1].x), (std::max)(ring[0].y, ring[1].y)};
      Box(*object.box, image);
    } else if (type == "polygon") {
      if (ring.size() < 3U) Malformed("labelme.polygon");
      ValidatePolygon(ring, state);
      object.polygons.push_back(std::move(ring));
    } else if (type == "point") {
      if (ring.size() != 1U) Malformed("labelme.point_count");
      object.keypoints.push_back({ring.front(), 2U});
    } else {
      throw NotImplemented("LabelMe shape type is not enabled",
                           Details("labelme.shape_type"));
    }
    if (shape.contains("flags")) {
      const auto& flags = shape.at("flags");
      if (!flags.is_object()) Malformed("labelme.flags");
      object.ignore = Flag(flags, "ignore");
      object.difficult = Flag(flags, "difficult");
    }
    image.objects.push_back(std::move(object));
  }
  result.images.push_back(std::move(image));
  return result;
}

}  // namespace jyppx::data::image::detail
