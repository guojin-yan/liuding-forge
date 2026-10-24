// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/geometry_trace.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace jyppx::liuding::data::image {
namespace {

[[noreturn]] void Invalid(const char* operation, const char* expected) {
  throw InvalidArgument("Invalid image geometry",
                        ErrorDetails{.component = "data.image.geometry",
                                     .operation = operation,
                                     .expected = expected});
}

void CheckExtent(ImageExtent extent) {
  if (extent.width == 0 || extent.height == 0)
    Invalid("geometry.extent", "positive width and height");
}

void CheckPoint(ImagePoint point) {
  if (!std::isfinite(point.x) || !std::isfinite(point.y))
    Invalid("geometry.point", "finite coordinates");
}

void CheckBox(ImageBox box) {
  CheckPoint({box.x_min, box.y_min});
  CheckPoint({box.x_max, box.y_max});
  if (box.x_max <= box.x_min || box.y_max <= box.y_min)
    Invalid("geometry.box", "positive width and height");
}

bool Contains(ImageBox box, ImagePoint point) {
  return point.x >= box.x_min && point.x <= box.x_max && point.y >= box.y_min &&
         point.y <= box.y_max;
}

std::optional<ImageBox> Intersect(ImageBox left, ImageBox right) {
  const ImageBox result{
      std::max(left.x_min, right.x_min), std::max(left.y_min, right.y_min),
      std::min(left.x_max, right.x_max), std::min(left.y_max, right.y_max)};
  if (result.x_max <= result.x_min || result.y_max <= result.y_min)
    return std::nullopt;
  return result;
}

ImageBox MapBox(ImageBox box, double sx, double sy, double tx, double ty) {
  const auto x1 = box.x_min * sx + tx;
  const auto x2 = box.x_max * sx + tx;
  const auto y1 = box.y_min * sy + ty;
  const auto y2 = box.y_max * sy + ty;
  return {std::min(x1, x2), std::min(y1, y2), std::max(x1, x2),
          std::max(y1, y2)};
}

ImageBox UnmapBox(ImageBox box, double sx, double sy, double tx, double ty) {
  const auto x1 = (box.x_min - tx) / sx;
  const auto x2 = (box.x_max - tx) / sx;
  const auto y1 = (box.y_min - ty) / sy;
  const auto y2 = (box.y_max - ty) / sy;
  return {std::min(x1, x2), std::min(y1, y2), std::max(x1, x2),
          std::max(y1, y2)};
}

}  // namespace

GeometryTrace::GeometryTrace(ImageExtent extent)
    : source_(extent),
      output_(extent),
      source_region_(ImageBox{0, 0, static_cast<double>(extent.width),
                              static_cast<double>(extent.height)}) {
  CheckExtent(extent);
}

GeometryTrace GeometryTrace::next() const {
  if (operations_ >= 128)
    throw ResourceExhausted("Image geometry operation limit exceeded",
                            ErrorDetails{.component = "data.image.geometry",
                                         .operation = "geometry.compose",
                                         .expected = "at most 128 operations"});
  auto result = *this;
  ++result.operations_;
  return result;
}

void GeometryTrace::validate_mapping() const {
  for (const auto value : {scale_x_, scale_y_, offset_x_, offset_y_,
                           1.0 / scale_x_, 1.0 / scale_y_}) {
    if (!std::isfinite(value))
      Invalid("geometry.compose", "finite invertible coordinate coefficients");
  }
  if (scale_x_ == 0 || scale_y_ == 0)
    Invalid("geometry.compose", "nonzero coordinate scales");
  if (const auto region = output_region()) {
    CheckBox(*region);
  }
}

GeometryTrace GeometryTrace::resized(ImageExtent extent) const {
  CheckExtent(extent);
  auto result = next();
  const auto sx = static_cast<double>(extent.width) / output_.width;
  const auto sy = static_cast<double>(extent.height) / output_.height;
  result.scale_x_ *= sx;
  result.offset_x_ *= sx;
  result.scale_y_ *= sy;
  result.offset_y_ *= sy;
  result.output_ = extent;
  result.validate_mapping();
  return result;
}

GeometryTrace GeometryTrace::cropped(std::uint32_t left, std::uint32_t top,
                                     ImageExtent extent) const {
  CheckExtent(extent);
  if (left > output_.width || top > output_.height ||
      extent.width > output_.width - left ||
      extent.height > output_.height - top)
    Invalid("geometry.crop", "a crop inside the current canvas");
  auto result = next();
  const ImageBox crop{static_cast<double>(left), static_cast<double>(top),
                      static_cast<double>(left) + extent.width,
                      static_cast<double>(top) + extent.height};
  const auto content = output_region();
  if (content && !(Contains(crop, {content->x_min, content->y_min}) &&
                   Contains(crop, {content->x_max, content->y_max}))) {
    result.source_region_ = inverse_box(crop);
  }
  result.offset_x_ -= left;
  result.offset_y_ -= top;
  result.output_ = extent;
  result.validate_mapping();
  return result;
}

GeometryTrace GeometryTrace::padded(std::uint32_t left, std::uint32_t top,
                                    std::uint32_t right,
                                    std::uint32_t bottom) const {
  const auto width = static_cast<std::uint64_t>(output_.width) + left + right;
  const auto height = static_cast<std::uint64_t>(output_.height) + top + bottom;
  if (width > std::numeric_limits<std::uint32_t>::max() ||
      height > std::numeric_limits<std::uint32_t>::max())
    Invalid("geometry.pad", "output dimensions within uint32 range");
  auto result = next();
  result.output_ = {static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height)};
  result.offset_x_ += left;
  result.offset_y_ += top;
  result.validate_mapping();
  return result;
}

GeometryTrace GeometryTrace::flipped_horizontal() const {
  auto result = next();
  result.scale_x_ = -scale_x_;
  result.offset_x_ = output_.width - offset_x_;
  result.validate_mapping();
  return result;
}

GeometryTrace GeometryTrace::flipped_vertical() const {
  auto result = next();
  result.scale_y_ = -scale_y_;
  result.offset_y_ = output_.height - offset_y_;
  result.validate_mapping();
  return result;
}

std::optional<ImagePoint> GeometryTrace::forward_point(ImagePoint point) const {
  CheckPoint(point);
  if (!source_region_ || !Contains(*source_region_, point)) return std::nullopt;
  return ImagePoint{point.x * scale_x_ + offset_x_,
                    point.y * scale_y_ + offset_y_};
}

std::optional<ImagePoint> GeometryTrace::inverse_point(ImagePoint point) const {
  CheckPoint(point);
  const auto region = output_region();
  if (!region || !Contains(*region, point)) return std::nullopt;
  return ImagePoint{(point.x - offset_x_) / scale_x_,
                    (point.y - offset_y_) / scale_y_};
}

std::optional<ImageBox> GeometryTrace::forward_box(ImageBox box) const {
  CheckBox(box);
  if (!source_region_) return std::nullopt;
  const auto clipped = Intersect(box, *source_region_);
  if (!clipped) return std::nullopt;
  return MapBox(*clipped, scale_x_, scale_y_, offset_x_, offset_y_);
}

std::optional<ImageBox> GeometryTrace::inverse_box(ImageBox box) const {
  CheckBox(box);
  const auto region = output_region();
  if (!region) return std::nullopt;
  const auto clipped = Intersect(box, *region);
  if (!clipped) return std::nullopt;
  const auto source =
      UnmapBox(*clipped, scale_x_, scale_y_, offset_x_, offset_y_);
  return Intersect(source, *source_region_);
}

ImageExtent GeometryTrace::source_extent() const noexcept { return source_; }
ImageExtent GeometryTrace::output_extent() const noexcept { return output_; }

std::optional<ImageBox> GeometryTrace::source_region() const noexcept {
  return source_region_;
}

std::optional<ImageBox> GeometryTrace::output_region() const noexcept {
  if (!source_region_) return std::nullopt;
  return MapBox(*source_region_, scale_x_, scale_y_, offset_x_, offset_y_);
}

bool GeometryTrace::retains_full_source() const noexcept {
  return source_region_ && source_region_->x_min == 0 &&
         source_region_->y_min == 0 && source_region_->x_max == source_.width &&
         source_region_->y_max == source_.height;
}

std::array<double, 9> GeometryTrace::matrix() const noexcept {
  return {scale_x_, 0, offset_x_, 0, scale_y_, offset_y_, 0, 0, 1};
}

}  // namespace jyppx::liuding::data::image
