// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/visualization/render.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace jyppx::liuding::visualization {
namespace {

using data::image::ImageBox;
using data::image::ImageLayout;
using data::image::ImagePoint;
using data::image::PixelAlphaMode;
using data::image::PixelColorModel;
using data::image::PixelSpec;
using data::image::PixelValueRange;
using data::image::VisionAnnotationRecord;
using data::image::VisionCategory;
using data::image::VisionObject;

constexpr std::array<RgbColor, 12> kPalette{{
    {230, 25, 75},
    {60, 180, 75},
    {255, 225, 25},
    {0, 130, 200},
    {245, 130, 48},
    {145, 30, 180},
    {70, 240, 240},
    {240, 50, 230},
    {210, 245, 60},
    {250, 190, 190},
    {0, 128, 128},
    {170, 110, 40},
}};

[[noreturn]] void Invalid(std::string operation, std::string actual = {}) {
  throw InvalidArgument("Invalid image render input",
                        {.component = "vision_render",
                         .operation = std::move(operation),
                         .expected = "valid bounded CPU image render",
                         .actual = std::move(actual)});
}

[[noreturn]] void Exhausted(std::string operation, std::string expected,
                            std::string actual) {
  throw ResourceExhausted("Image render resource limit exceeded",
                          {.component = "vision_render",
                           .operation = std::move(operation),
                           .expected = std::move(expected),
                           .actual = std::move(actual)});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Image render was cancelled");
}

std::uint64_t CheckedPixels(std::uint64_t width, std::uint64_t height) {
  if (width != 0U && height > std::numeric_limits<std::uint64_t>::max() / width)
    Exhausted("image.pixels", "representable pixel count", "overflow");
  return width * height;
}

std::uint64_t HashId(std::int64_t id) noexcept {
  auto value = static_cast<std::uint64_t>(id) + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

RgbColor CategoryColor(std::size_t category) noexcept {
  return kPalette[category % kPalette.size()];
}

RgbColor MaskColor(std::int64_t id) noexcept {
  return CategoryColor(static_cast<std::size_t>(HashId(id)));
}

void ValidatePoint(ImagePoint point);

bool ClipTest(double p, double q, double& first, double& last) noexcept {
  if (p == 0.0) return q >= 0.0;
  const auto ratio = q / p;
  if (p < 0.0) {
    if (ratio > last) return false;
    if (ratio > first) first = ratio;
  } else {
    if (ratio < first) return false;
    if (ratio < last) last = ratio;
  }
  return true;
}

class Canvas final {
 public:
  Canvas(std::uint8_t* data, std::uint64_t width, std::uint64_t height,
         const RenderOptions& options)
      : data_(data), width_(width), height_(height), options_(options) {}

  void blend(std::int64_t x, std::int64_t y, RgbColor color,
             std::uint8_t alpha) {
    if (x < 0 || y < 0 || static_cast<std::uint64_t>(x) >= width_ ||
        static_cast<std::uint64_t>(y) >= height_ || alpha == 0U)
      return;
    spend();
    const auto offset =
        (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
         static_cast<std::size_t>(x)) *
        3U;
    const auto inverse = static_cast<std::uint32_t>(255U - alpha);
    const auto weight = static_cast<std::uint32_t>(alpha);
    data_[offset] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(data_[offset]) * inverse +
         static_cast<std::uint32_t>(color.red) * weight + 127U) /
        255U);
    data_[offset + 1U] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(data_[offset + 1U]) * inverse +
         static_cast<std::uint32_t>(color.green) * weight + 127U) /
        255U);
    data_[offset + 2U] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(data_[offset + 2U]) * inverse +
         static_cast<std::uint32_t>(color.blue) * weight + 127U) /
        255U);
  }

  void line(ImagePoint from, ImagePoint to, RgbColor color,
            std::uint32_t thickness, std::stop_token stop) {
    ValidatePoint(from);
    ValidatePoint(to);
    double first = 0.0;
    double last = 1.0;
    const double dx = to.x - from.x;
    const double dy = to.y - from.y;
    const auto max_x = static_cast<double>(width_ - 1U);
    const auto max_y = static_cast<double>(height_ - 1U);
    if (!ClipTest(-dx, from.x, first, last) ||
        !ClipTest(dx, max_x - from.x, first, last) ||
        !ClipTest(-dy, from.y, first, last) ||
        !ClipTest(dy, max_y - from.y, first, last))
      return;

    const auto x0 =
        static_cast<std::int64_t>(std::llround(from.x + first * dx));
    const auto y0 =
        static_cast<std::int64_t>(std::llround(from.y + first * dy));
    const auto x1 = static_cast<std::int64_t>(std::llround(from.x + last * dx));
    const auto y1 = static_cast<std::int64_t>(std::llround(from.y + last * dy));
    auto x = x0;
    auto y = y0;
    const auto delta_x = std::llabs(x1 - x0);
    const auto step_x = x0 < x1 ? 1 : -1;
    const auto delta_y = -std::llabs(y1 - y0);
    const auto step_y = y0 < y1 ? 1 : -1;
    auto error = delta_x + delta_y;
    const auto before = static_cast<std::int64_t>(thickness / 2U);
    const auto after = static_cast<std::int64_t>(thickness) - before;
    for (;;) {
      CheckStop(stop);
      for (std::int64_t oy = -before; oy < after; ++oy)
        for (std::int64_t ox = -before; ox < after; ++ox)
          blend(x + ox, y + oy, color, 255U);
      if (x == x1 && y == y1) break;
      const auto twice = 2 * error;
      if (twice >= delta_y) {
        error += delta_y;
        x += step_x;
      }
      if (twice <= delta_x) {
        error += delta_x;
        y += step_y;
      }
    }
  }

  void point(ImagePoint center, RgbColor color, std::uint32_t radius,
             std::stop_token stop) {
    ValidatePoint(center);
    if (center.x < -static_cast<double>(radius) ||
        center.y < -static_cast<double>(radius) ||
        center.x > static_cast<double>(width_ - 1U + radius) ||
        center.y > static_cast<double>(height_ - 1U + radius))
      return;
    const auto x = static_cast<std::int64_t>(std::llround(center.x));
    const auto y = static_cast<std::int64_t>(std::llround(center.y));
    const auto radius_i = static_cast<std::int64_t>(radius);
    const auto radius_squared = radius_i * radius_i;
    for (auto dy = -radius_i; dy <= radius_i; ++dy) {
      CheckStop(stop);
      for (auto dx = -radius_i; dx <= radius_i; ++dx)
        if (dx * dx + dy * dy <= radius_squared)
          blend(x + dx, y + dy, color, 255U);
    }
  }

  void polygon(std::span<const ImagePoint> points, RgbColor color,
               std::uint32_t thickness, std::stop_token stop) {
    if (points.size() < 2U) return;
    for (std::size_t index = 0; index < points.size(); ++index)
      line(points[index], points[(index + 1U) % points.size()], color,
           thickness, stop);
  }

  void spend() {
    if (operations_ >= options_.max_raster_ops)
      Exhausted("raster.operations", std::to_string(options_.max_raster_ops),
                std::to_string(operations_ + 1U));
    ++operations_;
  }

 private:
  std::uint8_t* data_;
  std::uint64_t width_;
  std::uint64_t height_;
  const RenderOptions& options_;
  std::uint64_t operations_ = 0;
};

void ValidatePoint(ImagePoint point) {
  constexpr auto kCoordinateLimit = std::numeric_limits<double>::max() / 4.0;
  if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
      std::abs(point.x) > kCoordinateLimit ||
      std::abs(point.y) > kCoordinateLimit)
    Invalid("annotation.coordinate", "non-finite or unrepresentable");
}

void ValidateBox(const ImageBox& box) {
  ValidatePoint({box.x_min, box.y_min});
  ValidatePoint({box.x_max, box.y_max});
  if (box.x_min >= box.x_max || box.y_min >= box.y_max)
    Invalid("annotation.box", "non-positive extent");
}

void DrawBox(Canvas& canvas, const ImageBox& box, RgbColor color,
             std::uint32_t thickness, std::stop_token stop) {
  ValidateBox(box);
  const std::array<ImagePoint, 4> corners{{
      {box.x_min, box.y_min},
      {box.x_max, box.y_min},
      {box.x_max, box.y_max},
      {box.x_min, box.y_max},
  }};
  canvas.polygon(corners, color, thickness, stop);
}

void DrawRle(Canvas& canvas, const VisionObject& object, std::uint64_t width,
             std::uint64_t height, std::uint8_t alpha, std::stop_token stop) {
  if (!object.rle) return;
  const auto& rle = *object.rle;
  if (rle.width != width || rle.height != height)
    Invalid("annotation.rle.extent", "does not match image canvas");
  const auto pixel_count = CheckedPixels(width, height);
  std::uint64_t cursor = 0;
  bool foreground = false;
  const auto color = CategoryColor(object.category);
  for (const auto run : rle.counts) {
    if (run > pixel_count - cursor)
      Invalid("annotation.rle.counts", "run total exceeds canvas");
    if (foreground) {
      const auto end = cursor + run;
      for (auto offset = cursor; offset < end; ++offset) {
        if ((offset & 4095U) == 0U) CheckStop(stop);
        const auto x = offset / height;
        const auto y = offset % height;
        canvas.blend(static_cast<std::int64_t>(x), static_cast<std::int64_t>(y),
                     color, alpha);
      }
    }
    cursor += run;
    foreground = !foreground;
  }
  if (cursor != pixel_count)
    Invalid("annotation.rle.counts", "run total does not match canvas");
}

void DrawObject(Canvas& canvas, const VisionObject& object,
                std::span<const VisionCategory> categories,
                const RenderOptions& options, std::stop_token stop) {
  const auto color = CategoryColor(object.category);
  if (options.draw_boxes && object.box)
    DrawBox(canvas, *object.box, color, options.line_width, stop);
  if (options.draw_oriented_boxes && object.oriented_box) {
    for (std::size_t index = 0; index < object.oriented_box->size(); ++index)
      canvas.line((*object.oriented_box)[index],
                  (*object.oriented_box)[(index + 1U) % 4U], color,
                  options.line_width, stop);
  }
  if (options.draw_polygons)
    for (const auto& ring : object.polygons)
      canvas.polygon(ring, color, options.line_width, stop);
  if (!options.draw_keypoints) return;

  if (object.category < categories.size()) {
    for (const auto& [first, second] : categories[object.category].skeleton) {
      if (first >= object.keypoints.size() || second >= object.keypoints.size())
        Invalid("annotation.skeleton", "index exceeds keypoint count");
      const auto& a = object.keypoints[first];
      const auto& b = object.keypoints[second];
      if (a.visibility != 0U && b.visibility != 0U)
        canvas.line(a.point, b.point, color, options.line_width, stop);
    }
  }
  for (const auto& keypoint : object.keypoints) {
    if (keypoint.visibility > 2U)
      Invalid("annotation.keypoint.visibility", "expected 0, 1, or 2");
    if (keypoint.visibility != 0U)
      canvas.point(keypoint.point, color, options.point_radius, stop);
  }
}

RgbColor HeatColor(double value) noexcept {
  if (value < 0.5) {
    const auto t = value * 2.0;
    return {static_cast<std::uint8_t>(std::lround(255.0 * t)),
            static_cast<std::uint8_t>(std::lround(255.0 * t)),
            static_cast<std::uint8_t>(std::lround(255.0 * (1.0 - t)))};
  }
  const auto t = (value - 0.5) * 2.0;
  return {255, static_cast<std::uint8_t>(std::lround(255.0 * (1.0 - t))), 0};
}

}  // namespace

RenderedImage render_annotations(const Tensor& pixels,
                                 const PixelSpec& pixel_spec,
                                 const VisionAnnotationRecord& annotations,
                                 std::span<const VisionCategory> categories,
                                 const Tensor& mask, const Tensor& heatmap,
                                 const RenderOptions& options,
                                 std::stop_token stop) {
  if (options.line_width == 0U || options.line_width > 128U ||
      options.point_radius > 512U || options.max_pixels == 0U ||
      options.max_output_bytes == 0U || options.max_objects == 0U ||
      options.max_points == 0U || options.max_raster_ops == 0U)
    Invalid("options", "zero or out-of-range limit");
  data::image::validate_pixels(pixels, pixel_spec);
  if (!pixels.device().is_cpu() || pixels.requires_grad() ||
      pixels.scalar_type() != torch::kUInt8 ||
      pixel_spec.range != PixelValueRange::unsigned_full_scale)
    Invalid("image", "expected non-grad CPU uint8 full-scale pixels");

  const auto height_index = pixel_spec.layout == ImageLayout::chw ? 1 : 0;
  const auto width_index = pixel_spec.layout == ImageLayout::chw ? 2 : 1;
  const auto channel_index = pixel_spec.layout == ImageLayout::chw ? 0 : 2;
  const auto height = static_cast<std::uint64_t>(pixels.size(height_index));
  const auto width = static_cast<std::uint64_t>(pixels.size(width_index));
  if (width > std::numeric_limits<std::uint32_t>::max() ||
      height > std::numeric_limits<std::uint32_t>::max() ||
      annotations.width != width || annotations.height != height)
    Invalid("annotation.extent", "does not match image extent");
  const auto pixel_count = CheckedPixels(width, height);
  if (pixel_count > options.max_pixels)
    Exhausted("image.pixels", std::to_string(options.max_pixels),
              std::to_string(pixel_count));
  if (pixel_count > options.max_raster_ops)
    Exhausted("raster.operations", std::to_string(options.max_raster_ops),
              "minimum image conversion work exceeds the operation budget");
  if (pixel_count > options.max_output_bytes / 3U ||
      pixel_count > std::numeric_limits<std::size_t>::max() / 3U)
    Exhausted("image.output_bytes", std::to_string(options.max_output_bytes),
              "RGB8 output size overflow or limit exceeded");
  if (annotations.objects.size() > options.max_objects)
    Exhausted("annotations.objects", std::to_string(options.max_objects),
              std::to_string(annotations.objects.size()));
  if (categories.size() > options.max_objects)
    Exhausted("annotations.categories", std::to_string(options.max_objects),
              std::to_string(categories.size()));

  std::size_t point_count = 0;
  for (const auto& category : categories) {
    if (category.skeleton.size() > options.max_points - point_count)
      Exhausted("annotations.skeleton", std::to_string(options.max_points),
                "skeleton edge count exceeded");
    point_count += category.skeleton.size();
  }
  for (const auto& object : annotations.objects) {
    CheckStop(stop);
    if (object.category >= categories.size() && !categories.empty())
      Invalid("annotation.category", "index exceeds category count");
    if (object.box) {
      ValidateBox(*object.box);
      if (4U > options.max_points - point_count)
        Exhausted("annotations.points", std::to_string(options.max_points),
                  "box vertex count exceeded");
      point_count += 4U;
    }
    if (object.oriented_box) {
      for (const auto point : *object.oriented_box) ValidatePoint(point);
      if (4U > options.max_points - point_count)
        Exhausted("annotations.points", std::to_string(options.max_points),
                  "oriented-box vertex count exceeded");
      point_count += 4U;
    }
    for (const auto& ring : object.polygons) {
      const auto available = options.max_points - point_count;
      if (ring.size() >= available)
        Exhausted("annotations.points", std::to_string(options.max_points),
                  "polygon point count exceeded");
      point_count += ring.size() + 1U;
      for (const auto point : ring) ValidatePoint(point);
    }
    if (object.keypoints.size() > options.max_points - point_count)
      Exhausted("annotations.points", std::to_string(options.max_points),
                "keypoint count exceeded");
    point_count += object.keypoints.size();
    for (const auto& keypoint : object.keypoints) {
      ValidatePoint(keypoint.point);
      if (keypoint.visibility > 2U)
        Invalid("annotation.keypoint.visibility", "expected 0, 1, or 2");
    }
    if (object.rle) {
      if (object.rle->counts.size() > options.max_points - point_count)
        Exhausted("annotations.rle_runs", std::to_string(options.max_points),
                  "RLE run count exceeded");
      point_count += object.rle->counts.size();
    }
  }

  if (mask.defined() &&
      (!mask.device().is_cpu() || mask.requires_grad() || mask.dim() != 2 ||
       mask.scalar_type() != torch::kInt64 ||
       mask.size(0) != static_cast<std::int64_t>(height) ||
       mask.size(1) != static_cast<std::int64_t>(width) ||
       mask.layout() != torch::kStrided))
    Invalid("mask", "expected CPU int64 [H,W] IDs matching the image");
  if (heatmap.defined() &&
      (!heatmap.device().is_cpu() || heatmap.requires_grad() ||
       heatmap.dim() != 2 ||
       (heatmap.scalar_type() != torch::kFloat32 &&
        heatmap.scalar_type() != torch::kFloat64) ||
       heatmap.size(0) != static_cast<std::int64_t>(height) ||
       heatmap.size(1) != static_cast<std::int64_t>(width) ||
       heatmap.layout() != torch::kStrided))
    Invalid("heatmap", "expected CPU float32/float64 [H,W]");

  CheckStop(stop);
  auto output = torch::empty(
      {static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  const auto* input = pixels.const_data_ptr<std::uint8_t>();
  auto* rendered = output.data_ptr<std::uint8_t>();
  Canvas canvas(rendered, width, height, options);
  const auto channel_stride = pixels.stride(channel_index);
  const auto row_stride = pixels.stride(height_index);
  const auto column_stride = pixels.stride(width_index);
  const auto background = options.background;

  const bool gray = pixel_spec.color == PixelColorModel::gray ||
                    pixel_spec.color == PixelColorModel::gray_alpha;
  const bool bgr = pixel_spec.color == PixelColorModel::bgr ||
                   pixel_spec.color == PixelColorModel::bgra;
  const bool has_alpha = pixel_spec.alpha != PixelAlphaMode::none;
  const auto alpha_index = gray ? 1 : 3;
  for (std::uint64_t y = 0; y < height; ++y) {
    CheckStop(stop);
    for (std::uint64_t x = 0; x < width; ++x) {
      canvas.spend();
      const auto source_offset = static_cast<std::int64_t>(y) * row_stride +
                                 static_cast<std::int64_t>(x) * column_stride;
      const auto channel = [&](std::uint64_t index) {
        return input[source_offset +
                     static_cast<std::int64_t>(index) * channel_stride];
      };
      std::uint8_t red = channel(gray ? 0U : (bgr ? 2U : 0U));
      std::uint8_t green = gray ? red : channel(1U);
      std::uint8_t blue = gray ? red : channel(bgr ? 0U : 2U);
      const auto alpha = has_alpha
                             ? channel(static_cast<std::uint64_t>(alpha_index))
                             : static_cast<std::uint8_t>(255U);
      const auto composite = [&](std::uint8_t source, std::uint8_t back) {
        if (!has_alpha) return source;
        const auto foreground =
            pixel_spec.alpha == PixelAlphaMode::straight
                ? (static_cast<std::uint32_t>(source) * alpha + 127U) / 255U
                : source;
        return static_cast<std::uint8_t>(std::min(
            255U,
            foreground +
                (static_cast<std::uint32_t>(back) * (255U - alpha) + 127U) /
                    255U));
      };
      const auto target_offset =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)) *
          3U;
      rendered[target_offset] = composite(red, background.red);
      rendered[target_offset + 1U] = composite(green, background.green);
      rendered[target_offset + 2U] = composite(blue, background.blue);
    }
  }

  if (heatmap.defined()) {
    const auto row_stride_heat = heatmap.stride(0);
    const auto column_stride_heat = heatmap.stride(1);
    for (std::uint64_t y = 0; y < height; ++y) {
      CheckStop(stop);
      for (std::uint64_t x = 0; x < width; ++x) {
        canvas.spend();
        const auto offset = static_cast<std::int64_t>(y) * row_stride_heat +
                            static_cast<std::int64_t>(x) * column_stride_heat;
        const auto value =
            heatmap.scalar_type() == torch::kFloat32
                ? static_cast<double>(heatmap.const_data_ptr<float>()[offset])
                : heatmap.const_data_ptr<double>()[offset];
        if (!std::isfinite(value) || value < 0.0 || value > 1.0)
          Invalid("heatmap.value", "expected finite value in [0,1]");
        const auto alpha = static_cast<std::uint8_t>(
            std::lround(value * options.heatmap_alpha));
        canvas.blend(static_cast<std::int64_t>(x), static_cast<std::int64_t>(y),
                     HeatColor(value), alpha);
      }
    }
  }
  if (mask.defined()) {
    const auto* ids = mask.const_data_ptr<std::int64_t>();
    const auto row_stride_mask = mask.stride(0);
    const auto column_stride_mask = mask.stride(1);
    for (std::uint64_t y = 0; y < height; ++y) {
      CheckStop(stop);
      for (std::uint64_t x = 0; x < width; ++x) {
        canvas.spend();
        const auto id = ids[static_cast<std::int64_t>(y) * row_stride_mask +
                            static_cast<std::int64_t>(x) * column_stride_mask];
        if (options.mask_ignore_id && id == *options.mask_ignore_id) continue;
        canvas.blend(static_cast<std::int64_t>(x), static_cast<std::int64_t>(y),
                     MaskColor(id), options.mask_alpha);
      }
    }
  }
  for (const auto& object : annotations.objects) {
    CheckStop(stop);
    if (options.draw_instance_rle && object.rle)
      DrawRle(canvas, object, width, height, options.mask_alpha, stop);
    DrawObject(canvas, object, categories, options, stop);
  }
  CheckStop(stop);

  data::image::PixelSpec output_spec;
  output_spec.layout = ImageLayout::hwc;
  output_spec.color = PixelColorModel::rgb;
  output_spec.alpha = PixelAlphaMode::none;
  output_spec.dtype = torch::kUInt8;
  output_spec.range = PixelValueRange::unsigned_full_scale;
  return {.pixels = std::move(output), .spec = output_spec};
}

}  // namespace jyppx::liuding::visualization
