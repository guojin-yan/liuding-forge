// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/opencv_preparation.hpp>

#ifdef JYPPX_VISION_WITH_OPENCV
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <type_traits>
#endif

namespace jyppx::liuding::vision::tasks {
bool yolo_opencv_available() noexcept {
#ifdef JYPPX_VISION_WITH_OPENCV
  return true;
#else
  return false;
#endif
}
#ifdef JYPPX_VISION_WITH_OPENCV
namespace {
namespace im = data::image;
using M = std::array<float, 9>;
using D = std::array<double, 9>;
void Stop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("OpenCV preparation cancelled");
}
void Canvas(im::ImageExtent e, const im::JointTransformOptions& o) {
  const auto n = std::uint64_t(e.width) * e.height;
  if (!e.width || !e.height) throw InvalidArgument("Empty OpenCV canvas");
  if (e.width > 8192 || e.height > 8192 || e.width > o.limits.max_dimension ||
      e.height > o.limits.max_dimension || n > o.limits.max_pixels ||
      n > o.limits.max_output_bytes / 3)
    throw ResourceExhausted("OpenCV canvas/output budget exceeded");
}
void Check(const im::VisionSample& s, const im::JointTransformOptions& o) {
  Canvas({s.width, s.height}, o);
  im::validate_pixels(s.image, s.pixels);
  if (!s.image.is_cpu() || s.image.requires_grad() || s.image.dim() != 3 ||
      s.pixels.dtype != torch::kUInt8 ||
      s.pixels.color != im::PixelColorModel::rgb ||
      s.pixels.range != im::PixelValueRange::unsigned_full_scale ||
      s.paired_target || s.mask_target || s.task_payload || !s.annotation ||
      !s.categories || s.categories->empty())
    throw InvalidArgument(
        "OpenCV preparation requires annotated CPU uint8 RGB boxes");
  auto chw = s.pixels.layout == im::ImageLayout::chw
                 ? s.image
                 : s.image.permute({2, 0, 1});
  if (chw.size(0) != 3 || chw.size(1) != s.height || chw.size(2) != s.width ||
      s.annotation->width != s.width || s.annotation->height != s.height ||
      s.annotation->panoptic_mask_path)
    throw InvalidArgument("OpenCV preparation extent mismatch");
  if (s.image_path.empty() ||
      (s.geometry && (s.geometry->output_extent().width != s.width ||
                      s.geometry->output_extent().height != s.height)))
    throw InvalidArgument("OpenCV source identity or geometry mismatch");
  if (s.annotation->objects.size() > o.limits.max_objects)
    throw ResourceExhausted("OpenCV input object budget exceeded");
  for (const auto& x : s.annotation->objects) {
    if (!x.box || !x.polygons.empty() || !x.keypoints.empty() || x.rle ||
        x.oriented_box || x.category >= s.categories->size() ||
        x.augmentation_sources.size() > 16)
      throw InvalidArgument(
          "OpenCV preparation accepts axis-aligned boxes only");
    const auto& b = *x.box;
    for (auto v : {b.x_min, b.y_min, b.x_max, b.y_max})
      if (!std::isfinite(v))
        throw InvalidArgument("Nonfinite OpenCV input box");
    if (b.x_min < 0 || b.y_min < 0 || b.x_max > s.width || b.y_max > s.height ||
        b.x_max <= b.x_min || b.y_max <= b.y_min)
      throw InvalidArgument("OpenCV input box outside source canvas");
    for (const auto& source : x.augmentation_sources) {
      if (source.image_path.empty() || !std::isfinite(source.weight) ||
          source.weight <= 0 || source.weight > 1)
        throw InvalidArgument("Invalid OpenCV object provenance");
      for (const auto value : source.matrix)
        if (!std::isfinite(value))
          throw InvalidArgument("Nonfinite OpenCV object provenance");
    }
  }
}
void SameCategories(const im::VisionSample& a, const im::VisionSample& b) {
  if (a.categories->size() != b.categories->size())
    throw InvalidArgument("OpenCV category map mismatch");
  for (std::size_t i = 0; i < a.categories->size(); ++i)
    if ((*a.categories)[i].source_id != (*b.categories)[i].source_id ||
        (*a.categories)[i].name != (*b.categories)[i].name)
      throw InvalidArgument("OpenCV category map mismatch");
}
im::ImageResourceLease Reserve(const im::VisionSample& s,
                               const im::JointTransformOptions& o,
                               im::ImageResourceKind k, std::uint64_t n) {
  if (s.image_resource_leases)
    for (const auto& lease : *s.image_resource_leases)
      if (lease.active()) return lease.reserve_related(k, n);
  return o.resource_budget ? o.resource_budget->reserve(k, n)
                           : im::ImageResourceLease{};
}
struct Allocation {
  std::shared_ptr<im::ImageResourceLeaseSet> output;
  im::ImageResourceLease working;
};
Allocation Allocate(const im::VisionSample& s, im::ImageExtent canvas,
                    std::uint64_t working, const im::JointTransformOptions& o) {
  Canvas(canvas, o);
  if (working > o.limits.max_working_bytes)
    throw ResourceExhausted("OpenCV working budget exceeded");
  auto leases = std::make_shared<im::ImageResourceLeaseSet>();
  leases->push_back(Reserve(s, o, im::ImageResourceKind::pixels,
                            std::uint64_t(canvas.width) * canvas.height));
  leases->push_back(Reserve(s, o, im::ImageResourceKind::decoded_bytes,
                            std::uint64_t(canvas.width) * canvas.height * 3));
  auto scratch =
      Reserve(s, o, im::ImageResourceKind::codec_scratch_bytes, working);
  return {std::move(leases), std::move(scratch)};
}
struct Image {
  cv::Mat pixels;
  std::vector<im::VisionObject> objects;
};
cv::Mat Hwc(const im::VisionSample& s) {
  auto tensor =
      (s.pixels.layout == im::ImageLayout::chw ? s.image.permute({1, 2, 0})
                                               : s.image)
          .contiguous();
  return cv::Mat(int(s.height), int(s.width), CV_8UC3,
                 tensor.mutable_data_ptr<std::uint8_t>())
      .clone();
}
template <class T>
std::array<T, 9> Multiply(const std::array<T, 9>& a,
                          const std::array<T, 9>& b) {
  std::array<T, 9> out{};
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int k = 0; k < 3; ++k) {
        if constexpr (std::is_same_v<T, float>)
          out[y * 3 + x] = std::fma(a[y * 3 + k], b[k * 3 + x], out[y * 3 + x]);
        else
          out[y * 3 + x] += a[y * 3 + k] * b[k * 3 + x];
      }
  return out;
}
void Sources(im::VisionObject& object, const im::VisionSample& s) {
  if (object.augmentation_sources.empty())
    object.augmentation_sources.push_back(
        {.image_path = s.image_path,
         .object_id = object.source_id,
         .matrix =
             s.geometry ? s.geometry->matrix() : D{1, 0, 0, 0, 1, 0, 0, 0, 1}});
}
void MapSources(im::VisionObject& object, const D& m) {
  for (auto& source : object.augmentation_sources)
    source.matrix = Multiply(m, source.matrix);
}
void Bound(im::ImageBox& b, int w, int h) {
  b = {double(std::clamp(float(b.x_min), 0.f, float(w))),
       double(std::clamp(float(b.y_min), 0.f, float(h))),
       double(std::clamp(float(b.x_max), 0.f, float(w))),
       double(std::clamp(float(b.y_max), 0.f, float(h)))};
}
Image Loaded(const im::VisionSample& s, int side, std::stop_token stop) {
  Image result;
  const auto ratio = double(side) / std::max(s.width, s.height);
  const int w = std::min(side, int(std::ceil(s.width * ratio))),
            h = std::min(side, int(std::ceil(s.height * ratio)));
  auto input = Hwc(s);
  Stop(stop);
  cv::resize(input, result.pixels, {w, h}, 0, 0, cv::INTER_LINEAR);
  for (auto object : s.annotation->objects) {
    Stop(stop);
    if (object.ignore || object.crowd) continue;
    auto& box = *object.box;
    box = {double(float(box.x_min / s.width) * w),
           double(float(box.y_min / s.height) * h),
           double(float(box.x_max / s.width) * w),
           double(float(box.y_max / s.height) * h)};
    if (box.x_max <= box.x_min || box.y_max <= box.y_min) continue;
    Sources(object, s);
    MapSources(object, D{double(w) / s.width, 0, 0, 0, double(h) / s.height, 0,
                         0, 0, 1});
    result.objects.push_back(std::move(object));
  }
  return result;
}
void Append(std::vector<im::VisionObject>& out, im::VisionObject object,
            std::size_t maximum) {
  if (out.size() >= maximum)
    throw ResourceExhausted("OpenCV output object budget exceeded");
  out.push_back(std::move(object));
}
Image Mosaic(const std::array<Image, 4>& inputs, int side, int xc, int yc,
             const im::JointTransformOptions& o, std::stop_token stop) {
  Image out{cv::Mat(side * 2, side * 2, CV_8UC3, cv::Scalar::all(114)), {}};
  for (int i = 0; i < 4; ++i) {
    Stop(stop);
    const auto& input = inputs[i];
    int w = input.pixels.cols, h = input.pixels.rows;
    int ax1, ay1, ax2, ay2, bx1, by1;
    if (i == 0) {
      ax1 = std::max(xc - w, 0);
      ay1 = std::max(yc - h, 0);
      ax2 = xc;
      ay2 = yc;
      bx1 = w - (ax2 - ax1);
      by1 = h - (ay2 - ay1);
    } else if (i == 1) {
      ax1 = xc;
      ay1 = std::max(yc - h, 0);
      ax2 = std::min(xc + w, 2 * side);
      ay2 = yc;
      bx1 = 0;
      by1 = h - (ay2 - ay1);
    } else if (i == 2) {
      ax1 = std::max(xc - w, 0);
      ay1 = yc;
      ax2 = xc;
      ay2 = std::min(yc + h, 2 * side);
      bx1 = w - (ax2 - ax1);
      by1 = 0;
    } else {
      ax1 = xc;
      ay1 = yc;
      ax2 = std::min(xc + w, 2 * side);
      ay2 = std::min(yc + h, 2 * side);
      bx1 = by1 = 0;
    }
    input.pixels(cv::Rect(bx1, by1, ax2 - ax1, ay2 - ay1))
        .copyTo(out.pixels(cv::Rect(ax1, ay1, ax2 - ax1, ay2 - ay1)));
    const int dx = ax1 - bx1, dy = ay1 - by1;
    for (auto object : input.objects) {
      auto& b = *object.box;
      b = {double(float(b.x_min) + dx), double(float(b.y_min) + dy),
           double(float(b.x_max) + dx), double(float(b.y_max) + dy)};
      Bound(b, side * 2, side * 2);
      MapSources(object, D{1, 0, double(dx), 0, 1, double(dy), 0, 0, 1});
      if (b.x_max > b.x_min && b.y_max > b.y_min)
        Append(out.objects, std::move(object), o.limits.max_objects);
    }
  }
  return out;
}
M Matrix(const YoloOpenCvGeometryPlan& p, int w, int h) {
  const M center{1, 0, -float(w) / 2, 0, 1, -float(h) / 2, 0, 0, 1};
  const M perspective{
      1, 0, 0, 0, 1, 0, float(p.perspective_x), float(p.perspective_y), 1};
  const auto rotation =
      cv::getRotationMatrix2D({0, 0}, p.rotation_degrees, p.scale);
  M r{0, 0, 0, 0, 0, 0, 0, 0, 1};
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 3; ++x) r[3 * y + x] = float(rotation.at<double>(y, x));
  const M shear{1, float(std::tan(p.shear_x_degrees * std::numbers::pi / 180)),
                0, float(std::tan(p.shear_y_degrees * std::numbers::pi / 180)),
                1, 0,
                0, 0,
                1};
  const M translate{1, 0, float(p.translate_x * p.size),
                    0, 1, float(p.translate_y * p.size),
                    0, 0, 1};
  return Multiply(
      Multiply(Multiply(Multiply(translate, shear), r), perspective), center);
}
Image Warp(const Image& input, const YoloOpenCvGeometryPlan& p,
           std::stop_token stop) {
  const auto m = Matrix(p, input.pixels.cols, input.pixels.rows);
  cv::Mat matrix(3, 3, CV_32F, const_cast<float*>(m.data()));
  Image out;
  Stop(stop);
  if (p.perspective)
    cv::warpPerspective(input.pixels, out.pixels, matrix,
                        {int(p.size), int(p.size)}, cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar::all(114));
  else
    cv::warpAffine(input.pixels, out.pixels, matrix.rowRange(0, 2),
                   {int(p.size), int(p.size)}, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(114));
  D provenance;
  std::copy(m.begin(), m.end(), provenance.begin());
  for (auto object : input.objects) {
    Stop(stop);
    const auto before = *object.box;
    std::array<float, 4> box{INFINITY, INFINITY, -INFINITY, -INFINITY};
    for (int corner = 0; corner < 4; ++corner) {
      const float x = float((corner & 1) ? before.x_max : before.x_min),
                  y = float((corner & 2) ? before.y_max : before.y_min);
      const float z = p.perspective ? std::fma(y, m[7], x * m[6]) + m[8] : 1;
      if (!std::isfinite(z) || z == 0)
        throw InvalidArgument("OpenCV box crosses projective horizon");
      const float xx = (std::fma(y, m[1], x * m[0]) + m[2]) / z,
                  yy = (std::fma(y, m[4], x * m[3]) + m[5]) / z;
      if (!std::isfinite(xx) || !std::isfinite(yy))
        throw InvalidArgument("Nonfinite OpenCV mapped box");
      box[0] = std::min(box[0], xx);
      box[1] = std::min(box[1], yy);
      box[2] = std::max(box[2], xx);
      box[3] = std::max(box[3], yy);
    }
    auto& b = *object.box;
    b = {box[0], box[1], box[2], box[3]};
    Bound(b, p.size, p.size);
    const float scale = float(p.scale),
                w1 = float(before.x_max) * scale - float(before.x_min) * scale,
                h1 = float(before.y_max) * scale - float(before.y_min) * scale;
    const float w2 = float(b.x_max) - float(b.x_min),
                h2 = float(b.y_max) - float(b.y_min);
    if (w2 > 2 && h2 > 2 && w2 * h2 / (w1 * h1 + 1e-16f) > .1f &&
        std::max(w2 / (h2 + 1e-16f), h2 / (w2 + 1e-16f)) < 100) {
      MapSources(object, provenance);
      out.objects.push_back(std::move(object));
    }
  }
  return out;
}
im::VisionSample Publish(const im::VisionSample& base, Image image,
                         Allocation& allocation, std::string identity,
                         std::optional<im::GeometryTrace> trace,
                         std::stop_token stop) {
  Stop(stop);
  auto out = base;
  out.width = image.pixels.cols;
  out.height = image.pixels.rows;
  out.image =
      torch::from_blob(image.pixels.data,
                       {image.pixels.rows, image.pixels.cols, 3}, torch::kUInt8)
          .permute({2, 0, 1})
          .contiguous();
  out.pixels.layout = im::ImageLayout::chw;
  out.image_resource_leases = std::move(allocation.output);
  out.geometry = std::move(trace);
  auto ann = std::make_shared<im::VisionAnnotationRecord>(*base.annotation);
  ann->width = out.width;
  ann->height = out.height;
  ann->objects = std::move(image.objects);
  if (!out.geometry) {
    out.image_path =
        "synthetic/" + sha256_hex(std::as_bytes(std::span(identity)));
    ann->image_path = out.image_path;
    ann->source_image_id = 0;
    std::int64_t id = 1;
    for (auto& object : ann->objects) object.source_id = id++;
  }
  out.annotation = std::move(ann);
  Stop(stop);
  return out;
}
void GeometryPlan(const YoloOpenCvGeometryPlan& p) {
  if (p.size < 2 || p.size > 4096 || (p.mosaic && !p.warp))
    throw InvalidArgument("Invalid OpenCV geometry size or uncropped mosaic");
  for (auto v :
       {p.perspective_x, p.perspective_y, p.rotation_degrees, p.scale,
        p.shear_x_degrees, p.shear_y_degrees, p.translate_x, p.translate_y})
    if (!std::isfinite(v))
      throw InvalidArgument("Nonfinite OpenCV geometry plan");
  if (p.scale <= 0 || p.scale > 4 || std::abs(p.rotation_degrees) > 180 ||
      std::abs(p.shear_x_degrees) >= 89 || std::abs(p.shear_y_degrees) >= 89 ||
      std::abs(p.perspective_x) > .01 || std::abs(p.perspective_y) > .01 ||
      p.translate_x < 0 || p.translate_x > 1 || p.translate_y < 0 ||
      p.translate_y > 1 ||
      (!p.perspective && (p.perspective_x || p.perspective_y)))
    throw InvalidArgument("OpenCV geometry plan outside bounds");
  if (p.mosaic)
    for (auto center : p.mosaic_center)
      if (center < p.size / 2 || center > p.size * 3 / 2)
        throw InvalidArgument("OpenCV mosaic center outside bounds");
}
}  // namespace
#endif

data::image::VisionSample prepare_yolo_opencv_geometry(
    std::span<const data::image::VisionSample> samples,
    const YoloOpenCvGeometryPlan& p,
    const data::image::JointTransformOptions& options, std::stop_token stop) {
#ifndef JYPPX_VISION_WITH_OPENCV
  static_cast<void>(samples);
  static_cast<void>(p);
  static_cast<void>(options);
  static_cast<void>(stop);
  throw NotImplemented("OpenCV preparation was not enabled in this build");
#else
  Stop(stop);
  GeometryPlan(p);
  if (samples.size() != (p.mosaic ? 4U : 1U))
    throw InvalidArgument("OpenCV geometry requires one or four samples");
  std::uint64_t working = std::uint64_t(p.size) * p.size * 80;
  for (const auto& s : samples) {
    Check(s, options);
    SameCategories(samples.front(), s);
    working += std::uint64_t(s.width) * s.height * 6;
  }
  if (p.mosaic) Canvas({p.size * 2, p.size * 2}, options);
  auto allocation =
      Allocate(samples.front(), {p.size, p.size}, working, options);
  Stop(stop);
  std::array<Image, 4> loaded;
  for (std::size_t i = 0; i < samples.size(); ++i)
    loaded[i] = Loaded(samples[i], p.size, stop);
  Image image;
  std::optional<im::GeometryTrace> trace;
  if (p.mosaic)
    image = Mosaic(loaded, p.size, p.mosaic_center[0], p.mosaic_center[1],
                   options, stop);
  else {
    image = std::move(loaded[0]);
    const int left = (int(p.size) - image.pixels.cols) / 2,
              top = (int(p.size) - image.pixels.rows) / 2;
    trace =
        samples[0]
            .geometry
            .value_or(im::GeometryTrace({samples[0].width, samples[0].height}))
            .resized({std::uint32_t(image.pixels.cols),
                      std::uint32_t(image.pixels.rows)})
            .padded(left, top, p.size - image.pixels.cols - left,
                    p.size - image.pixels.rows - top);
    cv::copyMakeBorder(image.pixels, image.pixels, top,
                       p.size - image.pixels.rows - top, left,
                       p.size - image.pixels.cols - left, cv::BORDER_CONSTANT,
                       cv::Scalar::all(114));
    for (auto& object : image.objects) {
      auto& b = *object.box;
      b = {double(float(b.x_min) + left), double(float(b.y_min) + top),
           double(float(b.x_max) + left), double(float(b.y_max) + top)};
      MapSources(object, D{1, 0, double(left), 0, 1, double(top), 0, 0, 1});
    }
  }
  if (p.warp) {
    image = Warp(image, p, stop);
    trace.reset();
  }
  std::ostringstream identity;
  identity.imbue(std::locale::classic());
  identity << std::setprecision(17) << "opencv-yolo-geometry/fma-v2:";
  for (const auto& s : samples)
    identity << s.image_path.size() << ':' << s.image_path << ':';
  identity << p.size << ':' << p.mosaic << ':' << p.mosaic_center[0] << ':'
           << p.mosaic_center[1] << ':' << p.warp << ':' << p.perspective;
  for (auto v :
       {p.perspective_x, p.perspective_y, p.rotation_degrees, p.scale,
        p.shear_x_degrees, p.shear_y_degrees, p.translate_x, p.translate_y})
    identity << ':' << v;
  return Publish(samples.front(), std::move(image), allocation, identity.str(),
                 std::move(trace), stop);
#endif
}

data::image::VisionSample prepare_yolo_opencv_color(
    const data::image::VisionSample& sample, const YoloOpenCvColorPlan& p,
    const data::image::VisionSample* partner,
    const data::image::JointTransformOptions& options, std::stop_token stop) {
#ifndef JYPPX_VISION_WITH_OPENCV
  static_cast<void>(sample);
  static_cast<void>(p);
  static_cast<void>(partner);
  static_cast<void>(options);
  static_cast<void>(stop);
  throw NotImplemented("OpenCV preparation was not enabled in this build");
#else
  Stop(stop);
  Check(sample, options);
  for (auto v : {p.mixup_ratio, p.hue_shift, p.saturation_gain, p.value_gain})
    if (!std::isfinite(v)) throw InvalidArgument("Nonfinite OpenCV color plan");
  if (p.mixup_ratio < 0 || p.mixup_ratio > 1 || std::abs(p.hue_shift) > 1 ||
      p.saturation_gain < 0 || p.saturation_gain > 2 || p.value_gain < 0 ||
      p.value_gain > 2 || (partner != nullptr) != (p.mixup_ratio < 1))
    throw InvalidArgument("Invalid OpenCV color plan or partner");
  if (partner) {
    Check(*partner, options);
    SameCategories(sample, *partner);
    if (partner->width != sample.width || partner->height != sample.height)
      throw InvalidArgument("OpenCV MixUp extent mismatch");
  }
  auto allocation =
      Allocate(sample, {sample.width, sample.height},
               std::uint64_t(sample.width) * sample.height * 80, options);
  Image out{Hwc(sample), {}};
  for (auto object : sample.annotation->objects) {
    if (object.ignore || object.crowd) continue;
    Sources(object, sample);
    Append(out.objects, std::move(object), options.limits.max_objects);
  }
  auto trace = sample.geometry;
  if (partner) {
    const auto other = Hwc(*partner);
    for (int y = 0; y < out.pixels.rows; ++y) {
      Stop(stop);
      auto* a = out.pixels.ptr<std::uint8_t>(y);
      const auto* b = other.ptr<std::uint8_t>(y);
      for (int x = 0; x < out.pixels.cols * 3; ++x)
        a[x] = std::uint8_t(double(a[x]) * p.mixup_ratio +
                            double(b[x]) * (1 - p.mixup_ratio));
    }
    for (auto& object : out.objects)
      for (auto& source : object.augmentation_sources)
        source.weight *= p.mixup_ratio;
    if (p.mixup_ratio == 0) out.objects.clear();
    for (auto object : partner->annotation->objects) {
      if (object.ignore || object.crowd) continue;
      Sources(object, *partner);
      for (auto& source : object.augmentation_sources)
        source.weight *= 1 - p.mixup_ratio;
      Append(out.objects, std::move(object), options.limits.max_objects);
    }
    trace.reset();
  }
  if (p.hsv || p.hue_shift || p.saturation_gain != 1 || p.value_gain != 1) {
    Stop(stop);
    cv::Mat hsv;
    cv::cvtColor(out.pixels, hsv, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    for (int c = 0; c < 3; ++c) {
      cv::Mat lut(1, 256, CV_8U);
      for (int x = 0; x < 256; ++x) {
        double v =
            c ? std::clamp(x * (c == 1 ? p.saturation_gain : p.value_gain), 0.,
                           255.)
              : std::fmod(x + p.hue_shift * 180, 180.);
        if (v < 0) v += 180;
        lut.at<std::uint8_t>(x) = std::uint8_t(v);
      }
      if (c == 1) lut.at<std::uint8_t>(0) = 0;
      cv::LUT(channels[c], lut, channels[c]);
    }
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, out.pixels, cv::COLOR_HSV2RGB);
  }
  if (p.center_box_flips) {
    for (auto& object : out.objects) {
      auto& b = *object.box;
      const float w = float(b.x_max) - float(b.x_min),
                  h = float(b.y_max) - float(b.y_min);
      float cx = (float(b.x_min) + float(b.x_max)) / 2,
            cy = (float(b.y_min) + float(b.y_max)) / 2;
      if (p.vertical_flip) cy = float(sample.height) - cy;
      if (p.horizontal_flip) cx = float(sample.width) - cx;
      b = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
      Bound(b, sample.width, sample.height);
    }
  }
  if (p.horizontal_flip || p.vertical_flip) {
    Stop(stop);
    cv::flip(out.pixels, out.pixels,
             p.horizontal_flip ? (p.vertical_flip ? -1 : 1) : 0);
    for (auto& object : out.objects) {
      auto& b = *object.box;
      if (p.horizontal_flip) {
        if (!p.center_box_flips) {
          const auto old = b.x_min;
          b.x_min = double(float(sample.width) - float(b.x_max));
          b.x_max = double(float(sample.width) - float(old));
        }
        MapSources(object, D{-1, 0, double(sample.width), 0, 1, 0, 0, 0, 1});
      }
      if (p.vertical_flip) {
        if (!p.center_box_flips) {
          const auto old = b.y_min;
          b.y_min = double(float(sample.height) - float(b.y_max));
          b.y_max = double(float(sample.height) - float(old));
        }
        MapSources(object, D{1, 0, 0, 0, -1, double(sample.height), 0, 0, 1});
      }
    }
    if (trace) {
      if (p.horizontal_flip) trace = trace->flipped_horizontal();
      if (p.vertical_flip) trace = trace->flipped_vertical();
    }
  }
  std::ostringstream identity;
  identity.imbue(std::locale::classic());
  identity << std::setprecision(17)
           << "opencv-yolo-color/v1:" << sample.image_path.size() << ':'
           << sample.image_path << ':'
           << (partner ? partner->image_path.size() : 0) << ':'
           << (partner ? partner->image_path : "") << ':' << p.mixup_ratio
           << ':' << p.hsv << ':' << p.hue_shift << ':' << p.saturation_gain
           << ':' << p.value_gain << ':' << p.horizontal_flip << ':'
           << p.vertical_flip;
  if (p.center_box_flips) identity << ":center-box-flips/v1";
  return Publish(sample, std::move(out), allocation, identity.str(),
                 std::move(trace), stop);
#endif
}
}  // namespace jyppx::liuding::vision::tasks
