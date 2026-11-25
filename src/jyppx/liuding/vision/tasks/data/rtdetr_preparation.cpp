// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "rtdetr_preparation_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>

namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using I = std::int64_t;
void Stop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("RT-DETR data preparation cancelled");
}
void Probability(double p) {
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw InvalidArgument("RT-DETR probability outside [0,1]");
}
void Canvas(im::ImageExtent e, const im::JointTransformLimits& l) {
  const auto n = std::uint64_t(e.width) * e.height;
  if (!e.width || !e.height || e.width > l.max_dimension ||
      e.height > l.max_dimension || n > l.max_pixels ||
      n > l.max_output_bytes / 3)
    throw ResourceExhausted("RT-DETR canvas/output limit exceeded");
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
auto OutputLeases(const im::VisionSample& s, const im::JointTransformOptions& o,
                  im::ImageExtent e) {
  auto leases = std::make_shared<im::ImageResourceLeaseSet>();
  leases->push_back(Reserve(s, o, im::ImageResourceKind::pixels,
                            std::uint64_t(e.width) * e.height));
  leases->push_back(Reserve(s, o, im::ImageResourceKind::decoded_bytes,
                            std::uint64_t(e.width) * e.height * 3));
  return leases;
}
im::VisionSample Checked(const im::VisionSample& s,
                         const im::JointTransformOptions& o) {
  im::validate_pixels(s.image, s.pixels);
  if (!s.image.defined() || !s.image.is_cpu() || s.image.requires_grad() ||
      s.pixels.dtype != torch::kUInt8 ||
      s.pixels.color != im::PixelColorModel::rgb ||
      s.pixels.range != im::PixelValueRange::unsigned_full_scale ||
      s.image.dim() != 3 || s.paired_target || s.mask_target || s.task_payload)
    throw InvalidArgument(
        "RT-DETR profile requires CPU uint8 RGB box-only samples");
  Canvas({s.width, s.height}, o.limits);
  const auto chw = s.pixels.layout == im::ImageLayout::chw
                       ? s.image
                       : s.image.permute({2, 0, 1});
  if (chw.size(0) != 3 || chw.size(1) != s.height || chw.size(2) != s.width)
    throw InvalidArgument("RT-DETR pixel extents do not match metadata");
  auto result = s;
  if (s.annotation) {
    if (!s.categories || s.categories->empty() ||
        s.annotation->width != s.width || s.annotation->height != s.height ||
        s.annotation->panoptic_mask_path)
      throw InvalidArgument(
          "RT-DETR annotation extent or category map is missing");
    if (s.annotation->objects.size() > o.limits.max_objects)
      throw ResourceExhausted("RT-DETR annotation bound exceeded");
    auto a = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
    for (const auto& x : a->objects) {
      if (x.category >= s.categories->size() || !x.box || !x.polygons.empty() ||
          !x.keypoints.empty() || x.rle || x.oriented_box ||
          !x.augmentation_sources.empty())
        throw InvalidArgument(
            "RT-DETR profile accepts axis-aligned detection boxes only");
      const auto b = *x.box;
      if (!std::isfinite(b.x_min) || !std::isfinite(b.y_min) ||
          !std::isfinite(b.x_max) || !std::isfinite(b.y_max) ||
          b.x_max <= b.x_min || b.y_max <= b.y_min || b.x_min < 0 ||
          b.y_min < 0 || b.x_max > s.width || b.y_max > s.height)
        throw InvalidArgument("Invalid RT-DETR input box");
    }
    std::erase_if(a->objects,
                  [](const auto& x) { return x.ignore || x.crowd; });
    result.annotation = std::move(a);
  }
  return result;
}
void Sanitize(im::VisionSample& s) {
  if (!s.annotation) return;
  auto a = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
  std::erase_if(a->objects, [&](const auto& x) {
    const auto b = *x.box;
    return b.x_max - b.x_min < 1 || b.y_max - b.y_min < 1 || b.x_min < 0 ||
           b.y_min < 0 || b.x_max > s.width || b.y_max > s.height;
  });
  s.annotation = std::move(a);
}
std::uint8_t Byte(double value) {
  return static_cast<std::uint8_t>(std::clamp(value, 0., 255.));
}
std::uint8_t Luma(const std::array<int, 3>& rgb) {
  return static_cast<std::uint8_t>(
      (19595 * rgb[0] + 38470 * rgb[1] + 7471 * rgb[2] + 32768) >> 16);
}
std::array<int, 3> Hue(std::array<int, 3> rgb, double factor) {
  const float max =
      static_cast<float>(*std::max_element(rgb.begin(), rgb.end()));
  const float min =
      static_cast<float>(*std::min_element(rgb.begin(), rgb.end()));
  float h = 0, s = 0;
  if (max != min) {
    const float d = max - min;
    s = d / max;
    const float r = (max - rgb[0]) / d, g = (max - rgb[1]) / d,
                b = (max - rgb[2]) / d;
    h = rgb[0] == max ? b - g : rgb[1] == max ? 2 + r - b : 4 + g - r;
    h = static_cast<float>(std::fmod(static_cast<double>(h) / 6. + 1., 1.));
  }
  const auto hue = (static_cast<int>(static_cast<double>(h) * 255.) +
                    static_cast<int>(factor * 255) + 256) %
                   256;
  const auto sat = static_cast<int>(static_cast<double>(s) * 255.);
  const float sector = static_cast<float>(hue) * 6 / 255;
  const auto i = static_cast<int>(sector);
  const float f = sector - i, saturation = static_cast<float>(sat) / 255;
  const auto p = static_cast<int>(max * (1 - saturation) + .5f);
  const auto q = static_cast<int>(max * (1 - saturation * f) + .5f);
  const auto t = static_cast<int>(max * (1 - saturation * (1 - f)) + .5f);
  const int v = static_cast<int>(max);
  switch (i % 6) {
    case 0:
      return {v, t, p};
    case 1:
      return {q, v, p};
    case 2:
      return {p, v, t};
    case 3:
      return {p, q, v};
    case 4:
      return {t, p, v};
    default:
      return {v, p, q};
  }
}
im::VisionSample Photo(const im::VisionSample& s,
                       const detail::RtDetrSamplePlan& p,
                       const im::JointTransformOptions& o,
                       std::stop_token stop) {
  if (!p.brightness && !p.contrast && !p.saturation && !p.hue && !p.permutation)
    return s;
  for (auto factor : {p.brightness, p.contrast, p.saturation})
    if (factor && (!std::isfinite(*factor) || *factor < 0 || *factor > 2))
      throw InvalidArgument("Invalid RT-DETR photo factor");
  if (p.hue && (!std::isfinite(*p.hue) || std::abs(*p.hue) > .5))
    throw InvalidArgument("Invalid RT-DETR hue factor");
  if (p.permutation) {
    auto order = *p.permutation;
    std::sort(order.begin(), order.end());
    if (order != std::array<int, 3>{0, 1, 2})
      throw InvalidArgument("Invalid RGB permutation");
  }
  const auto bytes = std::uint64_t(s.width) * s.height * 3;
  if (bytes > o.limits.max_working_bytes)
    throw ResourceExhausted("RT-DETR photo working bound exceeded");
  auto leases = OutputLeases(s, o, {s.width, s.height});
  auto result = s;
  result.image = s.image.clone();
  result.image_resource_leases = std::move(leases);
  auto hwc = result.pixels.layout == im::ImageLayout::chw
                 ? result.image.permute({1, 2, 0})
                 : result.image;
  auto values = hwc.accessor<std::uint8_t, 3>();
  const auto blend = [&](double factor, bool gray, bool mean) {
    float base = 0;
    if (mean) {
      std::uint64_t sum = 0;
      for (I y = 0; y < s.height; ++y) {
        Stop(stop);
        for (I x = 0; x < s.width; ++x)
          sum += Luma({values[y][x][0], values[y][x][1], values[y][x][2]});
      }
      base = static_cast<float>(std::floor(
          static_cast<double>(sum) / (s.width * std::uint64_t(s.height)) + .5));
    }
    const auto f = static_cast<float>(factor);
    for (I y = 0; y < s.height; ++y) {
      Stop(stop);
      for (I x = 0; x < s.width; ++x) {
        const auto zero =
            gray ? static_cast<float>(Luma(
                       {values[y][x][0], values[y][x][1], values[y][x][2]}))
                 : base;
        for (int c = 0; c < 3; ++c)
          values[y][x][c] = Byte(zero + f * (values[y][x][c] - zero));
      }
    }
  };
  if (p.brightness) blend(*p.brightness, false, false);
  if (p.contrast && p.contrast_before) blend(*p.contrast, false, true);
  if (p.saturation) blend(*p.saturation, true, false);
  if (p.hue)
    for (I y = 0; y < s.height; ++y) {
      Stop(stop);
      for (I x = 0; x < s.width; ++x) {
        const auto rgb =
            Hue({values[y][x][0], values[y][x][1], values[y][x][2]}, *p.hue);
        for (int c = 0; c < 3; ++c)
          values[y][x][c] = static_cast<std::uint8_t>(rgb[c]);
      }
    }
  if (p.contrast && !p.contrast_before) blend(*p.contrast, false, true);
  if (p.permutation)
    for (I y = 0; y < s.height; ++y) {
      Stop(stop);
      for (I x = 0; x < s.width; ++x) {
        const std::array rgb{values[y][x][0], values[y][x][1], values[y][x][2]};
        for (int c = 0; c < 3; ++c) values[y][x][c] = rgb[(*p.permutation)[c]];
      }
    }
  return result;
}
struct Coefficients {
  I first;
  std::vector<std::int32_t> values;
};
Coefficients CoefficientsFor(I input, I output, I index) {
  const double scale = static_cast<double>(input) / output,
               support = std::max(1., scale), center = (index + .5) * scale;
  const auto first = std::max(I{0}, static_cast<I>(center - support + .5));
  const auto end = std::min(input, static_cast<I>(center + support + .5));
  std::vector<double> weights;
  double total = 0;
  for (I j = first; j < end; ++j) {
    auto w = std::max(0., 1 - std::abs((j - center + .5) / support));
    weights.push_back(w);
    total += w;
  }
  Coefficients result{first, {}};
  for (auto w : weights)
    result.values.push_back(
        static_cast<std::int32_t>(std::floor(w / total * (1 << 22) + .5)));
  return result;
}
im::VisionSample Stretch(const im::VisionSample& s, im::ImageExtent e,
                         const im::JointTransformOptions& o,
                         std::stop_token stop) {
  Canvas(e, o.limits);
  const auto intermediate = std::uint64_t(s.height) * e.width * 3;
  const auto scratch = intermediate + std::uint64_t(s.width) * s.height * 3 +
                       32ULL * (s.width + s.height);
  if (scratch > o.limits.max_working_bytes)
    throw ResourceExhausted("RT-DETR resize working bound exceeded");
  auto leases = OutputLeases(s, o, e);
  auto working =
      Reserve(s, o, im::ImageResourceKind::codec_scratch_bytes, scratch);
  auto input =
      (s.pixels.layout == im::ImageLayout::chw ? s.image
                                               : s.image.permute({2, 0, 1}))
          .contiguous();
  auto middle = torch::empty(
      {3, static_cast<I>(s.height), static_cast<I>(e.width)}, torch::kUInt8);
  const auto* a = input.const_data_ptr<std::uint8_t>();
  auto* b = middle.mutable_data_ptr<std::uint8_t>();
  for (I x = 0; x < e.width; ++x) {
    Stop(stop);
    const auto k = CoefficientsFor(s.width, e.width, x);
    for (I c = 0; c < 3; ++c)
      for (I y = 0; y < s.height; ++y) {
        I sum = 1 << 21;
        for (std::size_t j = 0; j < k.values.size(); ++j)
          sum += I(a[(c * s.height + y) * s.width + k.first + j]) * k.values[j];
        b[(c * s.height + y) * e.width + x] =
            static_cast<std::uint8_t>(std::clamp(sum >> 22, I{0}, I{255}));
      }
  }
  auto output = torch::empty(
      {3, static_cast<I>(e.height), static_cast<I>(e.width)}, torch::kUInt8);
  auto* out = output.mutable_data_ptr<std::uint8_t>();
  for (I y = 0; y < e.height; ++y) {
    Stop(stop);
    const auto k = CoefficientsFor(s.height, e.height, y);
    for (I c = 0; c < 3; ++c)
      for (I x = 0; x < e.width; ++x) {
        I sum = 1 << 21;
        for (std::size_t j = 0; j < k.values.size(); ++j)
          sum += I(b[(c * s.height + k.first + j) * e.width + x]) * k.values[j];
        out[(c * e.height + y) * e.width + x] =
            static_cast<std::uint8_t>(std::clamp(sum >> 22, I{0}, I{255}));
      }
  }
  auto result = s;
  result.image = std::move(output);
  result.pixels.layout = im::ImageLayout::chw;
  result.image_resource_leases = std::move(leases);
  result.width = e.width;
  result.height = e.height;
  result.geometry =
      s.geometry.value_or(im::GeometryTrace({s.width, s.height})).resized(e);
  if (s.annotation) {
    auto ann = std::make_shared<im::VisionAnnotationRecord>(*s.annotation);
    ann->width = e.width;
    ann->height = e.height;
    const double sx = static_cast<double>(e.width) / s.width,
                 sy = static_cast<double>(e.height) / s.height;
    for (auto& object : ann->objects) {
      auto& box = *object.box;
      // Checked input boxes lie inside the source canvas. Multiplying an
      // endpoint by a rounded scale can exceed the destination by one ULP;
      // contain that roundoff before sanitation and strict batch validation.
      box = {std::clamp(box.x_min * sx, 0., double(e.width)),
             std::clamp(box.y_min * sy, 0., double(e.height)),
             std::clamp(box.x_max * sx, 0., double(e.width)),
             std::clamp(box.y_max * sy, 0., double(e.height))};
    }
    result.annotation = std::move(ann);
  }
  return result;
}
}  // namespace

std::string rtdetr_data_identity(const RtDetrDataOptions& o) {
  for (double p : {o.photometric_probability, o.zoom_out_probability,
                   o.iou_crop_probability, o.horizontal_flip_probability})
    Probability(p);
  if (o.crop_trials < 1 || o.crop_trials > 1000 || o.max_crop_rounds < 1 ||
      o.max_crop_rounds > 1024)
    throw InvalidArgument("Invalid RT-DETR crop rejection bounds");
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(17) << "rtdetr-data/pil-u8-v2:" << o.enabled << ':'
       << o.training << ':' << o.photometric_probability << ':'
       << o.zoom_out_probability << ':' << o.iou_crop_probability << ':'
       << o.horizontal_flip_probability << ':' << o.crop_trials << ':'
       << o.max_crop_rounds;
  const auto value = text.str();
  return sha256_hex(std::as_bytes(std::span(value)));
}

im::VisionSample detail::apply_rtdetr_plan(
    const im::VisionSample& source, im::ImageExtent canvas,
    const RtDetrSamplePlan& p, const im::JointTransformOptions& geometry,
    std::stop_token stop) {
  Stop(stop);
  auto sample = Checked(source, geometry);
  sample = Photo(sample, p, geometry, stop);
  auto policy = geometry;
  policy.image_padding = 0;
  policy.min_box_area = 0;
  policy.min_box_visibility = 0;
  if (std::any_of(p.padding.begin(), p.padding.end(),
                  [](auto n) { return n != 0; })) {
    const std::array steps{
        im::ImageGeometryStep{.kind = im::ImageGeometryKind::pad,
                              .left = p.padding[0],
                              .top = p.padding[1],
                              .right = p.padding[2],
                              .bottom = p.padding[3]}};
    sample = im::transform_sample(sample, steps, policy, stop);
  }
  if (p.crop) {
    if (p.crop->kind != im::ImageGeometryKind::crop)
      throw InvalidArgument("RT-DETR crop plan has wrong operation");
    if (sample.annotation) {
      auto a = std::make_shared<im::VisionAnnotationRecord>(*sample.annotation);
      std::erase_if(a->objects, [&](const auto& x) {
        const auto b = *x.box;
        const auto cx = (b.x_min + b.x_max) * .5, cy = (b.y_min + b.y_max) * .5;
        return !(p.crop->left < cx &&
                 cx < p.crop->left + p.crop->extent.width && p.crop->top < cy &&
                 cy < p.crop->top + p.crop->extent.height);
      });
      sample.annotation = std::move(a);
    }
    sample = im::transform_sample(sample, std::array{*p.crop}, policy, stop);
  }
  if (p.sanitize_boxes) Sanitize(sample);
  if (p.horizontal_flip)
    sample = im::transform_sample(
        sample,
        std::array{im::ImageGeometryStep{
            .kind = im::ImageGeometryKind::horizontal_flip}},
        policy, stop);
  sample = Stretch(sample, canvas, geometry, stop);
  if (p.sanitize_boxes) Sanitize(sample);
  Stop(stop);
  return sample;
}

im::VisionSample prepare_rtdetr_sample(
    const im::VisionSample& source, im::ImageExtent canvas,
    const RtDetrDataOptions& o, const im::SampleRandom& random,
    const im::JointTransformOptions& geometry, std::stop_token stop) {
  static_cast<void>(rtdetr_data_identity(o));
  Stop(stop);
  const auto sample = Checked(source, geometry);
  detail::RtDetrSamplePlan p;
  p.sanitize_boxes = o.training;
  if (o.training) {
    std::uint64_t draw = 0;
    const auto unit = [&] { return random.unit(draw++); };
    const auto factor = [&](double lo, double hi) -> std::optional<double> {
      if (unit() >= o.photometric_probability) return {};
      return static_cast<double>(static_cast<float>(lo) +
                                 static_cast<float>(hi - lo) *
                                     static_cast<float>(unit()));
    };
    p.brightness = factor(.875, 1.125);
    p.contrast = factor(.5, 1.5);
    p.saturation = factor(.5, 1.5);
    p.hue = factor(-.05, .05);
    p.contrast_before = unit() < .5;
    if (unit() < o.photometric_probability) {
      std::array order{0, 1, 2};
      for (int i = 2; i > 0; --i)
        std::swap(order[i], order[random.bounded(i + 1, draw++)]);
      p.permutation = order;
    }
    auto width = sample.width, height = sample.height;
    if (unit() < o.zoom_out_probability) {
      const float r = 1 + 3 * static_cast<float>(unit());
      width = static_cast<std::uint32_t>(sample.width * r);
      height = static_cast<std::uint32_t>(sample.height * r);
      Canvas({width, height}, geometry.limits);
      const auto left = static_cast<std::uint32_t>((width - sample.width) *
                                                   unit()),
                 top = static_cast<std::uint32_t>((height - sample.height) *
                                                  unit());
      p.padding = {left, top, width - sample.width - left,
                   height - sample.height - top};
    }
    if (unit() < o.iou_crop_probability) {
      constexpr std::array thresholds{0., .1, .3, .5, .7, .9, 1.};
      bool done = false;
      for (std::uint32_t round = 0; round < o.max_crop_rounds && !done;
           ++round) {
        Stop(stop);
        const auto threshold =
            thresholds[random.bounded(thresholds.size(), draw++)];
        if (threshold >= 1) {
          done = true;
          break;
        }
        for (std::uint32_t trial = 0; trial < o.crop_trials; ++trial) {
          Stop(stop);
          const auto w = static_cast<std::uint32_t>(
                         width * (.3f + .7f * static_cast<float>(unit()))),
                     h = static_cast<std::uint32_t>(
                         height * (.3f + .7f * static_cast<float>(unit())));
          if (!w || !h || double(w) / h < .5 || double(w) / h > 2) continue;
          const auto left = static_cast<std::uint32_t>((width - w) * unit()),
                     top = static_cast<std::uint32_t>((height - h) * unit());
          double best = -1;
          if (sample.annotation)
            for (const auto& x : sample.annotation->objects) {
              auto b = *x.box;
              b.x_min += p.padding[0];
              b.x_max += p.padding[0];
              b.y_min += p.padding[1];
              b.y_max += p.padding[1];
              const auto cx = (b.x_min + b.x_max) * .5,
                         cy = (b.y_min + b.y_max) * .5;
              if (!(left < cx && cx < left + w && top < cy && cy < top + h))
                continue;
              const auto area =
                  std::max(0., std::min(b.x_max, double(left + w)) -
                                   std::max(b.x_min, double(left))) *
                  std::max(0., std::min(b.y_max, double(top + h)) -
                                   std::max(b.y_min, double(top)));
              best = std::max(
                  best, area / ((b.x_max - b.x_min) * (b.y_max - b.y_min) +
                                double(w) * h - area));
            }
          if (best < threshold) continue;
          p.crop = im::ImageGeometryStep{.kind = im::ImageGeometryKind::crop,
                                         .extent = {w, h},
                                         .left = left,
                                         .top = top};
          done = true;
          break;
        }
      }
      if (!done)
        throw ResourceExhausted(
            "RT-DETR IoU crop rejection-work bound exceeded");
    }
    p.horizontal_flip = unit() < o.horizontal_flip_probability;
  }
  return detail::apply_rtdetr_plan(sample, canvas, p, geometry, stop);
}
}  // namespace jyppx::liuding::vision::tasks
