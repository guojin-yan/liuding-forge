// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/augmentation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

namespace jyppx::liuding::vision::tasks {
namespace {
namespace image = data::image;

void Stop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Vision augmentation cancelled");
}

void Range(double value, double upper) {
  if (!std::isfinite(value) || value < 0 || value > upper)
    throw InvalidArgument(
        "Invalid vision augmentation magnitude or probability");
}

void Validate(const PhotometricOptions& options) {
  Range(options.hue_shift, 0.5);
  for (auto value : {options.saturation_gain, options.value_gain,
                     options.grayscale_probability, options.blur_probability,
                     options.median_blur_probability})
    Range(value, 1);
  Range(options.clahe_probability, 1);
  if (options.clahe_tiles < 1 || options.clahe_tiles > 32 ||
      !std::isfinite(options.clahe_clip) || options.clahe_clip < 1 ||
      options.clahe_clip > 64)
    throw InvalidArgument("Invalid CLAHE grid/clip profile");
  if (options.blur_kernel == 0 || options.blur_kernel > 31 ||
      options.blur_kernel % 2 == 0 || options.median_kernel == 0 ||
      options.median_kernel > 7 || options.median_kernel % 2 == 0 ||
      !std::isfinite(options.blur_sigma) || options.blur_sigma <= 0 ||
      options.limits.max_dimension == 0 || options.limits.max_pixels == 0 ||
      options.limits.max_output_bytes == 0 ||
      options.limits.max_working_bytes == 0 ||
      options.max_filter_operations == 0)
    throw InvalidArgument(
        "Invalid photo augmentation kernel, sigma or resource limit");
}

bool Enabled(const PhotometricOptions& options) {
  return options.hue_shift > 0 || options.saturation_gain > 0 ||
         options.value_gain > 0 || options.grayscale_probability > 0 ||
         options.blur_probability > 0 || options.median_blur_probability > 0 ||
         options.clahe_probability > 0;
}

std::string Hash(std::ostringstream& stream) {
  const auto text = stream.str();
  return sha256_hex(std::as_bytes(std::span(text)));
}

image::ImageResourceLease Reserve(const image::VisionSample& sample,
                                  const PhotometricOptions& options,
                                  image::ImageResourceKind kind,
                                  std::uint64_t amount) {
  if (sample.image_resource_leases)
    for (const auto& lease : *sample.image_resource_leases)
      if (lease.active()) return lease.reserve_related(kind, amount);
  return options.resource_budget
             ? options.resource_budget->reserve(kind, amount)
             : image::ImageResourceLease{};
}

std::uint8_t Quantize(double value) {
  return static_cast<std::uint8_t>(
      std::floor(std::clamp(value, 0.0, 255.0) + 0.5));
}

std::array<double, 3> Hsv(std::array<double, 3> rgb, double hue_shift,
                          double saturation, double value_gain) {
  const auto high = (std::max)({rgb[0], rgb[1], rgb[2]});
  const auto low = (std::min)({rgb[0], rgb[1], rgb[2]});
  const auto delta = high - low;
  double hue = 0;
  if (delta > 0) {
    if (high == rgb[0])
      hue = (rgb[1] - rgb[2]) / delta;
    else if (high == rgb[1])
      hue = (rgb[2] - rgb[0]) / delta + 2;
    else
      hue = (rgb[0] - rgb[1]) / delta + 4;
    hue /= 6;
  }
  hue += hue_shift;
  hue -= std::floor(hue);
  const auto sat =
      high == 0 ? 0 : std::clamp(delta / high * saturation, 0.0, 1.0);
  const auto value = std::clamp(high * value_gain, 0.0, 255.0);
  const auto sector = hue * 6;
  const auto index = static_cast<int>(std::floor(sector));
  const auto fraction = sector - index;
  const auto p = value * (1 - sat), q = value * (1 - sat * fraction);
  const auto t = value * (1 - sat * (1 - fraction));
  switch (index % 6) {
    case 0:
      return {value, t, p};
    case 1:
      return {q, value, p};
    case 2:
      return {p, value, t};
    case 3:
      return {p, q, value};
    case 4:
      return {t, p, value};
    default:
      return {value, p, q};
  }
}

void Gaussian(Tensor& photo, std::uint32_t kernel, double sigma,
              std::stop_token stop) {
  auto input =
      photo.accessor<std::uint8_t, 3>();  // HWC view; arbitrary strides.
  auto scratch =
      torch::empty(photo.sizes(), photo.options().dtype(torch::kFloat32));
  auto work = scratch.accessor<float, 3>();
  const auto height = photo.size(0), width = photo.size(1);
  const auto radius = static_cast<int>(kernel / 2);
  std::vector<double> weights(kernel);
  double sum = 0;
  for (int tap = -radius; tap <= radius; ++tap) {
    const auto distance = static_cast<double>(tap) / sigma;
    weights[static_cast<std::size_t>(tap + radius)] =
        std::exp(-0.5 * distance * distance);
    sum += weights[static_cast<std::size_t>(tap + radius)];
  }
  for (auto& weight : weights) weight /= sum;
  for (std::int64_t y = 0; y < height; ++y) {
    Stop(stop);
    for (std::int64_t x = 0; x < width; ++x)
      for (int channel = 0; channel < 3; ++channel) {
        double value = 0;
        for (int tap = -radius; tap <= radius; ++tap)
          value += input[y][std::clamp(x + tap, std::int64_t{0}, width - 1)]
                        [channel] *
                   weights[static_cast<std::size_t>(tap + radius)];
        work[y][x][channel] = static_cast<float>(value);
      }
  }
  for (std::int64_t y = 0; y < height; ++y) {
    Stop(stop);
    for (std::int64_t x = 0; x < width; ++x)
      for (int channel = 0; channel < 3; ++channel) {
        double value = 0;
        for (int tap = -radius; tap <= radius; ++tap)
          value += work[std::clamp(y + tap, std::int64_t{0}, height - 1)][x]
                       [channel] *
                   weights[static_cast<std::size_t>(tap + radius)];
        input[y][x][channel] = Quantize(value);
      }
  }
}

void Median(Tensor& photo, std::uint32_t kernel, std::stop_token stop) {
  auto scratch = torch::empty(photo.sizes(), photo.options());
  auto input = photo.accessor<std::uint8_t, 3>();
  auto output = scratch.accessor<std::uint8_t, 3>();
  const auto height = photo.size(0), width = photo.size(1);
  const auto radius = static_cast<int>(kernel / 2);
  const auto count = static_cast<std::size_t>(kernel * kernel);
  std::array<std::uint8_t, 49> values{};
  for (std::int64_t y = 0; y < height; ++y) {
    Stop(stop);
    for (std::int64_t x = 0; x < width; ++x)
      for (int channel = 0; channel < 3; ++channel) {
        std::size_t cursor = 0;
        for (int dy = -radius; dy <= radius; ++dy)
          for (int dx = -radius; dx <= radius; ++dx)
            values[cursor++] =
                input[std::clamp(y + dy, std::int64_t{0}, height - 1)]
                     [std::clamp(x + dx, std::int64_t{0}, width - 1)][channel];
        std::nth_element(values.begin(), values.begin() + count / 2,
                         values.begin() + count);
        output[y][x][channel] = values[count / 2];
      }
  }
  photo.copy_(scratch);
}
void Clahe(Tensor photo, std::uint32_t tiles, double clip,
           std::stop_token stop) {
  const auto h = photo.size(0), w = photo.size(1);
  const auto gy = (std::min)(static_cast<std::int64_t>(tiles), h),
             gx = (std::min)(static_cast<std::int64_t>(tiles), w);
  auto pixels = photo.accessor<std::uint8_t, 3>();
  std::vector<std::array<double, 256>> tables(
      static_cast<std::size_t>(gy * gx));
  const auto luma = [&](std::int64_t y, std::int64_t x) {
    return Quantize(.299 * pixels[y][x][0] + .587 * pixels[y][x][1] +
                    .114 * pixels[y][x][2]);
  };
  for (std::int64_t ty = 0; ty < gy; ++ty)
    for (std::int64_t tx = 0; tx < gx; ++tx) {
      Stop(stop);
      std::array<std::uint64_t, 256> hist{};
      const auto y0 = ty * h / gy, y1 = (ty + 1) * h / gy, x0 = tx * w / gx,
                 x1 = (tx + 1) * w / gx;
      for (auto y = y0; y < y1; ++y)
        for (auto x = x0; x < x1; ++x) ++hist[luma(y, x)];
      const auto count = (y1 - y0) * (x1 - x0);
      const auto cap = (std::max)(
          std::uint64_t{1}, static_cast<std::uint64_t>(clip * count / 256));
      std::uint64_t excess = 0;
      for (auto& value : hist)
        if (value > cap) {
          excess += value - cap;
          value = cap;
        }
      const auto quotient = excess / 256, remainder = excess % 256;
      for (std::size_t i = 0; i < 256; ++i)
        hist[i] += quotient + (i < remainder ? 1 : 0);
      std::uint64_t cumulative = 0;
      for (std::size_t i = 0; i < 256; ++i) {
        cumulative += hist[i];
        tables[static_cast<std::size_t>(ty * gx + tx)][i] =
            255. * cumulative / count;
      }
    }
  for (std::int64_t y = 0; y < h; ++y) {
    Stop(stop);
    const double yy = (y + .5) * gy / h - .5;
    const auto low_y = static_cast<std::int64_t>(std::floor(yy));
    const double fy = yy - low_y;
    for (std::int64_t x = 0; x < w; ++x) {
      const double xx = (x + .5) * gx / w - .5;
      const auto low_x = static_cast<std::int64_t>(std::floor(xx));
      const double fx = xx - low_x;
      const auto value = luma(y, x);
      double mapped = 0;
      for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          mapped +=
              tables[static_cast<std::size_t>(
                  std::clamp(low_y + dy, std::int64_t{0}, gy - 1) * gx +
                  std::clamp(low_x + dx, std::int64_t{0}, gx - 1))][value] *
              (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx);
      for (int c = 0; c < 3; ++c)
        pixels[y][x][c] = Quantize(pixels[y][x][c] + mapped - value);
    }
  }
}
}  // namespace

std::string photometric_identity(const PhotometricOptions& options) {
  Validate(options);
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(17)
       << "liuding.photo-rgb-u8/v1:" << options.hue_shift << ':'
       << options.saturation_gain << ':' << options.value_gain << ':'
       << options.grayscale_probability << ':' << options.blur_probability
       << ':' << options.blur_kernel << ':' << options.blur_sigma << ':'
       << options.median_blur_probability << ':' << options.median_kernel << ':'
       << options.clahe_probability << ':' << options.clahe_tiles << ':'
       << options.clahe_clip << ':' << options.limits.max_dimension << ':'
       << options.limits.max_pixels << ':' << options.limits.max_output_bytes
       << ':' << options.limits.max_working_bytes << ':'
       << options.max_filter_operations;
  return Hash(text);
}

image::VisionSample augment_photometric(const image::VisionSample& sample,
                                        const PhotometricOptions& options,
                                        const image::SampleRandom& random,
                                        std::stop_token stop) {
  Stop(stop);
  Validate(options);
  image::validate_pixels(sample.image, sample.pixels);
  if (!sample.image.is_cpu() || sample.pixels.dtype != torch::kUInt8 ||
      sample.pixels.color != image::PixelColorModel::rgb || sample.width == 0 ||
      sample.height == 0 ||
      sample.pixels.range != image::PixelValueRange::unsigned_full_scale)
    throw InvalidArgument(
        "Photo augmentation requires CPU uint8 RGB with positive extents");
  const auto source = sample.pixels.layout == image::ImageLayout::chw
                          ? sample.image.permute({1, 2, 0})
                          : sample.image;
  if (source.size(0) != sample.height || source.size(1) != sample.width)
    throw InvalidArgument(
        "Photo augmentation dimensions differ from pixel storage");
  const auto pixels = static_cast<std::uint64_t>(sample.width) * sample.height;
  if (sample.width > options.limits.max_dimension ||
      sample.height > options.limits.max_dimension ||
      pixels > options.limits.max_pixels ||
      pixels > options.limits.max_output_bytes / 3)
    throw ResourceExhausted("Photo augmentation canvas/output bound exceeded");
  if (!Enabled(options)) return sample;
  const auto gaussian =
      random.unit(4) < options.blur_probability && options.blur_kernel > 1;
  const auto median = random.unit(5) < options.median_blur_probability &&
                      options.median_kernel > 1;
  const auto bytes = pixels * 3;
  const auto working_factor = gaussian ? 5U : (median ? 2U : 1U);
  const auto taps =
      (gaussian ? 2ULL * options.blur_kernel : 0ULL) +
      (median ? static_cast<std::uint64_t>(options.median_kernel) *
                    options.median_kernel
              : 0ULL);
  if (bytes > options.limits.max_working_bytes / working_factor ||
      (taps != 0 && bytes > options.max_filter_operations / taps))
    throw ResourceExhausted(
        "Photo augmentation working payload/filter work bound exceeded");
  auto leases = std::make_shared<image::ImageResourceLeaseSet>();
  leases->push_back(
      Reserve(sample, options, image::ImageResourceKind::pixels, pixels));
  leases->push_back(
      Reserve(sample, options, image::ImageResourceKind::decoded_bytes, bytes));
  image::VisionSample result = sample;
  result.image = torch::empty(sample.image.sizes(),
                              sample.image.options().pinned_memory(false));
  result.image_resource_leases = std::move(leases);
  auto destination = sample.pixels.layout == image::ImageLayout::chw
                         ? result.image.permute({1, 2, 0})
                         : result.image;
  auto input = source.accessor<std::uint8_t, 3>();
  auto output = destination.accessor<std::uint8_t, 3>();
  const auto hue = (2 * random.unit(0) - 1) * options.hue_shift;
  const auto saturation =
      1 + (2 * random.unit(1) - 1) * options.saturation_gain;
  const auto value = 1 + (2 * random.unit(2) - 1) * options.value_gain;
  const auto gray = random.unit(3) < options.grayscale_probability;
  const auto hsv = options.hue_shift > 0 || options.saturation_gain > 0 ||
                   options.value_gain > 0;
  for (std::int64_t y = 0; y < source.size(0); ++y) {
    Stop(stop);
    for (std::int64_t x = 0; x < source.size(1); ++x) {
      std::array<double, 3> rgb{static_cast<double>(input[y][x][0]),
                                static_cast<double>(input[y][x][1]),
                                static_cast<double>(input[y][x][2])};
      if (hsv) rgb = Hsv(rgb, hue, saturation, value);
      for (auto& channel : rgb) channel = Quantize(channel);
      if (gray) {
        const auto luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        rgb.fill(luma);
      }
      for (int channel = 0; channel < 3; ++channel)
        output[y][x][channel] =
            Quantize(rgb[static_cast<std::size_t>(channel)]);
    }
  }
  if (gaussian) {
    auto scratch =
        Reserve(sample, options, image::ImageResourceKind::codec_scratch_bytes,
                bytes * 4);
    Gaussian(destination, options.blur_kernel, options.blur_sigma, stop);
  }
  if (median) {
    auto scratch = Reserve(
        sample, options, image::ImageResourceKind::codec_scratch_bytes, bytes);
    Median(destination, options.median_kernel, stop);
  }
  if (random.unit(6) < options.clahe_probability)
    Clahe(destination, options.clahe_tiles, options.clahe_clip, stop);
  Stop(stop);
  return result;
}

std::string scale_translate_identity(const ScaleTranslateOptions& options) {
  Range(options.scale_gain, 0.9);
  Range(options.translate_fraction, 0.5);
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::setprecision(17)
       << "liuding.scale-translate/v1:" << options.scale_gain << ':'
       << options.translate_fraction;
  return Hash(text);
}

image::VisionSample augment_scale_translate(
    const image::VisionSample& sample, const ScaleTranslateOptions& options,
    const image::SampleRandom& random,
    const image::JointTransformOptions& geometry, std::stop_token stop) {
  Stop(stop);
  static_cast<void>(scale_translate_identity(options));
  if (sample.width == 0 || sample.height == 0 || sample.width > 65536 ||
      sample.height > 65536)
    throw InvalidArgument(
        "Random scale/translate requires positive bounded extents");
  if (options.scale_gain == 0 && options.translate_fraction == 0) return sample;
  const auto factor = 1 + (2 * random.unit(0) - 1) * options.scale_gain;
  const auto width = (std::max)(
      std::int64_t{1},
      static_cast<std::int64_t>(std::llround(sample.width * factor)));
  const auto height = (std::max)(
      std::int64_t{1},
      static_cast<std::int64_t>(std::llround(sample.height * factor)));
  const auto canvas_w = static_cast<std::int64_t>(sample.width);
  const auto canvas_h = static_cast<std::int64_t>(sample.height);
  auto left =
      (canvas_w - width) / 2 +
      static_cast<std::int64_t>(std::llround(
          (2 * random.unit(1) - 1) * options.translate_fraction * canvas_w));
  auto top =
      (canvas_h - height) / 2 +
      static_cast<std::int64_t>(std::llround(
          (2 * random.unit(2) - 1) * options.translate_fraction * canvas_h));
  left = std::clamp(left, 1 - width, canvas_w - 1);
  top = std::clamp(top, 1 - height, canvas_h - 1);
  const auto crop_x = (std::max)(std::int64_t{0}, -left);
  const auto crop_y = (std::max)(std::int64_t{0}, -top);
  const auto pad_x = (std::max)(std::int64_t{0}, left);
  const auto pad_y = (std::max)(std::int64_t{0}, top);
  const auto visible_w = (std::min)(width - crop_x, canvas_w - pad_x);
  const auto visible_h = (std::min)(height - crop_y, canvas_h - pad_y);
  const std::array steps{
      image::ImageGeometryStep{.kind = image::ImageGeometryKind::resize,
                               .extent = {static_cast<std::uint32_t>(width),
                                          static_cast<std::uint32_t>(height)}},
      image::ImageGeometryStep{
          .kind = image::ImageGeometryKind::crop,
          .extent = {static_cast<std::uint32_t>(visible_w),
                     static_cast<std::uint32_t>(visible_h)},
          .left = static_cast<std::uint32_t>(crop_x),
          .top = static_cast<std::uint32_t>(crop_y)},
      image::ImageGeometryStep{
          .kind = image::ImageGeometryKind::pad,
          .left = static_cast<std::uint32_t>(pad_x),
          .top = static_cast<std::uint32_t>(pad_y),
          .right = static_cast<std::uint32_t>(canvas_w - visible_w - pad_x),
          .bottom = static_cast<std::uint32_t>(canvas_h - visible_h - pad_y)}};
  return image::transform_sample(sample, steps, geometry, stop);
}
}  // namespace jyppx::liuding::vision::tasks
