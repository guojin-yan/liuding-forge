// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "joint_annotation_detail.hpp"

#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

namespace jyppx::liuding::data::image {
namespace {
[[noreturn]] void Invalid(const char* operation, const char* expected) {
  throw InvalidArgument("Invalid joint image transform",
                        ErrorDetails{.component = "data.image.transform",
                                     .operation = operation,
                                     .expected = expected});
}

[[noreturn]] void Limit(const char* operation) {
  throw ResourceExhausted("Joint image transform budget exceeded",
                          ErrorDetails{.component = "data.image.transform",
                                       .operation = operation});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Joint image transform cancelled",
                    ErrorDetails{.component = "data.image.transform",
                                 .operation = "transform.cancel"});
}

bool SameExtent(ImageExtent a, ImageExtent b) {
  return a.width == b.width && a.height == b.height;
}

void CheckExtent(ImageExtent extent, const JointTransformLimits& limits) {
  if (extent.width == 0 || extent.height == 0)
    Invalid("transform.extent", "positive dimensions");
  if (extent.width > limits.max_dimension ||
      extent.height > limits.max_dimension ||
      static_cast<std::uint64_t>(extent.width) * extent.height >
          limits.max_pixels)
    Limit("transform.extent");
}

void Probability(double value) {
  if (!std::isfinite(value) || value < 0 || value > 1)
    Invalid("transform.probability", "finite value in [0,1]");
}

void CheckOptions(const JointTransformOptions& options) {
  if (options.interpolation != ImageInterpolation::nearest_exact &&
      options.interpolation != ImageInterpolation::bilinear)
    Invalid("transform.interpolation", "nearest_exact or bilinear");
  if (!std::isfinite(options.image_padding) ||
      !std::isfinite(options.paired_padding) ||
      !std::isfinite(options.min_box_area) || options.min_box_area < 0 ||
      !std::isfinite(options.min_region_area) || options.min_region_area < 0)
    Invalid("transform.options", "finite padding and nonnegative box area");
  Probability(options.min_box_visibility);
  Probability(options.min_region_visibility);
  if (options.oriented_box_crop != OrientedBoxCrop::to_polygon &&
      options.oriented_box_crop != OrientedBoxCrop::discard)
    Invalid("transform.obb_policy", "to_polygon or discard");
  const auto& limits = options.limits;
  if (limits.max_dimension == 0 || limits.max_pixels == 0 ||
      limits.max_output_bytes == 0 || limits.max_working_bytes == 0 ||
      limits.max_objects == 0 || limits.max_keypoints == 0 ||
      limits.max_polygon_points == 0 || limits.max_polygon_fragments == 0 ||
      limits.max_geometry_checks == 0 || limits.max_rle_runs == 0 ||
      limits.max_mask_work_pixels == 0)
    Invalid("transform.limits", "positive resource limits");
  for (const auto* table : {&options.horizontal_keypoint_permutations,
                            &options.vertical_keypoint_permutations}) {
    if (table->size() > limits.max_objects) Limit("transform.pose_categories");
    std::size_t total = 0;
    for (const auto& permutation : *table) {
      if (permutation.size() > limits.max_keypoints - total)
        Limit("transform.pose_permutations");
      total += permutation.size();
      for (std::size_t i = 0; i < permutation.size(); ++i) {
        const auto j = permutation[i];
        if (j >= permutation.size() || permutation[j] != i)
          Invalid("transform.pose_permutation", "an involutive permutation");
      }
    }
  }
}

void CheckRandomOptions(const RandomGeometryOptions& options) {
  for (const auto& extent : {options.resize, options.crop})
    if (extent && (extent->width == 0 || extent->height == 0))
      Invalid("transform.random_extent", "positive dimensions");
  Probability(options.horizontal_flip_probability);
  Probability(options.vertical_flip_probability);
}

GeometryTrace Append(const GeometryTrace& trace,
                     const ImageGeometryStep& step) {
  const bool extent_zero = step.extent.width == 0 && step.extent.height == 0;
  switch (step.kind) {
    case ImageGeometryKind::resize:
      if (step.left || step.top || step.right || step.bottom)
        Invalid("transform.step", "zero inactive resize fields");
      return trace.resized(step.extent);
    case ImageGeometryKind::crop:
      if (step.right || step.bottom)
        Invalid("transform.step", "zero inactive crop fields");
      return trace.cropped(step.left, step.top, step.extent);
    case ImageGeometryKind::pad:
      if (!extent_zero) Invalid("transform.step", "zero inactive pad extent");
      return trace.padded(step.left, step.top, step.right, step.bottom);
    case ImageGeometryKind::horizontal_flip:
    case ImageGeometryKind::vertical_flip:
      if (!extent_zero || step.left || step.top || step.right || step.bottom)
        Invalid("transform.step", "zero inactive flip fields");
      return step.kind == ImageGeometryKind::horizontal_flip
                 ? trace.flipped_horizontal()
                 : trace.flipped_vertical();
  }
  Invalid("transform.step", "a supported geometry kind");
}

void CheckPhoto(const Tensor& tensor, const PixelSpec& spec, ImageExtent extent,
                double padding) {
  validate_pixels(tensor, spec);
  const auto h = spec.layout == ImageLayout::chw ? 1 : 0;
  const auto w = spec.layout == ImageLayout::chw ? 2 : 1;
  if (!tensor.device().is_cpu() || tensor.size(h) != extent.height ||
      tensor.size(w) != extent.width || tensor.requires_grad())
    Invalid("transform.photo", "aligned CPU pixels without autograd");
  if (spec.range == PixelValueRange::unit_interval &&
      (padding < 0 || padding > 1))
    Invalid("transform.padding", "padding in the declared unit interval");
  if (spec.dtype == torch::kUInt8 || spec.dtype == torch::kUInt16) {
    const double maximum = spec.dtype == torch::kUInt8 ? 255 : 65535;
    if (padding < 0 || padding > maximum || std::trunc(padding) != padding)
      Invalid("transform.padding", "an exactly representable unsigned pixel");
  } else {
    const double maximum =
        spec.dtype == torch::kFloat16    ? 65504.0
        : spec.dtype == torch::kFloat64  ? std::numeric_limits<double>::max()
        : spec.dtype == torch::kBFloat16 ? 3.3895313892515355e38
                                         : std::numeric_limits<float>::max();
    if (std::abs(padding) > maximum)
      Invalid("transform.padding", "finite padding representable in the dtype");
  }
}

// Exact nearest-exact/crop/pad/flip uses byte copies, including uint16 and
// int64 IDs beyond 2^53. No label ever passes through a floating
// representation.
Tensor CopyStep(const Tensor& input, ImageLayout layout, bool mask,
                ImageExtent before, ImageExtent after,
                const ImageGeometryStep& step, const c10::Scalar& fill,
                std::stop_token stop) {
  const auto channels =
      mask ? 1 : input.size(layout == ImageLayout::chw ? 0 : 2);
  const auto h = static_cast<std::int64_t>(after.height);
  const auto w = static_cast<std::int64_t>(after.width);
  const std::vector<std::int64_t> shape =
      mask                         ? std::vector<std::int64_t>{h, w}
      : layout == ImageLayout::chw ? std::vector<std::int64_t>{channels, h, w}
                                   : std::vector<std::int64_t>{h, w, channels};
  auto output = torch::full(shape, fill, input.options().pinned_memory(false));
  const auto* source = static_cast<const std::byte*>(input.const_data_ptr());
  auto* destination = static_cast<std::byte*>(output.mutable_data_ptr());
  const auto scalar = input.element_size();
  const bool planar = mask || layout == ImageLayout::chw;
  const auto planes = planar ? channels : 1;
  const auto unit = scalar * (planar ? 1 : channels);
  for (std::int64_t c = 0; c < planes; ++c) {
    for (std::uint32_t y = 0; y < after.height; ++y) {
      CheckStop(stop);
      for (std::uint32_t x = 0; x < after.width; ++x) {
        std::int64_t sx = x, sy = y;
        switch (step.kind) {
          case ImageGeometryKind::resize:
            sx = (static_cast<std::uint64_t>(x) * before.width +
                  before.width / 2U) /
                 after.width;
            sy = (static_cast<std::uint64_t>(y) * before.height +
                  before.height / 2U) /
                 after.height;
            break;
          case ImageGeometryKind::crop:
            sx += step.left;
            sy += step.top;
            break;
          case ImageGeometryKind::pad:
            sx -= step.left;
            sy -= step.top;
            break;
          case ImageGeometryKind::horizontal_flip:
            sx = before.width - 1U - x;
            break;
          case ImageGeometryKind::vertical_flip:
            sy = before.height - 1U - y;
            break;
        }
        if (sx < 0 || sy < 0 || sx >= before.width || sy >= before.height)
          continue;
        const auto from = ((c * before.height + sy) * before.width + sx) * unit;
        const auto to = ((c * after.height + y) * after.width + x) * unit;
        std::memcpy(destination + to, source + from,
                    static_cast<std::size_t>(unit));
      }
    }
  }
  return output;
}

Tensor PhotoStep(const Tensor& input, const PixelSpec& spec, ImageExtent before,
                 ImageExtent after, const ImageGeometryStep& step,
                 const JointTransformOptions& options, double padding,
                 std::stop_token stop) {
  if (step.kind != ImageGeometryKind::resize ||
      options.interpolation == ImageInterpolation::nearest_exact)
    return CopyStep(input, spec.layout, false, before, after, step, padding,
                    stop);
  auto chw = spec.layout == ImageLayout::chw ? input : input.permute({2, 0, 1});
  const bool integer =
      spec.dtype == torch::kUInt8 || spec.dtype == torch::kUInt16;
  auto floating =
      chw.to(spec.dtype == torch::kFloat64 || integer ? torch::kFloat64
                                                      : torch::kFloat32);
  auto result =
      torch::nn::functional::interpolate(
          floating.unsqueeze(0),
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<std::int64_t>{after.height, after.width})
              .mode(torch::kBilinear)
              .align_corners(false)
              .antialias(true))
          .squeeze(0);
  CheckStop(stop);
  if (integer)
    result = result.round().clamp(0, spec.dtype == torch::kUInt8 ? 255 : 65535);
  result = result.to(spec.dtype);
  return (spec.layout == ImageLayout::chw ? result : result.permute({1, 2, 0}))
      .contiguous();
}

void HashInteger(Sha256& hash, std::uint64_t value) {
  std::array<std::byte, 8> bytes{};
  for (std::size_t i = 0; i < 8; ++i)
    bytes[i] = static_cast<std::byte>(value >> (56U - 8U * i));
  hash.update(bytes);
}

void HashDouble(Sha256& hash, double value) {
  HashInteger(hash, std::bit_cast<std::uint64_t>(value == 0 ? 0.0 : value));
}

Tensor OwnPixels(const Tensor& input) {
  auto result =
      torch::empty(input.sizes(), input.options().pinned_memory(false));
  result.copy_(input);
  return result;
}
}  // namespace

Tensor detail::TransformDiscrete(const Tensor& input,
                                 std::span<const ImageGeometryStep> steps,
                                 std::span<const ImageExtent> extents,
                                 std::int64_t fill, std::stop_token stop) {
  auto result = OwnPixels(input);
  for (std::size_t i = 0; i < steps.size(); ++i)
    result = CopyStep(result, ImageLayout::chw, true, extents[i],
                      extents[i + 1], steps[i], c10::Scalar(fill), stop);
  return result;
}

VisionSample transform_sample(const VisionSample& sample,
                              std::span<const ImageGeometryStep> steps,
                              const JointTransformOptions& options,
                              std::stop_token stop) {
  CheckStop(stop);
  CheckOptions(options);
  if (steps.size() > 128) Limit("transform.steps");
  ImageExtent input{sample.width, sample.height};
  CheckExtent(input, options.limits);
  CheckPhoto(sample.image, sample.pixels, input, options.image_padding);
  if (sample.labels.size() > options.limits.max_objects ||
      sample.image_path.size() > 16384 ||
      (sample.target_path && sample.target_path->size() > 16384) ||
      (sample.annotation && sample.annotation->image_path.size() > 16384))
    Limit("transform.metadata");
  if (sample.paired_target.has_value() !=
      sample.paired_target_pixels.has_value())
    Invalid("transform.pair", "paired photo and PixelSpec together");
  if (sample.paired_target)
    CheckPhoto(*sample.paired_target, *sample.paired_target_pixels, input,
               options.paired_padding);
  if (sample.mask_target && sample.paired_target)
    Invalid("transform.targets", "one discrete mask or one paired photo");
  if (sample.mask_target) {
    const auto& mask = *sample.mask_target;
    if (mask.palette.size() > 256) Limit("transform.palette");
    if (!mask.ids.defined() || mask.ids.dim() != 2 ||
        mask.ids.scalar_type() != torch::kInt64 ||
        !mask.ids.device().is_cpu() || mask.ids.layout() != torch::kStrided ||
        mask.ids.is_neg() || mask.ids.size(0) != input.height ||
        mask.ids.size(1) != input.width || mask.width != input.width ||
        mask.height != input.height)
      Invalid("transform.mask", "aligned CPU int64 [H,W] labels");
  }
  auto local = GeometryTrace(input);
  auto composed = sample.geometry.value_or(GeometryTrace(input));
  if (!SameExtent(composed.output_extent(), input))
    Invalid("transform.trace", "trace output equal to the sample canvas");
  const auto bytes_per_pixel =
      static_cast<std::uint64_t>(sample.pixels.channels()) *
          sample.image.element_size() +
      (sample.paired_target ? static_cast<std::uint64_t>(
                                  sample.paired_target_pixels->channels()) *
                                  sample.paired_target->element_size()
                            : 0) +
      (sample.mask_target ? 8 : 0);
  std::uint64_t maximum_pixels =
      static_cast<std::uint64_t>(input.width) * input.height;
  std::array<ImageExtent, 129> extents{};
  extents[0] = input;
  std::size_t extent_count = 1;
  for (const auto& step : steps) {
    CheckStop(stop);
    local = Append(local, step);
    composed = Append(composed, step);
    const auto extent = local.output_extent();
    extents[extent_count++] = extent;
    CheckExtent(extent, options.limits);
    maximum_pixels =
        std::max(maximum_pixels,
                 static_cast<std::uint64_t>(extent.width) * extent.height);
    if (step.kind == ImageGeometryKind::pad && sample.mask_target &&
        (step.left || step.top || step.right || step.bottom) &&
        !options.mask_padding_id)
      Invalid("transform.mask_padding", "explicit discrete padding ID");
  }
  if (maximum_pixels > options.limits.max_output_bytes / bytes_per_pixel ||
      maximum_pixels >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
              bytes_per_pixel)
    Limit("transform.output_bytes");
  // At most two tensor sets plus conservative interpolation temporaries. The
  // allocator and backend-internal workspace are deliberately not claimed here.
  const auto channels =
      sample.pixels.channels() +
      (sample.paired_target ? sample.paired_target_pixels->channels() : 0);
  const bool has_rle =
      sample.annotation &&
      std::any_of(sample.annotation->objects.begin(),
                  sample.annotation->objects.end(),
                  [](const auto& object) { return object.rle.has_value(); });
  const auto working_per_pixel =
      2 * bytes_per_pixel + (has_rle ? 4 : 0) +
      (options.interpolation == ImageInterpolation::bilinear ? 32ULL * channels
                                                             : 0);
  if (maximum_pixels > options.limits.max_working_bytes / working_per_pixel)
    Limit("transform.working_bytes");
  try {
    const auto canvases =
        std::span<const ImageExtent>(extents.data(), extent_count);
    auto annotation_budget = detail::CheckJointAnnotations(
        sample, steps, canvases, options, local, stop);
    VisionSample result = sample;
    result.image = OwnPixels(sample.image);
    if (sample.paired_target)
      result.paired_target = OwnPixels(*sample.paired_target);
    if (sample.mask_target)
      result.mask_target->ids = OwnPixels(sample.mask_target->ids);
    auto before = input;
    auto progress = GeometryTrace(input);
    for (const auto& step : steps) {
      CheckStop(stop);
      progress = Append(progress, step);
      const auto after = progress.output_extent();
      result.image = PhotoStep(result.image, result.pixels, before, after, step,
                               options, options.image_padding, stop);
      if (result.paired_target)
        result.paired_target = PhotoStep(
            *result.paired_target, *result.paired_target_pixels, before, after,
            step, options, options.paired_padding, stop);
      if (result.mask_target)
        result.mask_target->ids = CopyStep(
            result.mask_target->ids, ImageLayout::chw, true, before, after,
            step, c10::Scalar(options.mask_padding_id.value_or(0)), stop);
      before = after;
    }
    result.width = before.width;
    result.height = before.height;
    if (result.mask_target) {
      result.mask_target->width = before.width;
      result.mask_target->height = before.height;
    }
    detail::MapJointAnnotations(sample, result, steps, canvases, options, local,
                                annotation_budget, stop);
    result.geometry = composed;
    CheckStop(stop);
    return result;
  } catch (const std::bad_alloc&) {
    Limit("transform.allocate");
  } catch (const c10::OutOfMemoryError&) {
    Limit("transform.allocate");
  } catch (const c10::Error& error) {
    throw DeviceError("Joint image transform backend failed",
                      ErrorDetails{.component = "data.image.transform",
                                   .operation = "transform.execute",
                                   .actual = error.what_without_backtrace()});
  }
}

std::vector<ImageGeometryStep> sample_geometry(
    ImageExtent input, const RandomGeometryOptions& options,
    const SampleRandom& random) {
  CheckRandomOptions(options);
  static_cast<void>(GeometryTrace(input));
  std::vector<ImageGeometryStep> steps;
  if (options.resize) {
    steps.push_back(
        {.kind = ImageGeometryKind::resize, .extent = *options.resize});
    input = *options.resize;
  }
  if (options.crop) {
    if (options.crop->width > input.width ||
        options.crop->height > input.height)
      Invalid("transform.random_crop", "crop inside the resized input");
    steps.push_back(
        {.kind = ImageGeometryKind::crop,
         .extent = *options.crop,
         .left = static_cast<std::uint32_t>(random.bounded(
             static_cast<std::uint64_t>(input.width) - options.crop->width + 1,
             0)),
         .top = static_cast<std::uint32_t>(
             random.bounded(static_cast<std::uint64_t>(input.height) -
                                options.crop->height + 1,
                            1))});
  }
  if (random.unit(2) < options.horizontal_flip_probability)
    steps.push_back({.kind = ImageGeometryKind::horizontal_flip});
  if (random.unit(3) < options.vertical_flip_probability)
    steps.push_back({.kind = ImageGeometryKind::vertical_flip});
  return steps;
}

std::string joint_transform_identity(const RandomGeometryOptions& random,
                                     const JointTransformOptions& joint) {
  CheckRandomOptions(random);
  CheckOptions(joint);
  Sha256 hash;
  constexpr std::string_view domain = "liuding.joint-transform.cpu.v2";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  for (const auto& extent : {random.resize, random.crop}) {
    HashInteger(hash, extent.has_value());
    if (extent) {
      HashInteger(hash, extent->width);
      HashInteger(hash, extent->height);
    }
  }
  HashDouble(hash, random.horizontal_flip_probability);
  HashDouble(hash, random.vertical_flip_probability);
  HashInteger(hash, static_cast<std::uint8_t>(joint.interpolation));
  HashDouble(hash, joint.image_padding);
  HashDouble(hash, joint.paired_padding);
  HashInteger(hash, joint.mask_padding_id.has_value());
  if (joint.mask_padding_id)
    HashInteger(hash, static_cast<std::uint64_t>(*joint.mask_padding_id));
  HashDouble(hash, joint.min_box_area);
  HashDouble(hash, joint.min_box_visibility);
  HashDouble(hash, joint.min_region_area);
  HashDouble(hash, joint.min_region_visibility);
  HashInteger(hash, static_cast<std::uint8_t>(joint.oriented_box_crop));
  HashInteger(hash, joint.drop_empty_pose);
  for (const auto* table : {&joint.horizontal_keypoint_permutations,
                            &joint.vertical_keypoint_permutations}) {
    HashInteger(hash, table->size());
    for (const auto& permutation : *table) {
      HashInteger(hash, permutation.size());
      for (auto index : permutation) HashInteger(hash, index);
    }
  }
  HashInteger(hash, joint.limits.max_dimension);
  HashInteger(hash, joint.limits.max_pixels);
  HashInteger(hash, joint.limits.max_output_bytes);
  HashInteger(hash, joint.limits.max_working_bytes);
  HashInteger(hash, joint.limits.max_objects);
  HashInteger(hash, joint.limits.max_keypoints);
  HashInteger(hash, joint.limits.max_polygon_points);
  HashInteger(hash, joint.limits.max_polygon_fragments);
  HashInteger(hash, joint.limits.max_geometry_checks);
  HashInteger(hash, joint.limits.max_rle_runs);
  HashInteger(hash, joint.limits.max_mask_work_pixels);
  return sha256_digest_hex(hash.finish());
}

}  // namespace jyppx::liuding::data::image
