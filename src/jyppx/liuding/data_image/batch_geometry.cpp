// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "image_batch_detail.hpp"

#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>

namespace jyppx::liuding::data::image {
namespace {
using detail::BatchLimit;
using detail::InvalidBatch;

void Stop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Image batch geometry cancelled");
}

bool Equal(ImageExtent a, ImageExtent b) {
  return a.width == b.width && a.height == b.height;
}

void HashInteger(Sha256& hash, std::uint64_t value) {
  std::array<std::byte, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::byte>(value >> (8 * i));
  hash.update(bytes);
}

std::uint64_t Multiply(std::uint64_t a, std::uint64_t b,
                       const char* operation) {
  if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
    BatchLimit(operation);
  return a * b;
}

std::uint64_t Add(std::uint64_t a, std::uint64_t b,
                  const char* operation) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    BatchLimit(operation);
  return a + b;
}

std::uint64_t QuotaBytes(std::uint64_t total, std::size_t count) {
  return total / count;
}

std::size_t QuotaNodes(std::size_t total, std::size_t count,
                       std::size_t divisor = 1) {
  return std::max<std::size_t>(1, total / count / divisor);
}

void CheckOutputPolicy(ImageBatchLayout layout,
                       const ImageBatchGeometryOptions& options) {
  if (!options.output_size_divisor ||
      (layout != ImageBatchLayout::pad &&
       (options.output_padded_extent || options.output_size_divisor != 1)) ||
      (options.output_padded_extent &&
       (!options.output_padded_extent->width ||
        !options.output_padded_extent->height)))
    InvalidBatch("batch_geometry.output_policy");
}

GeometryTrace Append(GeometryTrace trace, const ImageGeometryStep& step) {
  const bool zero_extent = step.extent.width == 0 && step.extent.height == 0;
  switch (step.kind) {
    case ImageGeometryKind::resize:
      if (step.left || step.top || step.right || step.bottom)
        InvalidBatch("batch_geometry.resize_fields");
      return trace.resized(step.extent);
    case ImageGeometryKind::crop:
      if (step.right || step.bottom)
        InvalidBatch("batch_geometry.crop_fields");
      return trace.cropped(step.left, step.top, step.extent);
    case ImageGeometryKind::pad:
      if (!zero_extent) InvalidBatch("batch_geometry.pad_fields");
      return trace.padded(step.left, step.top, step.right, step.bottom);
    case ImageGeometryKind::horizontal_flip:
    case ImageGeometryKind::vertical_flip:
      if (!zero_extent || step.left || step.top || step.right || step.bottom)
        InvalidBatch("batch_geometry.flip_fields");
      return step.kind == ImageGeometryKind::horizontal_flip
                 ? trace.flipped_horizontal()
                 : trace.flipped_vertical();
  }
  InvalidBatch("batch_geometry.kind");
}

ImageExtent Extent(const Tensor& image) {
  return {static_cast<std::uint32_t>(image.size(2)),
          static_cast<std::uint32_t>(image.size(1))};
}

Tensor PhotoAt(const ImageBatch& batch, std::size_t index) {
  const auto i = static_cast<std::int64_t>(index);
  return batch.layout == ImageBatchLayout::ragged ? batch.image_list[index]
                                                   : batch.images.select(0, i);
}

Tensor PairedAt(const ImageBatch& batch, std::size_t index) {
  const auto i = static_cast<std::int64_t>(index);
  return batch.layout == ImageBatchLayout::ragged
             ? batch.paired_image_list[index]
             : batch.paired_images->select(0, i);
}

Tensor MaskAt(const ImageBatch& batch, std::size_t index) {
  const auto i = static_cast<std::int64_t>(index);
  return batch.layout == ImageBatchLayout::ragged ? batch.mask_list[index]
                                                   : batch.masks->select(0, i);
}

Tensor ValidAt(const ImageBatch& batch, std::size_t index) {
  const auto i = static_cast<std::int64_t>(index);
  return batch.layout == ImageBatchLayout::ragged
             ? batch.valid_list[index]
             : batch.valid.select(0, i);
}

void CheckValidRectangle(const Tensor& valid, ImageExtent canvas,
                         ImageExtent valid_extent, std::stop_token stop) {
  const auto* values = valid.const_data_ptr<bool>();
  for (std::uint32_t y = 0; y < canvas.height; ++y) {
    Stop(stop);
    for (std::uint32_t x = 0; x < canvas.width; ++x) {
      const bool expected = x < valid_extent.width && y < valid_extent.height;
      if (values[static_cast<std::size_t>(y) * canvas.width + x] != expected)
        InvalidBatch("batch_geometry.valid_rectangle");
    }
  }
}

VisionSample SampleAt(const ImageBatch& batch, std::size_t index,
                     const Tensor& image, ImageExtent canvas) {
  const auto& item = batch.items[index];
  VisionSample sample;
  sample.image = image;
  sample.width = canvas.width;
  sample.height = canvas.height;
  sample.pixels = batch.pixels;
  sample.pixels.layout = ImageLayout::chw;
  sample.image_path = item.image_path;
  sample.target_path = item.target_path;
  sample.label_categories = item.label_categories;
  sample.categories = item.categories;
  sample.annotation = item.annotation;
  sample.geometry = item.geometry;
  if (batch.paired_pixels) {
    sample.paired_target = PairedAt(batch, index);
    sample.paired_target_pixels = *batch.paired_pixels;
    sample.paired_target_pixels->layout = ImageLayout::chw;
  }
  if (item.mask) {
    const auto& info = *item.mask;
    DecodedMask mask;
    mask.ids = MaskAt(batch, index);
    mask.encoding = info.encoding;
    mask.width = canvas.width;
    mask.height = canvas.height;
    mask.source_bit_depth = info.source_bit_depth;
    mask.source_interlaced = info.source_interlaced;
    mask.palette = info.palette;
    mask.ignore_id = info.ignore_id;
    mask.void_id = info.void_id;
    sample.mask_target = std::move(mask);
  }
  const auto begin = batch.label_offsets[index];
  const auto end = batch.label_offsets[index + 1];
  const auto* labels = batch.labels.const_data_ptr<std::int64_t>();
  sample.labels.reserve(static_cast<std::size_t>(end - begin));
  for (auto offset = begin; offset < end; ++offset) {
    const auto label = labels[offset];
    if (label < 0 || !sample.label_categories ||
        static_cast<std::uint64_t>(label) >= sample.label_categories->size())
      InvalidBatch("batch_geometry.label");
    sample.labels.push_back(static_cast<std::size_t>(label));
  }
  return sample;
}

void CheckCpu(const ImageBatch& batch) {
  const auto cpu = [](const Tensor& tensor) {
    if (!tensor.device().is_cpu()) InvalidBatch("batch_geometry.cpu_only");
  };
  cpu(batch.labels);
  if (batch.layout == ImageBatchLayout::ragged) {
    for (const auto* list : {&batch.image_list, &batch.valid_list,
                             &batch.paired_image_list, &batch.mask_list})
      for (const auto& tensor : *list) cpu(tensor);
  } else {
    cpu(batch.images);
    cpu(batch.valid);
    if (batch.paired_images) cpu(*batch.paired_images);
    if (batch.masks) cpu(*batch.masks);
  }
}

void BoundPerSampleOptions(JointTransformOptions& joint,
                           const ImageBatchLimits& limits,
                           std::size_t samples) {
  joint.limits.max_output_bytes = std::min<std::uint64_t>(
      joint.limits.max_output_bytes,
      QuotaBytes(limits.max_output_bytes, samples));
  joint.limits.max_working_bytes = std::min<std::uint64_t>(
      joint.limits.max_working_bytes,
      QuotaBytes(limits.max_working_bytes, samples));
  joint.limits.max_objects = std::min(
      joint.limits.max_objects,
      QuotaNodes(limits.max_metadata_nodes, samples, 20));
  joint.limits.max_keypoints = std::min(
      joint.limits.max_keypoints,
      QuotaNodes(limits.max_metadata_nodes, samples, 4));
  joint.limits.max_polygon_points = std::min(
      joint.limits.max_polygon_points,
      QuotaNodes(limits.max_metadata_nodes, samples, 3));
  joint.limits.max_polygon_fragments = std::min(
      joint.limits.max_polygon_fragments,
      QuotaNodes(limits.max_metadata_nodes, samples, 16));
  joint.limits.max_geometry_checks = std::min<std::size_t>(
      joint.limits.max_geometry_checks,
      QuotaNodes(limits.max_geometry_checks, samples));
  joint.limits.max_rle_runs = std::min(
      joint.limits.max_rle_runs,
      QuotaNodes(limits.max_metadata_nodes, samples));
  joint.limits.max_mask_work_pixels = std::min<std::uint64_t>(
      joint.limits.max_mask_work_pixels,
      QuotaBytes(limits.max_pixels, samples));
}
}  // namespace

std::string image_batch_geometry_identity(
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options) {
  detail::CheckBatchLimits(options.limits);
  const auto joint_identity =
      joint_transform_identity(random, options.joint);
  if (!options.output_size_divisor ||
      (options.output_padded_extent &&
       (!options.output_padded_extent->width ||
        !options.output_padded_extent->height)))
    InvalidBatch("batch_geometry.output_policy");
  Sha256 hash;
  constexpr std::string_view domain = "liuding.image-batch-geometry.v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  hash.update(std::as_bytes(std::span(joint_identity.data(),
                                      joint_identity.size())));
  const auto integer = [&](std::uint64_t value) { HashInteger(hash, value); };
  integer(options.limits.max_samples);
  integer(options.limits.max_dimension);
  integer(options.limits.max_pixels);
  integer(options.limits.max_output_bytes);
  integer(options.limits.max_working_bytes);
  integer(options.limits.max_metadata_nodes);
  integer(options.limits.max_metadata_bytes);
  integer(options.limits.max_string_bytes);
  integer(options.limits.max_geometry_checks);
  integer(options.output_padded_extent.has_value());
  if (options.output_padded_extent) {
    integer(options.output_padded_extent->width);
    integer(options.output_padded_extent->height);
  }
  integer(options.output_size_divisor);
  return sha256_digest_hex(hash.finish());
}

std::vector<std::vector<ImageGeometryStep>> sample_image_batch_geometry(
    const ImageBatch& batch, std::string_view dataset_id,
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options, std::uint64_t seed,
    std::uint64_t epoch) {
  detail::CheckBatchLimits(options.limits);
  if (batch.items.empty() || batch.items.size() > options.limits.max_samples ||
      (batch.layout != ImageBatchLayout::stack &&
          batch.layout != ImageBatchLayout::pad &&
          batch.layout != ImageBatchLayout::ragged))
    InvalidBatch("batch_geometry.sample_input");
  CheckOutputPolicy(batch.layout, options);
  detail::CheckImageBatch(batch, options.limits);
  const auto transform_id = image_batch_geometry_identity(random, options);
  std::vector<std::vector<ImageGeometryStep>> result;
  result.reserve(batch.items.size());
  for (std::size_t i = 0; i < batch.items.size(); ++i) {
    const auto& item = batch.items[i];
    SampleRandom stream({.dataset_id = std::string(dataset_id),
                         .sample_id = item.key.id,
                         .transform_id = transform_id,
                         .seed = seed,
                         .epoch = epoch,
                         .occurrence = item.key.occurrence,
                         .view = item.key.view});
    result.push_back(sample_geometry(item.valid_extent, random, stream));
  }
  return result;
}

ImageBatch transform_image_batch_geometry(
    const ImageBatch& batch,
    std::span<const std::vector<ImageGeometryStep>> sample_steps,
    const ImageBatchGeometryOptions& options, std::stop_token stop) {
  Stop(stop);
  try {
    detail::CheckImageBatch(batch, options.limits, stop);
    CheckCpu(batch);
    if (sample_steps.size() != batch.items.size())
      InvalidBatch("batch_geometry.step_count");
    const auto count = batch.items.size();
    JointTransformOptions joint = options.joint;
    BoundPerSampleOptions(joint, options.limits, count);
    CheckOutputPolicy(batch.layout, options);

    std::vector<ImageExtent> input_canvases;
    std::vector<ImageExtent> output_extents;
    input_canvases.reserve(count);
    output_extents.reserve(count);
    ImageExtent largest{0, 0};
    std::uint64_t max_intermediate_bytes = 0;
    for (std::size_t i = 0; i < count; ++i) {
      Stop(stop);
      const auto image = PhotoAt(batch, i);
      const auto canvas = Extent(image);
      const auto& item = batch.items[i];
      CheckValidRectangle(ValidAt(batch, i), canvas, item.valid_extent, stop);
      const bool remove_collation_padding =
          !Equal(item.valid_extent, canvas);
      if (sample_steps[i].size() + remove_collation_padding > 128)
        InvalidBatch("batch_geometry.steps");
      auto trace = GeometryTrace(canvas);
      if (remove_collation_padding)
        trace = trace.cropped(0, 0, item.valid_extent);
      std::uint64_t maximum =
          static_cast<std::uint64_t>(canvas.width) * canvas.height;
      for (const auto& step : sample_steps[i]) {
        Stop(stop);
        trace = Append(std::move(trace), step);
        const auto extent = trace.output_extent();
        if (extent.width > joint.limits.max_dimension ||
            extent.height > joint.limits.max_dimension ||
            extent.width > options.limits.max_dimension ||
            extent.height > options.limits.max_dimension)
          BatchLimit("batch_geometry.dimension");
        const auto pixels = Multiply(extent.width, extent.height,
                                     "batch_geometry.pixels");
        if (pixels > joint.limits.max_pixels ||
            pixels > options.limits.max_pixels)
          BatchLimit("batch_geometry.sample_pixels");
        maximum = std::max(maximum, pixels);
      }
      if (batch.layout != ImageBatchLayout::ragged && i > 0 &&
          batch.layout == ImageBatchLayout::stack &&
          !Equal(trace.output_extent(), output_extents.front()))
        InvalidBatch("batch_geometry.stack_shape");
      input_canvases.push_back(canvas);
      output_extents.push_back(trace.output_extent());
      largest.width = std::max(largest.width, trace.output_extent().width);
      largest.height = std::max(largest.height, trace.output_extent().height);
      const auto channels = batch.pixels.channels() +
                            (batch.paired_pixels
                                 ? batch.paired_pixels->channels()
                                 : 0);
      const auto bytes_per_pixel =
          static_cast<std::uint64_t>(batch.pixels.channels()) *
              c10::elementSize(batch.pixels.dtype) +
          (batch.paired_pixels
               ? static_cast<std::uint64_t>(batch.paired_pixels->channels()) *
                     c10::elementSize(batch.paired_pixels->dtype)
               : 0) +
          (batch.items[i].mask ? 8 : 0);
      const bool sample_has_rle =
          item.annotation &&
          std::any_of(item.annotation->objects.begin(),
                      item.annotation->objects.end(), [](const auto& object) {
                        return object.rle.has_value();
                      });
      auto working_per_pixel =
          2 * bytes_per_pixel + (sample_has_rle ? 4 : 0);
      if (joint.interpolation == ImageInterpolation::bilinear)
        working_per_pixel += 32ULL * channels;
      max_intermediate_bytes = Add(
          max_intermediate_bytes,
          Multiply(maximum, working_per_pixel,
                   "batch_geometry.working_bytes"),
          "batch_geometry.working_bytes");
    }

    ImageExtent output_canvas = largest;
    if (batch.layout == ImageBatchLayout::pad) {
      if (options.output_padded_extent) {
        if (options.output_padded_extent->width < largest.width ||
            options.output_padded_extent->height < largest.height)
          InvalidBatch("batch_geometry.output_extent");
        output_canvas = *options.output_padded_extent;
      }
      const auto round = [&](std::uint32_t side) {
        const auto rounded =
            (static_cast<std::uint64_t>(side) +
             options.output_size_divisor - 1) /
            options.output_size_divisor * options.output_size_divisor;
        if (rounded > options.limits.max_dimension ||
            rounded > joint.limits.max_dimension)
          BatchLimit("batch_geometry.output_dimension");
        return static_cast<std::uint32_t>(rounded);
      };
      output_canvas = {round(output_canvas.width), round(output_canvas.height)};
    }
    std::uint64_t output_pixels = 0;
    if (batch.layout == ImageBatchLayout::stack ||
        batch.layout == ImageBatchLayout::pad) {
      output_pixels = Multiply(
          Multiply(output_canvas.width, output_canvas.height,
                   "batch_geometry.output_pixels"),
          count, "batch_geometry.output_pixels");
    } else {
      for (const auto extent : output_extents)
        output_pixels = Add(
            output_pixels,
            Multiply(extent.width, extent.height,
                     "batch_geometry.output_pixels"),
            "batch_geometry.output_pixels");
    }
    if (output_pixels > options.limits.max_pixels)
      BatchLimit("batch_geometry.output_pixels");
    if (batch.items.front().mask && batch.layout == ImageBatchLayout::pad &&
        !options.joint.mask_padding_id) {
      for (const auto extent : output_extents)
        if (!Equal(extent, output_canvas))
          InvalidBatch("batch_geometry.mask_padding_id");
    }
    const auto pixel_bytes =
        static_cast<std::uint64_t>(batch.pixels.channels()) *
            c10::elementSize(batch.pixels.dtype) +
        (batch.paired_pixels
             ? static_cast<std::uint64_t>(batch.paired_pixels->channels()) *
                   c10::elementSize(batch.paired_pixels->dtype)
             : 0) +
        1 + (batch.items.front().mask ? 8 : 0);
    const auto output_bytes = Add(
        Multiply(output_pixels, pixel_bytes, "batch_geometry.output_bytes"),
        Multiply(static_cast<std::uint64_t>(batch.labels.numel()), 8,
                 "batch_geometry.output_bytes"),
        "batch_geometry.output_bytes");
    if (output_bytes > options.limits.max_output_bytes)
      BatchLimit("batch_geometry.output_bytes");
    const auto source_metadata = detail::CheckBatchMetadata(
        batch, input_canvases, options.limits, false, stop);
    const auto working = Add(
        Add(max_intermediate_bytes, output_bytes,
            "batch_geometry.working_bytes"),
        Multiply(source_metadata, 2, "batch_geometry.working_bytes"),
        "batch_geometry.working_bytes");
    if (working > options.limits.max_working_bytes)
      BatchLimit("batch_geometry.working_bytes");

    std::vector<VisionSample> transformed;
    std::vector<ImageBatchKey> keys;
    transformed.reserve(count);
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      Stop(stop);
      auto sample = SampleAt(batch, i, PhotoAt(batch, i), input_canvases[i]);
      std::vector<ImageGeometryStep> effective;
      effective.reserve(sample_steps[i].size() + 1);
      if (!Equal(batch.items[i].valid_extent, input_canvases[i]))
        effective.push_back({.kind = ImageGeometryKind::crop,
                             .extent = batch.items[i].valid_extent});
      effective.insert(effective.end(), sample_steps[i].begin(),
                       sample_steps[i].end());
      transformed.push_back(
          transform_sample(sample, effective, joint, stop));
      keys.push_back(batch.items[i].key);
    }
    Stop(stop);
    ImageCollateOptions collate;
    collate.layout = batch.layout;
    collate.image_padding = joint.image_padding;
    collate.paired_padding = joint.paired_padding;
    collate.mask_padding_id = joint.mask_padding_id;
    if (batch.layout == ImageBatchLayout::pad)
      collate.padded_extent = output_canvas;
    collate.limits = options.limits;
    return collate_images(transformed, keys, collate, stop);
  } catch (const std::bad_alloc&) {
    BatchLimit("batch_geometry.allocate");
  } catch (const c10::OutOfMemoryError&) {
    BatchLimit("batch_geometry.allocate");
  } catch (const c10::Error& error) {
    throw DeviceError("Image batch geometry failed",
                      ErrorDetails{.component = "data.image.batch_geometry",
                                   .operation = "batch_geometry.execute",
                                   .actual = error.what_without_backtrace()});
  }
}
}  // namespace jyppx::liuding::data::image
