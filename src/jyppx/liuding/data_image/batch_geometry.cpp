// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "image_batch_detail.hpp"
#include "joint_annotation_detail.hpp"

#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>

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

std::uint64_t Add(std::uint64_t a, std::uint64_t b, const char* operation) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) BatchLimit(operation);
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
      (options.output_padded_extent && (!options.output_padded_extent->width ||
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
      if (step.right || step.bottom) InvalidBatch("batch_geometry.crop_fields");
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
  return batch.layout == ImageBatchLayout::ragged ? batch.valid_list[index]
                                                  : batch.valid.select(0, i);
}

void CheckValidRectangle(const Tensor& valid, ImageExtent canvas,
                         ImageExtent valid_extent, std::stop_token stop) {
  if (!valid.device().is_cpu()) {
    // A CUDA batch is produced by the validated CPU collator or by an
    // explicit device transfer. Reading it back to validate every padding
    // bit would synchronize the caller's stream, so keep this check shape
    // only; CheckImageBatch already validated the rectangle dimensions.
    Stop(stop);
    return;
  }
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

bool IsPanoptic(const ImageBatchItem& item) {
  return item.annotation && item.annotation->panoptic_mask_path.has_value();
}

void CheckCudaMetadata(const ImageBatch& batch, bool allow_summary_transfer) {
  for (const auto& item : batch.items)
    if (IsPanoptic(item) && !allow_summary_transfer)
      InvalidBatch("batch_geometry.cuda_panoptic");
}

std::shared_ptr<const VisionAnnotationRecord> MapCudaAnnotation(
    const ImageBatch& batch, std::size_t index, ImageExtent canvas,
    std::span<const ImageGeometryStep> steps,
    const JointTransformOptions& options, std::stop_token stop) {
  const auto& item = batch.items[index];
  if (!item.annotation) return {};
  VisionSample source;
  source.width = canvas.width;
  source.height = canvas.height;
  source.categories = item.categories;
  source.annotation = item.annotation;
  if (item.mask) {
    DecodedMask mask;
    mask.ids = MaskAt(batch, index);
    mask.width = canvas.width;
    mask.height = canvas.height;
    source.mask_target = std::move(mask);
  }
  auto local = GeometryTrace(canvas);
  std::vector<ImageExtent> extents{canvas};
  extents.reserve(steps.size() + 1);
  for (const auto& step : steps) {
    Stop(stop);
    local = Append(local, step);
    extents.push_back(local.output_extent());
  }
  auto budget = detail::CheckJointAnnotations(source, steps, extents, options,
                                              local, stop);
  VisionSample output;
  output.width = local.output_extent().width;
  output.height = local.output_extent().height;
  detail::MapJointAnnotations(source, output, steps, extents, options, local,
                              budget, stop);
  return output.annotation;
}

struct CudaPanopticResult {
  std::shared_ptr<const VisionAnnotationRecord> annotation;
  Tensor mask;
};

[[noreturn]] void PanopticMismatch(const char* operation) {
  throw SerializationError(
      "CUDA panoptic mask/table mismatch",
      ErrorDetails{.component = "data.image.batch_geometry",
                   .operation = operation});
}

CudaPanopticResult MapCudaPanoptic(const ImageBatch& batch, std::size_t index,
                                   const Tensor& source_mask,
                                   const Tensor& output_mask,
                                   ImageExtent source_canvas,
                                   std::span<const ImageGeometryStep> steps,
                                   const JointTransformOptions& options,
                                   std::stop_token stop) {
  const auto& item = batch.items[index];
  if (!item.mask || item.mask->encoding != MaskEncoding::panoptic_rgb ||
      !item.mask->void_id || !item.annotation ||
      !item.annotation->panoptic_mask_path ||
      item.annotation->panoptic_mask_path->empty() ||
      !item.annotation->panoptic_mask_materialized)
    InvalidBatch("batch_geometry.cuda_panoptic_source");
  auto local = GeometryTrace(source_canvas);
  std::vector<ImageExtent> extents{source_canvas};
  extents.reserve(steps.size() + 1);
  for (const auto& step : steps) {
    Stop(stop);
    local = Append(local, step);
    extents.push_back(local.output_extent());
  }
  const auto destination = extents.back();
  VisionSample source;
  source.width = source_canvas.width;
  source.height = source_canvas.height;
  source.categories = item.categories;
  source.annotation = item.annotation;
  DecodedMask mask;
  mask.ids = source_mask;
  mask.width = source_canvas.width;
  mask.height = source_canvas.height;
  mask.encoding = item.mask->encoding;
  mask.ignore_id = item.mask->ignore_id;
  mask.void_id = item.mask->void_id;
  source.mask_target = std::move(mask);
  static_cast<void>(detail::CheckJointAnnotations(source, steps, extents,
                                                  options, local, stop, true));

  const auto& bound = *item.annotation;
  const auto source_pixels = Multiply(source_canvas.width, source_canvas.height,
                                      "batch_geometry.cuda_panoptic_pixels");
  const auto output_pixels = Multiply(destination.width, destination.height,
                                      "batch_geometry.cuda_panoptic_pixels");
  const auto scans = Multiply(
      Add(source_pixels, output_pixels, "batch_geometry.cuda_panoptic_work"),
      Multiply(
          Add(bound.objects.size(), 1U, "batch_geometry.cuda_panoptic_work"),
          8U, "batch_geometry.cuda_panoptic_work"),
      "batch_geometry.cuda_panoptic_work");
  if (scans > options.limits.max_mask_work_pixels)
    BatchLimit("batch_geometry.cuda_panoptic_work");
  const auto summary_bytes =
      Multiply(Add(Multiply(bound.objects.size(), 11U,
                            "batch_geometry.cuda_panoptic_summary"),
                   2U, "batch_geometry.cuda_panoptic_summary"),
               sizeof(double), "batch_geometry.cuda_panoptic_summary");
  const auto reduction_bytes =
      Add(Multiply(Add(source_pixels, output_pixels,
                       "batch_geometry.cuda_panoptic_working"),
                   64U, "batch_geometry.cuda_panoptic_working"),
          summary_bytes, "batch_geometry.cuda_panoptic_working");
  if (reduction_bytes > options.limits.max_working_bytes ||
      summary_bytes > std::numeric_limits<std::size_t>::max())
    BatchLimit("batch_geometry.cuda_panoptic_summary");

  const auto coordinates = source_mask.options().dtype(torch::kInt64);
  const auto source_x =
      torch::arange(source_canvas.width, coordinates)
          .unsqueeze(0)
          .expand({source_canvas.height, source_canvas.width});
  const auto source_y =
      torch::arange(source_canvas.height, coordinates)
          .unsqueeze(1)
          .expand({source_canvas.height, source_canvas.width});
  const auto output_x = torch::arange(destination.width, coordinates)
                            .unsqueeze(0)
                            .expand({destination.height, destination.width});
  const auto output_y = torch::arange(destination.height, coordinates)
                            .unsqueeze(1)
                            .expand({destination.height, destination.width});
  Tensor visibility;
  if (local.source_region()) {
    const auto region = *local.source_region();
    auto x =
        torch::arange(source_canvas.width, coordinates.dtype(torch::kFloat64));
    auto y =
        torch::arange(source_canvas.height, coordinates.dtype(torch::kFloat64));
    auto x_weight =
        ((x + 1).clamp_max(region.x_max) - x.clamp_min(region.x_min))
            .clamp_min(0);
    auto y_weight =
        ((y + 1).clamp_max(region.y_max) - y.clamp_min(region.y_min))
            .clamp_min(0);
    visibility = y_weight.unsqueeze(1) * x_weight.unsqueeze(0);
  } else {
    visibility = torch::zeros({source_canvas.height, source_canvas.width},
                              coordinates.dtype(torch::kFloat64));
  }

  const auto void_id = *item.mask->void_id;
  const auto ignore_id = item.mask->ignore_id;
  auto source_known = source_mask.eq(void_id);
  auto output_known = output_mask.eq(void_id);
  if (ignore_id) {
    source_known.logical_or_(source_mask.eq(*ignore_id));
    output_known.logical_or_(output_mask.eq(*ignore_id));
  }
  std::vector<Tensor> summary;
  summary.reserve(static_cast<std::size_t>(summary_bytes / sizeof(double)));
  const auto collect = [&](const Tensor& ids, const Tensor& x, const Tensor& y,
                           Tensor& known, std::int64_t id,
                           const Tensor* weights) {
    auto match = ids.eq(id);
    known.logical_or_(match);
    auto absent = match.logical_not();
    summary.push_back(match.sum().to(torch::kFloat64));
    summary.push_back(
        x.masked_fill(absent, ids.size(1)).min().to(torch::kFloat64));
    summary.push_back(
        y.masked_fill(absent, ids.size(0)).min().to(torch::kFloat64));
    summary.push_back(x.masked_fill(absent, -1).max().to(torch::kFloat64));
    summary.push_back(y.masked_fill(absent, -1).max().to(torch::kFloat64));
    if (weights)
      summary.push_back((match.to(torch::kFloat64) * *weights).sum());
  };
  for (const auto& object : bound.objects) {
    Stop(stop);
    collect(source_mask, source_x, source_y, source_known, object.source_id,
            &visibility);
    collect(output_mask, output_x, output_y, output_known, object.source_id,
            nullptr);
  }
  summary.push_back(source_known.logical_not().sum().to(torch::kFloat64));
  summary.push_back(output_known.logical_not().sum().to(torch::kFloat64));
  Stop(stop);
  // The explicit option authorizes this single bounded D2H summary and sync.
  const auto host = torch::stack(summary).cpu().contiguous();
  const auto* values = host.const_data_ptr<double>();
  const auto unknown_offset = summary.size() - 2U;
  if (values[unknown_offset] != 0 || values[unknown_offset + 1U] != 0)
    PanopticMismatch("batch_geometry.cuda_panoptic_unknown_id");

  auto record = std::make_shared<VisionAnnotationRecord>();
  record->source_image_id = bound.source_image_id;
  record->image_path = bound.image_path;
  record->panoptic_mask_path = bound.panoptic_mask_path;
  record->width = destination.width;
  record->height = destination.height;
  record->panoptic_mask_materialized = true;
  std::vector<std::int64_t> kept;
  kept.reserve(bound.objects.size());
  for (std::size_t i = 0; i < bound.objects.size(); ++i) {
    Stop(stop);
    const auto* row = values + i * 11U;
    const auto& original = bound.objects[i];
    const auto source_count = static_cast<std::uint64_t>(row[0]);
    if (!source_count || original.mask_pixel_area != source_count ||
        !original.box || original.box->x_min != row[1] ||
        original.box->y_min != row[2] || original.box->x_max != row[3] + 1 ||
        original.box->y_max != row[4] + 1)
      PanopticMismatch("batch_geometry.cuda_panoptic_source_binding");
    const auto output_count = static_cast<std::uint64_t>(row[6]);
    if (!output_count) continue;
    auto object = original;
    object.box = ImageBox{row[7], row[8], row[9] + 1, row[10] + 1};
    object.mask_pixel_area = output_count;
    if (static_cast<long double>(output_count) < options.min_region_area ||
        row[5] / source_count < options.min_region_visibility ||
        !detail::PanopticBoxKeeps(original, object, local, options))
      continue;
    kept.push_back(object.source_id);
    record->objects.push_back(std::move(object));
  }
  if (kept.size() == bound.objects.size()) return {record, output_mask};
  auto keep = output_mask.eq(void_id);
  if (ignore_id) keep.logical_or_(output_mask.eq(*ignore_id));
  for (const auto id : kept) {
    Stop(stop);
    keep.logical_or_(output_mask.eq(id));
  }
  auto filtered =
      torch::where(keep, output_mask, torch::full_like(output_mask, void_id));
  return {record, std::move(filtered)};
}

Tensor CudaNearestResize(const Tensor& input, bool mask, ImageExtent before,
                         ImageExtent after, std::stop_token stop) {
  Stop(stop);
  const auto index_options =
      input.options().dtype(torch::kInt64).requires_grad(false);
  const auto x =
      (torch::arange(static_cast<std::int64_t>(after.width), index_options) *
           static_cast<std::int64_t>(before.width) +
       static_cast<std::int64_t>(before.width / 2U))
          .floor_divide(static_cast<std::int64_t>(after.width));
  const auto y =
      (torch::arange(static_cast<std::int64_t>(after.height), index_options) *
           static_cast<std::int64_t>(before.height) +
       static_cast<std::int64_t>(before.height / 2U))
          .floor_divide(static_cast<std::int64_t>(after.height));
  // The smaller first gather avoids materializing a broadcast HxW index grid.
  const bool rows_first =
      static_cast<std::uint64_t>(after.height) * before.width <=
      static_cast<std::uint64_t>(after.width) * before.height;
  const auto row_dim = mask ? 0 : 1;
  const auto column_dim = mask ? 1 : 2;
  if (rows_first)
    return input.index_select(row_dim, y).index_select(column_dim, x);
  return input.index_select(column_dim, x).index_select(row_dim, y);
}

Tensor CudaCopyStep(const Tensor& input, bool mask, ImageExtent before,
                    ImageExtent after, const ImageGeometryStep& step,
                    const c10::Scalar& fill, std::stop_token stop) {
  Stop(stop);
  switch (step.kind) {
    case ImageGeometryKind::resize:
      return CudaNearestResize(input, mask, before, after, stop);
    case ImageGeometryKind::crop:
      return input
          .narrow(mask ? 0 : 1, step.top, after.height)
          .narrow(mask ? 1 : 2, step.left, after.width)
          .clone();
    case ImageGeometryKind::pad:
      return torch::constant_pad_nd(
          input,
          {static_cast<std::int64_t>(step.left),
           static_cast<std::int64_t>(step.right),
           static_cast<std::int64_t>(step.top),
           static_cast<std::int64_t>(step.bottom)},
          fill)
          .contiguous();
    case ImageGeometryKind::horizontal_flip:
      return input.flip({mask ? 1 : 2}).contiguous();
    case ImageGeometryKind::vertical_flip:
      return input.flip({mask ? 0 : 1}).contiguous();
  }
  InvalidBatch("batch_geometry.kind");
}

Tensor CudaPhotoStep(const Tensor& input, ImageExtent before,
                     ImageExtent after, const ImageGeometryStep& step,
                     const JointTransformOptions& options, double fill,
                     std::stop_token stop) {
  if (step.kind != ImageGeometryKind::resize ||
      options.interpolation == ImageInterpolation::nearest_exact)
    return CudaCopyStep(input, false, before, after, step, fill, stop);
  Stop(stop);
  const bool integer = input.scalar_type() == torch::kUInt8 ||
                       input.scalar_type() == torch::kUInt16;
  auto floating = integer ? input.to(torch::kFloat64) : input;
  auto result = torch::nn::functional::interpolate(
                    floating.unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<std::int64_t>{
                            static_cast<std::int64_t>(after.height),
                            static_cast<std::int64_t>(after.width)})
                        .mode(torch::kBilinear)
                        .align_corners(false)
                        .antialias(true))
                    .squeeze(0);
  Stop(stop);
  if (integer) {
    const auto maximum = input.scalar_type() == torch::kUInt8 ? 255 : 65535;
    result = result.round().clamp(0, maximum).to(input.scalar_type());
  }
  return result.contiguous();
}

Tensor CudaPadToCanvas(const Tensor& input, ImageExtent extent,
                       ImageExtent canvas, const c10::Scalar& fill) {
  if (Equal(extent, canvas)) return input.contiguous();
  return torch::constant_pad_nd(
             input,
             {0, static_cast<std::int64_t>(canvas.width - extent.width), 0,
              static_cast<std::int64_t>(canvas.height - extent.height)},
             fill)
      .contiguous();
}

Tensor CudaValid(ImageExtent extent, ImageExtent canvas,
                 const torch::Device& device) {
  auto valid = torch::zeros(
      {static_cast<std::int64_t>(canvas.height),
       static_cast<std::int64_t>(canvas.width)},
      torch::TensorOptions().device(device).dtype(torch::kBool));
  valid.narrow(0, 0, extent.height).narrow(1, 0, extent.width).fill_(true);
  return valid;
}

void BoundPerSampleOptions(JointTransformOptions& joint,
                           const ImageBatchLimits& limits,
                           std::size_t samples) {
  joint.limits.max_output_bytes =
      std::min<std::uint64_t>(joint.limits.max_output_bytes,
                              QuotaBytes(limits.max_output_bytes, samples));
  joint.limits.max_working_bytes =
      std::min<std::uint64_t>(joint.limits.max_working_bytes,
                              QuotaBytes(limits.max_working_bytes, samples));
  joint.limits.max_objects =
      std::min(joint.limits.max_objects,
               QuotaNodes(limits.max_metadata_nodes, samples, 20));
  joint.limits.max_keypoints =
      std::min(joint.limits.max_keypoints,
               QuotaNodes(limits.max_metadata_nodes, samples, 4));
  joint.limits.max_polygon_points =
      std::min(joint.limits.max_polygon_points,
               QuotaNodes(limits.max_metadata_nodes, samples, 3));
  joint.limits.max_polygon_fragments =
      std::min(joint.limits.max_polygon_fragments,
               QuotaNodes(limits.max_metadata_nodes, samples, 16));
  joint.limits.max_geometry_checks =
      std::min<std::size_t>(joint.limits.max_geometry_checks,
                            QuotaNodes(limits.max_geometry_checks, samples));
  joint.limits.max_rle_runs =
      std::min(joint.limits.max_rle_runs,
               QuotaNodes(limits.max_metadata_nodes, samples));
  joint.limits.max_mask_work_pixels =
      std::min<std::uint64_t>(joint.limits.max_mask_work_pixels,
                              QuotaBytes(limits.max_pixels, samples));
}
}  // namespace

std::string image_batch_geometry_identity(
    const RandomGeometryOptions& random,
    const ImageBatchGeometryOptions& options) {
  detail::CheckBatchLimits(options.limits);
  const auto joint_identity = joint_transform_identity(random, options.joint);
  if (!options.output_size_divisor ||
      (options.output_padded_extent && (!options.output_padded_extent->width ||
                                        !options.output_padded_extent->height)))
    InvalidBatch("batch_geometry.output_policy");
  Sha256 hash;
  constexpr std::string_view domain = "liuding.image-batch-geometry.v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  hash.update(
      std::as_bytes(std::span(joint_identity.data(), joint_identity.size())));
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
  integer(options.allow_cuda_panoptic_summary_transfer);
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
    if (sample_steps.size() != batch.items.size())
      InvalidBatch("batch_geometry.step_count");
    const auto count = batch.items.size();
    const auto device = PhotoAt(batch, 0).device();
    if (device.is_cpu()) {
      CheckCpu(batch);
    } else if (device.is_cuda()) {
      CheckCudaMetadata(batch, options.allow_cuda_panoptic_summary_transfer);
    } else {
      InvalidBatch("batch_geometry.device");
    }
    if (device.is_cuda())
      static_cast<void>(joint_transform_identity({}, options.joint));
    JointTransformOptions joint = options.joint;
    if (!joint.resource_budget) joint.resource_budget = options.resource_budget;
    BoundPerSampleOptions(joint, options.limits, count);
    CheckOutputPolicy(batch.layout, options);

    std::vector<ImageExtent> input_canvases;
    std::vector<ImageExtent> output_extents;
    input_canvases.reserve(count);
    output_extents.reserve(count);
    ImageExtent largest{0, 0};
    std::uint64_t max_intermediate_bytes = 0;
    std::uint64_t cuda_index_bytes = 0;
    for (std::size_t i = 0; i < count; ++i) {
      Stop(stop);
      const auto image = PhotoAt(batch, i);
      const auto canvas = Extent(image);
      const auto& item = batch.items[i];
      CheckValidRectangle(ValidAt(batch, i), canvas, item.valid_extent, stop);
      const bool remove_collation_padding = !Equal(item.valid_extent, canvas);
      if (sample_steps[i].size() + remove_collation_padding > 128)
        InvalidBatch("batch_geometry.steps");
      auto trace = GeometryTrace(canvas);
      if (remove_collation_padding)
        trace = trace.cropped(0, 0, item.valid_extent);
      std::uint64_t maximum =
          static_cast<std::uint64_t>(canvas.width) * canvas.height;
      std::uint64_t sample_index_bytes = 0;
      for (const auto& step : sample_steps[i]) {
        Stop(stop);
        trace = Append(trace, step);
        const auto extent = trace.output_extent();
        if (extent.width > joint.limits.max_dimension ||
            extent.height > joint.limits.max_dimension ||
            extent.width > options.limits.max_dimension ||
            extent.height > options.limits.max_dimension)
          BatchLimit("batch_geometry.dimension");
        const auto pixels =
            Multiply(extent.width, extent.height, "batch_geometry.pixels");
        if (pixels > joint.limits.max_pixels ||
            pixels > options.limits.max_pixels)
          BatchLimit("batch_geometry.sample_pixels");
        maximum = std::max(maximum, pixels);
        if (device.is_cuda() && step.kind == ImageGeometryKind::resize &&
            (joint.interpolation == ImageInterpolation::nearest_exact ||
             item.mask)) {
          const auto axes = Add(extent.width, extent.height,
                                "batch_geometry.cuda_index_bytes");
          sample_index_bytes = std::max(
              sample_index_bytes, Multiply(axes, 4U * sizeof(std::int64_t),
                                           "batch_geometry.cuda_index_bytes"));
        }
      }
      if (batch.layout != ImageBatchLayout::ragged && i > 0 &&
          batch.layout == ImageBatchLayout::stack &&
          !Equal(trace.output_extent(), output_extents.front()))
        InvalidBatch("batch_geometry.stack_shape");
      input_canvases.push_back(canvas);
      output_extents.push_back(trace.output_extent());
      largest.width = std::max(largest.width, trace.output_extent().width);
      largest.height = std::max(largest.height, trace.output_extent().height);
      const auto channels =
          batch.pixels.channels() +
          (batch.paired_pixels ? batch.paired_pixels->channels() : 0);
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
          std::any_of(
              item.annotation->objects.begin(), item.annotation->objects.end(),
              [](const auto& object) { return object.rle.has_value(); });
      auto working_per_pixel = 2 * bytes_per_pixel + (sample_has_rle ? 4 : 0);
      if (joint.interpolation == ImageInterpolation::bilinear)
        working_per_pixel += 32ULL * channels;
      max_intermediate_bytes = Add(
          max_intermediate_bytes,
          Multiply(maximum, working_per_pixel, "batch_geometry.working_bytes"),
          "batch_geometry.working_bytes");
      cuda_index_bytes = Add(cuda_index_bytes, sample_index_bytes,
                             "batch_geometry.cuda_index_bytes");
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
        const auto rounded = (static_cast<std::uint64_t>(side) +
                              options.output_size_divisor - 1) /
                             options.output_size_divisor *
                             options.output_size_divisor;
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
      output_pixels =
          Multiply(Multiply(output_canvas.width, output_canvas.height,
                            "batch_geometry.output_pixels"),
                   count, "batch_geometry.output_pixels");
    } else {
      for (const auto extent : output_extents)
        output_pixels = Add(output_pixels,
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
    const auto output_bytes =
        Add(Multiply(output_pixels, pixel_bytes, "batch_geometry.output_bytes"),
            Multiply(static_cast<std::uint64_t>(batch.labels.numel()), 8,
                     "batch_geometry.output_bytes"),
            "batch_geometry.output_bytes");
    if (output_bytes > options.limits.max_output_bytes)
      BatchLimit("batch_geometry.output_bytes");
    const auto source_metadata = detail::CheckBatchMetadata(
        batch, input_canvases, options.limits, false, stop);
    const auto cuda_pixel_working_bytes =
        device.is_cuda()
            ? Add(Add(max_intermediate_bytes, output_bytes,
                      "batch_geometry.cuda_pixel_working"),
                  cuda_index_bytes, "batch_geometry.cuda_pixel_working")
            : 0;
    auto working =
        Add(Add(max_intermediate_bytes, output_bytes,
                "batch_geometry.working_bytes"),
            Multiply(source_metadata, 2, "batch_geometry.working_bytes"),
            "batch_geometry.working_bytes");
    if (device.is_cuda())
      working = Add(working, cuda_index_bytes, "batch_geometry.working_bytes");
    if (device.is_cuda()) {
      for (std::size_t i = 0; i < count; ++i) {
        if (!IsPanoptic(batch.items[i])) continue;
        const auto source_pixels =
            Multiply(input_canvases[i].width, input_canvases[i].height,
                     "batch_geometry.cuda_panoptic_working");
        const auto destination = batch.layout == ImageBatchLayout::pad
                                     ? output_canvas
                                     : output_extents[i];
        const auto destination_pixels =
            Multiply(destination.width, destination.height,
                     "batch_geometry.cuda_panoptic_working");
        const auto summary_bytes =
            Multiply(Add(Multiply(batch.items[i].annotation->objects.size(),
                                  11U, "batch_geometry.cuda_panoptic_working"),
                         2U, "batch_geometry.cuda_panoptic_working"),
                     sizeof(double), "batch_geometry.cuda_panoptic_working");
        working =
            Add(working,
                Add(Multiply(Add(source_pixels, destination_pixels,
                                 "batch_geometry.cuda_panoptic_working"),
                             64U, "batch_geometry.cuda_panoptic_working"),
                    summary_bytes, "batch_geometry.cuda_panoptic_working"),
                "batch_geometry.cuda_panoptic_working");
      }
    }
    if (working > options.limits.max_working_bytes)
      BatchLimit("batch_geometry.working_bytes");

    if (device.is_cuda()) {
      std::shared_ptr<ImageResourceLeaseSet> output_gpu_lease;
      ImageResourceLease pixel_working_lease;
      if (options.resource_budget) {
        output_gpu_lease = std::make_shared<ImageResourceLeaseSet>();
        output_gpu_lease->push_back(options.resource_budget->reserve(
            ImageResourceKind::gpu_bytes, output_bytes));
        pixel_working_lease = options.resource_budget->reserve(
            ImageResourceKind::gpu_bytes, cuda_pixel_working_bytes);
      }
      ImageBatch result;
      if (output_gpu_lease)
        result.resource_lease_sets.push_back(std::move(output_gpu_lease));
      result.layout = batch.layout;
      result.pixels = batch.pixels;
      result.paired_pixels = batch.paired_pixels;
      result.labels = batch.labels.clone();
      result.label_offsets = batch.label_offsets;
      result.items = batch.items;
      result.memory = {};

      std::vector<Tensor> images;
      std::vector<Tensor> paired;
      std::vector<Tensor> masks;
      std::vector<Tensor> valid;
      images.reserve(count);
      valid.reserve(count);
      if (batch.paired_pixels) paired.reserve(count);
      if (batch.items.front().mask) masks.reserve(count);

      for (std::size_t i = 0; i < count; ++i) {
        Stop(stop);
        auto image = PhotoAt(batch, i);
        auto paired_image =
            batch.paired_pixels ? std::optional<Tensor>(PairedAt(batch, i))
                                : std::nullopt;
        auto mask = batch.items[i].mask
                        ? std::optional<Tensor>(MaskAt(batch, i))
                        : std::nullopt;
        auto before = input_canvases[i];
        auto progress = GeometryTrace(before);
        std::vector<ImageGeometryStep> effective;
        if (!Equal(batch.items[i].valid_extent, before))
          effective.push_back({.kind = ImageGeometryKind::crop,
                               .extent = batch.items[i].valid_extent});
        effective.insert(effective.end(), sample_steps[i].begin(),
                         sample_steps[i].end());
        const bool pad_output = batch.layout == ImageBatchLayout::pad &&
                                !Equal(output_extents[i], output_canvas);
        if (pad_output)
          effective.push_back(
              {.kind = ImageGeometryKind::pad,
               .right = output_canvas.width - output_extents[i].width,
               .bottom = output_canvas.height - output_extents[i].height});
        const bool panoptic = IsPanoptic(batch.items[i]);
        if (!panoptic)
          result.items[i].annotation =
              MapCudaAnnotation(batch, i, before, effective, joint, stop);
        const auto annotation_steps =
            panoptic ? effective : std::vector<ImageGeometryStep>{};
        if (pad_output) effective.pop_back();
        for (const auto& step : effective) {
          const auto after = Append(progress, step).output_extent();
          image = CudaPhotoStep(image, before, after, step, joint,
                                joint.image_padding, stop);
          if (paired_image)
            *paired_image = CudaPhotoStep(*paired_image, before, after, step,
                                          joint, joint.paired_padding, stop);
          if (mask)
            *mask = CudaCopyStep(*mask, true, before, after, step,
                                 c10::Scalar(joint.mask_padding_id.value_or(0)),
                                 stop);
          progress = Append(progress, step);
          before = after;
        }
        result.items[i].valid_extent = output_extents[i];
        auto composed = batch.items[i].geometry.value_or(
            GeometryTrace(input_canvases[i]));
        for (const auto& step : effective) composed = Append(composed, step);
        if (batch.layout == ImageBatchLayout::pad &&
            !Equal(output_extents[i], output_canvas))
          composed = composed.padded(
              0, 0, output_canvas.width - output_extents[i].width,
              output_canvas.height - output_extents[i].height);
        result.items[i].geometry = composed;

        if (batch.layout == ImageBatchLayout::pad) {
          image = CudaPadToCanvas(image, output_extents[i], output_canvas,
                                  joint.image_padding);
          if (paired_image)
            *paired_image = CudaPadToCanvas(
                *paired_image, output_extents[i], output_canvas,
                joint.paired_padding);
          if (mask)
            *mask = CudaPadToCanvas(*mask, output_extents[i], output_canvas,
                                    c10::Scalar(
                                        joint.mask_padding_id.value_or(0)));
          valid.push_back(
              CudaValid(output_extents[i], output_canvas, device));
        } else {
          valid.push_back(
              CudaValid(output_extents[i], output_extents[i], device));
        }
        if (panoptic) {
          auto mapped =
              MapCudaPanoptic(batch, i, MaskAt(batch, i), *mask,
                              input_canvases[i], annotation_steps, joint, stop);
          result.items[i].annotation = std::move(mapped.annotation);
          mask = std::move(mapped.mask);
        }
        images.push_back(std::move(image));
        if (paired_image) paired.push_back(std::move(*paired_image));
        if (mask) masks.push_back(std::move(*mask));
      }

      if (batch.layout == ImageBatchLayout::ragged) {
        result.image_list = std::move(images);
        result.valid_list = std::move(valid);
        if (batch.paired_pixels) result.paired_image_list = std::move(paired);
        if (batch.items.front().mask) result.mask_list = std::move(masks);
      } else {
        result.images = torch::stack(images);
        result.valid = torch::stack(valid);
        if (batch.paired_pixels) result.paired_images = torch::stack(paired);
        if (batch.items.front().mask) result.masks = torch::stack(masks);
      }
      detail::CheckImageBatch(result, options.limits, stop);
      if (options.resource_budget)
        result.resource_lease_sets.push_back(detail::ReserveBatchResources(
            result, options.resource_budget, options.limits, stop, false));
      return result;
    }

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
      transformed.push_back(transform_sample(sample, effective, joint, stop));
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
    collate.resource_budget = options.resource_budget;
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
