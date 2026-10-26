// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "image_batch_detail.hpp"

#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <string_view>

namespace jyppx::liuding::data::image {
namespace {
using detail::BatchLimit;
using detail::InvalidBatch;
void Stop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Image batch transform cancelled");
}
std::uint64_t ScalarBytes(Dtype dtype) {
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16) return 2;
  if (dtype == torch::kFloat32) return 4;
  if (dtype == torch::kFloat64) return 8;
  InvalidBatch("batch_transform.dtype");
}
PixelSpec Policy(const PixelSpec& source, const BatchPixelTransform& p) {
  static_cast<void>(ScalarBytes(p.dtype));
  if (p.scale != BatchPixelScale::preserve &&
      p.scale != BatchPixelScale::to_unit)
    InvalidBatch("batch_transform.scale");
  if (p.scale == BatchPixelScale::to_unit &&
      source.range == PixelValueRange::unbounded)
    InvalidBatch("batch_transform.unbounded_to_unit");
  auto output = source;
  output.dtype = p.dtype;
  output.range = p.scale == BatchPixelScale::to_unit
                     ? PixelValueRange::unit_interval
                 : source.range == PixelValueRange::unsigned_full_scale
                     ? PixelValueRange::unbounded
                     : source.range;
  if (p.swap_red_blue) {
    switch (source.color) {
      case PixelColorModel::rgb:
        output.color = PixelColorModel::bgr;
        break;
      case PixelColorModel::bgr:
        output.color = PixelColorModel::rgb;
        break;
      case PixelColorModel::rgba:
        output.color = PixelColorModel::bgra;
        break;
      case PixelColorModel::bgra:
        output.color = PixelColorModel::rgba;
        break;
      default:
        InvalidBatch("batch_transform.channel_swap");
    }
  }
  if (p.mean.size() != p.std.size() ||
      (!p.mean.empty() && (p.mean.size() != source.channels() ||
                           source.alpha != PixelAlphaMode::none)))
    InvalidBatch("batch_transform.normalization_channels");
  const double maximum = p.dtype == torch::kFloat64
                             ? std::numeric_limits<double>::max()
                             : std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < p.mean.size(); ++i)
    if (!std::isfinite(p.mean[i]) || !std::isfinite(p.std[i]) ||
        std::abs(p.mean[i]) > maximum || p.std[i] > maximum || p.std[i] <= 0 ||
        (p.dtype != torch::kFloat64 && static_cast<float>(p.std[i]) == 0))
      InvalidBatch("batch_transform.normalization_parameters");
  if (!p.mean.empty()) output.range = PixelValueRange::unbounded;
  return output;
}
std::uint64_t Count(const Tensor& packed, const std::vector<Tensor>& ragged) {
  std::uint64_t count =
      packed.defined() ? static_cast<std::uint64_t>(packed.numel()) : 0;
  for (const auto& t : ragged) {
    if (static_cast<std::uint64_t>(t.numel()) >
        std::numeric_limits<std::uint64_t>::max() - count)
      BatchLimit("batch_transform.elements");
    count += static_cast<std::uint64_t>(t.numel());
  }
  return count;
}
Tensor Numeric(const Tensor& source, const PixelSpec& pixels,
               const BatchPixelTransform& p) {
  const auto compute =
      p.dtype == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
  auto result = source.to(source.options().dtype(compute).pinned_memory(false),
                          false, true);
  if (p.scale == BatchPixelScale::to_unit &&
      pixels.range == PixelValueRange::unsigned_full_scale)
    result.div_(pixels.dtype == torch::kUInt8 ? 255.0 : 65535.0);
  const auto axis = source.dim() == 4 ? 1 : 0;
  if (p.swap_red_blue) {
    auto swapped = torch::empty_like(result);
    for (int c = 0; c < pixels.channels(); ++c)
      swapped.select(axis, c).copy_(result.select(axis, c == 0   ? 2
                                                        : c == 2 ? 0
                                                                 : c));
    result = std::move(swapped);
  }
  // A channel operation covers every N/H/W element, without reading GPU
  // scalars or allocating a host-generated parameter tensor per sample.
  for (std::size_t c = 0; c < p.mean.size(); ++c)
    result.select(axis, static_cast<std::int64_t>(c))
        .sub_(p.mean[c])
        .div_(p.std[c]);
  return result.to(result.options().dtype(p.dtype), false, false);
}
}  // namespace

ImageBatch transform_image_batch(const ImageBatch& b,
                                 const ImageBatchTransformOptions& o,
                                 std::stop_token stop) {
  Stop(stop);
  try {
    detail::CheckImageBatch(b, o.limits, stop);
    const auto pixels = Policy(b.pixels, o.image);
    if (o.paired && !b.paired_pixels)
      InvalidBatch("batch_transform.missing_paired");
    const auto paired_pixels =
        o.paired ? std::optional(Policy(*b.paired_pixels, *o.paired))
                 : b.paired_pixels;
    const auto source_bytes = detail::BatchTensorBytes(b);
    std::uint64_t output_bytes = source_bytes, scratch_bytes = 0;
    const auto reserve = [&](std::uint64_t count, const PixelSpec& spec,
                             const BatchPixelTransform& p) {
      const auto old_bytes =
          count * c10::elementSize(spec.dtype);  // bounded by input validation
      const auto scalar = ScalarBytes(p.dtype);
      if (count > o.limits.max_output_bytes / scalar)
        BatchLimit("batch_transform.output");
      output_bytes -= old_bytes;
      if (count * scalar > o.limits.max_output_bytes - output_bytes)
        BatchLimit("batch_transform.output");
      output_bytes += count * scalar;
      // Reserve two full compute buffers (conversion and channel exchange),
      // conservatively across both fields and every ragged image.
      const auto compute = p.dtype == torch::kFloat64 ? 8U : 4U;
      if (count > (o.limits.max_working_bytes - scratch_bytes) / (2 * compute))
        BatchLimit("batch_transform.working");
      scratch_bytes += count * 2 * compute;
    };
    const auto image_elements = Count(b.images, b.image_list);
    reserve(image_elements, b.pixels, o.image);
    const auto paired_elements =
        (o.paired || (o.resource_budget && b.paired_pixels))
            ? Count(b.paired_images.value_or(Tensor()), b.paired_image_list)
            : 0;
    if (o.paired) reserve(paired_elements, *b.paired_pixels, *o.paired);
    std::vector<ImageExtent> canvases;
    for (std::size_t i = 0; i < b.items.size(); ++i) {
      const auto t = b.layout == ImageBatchLayout::ragged
                         ? b.image_list[i]
                         : b.images[static_cast<std::int64_t>(i)];
      canvases.push_back({static_cast<std::uint32_t>(t.size(2)),
                          static_cast<std::uint32_t>(t.size(1))});
    }
    const auto metadata =
        detail::CheckBatchMetadata(b, canvases, o.limits, false, stop);
    if (scratch_bytes > o.limits.max_working_bytes - metadata ||
        output_bytes > o.limits.max_working_bytes - metadata - scratch_bytes)
      BatchLimit("batch_transform.working");
    const auto device = b.layout == ImageBatchLayout::ragged
                            ? b.image_list.front().device()
                            : b.images.device();
    std::shared_ptr<ImageResourceLeaseSet> output_leases;
    if (o.resource_budget) {
      const auto image_pixels = image_elements / b.pixels.channels();
      const auto paired_pixels_count =
          b.paired_pixels ? paired_elements / b.paired_pixels->channels() : 0;
      if (paired_pixels_count >
          std::numeric_limits<std::uint64_t>::max() - image_pixels)
        BatchLimit("batch_transform.resource_pixels");
      const auto label_count = static_cast<std::uint64_t>(b.labels.numel());
      if (label_count >
          std::numeric_limits<std::uint64_t>::max() / sizeof(std::int64_t))
        BatchLimit("batch_transform.resource_index");
      auto index_bytes = label_count * sizeof(std::int64_t);
      if (b.label_offsets.size() >
          (std::numeric_limits<std::uint64_t>::max() - index_bytes) /
              sizeof(std::int64_t))
        BatchLimit("batch_transform.resource_index");
      index_bytes += b.label_offsets.size() * sizeof(std::int64_t);
      std::uint64_t pinned_bytes = 0;
      const auto charge_pinned = [&](const Tensor& tensor) {
        if (!tensor.defined() || !tensor.is_pinned()) return;
        const auto bytes =
            static_cast<std::uint64_t>(tensor.numel()) * tensor.element_size();
        if (bytes > std::numeric_limits<std::uint64_t>::max() - pinned_bytes)
          BatchLimit("batch_transform.resource_pinned");
        pinned_bytes += bytes;
      };
      if (device.is_cpu()) {
        charge_pinned(b.valid);
        charge_pinned(b.labels);
        if (b.masks) charge_pinned(*b.masks);
        for (const auto& tensor : b.valid_list) charge_pinned(tensor);
        for (const auto& tensor : b.mask_list) charge_pinned(tensor);
        if (!o.paired) {
          if (b.paired_images) charge_pinned(*b.paired_images);
          for (const auto& tensor : b.paired_image_list) charge_pinned(tensor);
        }
      }
      output_leases = detail::ReserveBatchTotals(
          o.resource_budget, output_bytes, image_pixels + paired_pixels_count,
          metadata, index_bytes, pinned_bytes,
          device.is_cuda() ? output_bytes : 0, stop);
    }
    const auto numeric = [&](const Tensor& source, const PixelSpec& spec,
                             const BatchPixelTransform& policy) {
      ImageResourceLease scratch;
      if (o.resource_budget) {
        const auto compute = policy.dtype == torch::kFloat64 ? 8U : 4U;
        const auto amount =
            static_cast<std::uint64_t>(source.numel()) * 2 * compute;
        scratch = o.resource_budget->reserve(
            device.is_cuda() ? ImageResourceKind::gpu_bytes
                             : ImageResourceKind::decoded_bytes,
            amount);
      }
      return Numeric(source, spec, policy);
    };
    auto result = b;
    result.pixels = pixels;
    result.paired_pixels = paired_pixels;
    result.memory = {};
    result.resource_lease_sets.clear();
    if (output_leases)
      result.resource_lease_sets.push_back(std::move(output_leases));
    if (b.layout == ImageBatchLayout::ragged) {
      result.image_list.clear();
      result.paired_image_list =
          o.paired ? std::vector<Tensor>() : b.paired_image_list;
      for (std::size_t i = 0; i < b.items.size(); ++i) {
        Stop(stop);
        result.image_list.push_back(
            numeric(b.image_list[i], b.pixels, o.image));
        if (o.paired)
          result.paired_image_list.push_back(
              numeric(b.paired_image_list[i], *b.paired_pixels, *o.paired));
      }
    } else {
      result.images = numeric(b.images, b.pixels, o.image);
      Stop(stop);
      if (o.paired)
        result.paired_images =
            numeric(*b.paired_images, *b.paired_pixels, *o.paired);
    }
    Stop(stop);
    detail::CheckImageBatch(result, o.limits, stop);
    return result;
  } catch (const std::bad_alloc&) {
    BatchLimit("batch_transform.allocate");
  } catch (const c10::OutOfMemoryError&) {
    BatchLimit("batch_transform.allocate");
  } catch (const c10::Error&) {
    throw DeviceError("Image batch numeric operation failed");
  }
}

std::string image_batch_identity(
    const ImageCollateOptions& c,
    const std::optional<ImageBatchTransformOptions>& n) {
  detail::CheckBatchLimits(c.limits);
  if ((c.layout != ImageBatchLayout::stack &&
       c.layout != ImageBatchLayout::pad &&
       c.layout != ImageBatchLayout::ragged) ||
      !c.size_divisor ||
      (c.layout != ImageBatchLayout::pad &&
       (c.padded_extent || c.size_divisor != 1)) ||
      (c.padded_extent &&
       (!c.padded_extent->width || !c.padded_extent->height)))
    InvalidBatch("batch_identity.collate");
  Sha256 hash;
  constexpr std::string_view domain = "liuding.image-batch.v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  const auto integer = [&](std::uint64_t v) {
    std::array<std::byte, 8> bytes{};
    for (std::size_t i = 0; i < 8; ++i)
      bytes[i] = static_cast<std::byte>((v >> (8 * i)) & 255U);
    hash.update(bytes);
  };
  const auto real = [&](double v) {
    if (!std::isfinite(v)) InvalidBatch("batch_identity.finite");
    integer(std::bit_cast<std::uint64_t>(v == 0 ? 0.0 : v));
  };
  const auto limits = [&](const ImageBatchLimits& l) {
    detail::CheckBatchLimits(l);
    integer(l.max_samples);
    integer(l.max_dimension);
    integer(l.max_pixels);
    integer(l.max_output_bytes);
    integer(l.max_working_bytes);
    integer(l.max_metadata_nodes);
    integer(l.max_metadata_bytes);
    integer(l.max_string_bytes);
    integer(l.max_geometry_checks);
  };
  integer(static_cast<std::uint8_t>(c.layout));
  integer(c.padded_extent.has_value());
  if (c.padded_extent) {
    integer(c.padded_extent->width);
    integer(c.padded_extent->height);
  }
  integer(c.size_divisor);
  real(c.image_padding);
  real(c.paired_padding);
  integer(c.mask_padding_id.has_value());
  if (c.mask_padding_id)
    integer(static_cast<std::uint64_t>(*c.mask_padding_id));
  limits(c.limits);
  integer(n.has_value());
  if (n) {
    const auto pixel = [&](const BatchPixelTransform& p) {
      static_cast<void>(ScalarBytes(p.dtype));
      if ((p.scale != BatchPixelScale::preserve &&
           p.scale != BatchPixelScale::to_unit) ||
          p.mean.size() != p.std.size() || p.mean.size() > 4)
        InvalidBatch("batch_identity.numeric");
      integer(static_cast<std::uint64_t>(p.dtype));
      integer(static_cast<std::uint8_t>(p.scale));
      integer(p.swap_red_blue);
      integer(p.mean.size());
      for (const auto v : p.mean) real(v);
      for (const auto v : p.std) {
        if (v <= 0) InvalidBatch("batch_identity.std");
        real(v);
      }
    };
    pixel(n->image);
    integer(n->paired.has_value());
    if (n->paired) pixel(*n->paired);
    limits(n->limits);
  }
  return sha256_digest_hex(hash.finish());
}
}  // namespace jyppx::liuding::data::image
