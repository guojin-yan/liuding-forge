// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"
#include "image_batch_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <unordered_set>

namespace jyppx::liuding::data::image {
namespace detail {
[[noreturn]] void InvalidBatch(const char* operation) {
  throw InvalidArgument(
      "Invalid image batch",
      ErrorDetails{.component = "data.image.batch", .operation = operation});
}
[[noreturn]] void BatchLimit(const char* operation) {
  throw ResourceExhausted(
      "Image batch budget exceeded",
      ErrorDetails{.component = "data.image.batch", .operation = operation});
}
void CheckBatchLimits(const ImageBatchLimits& l) {
  if (!l.max_samples || !l.max_dimension || !l.max_pixels ||
      !l.max_output_bytes || !l.max_working_bytes || !l.max_metadata_nodes ||
      !l.max_metadata_bytes || !l.max_string_bytes || !l.max_geometry_checks)
    InvalidBatch("batch.limits");
}
bool SamePixels(const PixelSpec& a, const PixelSpec& b) {
  return a.dtype == b.dtype && a.color == b.color && a.alpha == b.alpha &&
         a.range == b.range;
}
void CheckImageFill(const PixelSpec& p, double value) {
  if (!std::isfinite(value)) InvalidBatch("batch.fill_finite");
  if (p.range == PixelValueRange::unit_interval && (value < 0 || value > 1))
    InvalidBatch("batch.fill_range");
  if (p.dtype == torch::kUInt8 || p.dtype == torch::kUInt16) {
    if (value < 0 || value > (p.dtype == torch::kUInt8 ? 255 : 65535) ||
        std::trunc(value) != value)
      InvalidBatch("batch.fill_integer");
  } else {
    const double maximum = p.dtype == torch::kFloat16    ? 65504.0
                           : p.dtype == torch::kBFloat16 ? 3.3895313892515355e38
                           : p.dtype == torch::kFloat64
                               ? std::numeric_limits<double>::max()
                               : std::numeric_limits<float>::max();
    if (std::abs(value) > maximum) InvalidBatch("batch.fill_dtype");
  }
}
namespace {
bool Equal(ImageExtent a, ImageExtent b) {
  return a.width == b.width && a.height == b.height;
}
void Extent(ImageExtent e, const ImageBatchLimits& l) {
  if (!e.width || !e.height) InvalidBatch("batch.extent");
  if (e.width > l.max_dimension || e.height > l.max_dimension ||
      static_cast<std::uint64_t>(e.width) * e.height > l.max_pixels)
    BatchLimit("batch.extent");
}

// Logical scalar/container payload, not allocator overhead or measured RSS.
// The conservative container charges also bound allocations before copying
// caller-owned annotations or constructing the recursive IValue tree.
struct Metadata {
  const ImageBatchLimits& l;
  std::size_t nodes = 0, bytes = 0;
  void Add(std::size_t n, std::size_t payload = 0) {
    if (n > l.max_metadata_nodes - nodes ||
        n > (l.max_metadata_bytes - bytes) / 8)
      BatchLimit("batch.metadata");
    nodes += n;
    bytes += n * 8;
    if (payload > l.max_metadata_bytes - bytes) BatchLimit("batch.metadata");
    bytes += payload;
  }
  void String(const std::string& s) {
    if (s.size() > l.max_string_bytes) BatchLimit("batch.string");
    Add(1, s.size());
    if (!Utf8(s)) InvalidBatch("batch.utf8");
  }
  void Categories(
      const std::shared_ptr<const std::vector<VisionCategory>>& cs) {
    Add(1);
    if (!cs) return;
    if (cs->size() > l.max_metadata_nodes - nodes)
      BatchLimit("batch.categories");
    std::unordered_set<std::int64_t> ids;
    for (const auto& c : *cs) {
      Add(6);
      String(c.name);
      if (c.name.empty() || !ids.insert(c.source_id).second)
        InvalidBatch("batch.category");
      for (const auto& name : c.keypoint_names) String(name);
      for (const auto& link : c.skeleton) {
        Add(3);
        if (link[0] >= c.keypoint_names.size() ||
            link[1] >= c.keypoint_names.size())
          InvalidBatch("batch.skeleton");
      }
    }
  }
};
void PointBounds(ImagePoint p, ImageExtent e) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
      p.x > e.width || p.y > e.height)
    InvalidBatch("batch.point");
}
void BoxBounds(ImageBox b, ImageExtent e) {
  PointBounds({b.x_min, b.y_min}, e);
  PointBounds({b.x_max, b.y_max}, e);
  if (b.x_max <= b.x_min || b.y_max <= b.y_min) InvalidBatch("batch.box");
}
}  // namespace

std::size_t CheckBatchMetadata(const ImageBatch& b,
                               std::span<const ImageExtent> canvases,
                               const ImageBatchLimits& l, bool source_panoptic,
                               std::stop_token stop) {
  CheckBatchLimits(l);
  if (b.items.empty() || canvases.size() != b.items.size())
    InvalidBatch("batch.items");
  if (b.items.size() > l.max_samples) BatchLimit("batch.samples");
  Metadata m{l};
  m.Add(64);
  AnnotationReadOptions options;
  options.limits.max_geometry_checks = l.max_geometry_checks;
  ParseState geometry{options, stop};
  for (std::size_t i = 0; i < b.items.size(); ++i) {
    CheckStop(stop);
    const auto& item = b.items[i];
    const auto canvas = canvases[i];
    Extent(canvas, l);
    Extent(item.valid_extent, l);
    if (item.valid_extent.width > canvas.width ||
        item.valid_extent.height > canvas.height)
      InvalidBatch("batch.valid_extent");
    if (item.key.id.empty() || item.key.id.size() > 4096)
      InvalidBatch("batch.sample_id");
    m.Add(32);
    m.String(item.key.id);
    m.String(item.image_path);
    if (item.target_path) m.String(*item.target_path);
    m.Categories(item.categories);
    m.Categories(item.label_categories);
    if (item.geometry) {
      m.Add(24);
      if (!Equal(item.geometry->output_extent(), canvas))
        InvalidBatch("batch.trace_canvas");
      static_cast<void>(GeometryTrace::from_state(item.geometry->state()));
    }
    if (item.mask) {
      const auto& mask = *item.mask;
      m.Add(8);
      if (mask.encoding != MaskEncoding::class_index &&
          mask.encoding != MaskEncoding::palette_index &&
          mask.encoding != MaskEncoding::rgb_mapping &&
          mask.encoding != MaskEncoding::panoptic_rgb)
        InvalidBatch("batch.mask_encoding");
      if (mask.palette.size() > 256 ||
          (mask.encoding != MaskEncoding::palette_index &&
           !mask.palette.empty()))
        InvalidBatch("batch.palette");
      for (const auto& entry : mask.palette) {
        static_cast<void>(entry);
        m.Add(5);
      }
    }
    if (!item.annotation) continue;
    const auto& r = *item.annotation;
    m.Add(8);
    m.String(r.image_path);
    if (!item.categories || !Equal({r.width, r.height}, canvas))
      InvalidBatch("batch.annotation_canvas");
    if (r.panoptic_mask_path) m.String(*r.panoptic_mask_path);
    const bool panoptic = r.panoptic_mask_path.has_value();
    if ((!panoptic && r.panoptic_mask_materialized) ||
        (panoptic && !r.panoptic_mask_materialized && !source_panoptic) ||
        (panoptic &&
         (!item.mask || item.mask->encoding != MaskEncoding::panoptic_rgb ||
          !item.mask->void_id)))
      InvalidBatch("batch.panoptic_binding");
    std::unordered_set<std::int64_t> segment_ids;
    for (const auto& o : r.objects) {
      CheckStop(stop);
      m.Add(20);
      if (o.category >= item.categories->size())
        InvalidBatch("batch.category_index");
      if (o.area && (!std::isfinite(*o.area) || *o.area < 0))
        InvalidBatch("batch.area");
      if (o.box) {
        m.Add(5);
        BoxBounds(*o.box, canvas);
      }
      const int regions = (!o.polygons.empty() ? 1 : 0) +
                          (o.oriented_box ? 1 : 0) + (o.rle ? 1 : 0);
      if (regions > 1) InvalidBatch("batch.mixed_regions");
      const auto ring = [&](std::span<const ImagePoint> points) {
        m.Add(1);
        for (const auto p : points) {
          m.Add(3);
          PointBounds(p, canvas);
        }
        try {
          ValidatePolygon(points, geometry);
        } catch (const SerializationError&) {
          InvalidBatch("batch.polygon");
        }
      };
      for (const auto& points : o.polygons) ring(points);
      if (o.oriented_box) {
        ring(*o.oriented_box);
        try {
          ValidateConvexQuad(*o.oriented_box);
        } catch (const SerializationError&) {
          InvalidBatch("batch.obb");
        }
      }
      if (o.rle) {
        m.Add(4);
        const auto& rle = *o.rle;
        if (!Equal({rle.width, rle.height}, canvas) || rle.counts.empty())
          InvalidBatch("batch.rle_canvas");
        std::uint64_t total = 0, foreground = 0;
        const auto pixels =
            static_cast<std::uint64_t>(canvas.width) * canvas.height;
        bool value = false;
        for (const auto run : rle.counts) {
          CheckStop(stop);
          m.Add(1);
          if (run > pixels - total) InvalidBatch("batch.rle_runs");
          total += run;
          if (value) foreground += run;
          value = !value;
        }
        if (total != pixels ||
            (o.mask_pixel_area && *o.mask_pixel_area != foreground))
          InvalidBatch("batch.rle_pixels");
      }
      if (o.mask_pixel_area &&
          *o.mask_pixel_area >
              static_cast<std::uint64_t>(canvas.width) * canvas.height)
        InvalidBatch("batch.mask_area");
      const auto& category = (*item.categories)[o.category];
      if (!o.keypoints.empty() && !category.keypoint_names.empty() &&
          o.keypoints.size() != category.keypoint_names.size())
        InvalidBatch("batch.pose_schema");
      for (const auto& k : o.keypoints) {
        m.Add(4);
        if (k.visibility > 2 || !std::isfinite(k.point.x) ||
            !std::isfinite(k.point.y))
          InvalidBatch("batch.keypoint");
        if (k.visibility) PointBounds(k.point, canvas);
      }
      if (panoptic &&
          (regions || !o.keypoints.empty() ||
           (r.panoptic_mask_materialized &&
            (!o.box || !o.mask_pixel_area || !*o.mask_pixel_area)) ||
           o.source_id <= 0 || o.source_id > 0xffffff ||
           o.source_id == item.mask->void_id ||
           o.source_id == item.mask->ignore_id ||
           !segment_ids.insert(o.source_id).second))
        InvalidBatch("batch.panoptic_segment");
    }
  }
  for (const auto offset : b.label_offsets) {
    static_cast<void>(offset);
    m.Add(1);
  }
  if (m.bytes > l.max_working_bytes) BatchLimit("batch.metadata_working");
  return m.bytes;
}

std::uint64_t BatchTensorBytes(const ImageBatch& b) {
  std::uint64_t result = 0;
  const auto add = [&](const Tensor& t) {
    if (!t.defined()) return;
    const auto count = static_cast<std::uint64_t>(t.numel());
    if (count >
        (std::numeric_limits<std::uint64_t>::max() - result) / t.element_size())
      BatchLimit("batch.tensor_bytes");
    result += count * t.element_size();
  };
  add(b.images);
  add(b.valid);
  add(b.labels);
  if (b.paired_images) add(*b.paired_images);
  if (b.masks) add(*b.masks);
  for (const auto* list :
       {&b.image_list, &b.paired_image_list, &b.mask_list, &b.valid_list})
    for (const auto& t : *list) add(t);
  return result;
}

std::shared_ptr<ImageResourceLeaseSet> ReserveBatchTotals(
    const std::shared_ptr<ImageResourceBudget>& budget,
    std::uint64_t tensor_bytes, std::uint64_t pixel_count,
    std::uint64_t metadata_bytes, std::uint64_t index_bytes,
    std::uint64_t pinned_bytes, std::uint64_t gpu_bytes, std::stop_token stop) {
  if (!budget) return {};
  CheckStop(stop);
  auto leases = std::make_shared<ImageResourceLeaseSet>();
  const auto reserve = [&](ImageResourceKind kind, std::uint64_t amount) {
    if (amount) leases->emplace_back(budget->reserve(kind, amount));
  };
  reserve(ImageResourceKind::staged_batches, 1);
  reserve(ImageResourceKind::decoded_bytes, tensor_bytes);
  reserve(ImageResourceKind::pixels, pixel_count);
  reserve(ImageResourceKind::annotation_bytes, metadata_bytes);
  reserve(ImageResourceKind::index_bytes, index_bytes);
  reserve(ImageResourceKind::pinned_bytes, pinned_bytes);
  reserve(ImageResourceKind::gpu_bytes, gpu_bytes);
  return leases;
}

std::shared_ptr<ImageResourceLeaseSet> ReserveBatchResources(
    const ImageBatch& b, const std::shared_ptr<ImageResourceBudget>& budget,
    const ImageBatchLimits& l, std::stop_token stop) {
  if (!budget) return {};
  CheckStop(stop);

  const auto add = [](std::uint64_t current, std::uint64_t value,
                      const char* operation) {
    if (value > std::numeric_limits<std::uint64_t>::max() - current)
      BatchLimit(operation);
    return current + value;
  };
  const auto tensor_bytes = BatchTensorBytes(b);
  std::uint64_t pixel_count = 0;
  std::uint64_t pinned_bytes = 0;
  std::uint64_t gpu_bytes = 0;
  const auto inspect = [&](const Tensor& tensor) {
    if (!tensor.defined()) return;
    const auto bytes =
        static_cast<std::uint64_t>(tensor.numel()) * tensor.element_size();
    if (tensor.device().is_cuda()) {
      gpu_bytes = add(gpu_bytes, bytes, "batch.resource.gpu_bytes");
    } else if (tensor.is_pinned()) {
      pinned_bytes = add(pinned_bytes, bytes, "batch.resource.pinned_bytes");
    }
  };
  const auto inspect_image = [&](const Tensor& tensor) {
    if (!tensor.defined()) return;
    if (tensor.dim() < 2) InvalidBatch("batch.resource.image_rank");
    const auto height =
        static_cast<std::uint64_t>(tensor.size(tensor.dim() - 2));
    const auto width =
        static_cast<std::uint64_t>(tensor.size(tensor.dim() - 1));
    if (height && width > std::numeric_limits<std::uint64_t>::max() / height)
      BatchLimit("batch.resource.pixels");
    pixel_count = add(pixel_count, height * width, "batch.resource.pixels");
    inspect(tensor);
  };
  const auto inspect_packed_image = [&](const Tensor& tensor) {
    if (!tensor.defined() || tensor.dim() != 4)
      InvalidBatch("batch.resource.image_rank");
    const auto count = static_cast<std::uint64_t>(tensor.size(0));
    const auto height = static_cast<std::uint64_t>(tensor.size(2));
    const auto width = static_cast<std::uint64_t>(tensor.size(3));
    if (height && width > std::numeric_limits<std::uint64_t>::max() / height)
      BatchLimit("batch.resource.pixels");
    const auto area = height * width;
    if (count && area > std::numeric_limits<std::uint64_t>::max() / count)
      BatchLimit("batch.resource.pixels");
    pixel_count = add(pixel_count, count * area, "batch.resource.pixels");
    inspect(tensor);
  };
  if (b.layout == ImageBatchLayout::ragged) {
    for (const auto& tensor : b.image_list) inspect_image(tensor);
    for (const auto& tensor : b.paired_image_list) inspect_image(tensor);
    for (const auto& tensor : b.mask_list) inspect(tensor);
    for (const auto& tensor : b.valid_list) inspect(tensor);
  } else {
    if (b.images.defined()) inspect_packed_image(b.images);
    if (b.paired_images) inspect_packed_image(*b.paired_images);
    if (b.masks) inspect(*b.masks);
    inspect(b.valid);
  }
  inspect(b.labels);

  std::vector<ImageExtent> canvases;
  canvases.reserve(b.items.size());
  for (std::size_t i = 0; i < b.items.size(); ++i) {
    const auto tensor = b.layout == ImageBatchLayout::ragged
                            ? b.image_list[i]
                            : b.images[static_cast<std::int64_t>(i)];
    canvases.push_back(
        {static_cast<std::uint32_t>(tensor.size(tensor.dim() - 1)),
         static_cast<std::uint32_t>(tensor.size(tensor.dim() - 2))});
  }
  const auto metadata_bytes = CheckBatchMetadata(b, canvases, l, false, stop);
  const auto label_count = static_cast<std::uint64_t>(b.labels.numel());
  if (label_count >
      std::numeric_limits<std::uint64_t>::max() / sizeof(std::int64_t))
    BatchLimit("batch.resource.index_bytes");
  std::uint64_t index_bytes = label_count * sizeof(std::int64_t);
  if (b.label_offsets.size() >
      (std::numeric_limits<std::uint64_t>::max() - index_bytes) /
          sizeof(std::int64_t))
    BatchLimit("batch.resource.index_bytes");
  index_bytes +=
      static_cast<std::uint64_t>(b.label_offsets.size()) * sizeof(std::int64_t);

  return ReserveBatchTotals(budget, tensor_bytes, pixel_count, metadata_bytes,
                            index_bytes, pinned_bytes, gpu_bytes, stop);
}

void CheckImageBatch(const ImageBatch& b, const ImageBatchLimits& l,
                     std::stop_token stop) {
  CheckBatchLimits(l);
  if (b.layout != ImageBatchLayout::stack &&
      b.layout != ImageBatchLayout::pad && b.layout != ImageBatchLayout::ragged)
    InvalidBatch("batch.layout");
  const auto n = b.items.size();
  if (!n) InvalidBatch("batch.empty");
  if (n > l.max_samples) BatchLimit("batch.samples");
  if (b.pixels.layout != ImageLayout::chw ||
      (b.paired_pixels && b.paired_pixels->layout != ImageLayout::chw))
    InvalidBatch("batch.layout_pixels");
  const bool ragged = b.layout == ImageBatchLayout::ragged;
  const bool paired = b.paired_pixels.has_value(),
             masks = b.items[0].mask.has_value();
  if (ragged
          ? (b.images.defined() || b.valid.defined() || b.paired_images ||
             b.masks || b.image_list.size() != n || b.valid_list.size() != n ||
             b.paired_image_list.size() != (paired ? n : 0) ||
             b.mask_list.size() != (masks ? n : 0))
          : (!b.image_list.empty() || !b.paired_image_list.empty() ||
             !b.mask_list.empty() || !b.valid_list.empty() ||
             b.paired_images.has_value() != paired ||
             b.masks.has_value() != masks))
    InvalidBatch("batch.storage");
  const auto& first = ragged ? b.image_list[0] : b.images;
  if (!first.defined()) InvalidBatch("batch.images");
  const auto device = first.device();
  const auto tensor = [&](const Tensor& t, Dtype dtype, int rank) {
    if (!t.defined() || t.layout() != torch::kStrided || t.device() != device ||
        t.scalar_type() != dtype || t.dim() != rank || !t.is_contiguous() ||
        t.requires_grad())
      InvalidBatch("batch.tensor");
  };
  if (!device.is_cpu() && !device.is_cuda()) InvalidBatch("batch.device");
  if (!ragged) {
    tensor(b.images, b.pixels.dtype, 4);
    tensor(b.valid, torch::kBool, 3);
    if (b.images.size(0) != static_cast<std::int64_t>(n) ||
        b.valid.size(0) != static_cast<std::int64_t>(n))
      InvalidBatch("batch.sample_axis");
    if (paired) tensor(*b.paired_images, b.paired_pixels->dtype, 4);
    if (masks) tensor(*b.masks, torch::kInt64, 3);
    if ((paired && b.paired_images->size(0) != static_cast<std::int64_t>(n)) ||
        (masks && b.masks->size(0) != static_cast<std::int64_t>(n)))
      InvalidBatch("batch.target_sample_axis");
  }
  std::vector<ImageExtent> canvases;
  std::uint64_t pixels = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto photo =
        ragged ? b.image_list[i] : b.images[static_cast<std::int64_t>(i)];
    tensor(photo, b.pixels.dtype, 3);
    validate_pixels(photo, b.pixels);
    if (photo.size(1) > l.max_dimension || photo.size(2) > l.max_dimension)
      BatchLimit("batch.dimension");
    const ImageExtent canvas{static_cast<std::uint32_t>(photo.size(2)),
                             static_cast<std::uint32_t>(photo.size(1))};
    Extent(canvas, l);
    canvases.push_back(canvas);
    const auto count = static_cast<std::uint64_t>(canvas.width) * canvas.height;
    if (count > l.max_pixels - pixels) BatchLimit("batch.pixels");
    pixels += count;
    const auto valid =
        ragged ? b.valid_list[i] : b.valid[static_cast<std::int64_t>(i)];
    tensor(valid, torch::kBool, 2);
    if (valid.size(0) != canvas.height || valid.size(1) != canvas.width)
      InvalidBatch("batch.valid_shape");
    if (b.items[i].mask.has_value() != masks)
      InvalidBatch("batch.mask_presence");
    if (b.layout != ImageBatchLayout::pad &&
        !Equal(canvas, b.items[i].valid_extent))
      InvalidBatch("batch.valid_extent");
    if (paired) {
      const auto p = ragged ? b.paired_image_list[i]
                            : (*b.paired_images)[static_cast<std::int64_t>(i)];
      tensor(p, b.paired_pixels->dtype, 3);
      validate_pixels(p, *b.paired_pixels);
      if (p.size(1) != canvas.height || p.size(2) != canvas.width)
        InvalidBatch("batch.paired_shape");
    }
    if (masks) {
      const auto mask =
          ragged ? b.mask_list[i] : (*b.masks)[static_cast<std::int64_t>(i)];
      tensor(mask, torch::kInt64, 2);
      if (mask.size(0) != canvas.height || mask.size(1) != canvas.width)
        InvalidBatch("batch.mask_shape");
    }
  }
  if (!ragged &&
      (b.valid.size(0) != static_cast<std::int64_t>(n) ||
       (paired && b.paired_images->size(0) != static_cast<std::int64_t>(n)) ||
       (masks && b.masks->size(0) != static_cast<std::int64_t>(n))))
    InvalidBatch("batch.target_sample_axis");
  tensor(b.labels, torch::kInt64, 1);
  if (b.label_offsets.size() != n + 1 || b.label_offsets[0] != 0 ||
      b.label_offsets.back() != b.labels.numel())
    InvalidBatch("batch.label_offsets");
  for (std::size_t i = 1; i < b.label_offsets.size(); ++i)
    if (b.label_offsets[i] < b.label_offsets[i - 1])
      InvalidBatch("batch.label_offsets");
  const auto bytes = BatchTensorBytes(b);
  if (bytes > l.max_output_bytes || bytes > l.max_working_bytes)
    BatchLimit("batch.tensor_bytes");
  const auto metadata = CheckBatchMetadata(b, canvases, l, false, stop);
  if (bytes > l.max_working_bytes - metadata) BatchLimit("batch.working_bytes");
}
}  // namespace detail

namespace {
using detail::BatchLimit;
using detail::InvalidBatch;
void Stop(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("Image batch cancelled");
}
bool Equal(ImageExtent a, ImageExtent b) {
  return a.width == b.width && a.height == b.height;
}
Tensor Chw(const Tensor& tensor, const PixelSpec& p) {
  return p.layout == ImageLayout::chw ? tensor : tensor.permute({2, 0, 1});
}
void Photo(const Tensor& t, const PixelSpec& p, ImageExtent e) {
  validate_pixels(t, p);
  if (!t.device().is_cpu() || t.requires_grad() ||
      t.size(p.layout == ImageLayout::chw ? 1 : 0) != e.height ||
      t.size(p.layout == ImageLayout::chw ? 2 : 1) != e.width)
    InvalidBatch("batch.source_photo");
}
ImageBatchMaskInfo MaskInfo(const DecodedMask& m) {
  return {m.encoding, m.source_bit_depth, m.source_interlaced,
          m.palette,  m.ignore_id,        m.void_id};
}

// Column-major RLE padding never materializes a raster or converts an ID to
// floating point. Consecutive equal values are coalesced, including zero runs.
ImageMaskRle PadRuns(const ImageMaskRle& input, ImageExtent output,
                     const ImageBatchLimits& l, std::size_t& output_runs,
                     std::stop_token stop) {
  const auto charge = [&] {
    if (output_runs >= l.max_metadata_nodes ||
        output_runs >= l.max_metadata_bytes / 8)
      BatchLimit("batch.padded_runs");
    ++output_runs;
  };
  charge();
  ImageMaskRle result{output.width, output.height, {0}};
  bool last = false;
  const auto emit = [&](bool value, std::uint64_t count) {
    if (!count) return;
    if (last == value)
      result.counts.back() += count;
    else {
      charge();
      result.counts.push_back(count);
      last = value;
    }
  };
  std::size_t run = 0;
  std::uint64_t remaining = input.counts[0];
  bool value = false;
  for (std::uint32_t x = 0; x < input.width; ++x) {
    Stop(stop);
    std::uint64_t needed = input.height;
    while (needed) {
      if (!remaining) {
        value = !value;
        remaining = input.counts[++run];
        continue;
      }
      const auto take = std::min(remaining, needed);
      emit(value, take);
      remaining -= take;
      needed -= take;
    }
    emit(false, output.height - input.height);
  }
  emit(false,
       static_cast<std::uint64_t>(output.width - input.width) * output.height);
  return result;
}
}  // namespace

ImageBatch collate_images(std::span<const VisionSample> samples,
                          std::span<const ImageBatchKey> keys,
                          const ImageCollateOptions& o, std::stop_token stop) {
  detail::CheckBatchLimits(o.limits);
  Stop(stop);
  if (samples.empty() || samples.size() != keys.size())
    InvalidBatch("batch.source_count");
  if (samples.size() > o.limits.max_samples) BatchLimit("batch.samples");
  if (o.layout != ImageBatchLayout::stack &&
      o.layout != ImageBatchLayout::pad && o.layout != ImageBatchLayout::ragged)
    InvalidBatch("batch.layout");
  if (!o.size_divisor || (o.layout != ImageBatchLayout::pad &&
                          (o.padded_extent || o.size_divisor != 1)))
    InvalidBatch("batch.padding_policy");
  try {
    ImageBatch b;
    b.layout = o.layout;
    b.pixels = samples[0].pixels;
    b.pixels.layout = ImageLayout::chw;
    b.paired_pixels = samples[0].paired_target_pixels;
    if (b.paired_pixels) b.paired_pixels->layout = ImageLayout::chw;
    const bool paired = b.paired_pixels.has_value(),
               masks = samples[0].mask_target.has_value();
    if (paired && masks) InvalidBatch("batch.target_ambiguity");
    detail::CheckImageFill(b.pixels, o.image_padding);
    if (paired) detail::CheckImageFill(*b.paired_pixels, o.paired_padding);
    ImageExtent largest{0, 0};
    std::vector<ImageExtent> canvases;
    std::uint64_t labels = 0;
    b.label_offsets.push_back(0);
    for (std::size_t i = 0; i < samples.size(); ++i) {
      Stop(stop);
      const auto& s = samples[i];
      const ImageExtent e{s.width, s.height};
      if (!e.width || !e.height) InvalidBatch("batch.source_extent");
      if (e.width > o.limits.max_dimension || e.height > o.limits.max_dimension)
        BatchLimit("batch.dimension");
      Photo(s.image, s.pixels, e);
      if (!detail::SamePixels(s.pixels, b.pixels) ||
          s.paired_target.has_value() != paired ||
          s.paired_target_pixels.has_value() != paired ||
          s.mask_target.has_value() != masks)
        InvalidBatch("batch.source_semantics");
      if (paired) {
        Photo(*s.paired_target, *s.paired_target_pixels, e);
        if (!detail::SamePixels(*s.paired_target_pixels, *b.paired_pixels))
          InvalidBatch("batch.paired_semantics");
      }
      if (masks) {
        const auto& m = *s.mask_target;
        if (m.palette.size() > 256) InvalidBatch("batch.palette");
        if (!m.ids.defined() || !m.ids.device().is_cpu() ||
            m.ids.scalar_type() != torch::kInt64 ||
            m.ids.layout() != torch::kStrided || m.ids.dim() != 2 ||
            m.ids.requires_grad() || m.ids.size(0) != e.height ||
            m.ids.size(1) != e.width || !Equal({m.width, m.height}, e))
          InvalidBatch("batch.source_mask");
      }
      if (s.labels.size() >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
              labels)
        BatchLimit("batch.labels");
      for (const auto label : s.labels)
        if (!s.label_categories || label >= s.label_categories->size())
          InvalidBatch("batch.label_index");
      labels += s.labels.size();
      b.label_offsets.push_back(static_cast<std::int64_t>(labels));
      if (keys[i].id.empty() || keys[i].id.size() > 4096)
        InvalidBatch("batch.sample_id");
      for (const auto* text : {&keys[i].id, &s.image_path,
                               s.target_path ? &*s.target_path : &s.image_path})
        if (text->size() > o.limits.max_string_bytes)
          BatchLimit("batch.string");
      ImageBatchItem item{keys[i],
                          e,
                          s.image_path,
                          s.target_path,
                          s.label_categories,
                          s.categories,
                          s.annotation,
                          s.geometry,
                          {}};
      if (!item.geometry) item.geometry.emplace(e);
      if (masks) item.mask = MaskInfo(*s.mask_target);
      b.items.push_back(std::move(item));
      canvases.push_back(e);
      largest.width = std::max(largest.width, e.width);
      largest.height = std::max(largest.height, e.height);
      if (o.layout == ImageBatchLayout::stack && !Equal(e, canvases[0]))
        InvalidBatch("batch.stack_shape");
    }
    // Bound source metadata before any annotation copy or pixel output. Raw
    // panoptic tables are permitted only for this CPU producer preflight.
    const auto source_metadata =
        detail::CheckBatchMetadata(b, canvases, o.limits, true, stop);
    if (o.layout == ImageBatchLayout::pad) {
      auto canvas = o.padded_extent.value_or(largest);
      if (canvas.width < largest.width || canvas.height < largest.height)
        InvalidBatch("batch.padding_extent");
      const auto round = [&](std::uint32_t v) {
        const auto n = (static_cast<std::uint64_t>(v) + o.size_divisor - 1) /
                       o.size_divisor * o.size_divisor;
        if (n > o.limits.max_dimension) BatchLimit("batch.padded_dimension");
        return static_cast<std::uint32_t>(n);
      };
      canvas = {round(canvas.width), round(canvas.height)};
      for (auto& e : canvases) e = canvas;
    }
    std::uint64_t pixels = 0, bytes = 0;
    if (labels > o.limits.max_output_bytes / 8) BatchLimit("batch.labels");
    bytes = labels * 8;
    const auto per_pixel =
        samples[0].image.element_size() * b.pixels.channels() + 1 +
        (paired ? samples[0].paired_target->element_size() *
                      b.paired_pixels->channels()
                : 0) +
        (masks ? 8 : 0);
    for (const auto e : canvases) {
      const auto count = static_cast<std::uint64_t>(e.width) * e.height;
      if (count > o.limits.max_pixels - pixels ||
          count > (o.limits.max_output_bytes - bytes) / per_pixel)
        BatchLimit("batch.output_bytes");
      pixels += count;
      bytes += count * per_pixel;
    }
    if (bytes > o.limits.max_working_bytes - source_metadata)
      BatchLimit("batch.working_bytes");
    // Pixel and aggregate metadata limits precede even a panoptic binding scan.
    for (std::size_t i = 0; i < b.items.size(); ++i) {
      auto& item = b.items[i];
      if (item.annotation && item.annotation->panoptic_mask_path) {
        if (!masks || !item.categories) InvalidBatch("batch.panoptic_source");
        PanopticBindOptions bind;
        bind.limits.max_dimension = o.limits.max_dimension;
        bind.limits.max_pixels = o.limits.max_pixels;
        bind.limits.max_objects = o.limits.max_metadata_nodes;
        bind.limits.max_categories = o.limits.max_metadata_nodes;
        bind.max_working_bytes =
            static_cast<std::size_t>(std::min<std::uint64_t>(
                (o.limits.max_working_bytes - source_metadata) / 2,
                std::numeric_limits<std::size_t>::max()));
        if (!bind.max_working_bytes) BatchLimit("batch.panoptic_working");
        item.annotation = std::make_shared<VisionAnnotationRecord>(
            bind_panoptic_mask(*item.annotation, *samples[i].mask_target,
                               item.categories->size(), bind, stop));
      }
    }
    std::size_t output_runs = 0;
    for (std::size_t i = 0; i < canvases.size(); ++i) {
      const auto e = canvases[i];
      auto& item = b.items[i];
      if (Equal(item.valid_extent, e)) continue;
      if (masks && (!o.mask_padding_id ||
                    (item.annotation && item.annotation->panoptic_mask_path &&
                     o.mask_padding_id != item.mask->void_id)))
        InvalidBatch("batch.mask_padding");
      item.geometry =
          item.geometry->padded(0, 0, e.width - item.valid_extent.width,
                                e.height - item.valid_extent.height);
      if (item.annotation) {
        auto record =
            std::make_shared<VisionAnnotationRecord>(*item.annotation);
        record->width = e.width;
        record->height = e.height;
        for (auto& object : record->objects)
          if (object.rle)
            object.rle = PadRuns(*object.rle, e, o.limits, output_runs, stop);
        item.annotation = std::move(record);
      }
    }
    if (bytes > o.limits.max_working_bytes) BatchLimit("batch.working_bytes");
    const auto metadata_bytes =
        detail::CheckBatchMetadata(b, canvases, o.limits, false, stop);
    if (bytes > o.limits.max_working_bytes - metadata_bytes)
      BatchLimit("batch.working_bytes");
    if (o.resource_budget) {
      if (labels > std::numeric_limits<std::uint64_t>::max() / 8 ||
          b.label_offsets.size() >
              (std::numeric_limits<std::uint64_t>::max() - labels * 8) / 8)
        BatchLimit("batch.resource.index_bytes");
      const auto index_bytes = labels * 8 + b.label_offsets.size() * 8;
      if (paired && pixels > std::numeric_limits<std::uint64_t>::max() / 2)
        BatchLimit("batch.resource.pixels");
      const auto pixel_charge = paired ? 2 * pixels : pixels;
      b.resource_lease_sets.push_back(
          detail::ReserveBatchTotals(o.resource_budget, bytes, pixel_charge,
                                     metadata_bytes, index_bytes, 0, 0, stop));
    }
    Stop(stop);
    const auto photo_options = torch::TensorOptions()
                                   .dtype(b.pixels.dtype)
                                   .device(torch::kCPU)
                                   .requires_grad(false);
    const auto paired_options =
        paired ? photo_options.dtype(b.paired_pixels->dtype) : photo_options;
    const auto n = static_cast<std::int64_t>(samples.size());
    if (o.layout != ImageBatchLayout::ragged) {
      const auto e = canvases[0];
      b.images = torch::full({n, b.pixels.channels(), e.height, e.width},
                             o.image_padding, photo_options);
      b.valid = torch::zeros({n, e.height, e.width},
                             photo_options.dtype(torch::kBool));
      if (paired)
        b.paired_images =
            torch::full({n, b.paired_pixels->channels(), e.height, e.width},
                        o.paired_padding, paired_options);
      if (masks)
        b.masks =
            torch::full({n, e.height, e.width}, o.mask_padding_id.value_or(0),
                        photo_options.dtype(torch::kInt64));
    }
    b.labels = torch::empty({static_cast<std::int64_t>(labels)},
                            photo_options.dtype(torch::kInt64));
    for (std::size_t i = 0; i < samples.size(); ++i) {
      Stop(stop);
      const auto& s = samples[i];
      const auto index = static_cast<std::int64_t>(i);
      if (o.layout == ImageBatchLayout::ragged) {
        b.image_list.push_back(
            Chw(s.image, s.pixels).clone(torch::MemoryFormat::Contiguous));
        b.valid_list.push_back(torch::ones({s.height, s.width},
                                           photo_options.dtype(torch::kBool)));
        if (paired)
          b.paired_image_list.push_back(
              Chw(*s.paired_target, *s.paired_target_pixels)
                  .clone(torch::MemoryFormat::Contiguous));
        if (masks)
          b.mask_list.push_back(
              s.mask_target->ids.clone(torch::MemoryFormat::Contiguous));
      } else {
        b.images[index]
            .narrow(1, 0, s.height)
            .narrow(2, 0, s.width)
            .copy_(Chw(s.image, s.pixels));
        b.valid[index].narrow(0, 0, s.height).narrow(1, 0, s.width).fill_(true);
        if (paired)
          (*b.paired_images)[index]
              .narrow(1, 0, s.height)
              .narrow(2, 0, s.width)
              .copy_(Chw(*s.paired_target, *s.paired_target_pixels));
        if (masks)
          (*b.masks)[index]
              .narrow(0, 0, s.height)
              .narrow(1, 0, s.width)
              .copy_(s.mask_target->ids);
      }
      auto* out = b.labels.data_ptr<std::int64_t>();
      for (std::size_t j = 0; j < s.labels.size(); ++j)
        out[b.label_offsets[i] + static_cast<std::int64_t>(j)] =
            static_cast<std::int64_t>(s.labels[j]);
    }
    Stop(stop);
    detail::CheckImageBatch(b, o.limits, stop);
    return b;
  } catch (const std::bad_alloc&) {
    BatchLimit("batch.allocate");
  } catch (const c10::OutOfMemoryError&) {
    BatchLimit("batch.allocate");
  } catch (const c10::Error&) {
    throw DeviceError("Image batch tensor operation failed");
  }
}
}  // namespace jyppx::liuding::data::image
