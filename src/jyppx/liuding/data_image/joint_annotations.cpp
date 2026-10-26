// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"
#include "joint_annotation_detail.hpp"
#include "panoptic_detail.hpp"

#include <jyppx/liuding/data_image/panoptic.hpp>

#include <limits>
#include <numeric>
#include <unordered_map>

namespace jyppx::liuding::data::image::detail {
namespace {
[[noreturn]] void Invalid(const char* operation) {
  throw InvalidArgument("Invalid joint annotation geometry",
                        Details(operation));
}

void Charge(std::uint64_t& total, std::uint64_t count,
            const JointTransformOptions& options) {
  if (count > options.limits.max_mask_work_pixels - total)
    Resource("transform.mask_work_pixels");
  total += count;
}

ImageResourceLease ReserveAnnotation(
    const std::shared_ptr<ImageResourceBudget>& resource_budget,
    std::size_t count, std::size_t element_bytes, const char* operation) {
  if (!resource_budget) return {};
  if (count > std::numeric_limits<std::uint64_t>::max() / element_bytes)
    Resource(operation);
  return resource_budget->reserve(ImageResourceKind::annotation_bytes,
                                  count * element_bytes);
}

std::size_t NextCapacity(std::size_t current, std::size_t maximum,
                         const char* operation) {
  if (current >= maximum) Resource(operation);
  return current > maximum / 2 ? maximum
                               : std::max<std::size_t>(1, current * 2);
}

struct NonPanopticRecordOwner {
  ImageResourceLease object_storage;
  ImageResourceLease run_lease_storage;
  ImageResourceLeaseSet run_leases;
  ImageResourceLease keypoint_lease_storage;
  ImageResourceLeaseSet keypoint_leases;
  VisionAnnotationRecord record;
};

// Polygon geometry is rebuilt below; copying source rings only raises peak
// memory.
VisionObject CopyForMapping(const VisionObject& source) {
  return {.source_id = source.source_id,
          .category = source.category,
          .box = source.box,
          .polygons = {},
          .oriented_box = source.oriented_box,
          .rle = source.rle,
          .keypoints = source.keypoints,
          .area = source.area,
          .mask_pixel_area = source.mask_pixel_area,
          .oriented_box_to_polygon = source.oriented_box_to_polygon,
          .polygon_partitioned = source.polygon_partitioned,
          .group_id = source.group_id,
          .crowd = source.crowd,
          .ignore = source.ignore,
          .difficult = source.difficult,
          .truncated = source.truncated};
}

std::uint64_t Pixels(ImageExtent extent) {
  return static_cast<std::uint64_t>(extent.width) * extent.height;
}

long double Cross(ImagePoint a, ImagePoint b, ImagePoint c) {
  return (static_cast<long double>(b.x) - a.x) *
             (static_cast<long double>(c.y) - a.y) -
         (static_cast<long double>(b.y) - a.y) *
             (static_cast<long double>(c.x) - a.x);
}

long double SignedArea(std::span<const ImagePoint> ring) {
  long double area = 0;
  // A fan around one vertex avoids cancellation of large absolute offsets.
  for (std::size_t i = 1; i + 1 < ring.size(); ++i)
    area += Cross(ring[0], ring[i], ring[i + 1]);
  return area / 2;
}

bool Equal(ImagePoint a, ImagePoint b) { return a.x == b.x && a.y == b.y; }

bool Inside(ImagePoint p, const ImageBox& box) {
  return p.x >= box.x_min && p.y >= box.y_min && p.x <= box.x_max &&
         p.y <= box.y_max;
}

bool Convex(std::span<const ImagePoint> ring, bool strict = false) {
  int sign = 0;
  for (std::size_t i = 0; i < ring.size(); ++i) {
    const auto cross = Cross(ring[i], ring[(i + 1) % ring.size()],
                             ring[(i + 2) % ring.size()]);
    if (cross == 0) {
      if (strict) return false;
      continue;
    }
    const int next = cross > 0 ? 1 : -1;
    if (sign && sign != next) return false;
    sign = next;
  }
  return sign != 0;
}

void Tick(JointAnnotationState& budget, const JointTransformOptions& options,
          std::stop_token stop) {
  CheckStop(stop);
  Add(budget.geometry_checks, 1, options.limits.max_geometry_checks,
      "transform.geometry_checks");
}

using Ring = std::vector<ImagePoint>;

Ring ClipConvex(Ring ring, const ImageBox& box, std::stop_token stop) {
  for (int edge = 0; edge < 4 && !ring.empty(); ++edge) {
    CheckStop(stop);
    const bool horizontal = edge >= 2;
    const bool lower = edge % 2 == 0;
    const double boundary = edge == 0   ? box.x_min
                            : edge == 1 ? box.x_max
                            : edge == 2 ? box.y_min
                                        : box.y_max;
    const auto coordinate = [&](ImagePoint p) {
      return horizontal ? p.y : p.x;
    };
    const auto inside = [&](ImagePoint p) {
      return lower ? coordinate(p) >= boundary : coordinate(p) <= boundary;
    };
    Ring next;
    next.reserve(ring.size() + 1);
    auto previous = ring.back();
    bool previous_inside = inside(previous);
    for (const auto current : ring) {
      CheckStop(stop);
      const bool current_inside = inside(current);
      if (current_inside != previous_inside) {
        const long double fraction =
            (static_cast<long double>(boundary) - coordinate(previous)) /
            (static_cast<long double>(coordinate(current)) -
             coordinate(previous));
        ImagePoint intersection{
            static_cast<double>(previous.x +
                                fraction * (current.x - previous.x)),
            static_cast<double>(previous.y +
                                fraction * (current.y - previous.y))};
        if (horizontal)
          intersection.y = boundary;
        else
          intersection.x = boundary;
        next.push_back(intersection);
      }
      if (current_inside) next.push_back(current);
      previous = current;
      previous_inside = current_inside;
    }
    ring.clear();
    for (const auto p : next)
      if (ring.empty() || !Equal(p, ring.back())) ring.push_back(p);
    if (ring.size() > 1 && Equal(ring.front(), ring.back())) ring.pop_back();
  }
  if (ring.size() < 3 || SignedArea(ring) == 0) ring.clear();
  return ring;
}

ImagePoint Mapped(ImagePoint p, const GeometryTrace& local) {
  const auto m = local.matrix();
  const auto extent = local.output_extent();
  return {
      std::clamp(m[0] * p.x + m[2], 0.0, static_cast<double>(extent.width)),
      std::clamp(m[4] * p.y + m[5], 0.0, static_cast<double>(extent.height))};
}

// Ear triangles have disjoint interiors. Clipping them separately preserves
// disconnected parts of a concave ring without introducing artificial bridges.
std::vector<Ring> ClipRing(std::span<const ImagePoint> input,
                           const GeometryTrace& local,
                           const JointTransformOptions& options,
                           JointAnnotationState& budget, bool& partitioned,
                           std::stop_token stop) {
  if (!local.source_region()) return {};
  const auto region = *local.source_region();
  if (input.size() > 1 && Equal(input.front(), input.back()))
    input = input.first(input.size() - 1);
  const auto original_area = std::abs(SignedArea(input));
  std::vector<Ring> pieces;
  long double visible_area = 0;
  long double output_area = 0;
  const auto emit = [&](Ring ring) {
    if (ring.empty()) return;
    visible_area += std::abs(SignedArea(ring));
    for (auto& point : ring) point = Mapped(point, local);
    const auto area = std::abs(SignedArea(ring));
    if (area == 0) return;
    output_area += area;
    Add(budget.output_fragments, 1, options.limits.max_polygon_fragments,
        "transform.output_fragments");
    Add(budget.output_points, ring.size(), options.limits.max_polygon_points,
        "transform.output_points");
    pieces.push_back(std::move(ring));
  };
  if (std::all_of(input.begin(), input.end(),
                  [&](ImagePoint p) { return Inside(p, region); })) {
    emit(Ring(input.begin(), input.end()));
  } else if (Convex(input)) {
    emit(ClipConvex(Ring(input.begin(), input.end()), region, stop));
  } else {
    partitioned = true;
    std::vector<std::size_t> vertices(input.size());
    std::iota(vertices.begin(), vertices.end(), 0U);
    // Remove harmless collinear middle vertices before ear selection.
    for (std::size_t i = 0; i < vertices.size() && vertices.size() > 3;) {
      Tick(budget, options, stop);
      const auto n = vertices.size();
      if (Cross(input[vertices[(i + n - 1) % n]], input[vertices[i]],
                input[vertices[(i + 1) % n]]) == 0) {
        vertices.erase(vertices.begin() + static_cast<std::ptrdiff_t>(i));
        i = 0;
      } else
        ++i;
    }
    const int sign = SignedArea(input) > 0 ? 1 : -1;
    while (vertices.size() > 3) {
      bool found = false;
      const auto n = vertices.size();
      for (std::size_t i = 0; i < n; ++i) {
        Tick(budget, options, stop);
        const auto a = vertices[(i + n - 1) % n], b = vertices[i],
                   c = vertices[(i + 1) % n];
        if (sign * Cross(input[a], input[b], input[c]) <= 0) continue;
        bool contains = false;
        for (const auto j : vertices) {
          if (j == a || j == b || j == c) continue;
          Tick(budget, options, stop);
          if (sign * Cross(input[a], input[b], input[j]) >= 0 &&
              sign * Cross(input[b], input[c], input[j]) >= 0 &&
              sign * Cross(input[c], input[a], input[j]) >= 0) {
            contains = true;
            break;
          }
        }
        if (contains) continue;
        emit(ClipConvex({input[a], input[b], input[c]}, region, stop));
        vertices.erase(vertices.begin() + static_cast<std::ptrdiff_t>(i));
        found = true;
        break;
      }
      if (!found) Invalid("transform.polygon_triangulation");
    }
    emit(
        ClipConvex({input[vertices[0]], input[vertices[1]], input[vertices[2]]},
                   region, stop));
  }
  if (output_area < options.min_region_area ||
      std::clamp(visible_area / original_area, 0.0L, 1.0L) <
          options.min_region_visibility)
    pieces.clear();
  return pieces;
}

std::optional<ImageBox> Bounds(const std::vector<Ring>& rings) {
  std::optional<ImageBox> result;
  for (const auto& ring : rings)
    for (const auto p : ring) {
      if (!result)
        result = ImageBox{p.x, p.y, p.x, p.y};
      else
        *result = {std::min(result->x_min, p.x), std::min(result->y_min, p.y),
                   std::max(result->x_max, p.x), std::max(result->y_max, p.y)};
    }
  return result;
}

double CellVisibility(std::uint32_t x, std::uint32_t y,
                      const GeometryTrace& local) {
  if (!local.source_region()) return 0;
  const auto r = *local.source_region();
  return std::max(0.0, std::min(static_cast<double>(x) + 1, r.x_max) -
                           std::max(static_cast<double>(x), r.x_min)) *
         std::max(0.0, std::min(static_cast<double>(y) + 1, r.y_max) -
                           std::max(static_cast<double>(y), r.y_min));
}

bool BoxKeeps(const VisionObject& source, const VisionObject& output,
              const GeometryTrace& local,
              const JointTransformOptions& options) {
  if (output.box) {
    const auto b = *output.box;
    if ((b.x_max - b.x_min) * (b.y_max - b.y_min) < options.min_box_area)
      return false;
  }
  if (!source.box) return true;
  if (!output.box || !local.source_region()) return false;
  const auto b = *source.box, r = *local.source_region();
  const auto w =
      std::max(0.0, std::min(b.x_max, r.x_max) - std::max(b.x_min, r.x_min));
  const auto h =
      std::max(0.0, std::min(b.y_max, r.y_max) - std::max(b.y_min, r.y_min));
  return std::clamp((w / (b.x_max - b.x_min)) * (h / (b.y_max - b.y_min)), 0.0,
                    1.0) >= options.min_box_visibility;
}

bool MapPose(VisionObject& object, std::span<const ImageGeometryStep> steps,
             const JointTransformOptions& options, const GeometryTrace& local) {
  for (auto& keypoint : object.keypoints) {
    const auto p = keypoint.visibility ? local.forward_point(keypoint.point)
                                       : std::nullopt;
    if (p)
      keypoint.point = *p;
    else
      keypoint = {};
  }
  ImageResourceLease permutation_lease;
  std::vector<ImageKeypoint> previous;
  for (const auto& step : steps) {
    const auto* table = step.kind == ImageGeometryKind::horizontal_flip
                            ? &options.horizontal_keypoint_permutations
                        : step.kind == ImageGeometryKind::vertical_flip
                            ? &options.vertical_keypoint_permutations
                            : nullptr;
    if (!table || object.keypoints.empty()) continue;
    if (previous.empty()) {
      permutation_lease = ReserveAnnotation(
          options.resource_budget, object.keypoints.size(),
          sizeof(ImageKeypoint), "transform.pose_permutation_bytes");
      previous.resize(object.keypoints.size());
    }
    std::copy(object.keypoints.begin(), object.keypoints.end(),
              previous.begin());
    const auto& permutation = (*table)[object.category];
    for (std::size_t i = 0; i < permutation.size(); ++i)
      object.keypoints[i] = previous[permutation[i]];
  }
  return !options.drop_empty_pose || object.keypoints.empty() ||
         std::any_of(object.keypoints.begin(), object.keypoints.end(),
                     [](const auto& p) { return p.visibility != 0; });
}

void KeepOutputLease(ImageResourceLeaseSet& leases, ImageResourceLease& storage,
                     ImageResourceLease lease,
                     const JointTransformOptions& options,
                     const char* capacity_operation,
                     const char* bytes_operation) {
  if (!lease.active()) return;
  if (leases.size() == leases.capacity()) {
    const auto maximum =
        std::min(options.limits.max_objects, leases.max_size());
    const auto capacity =
        NextCapacity(leases.capacity(), maximum, capacity_operation);
    auto next_lease =
        ReserveAnnotation(options.resource_budget, capacity,
                          sizeof(ImageResourceLease), bytes_operation);
    leases.reserve(capacity);
    storage = std::move(next_lease);
  }
  leases.push_back(std::move(lease));
}

bool MapRle(VisionObject& object, std::span<const ImageGeometryStep> steps,
            std::span<const ImageExtent> extents,
            const JointTransformOptions& options, const GeometryTrace& local,
            JointAnnotationState& budget, ImageResourceLease& copied_runs,
            ImageResourceLease& output_runs, std::stop_token stop) {
  const auto& source = *object.rle;
  BudgetedDiscreteRaster raster;
  if (options.resource_budget)
    raster.lease = options.resource_budget->reserve(
        ImageResourceKind::annotation_bytes,
        static_cast<std::uint64_t>(source.width) * source.height);
  raster.pixels = torch::empty({source.height, source.width}, torch::kUInt8);
  auto* pixels = raster.pixels.data_ptr<std::uint8_t>();
  std::uint64_t position = 0, foreground = 0;
  long double visible = 0;
  bool value = false;
  for (const auto run : source.counts) {
    for (std::uint64_t i = 0; i < run; ++i, ++position) {
      if (position % source.height == 0) CheckStop(stop);
      const auto x = static_cast<std::uint32_t>(position / source.height);
      const auto y = static_cast<std::uint32_t>(position % source.height);
      pixels[static_cast<std::size_t>(y) * source.width + x] = value ? 1 : 0;
      if (value) {
        ++foreground;
        visible += CellVisibility(x, y, local);
      }
    }
    value = !value;
  }
  if (!foreground || visible / foreground < options.min_region_visibility)
    return false;
  const auto transformed = TransformDiscrete(std::move(raster), steps, extents,
                                             0, options.resource_budget, stop);
  const auto* data = transformed.pixels.const_data_ptr<std::uint8_t>();
  ImageMaskRle result;
  result.width = extents.back().width;
  result.height = extents.back().height;
  std::uint64_t count = 0, area = 0;
  value = false;
  std::optional<ImageBox> box;
  ImageResourceLease counts_lease;
  const auto emit = [&] {
    Add(budget.output_runs, 1, options.limits.max_rle_runs,
        "transform.output_runs");
    if (options.resource_budget &&
        result.counts.size() == result.counts.capacity()) {
      const auto maximum =
          std::min(options.limits.max_rle_runs, result.counts.max_size());
      const auto capacity = NextCapacity(result.counts.capacity(), maximum,
                                         "transform.rle_runs_capacity");
      auto next_lease =
          ReserveAnnotation(options.resource_budget, capacity,
                            sizeof(std::uint64_t), "transform.rle_runs_bytes");
      result.counts.reserve(capacity);
      counts_lease = std::move(next_lease);
    }
    result.counts.push_back(count);
    count = 0;
  };
  for (std::uint32_t x = 0; x < result.width; ++x) {
    CheckStop(stop);
    for (std::uint32_t y = 0; y < result.height; ++y) {
      const bool next =
          data[static_cast<std::size_t>(y) * result.width + x] != 0;
      if (next != value) {
        emit();
        value = next;
      }
      ++count;
      if (!next) continue;
      ++area;
      if (!box)
        box = ImageBox{static_cast<double>(x), static_cast<double>(y),
                       static_cast<double>(x) + 1, static_cast<double>(y) + 1};
      else
        *box = {std::min(box->x_min, static_cast<double>(x)),
                std::min(box->y_min, static_cast<double>(y)),
                std::max(box->x_max, static_cast<double>(x) + 1),
                std::max(box->y_max, static_cast<double>(y) + 1)};
    }
  }
  emit();
  if (!area || static_cast<long double>(area) < options.min_region_area)
    return false;
  object.rle = std::move(result);
  copied_runs.release();
  output_runs = std::move(counts_lease);
  object.mask_pixel_area = area;
  object.box = box;
  return true;
}

void MapPanoptic(const VisionSample& source, VisionSample& output,
                 VisionAnnotationRecord& record,
                 const JointTransformOptions& options,
                 const GeometryTrace& local,
                 const VisionAnnotationRecord& bound, std::stop_token stop) {
  struct Stats {
    long double visible = 0;
    std::uint64_t count = 0;
    ImageBox box;
    bool keep = false;
  };
  if (bound.objects.size() > std::numeric_limits<std::size_t>::max() / 128U)
    Resource("transform.panoptic_working_bytes");
  ImageResourceLease scratch_lease;
  if (options.resource_budget)
    scratch_lease = options.resource_budget->reserve(
        ImageResourceKind::annotation_bytes, bound.objects.size() * 128U);
  std::unordered_map<std::int64_t, std::size_t> lookup;
  lookup.reserve(bound.objects.size());
  std::vector<Stats> stats(bound.objects.size());
  for (std::size_t i = 0; i < bound.objects.size(); ++i)
    lookup.emplace(bound.objects[i].source_id, i);
  const auto input = source.mask_target->ids.accessor<std::int64_t, 2>();
  const auto& mask = *source.mask_target;
  for (std::uint32_t y = 0; y < source.height; ++y) {
    CheckStop(stop);
    for (std::uint32_t x = 0; x < source.width; ++x) {
      const auto id = input[y][x];
      if (id == mask.void_id || id == mask.ignore_id) continue;
      stats[lookup.at(id)].visible += CellVisibility(x, y, local);
    }
  }
  auto ids = output.mask_target->ids.accessor<std::int64_t, 2>();
  for (std::uint32_t y = 0; y < output.height; ++y) {
    CheckStop(stop);
    for (std::uint32_t x = 0; x < output.width; ++x) {
      const auto id = ids[y][x];
      if (id == mask.void_id || id == mask.ignore_id) continue;
      const auto found = lookup.find(id);
      if (found == lookup.end()) Invalid("transform.panoptic_output_id");
      auto& s = stats[found->second];
      const double left = x, top = y, right = static_cast<double>(x) + 1,
                   bottom = static_cast<double>(y) + 1;
      if (s.count++ == 0)
        s.box = {left, top, right, bottom};
      else
        s.box = {std::min(s.box.x_min, left), std::min(s.box.y_min, top),
                 std::max(s.box.x_max, right), std::max(s.box.y_max, bottom)};
    }
  }
  std::size_t kept_count = 0;
  for (std::size_t i = 0; i < stats.size(); ++i) {
    CheckStop(stop);
    auto& s = stats[i];
    auto object = bound.objects[i];
    object.box = s.box;
    object.mask_pixel_area = s.count;
    if (!s.count ||
        static_cast<long double>(s.count) < options.min_region_area ||
        s.visible / *bound.objects[i].mask_pixel_area <
            options.min_region_visibility ||
        !BoxKeeps(bound.objects[i], object, local, options))
      continue;
    s.keep = true;
    ++kept_count;
  }
  record.objects.reserve(kept_count);
  for (std::size_t i = 0; i < stats.size(); ++i) {
    if (!stats[i].keep) continue;
    auto object = bound.objects[i];
    object.box = stats[i].box;
    object.mask_pixel_area = stats[i].count;
    record.objects.push_back(std::move(object));
  }
  for (std::uint32_t y = 0; y < output.height; ++y) {
    CheckStop(stop);
    for (std::uint32_t x = 0; x < output.width; ++x) {
      const auto id = ids[y][x];
      if (id != mask.void_id && id != mask.ignore_id &&
          !stats[lookup.at(id)].keep)
        ids[y][x] = *mask.void_id;
    }
  }
  record.panoptic_mask_materialized = true;
}
}  // namespace

bool PanopticBoxKeeps(const VisionObject& source, const VisionObject& output,
                      const GeometryTrace& local,
                      const JointTransformOptions& options) {
  return BoxKeeps(source, output, local, options);
}

JointAnnotationState CheckJointAnnotations(
    const VisionSample& sample, std::span<const ImageGeometryStep> steps,
    std::span<const ImageExtent> extents, const JointTransformOptions& options,
    const GeometryTrace& local, std::stop_token stop,
    bool defer_panoptic_binding) {
  JointAnnotationState result;
  std::uint64_t raster_work = 0;
  const auto sequence_work = [&] {
    for (const auto extent : extents)
      Charge(raster_work, Pixels(extent), options);
  };
  if (sample.mask_target) sequence_work();
  if (!sample.annotation) return result;
  const auto& record = *sample.annotation;
  if (!sample.categories || record.width != sample.width ||
      record.height != sample.height)
    Invalid("transform.annotation_alignment");
  if (record.objects.size() > options.limits.max_objects)
    Resource("transform.objects");
  AnnotationReadOptions validation;
  validation.limits.max_geometry_checks = options.limits.max_geometry_checks;
  ParseState state{validation, stop};
  std::size_t points = 0, fragments = 0, runs = 0, keypoints = 0;
  for (const auto& object : record.objects) {
    CheckStop(stop);
    if (object.category >= sample.categories->size())
      Invalid("transform.category");
    if (object.box) {
      const auto b = *object.box;
      if (!std::isfinite((b.x_max - b.x_min) * (b.y_max - b.y_min)))
        Invalid("transform.box_area");
      static_cast<void>(local.forward_box(b));
    }
    const int rich = (!object.polygons.empty() ? 1 : 0) +
                     (object.oriented_box ? 1 : 0) + (object.rle ? 1 : 0);
    if (rich > 1) Invalid("transform.mixed_region_geometry");
    const auto validate_ring = [&](std::span<const ImagePoint> ring) {
      Add(points, ring.size(), options.limits.max_polygon_points,
          "transform.points");
      Add(fragments, 1, options.limits.max_polygon_fragments,
          "transform.fragments");
      for (const auto p : ring)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
            p.x > sample.width || p.y > sample.height)
          Invalid("transform.polygon_bounds");
      try {
        ValidatePolygon(ring, state);
      } catch (const SerializationError&) {
        Invalid("transform.polygon");
      }
    };
    for (const auto& ring : object.polygons) validate_ring(ring);
    if (object.oriented_box) {
      validate_ring(*object.oriented_box);
      if (!Convex(*object.oriented_box, true))
        Invalid("transform.obb_convex_order");
    }
    if (object.rle) {
      const auto& rle = *object.rle;
      if (rle.width != sample.width || rle.height != sample.height ||
          rle.counts.empty())
        Invalid("transform.rle_dimensions");
      Add(runs, rle.counts.size(), options.limits.max_rle_runs,
          "transform.input_runs");
      std::uint64_t total = 0, foreground = 0;
      bool value = false;
      for (const auto run : rle.counts) {
        CheckStop(stop);
        if (run > Pixels(extents.front()) - total)
          Invalid("transform.rle_runs");
        total += run;
        if (value) foreground += run;
        value = !value;
      }
      if (total != Pixels(extents.front()) ||
          (object.mask_pixel_area && *object.mask_pixel_area != foreground))
        Invalid("transform.rle_pixels");
      sequence_work();
      Charge(raster_work, Pixels(extents.front()), options);  // Decode.
      Charge(raster_work, Pixels(extents.back()), options);   // Encode/count.
    }
    Add(keypoints, object.keypoints.size(), options.limits.max_keypoints,
        "transform.keypoints");
    for (const auto& p : object.keypoints)
      if (p.visibility > 2 || (p.visibility && (!std::isfinite(p.point.x) ||
                                                !std::isfinite(p.point.y))))
        Invalid("transform.keypoint");
    if (object.keypoints.empty()) continue;
    const auto& names = (*sample.categories)[object.category].keypoint_names;
    if (!names.empty() && names.size() != object.keypoints.size())
      Invalid("transform.pose_schema");
    for (const auto& step : steps) {
      const auto* table = step.kind == ImageGeometryKind::horizontal_flip
                              ? &options.horizontal_keypoint_permutations
                          : step.kind == ImageGeometryKind::vertical_flip
                              ? &options.vertical_keypoint_permutations
                              : nullptr;
      if (table &&
          (table->size() != sample.categories->size() ||
           (*table)[object.category].size() != object.keypoints.size()))
        Invalid("transform.pose_flip");
    }
  }
  result.geometry_checks = state.geometry_checks;
  if (record.panoptic_mask_path) {
    if (!sample.mask_target) Invalid("transform.panoptic_mask_missing");
    for (const auto& step : steps)
      if (step.kind == ImageGeometryKind::pad &&
          (step.left || step.top || step.right || step.bottom) &&
          options.mask_padding_id != sample.mask_target->void_id)
        Invalid("transform.panoptic_padding_void");
    Charge(raster_work, Pixels(extents.front()), options);  // Binding scan.
    Charge(raster_work, Pixels(extents.front()),
           options);                                       // Source visibility.
    Charge(raster_work, Pixels(extents.back()), options);  // Output statistics.
    Charge(raster_work, Pixels(extents.back()),
           options);  // Removed IDs -> void.
    PanopticBindOptions bind;
    bind.limits.max_objects = options.limits.max_objects;
    bind.limits.max_categories = options.limits.max_objects;
    bind.limits.max_dimension = options.limits.max_dimension;
    bind.limits.max_pixels = options.limits.max_pixels;
    bind.max_working_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(options.limits.max_working_bytes,
                                std::numeric_limits<std::size_t>::max()));
    if (!defer_panoptic_binding) {
      auto bound = BindPanopticMask(record, *sample.mask_target,
                                    sample.categories->size(), bind,
                                    options.resource_budget, stop);
      result.panoptic_lease = std::move(bound.output_lease);
      result.panoptic = std::move(bound.record);
    }
  } else if (record.panoptic_mask_materialized)
    Invalid("transform.panoptic_reference");
  return result;
}

void MapJointAnnotations(const VisionSample& sample, VisionSample& output,
                         std::span<const ImageGeometryStep> steps,
                         std::span<const ImageExtent> extents,
                         const JointTransformOptions& options,
                         const GeometryTrace& local,
                         JointAnnotationState& budget, std::stop_token stop) {
  if (!sample.annotation) return;
  ImageResourceLease output_lease;
  if (budget.panoptic && options.resource_budget)
    output_lease =
        options.resource_budget->reserve(ImageResourceKind::annotation_bytes,
                                         PanopticRecordBytes(*budget.panoptic));
  std::shared_ptr<VisionAnnotationRecord> record;
  std::shared_ptr<NonPanopticRecordOwner> mapped_owner;
  if (budget.panoptic) {
    auto owner = std::make_shared<PanopticRecordOwner>();
    owner->output_lease = std::move(output_lease);
    record = std::shared_ptr<VisionAnnotationRecord>(owner, &owner->record);
  } else if (options.resource_budget) {
    mapped_owner = std::make_shared<NonPanopticRecordOwner>();
    record = std::shared_ptr<VisionAnnotationRecord>(mapped_owner,
                                                     &mapped_owner->record);
  } else {
    record = std::make_shared<VisionAnnotationRecord>();
  }
  record->source_image_id = sample.annotation->source_image_id;
  record->image_path = sample.annotation->image_path;
  record->panoptic_mask_path = sample.annotation->panoptic_mask_path;
  record->width = output.width;
  record->height = output.height;
  if (budget.panoptic) {
    MapPanoptic(sample, output, *record, options, local, *budget.panoptic,
                stop);
    output.annotation = std::move(record);
    return;
  }
  for (const auto& source : sample.annotation->objects) {
    CheckStop(stop);
    auto copied_runs =
        source.rle ? ReserveAnnotation(
                         options.resource_budget, source.rle->counts.size(),
                         sizeof(std::uint64_t), "transform.rle_source_copy")
                   : ImageResourceLease{};
    auto copied_keypoints =
        source.keypoints.empty()
            ? ImageResourceLease{}
            : ReserveAnnotation(options.resource_budget,
                                source.keypoints.size(), sizeof(ImageKeypoint),
                                "transform.pose_source_copy");
    ImageResourceLease output_runs;
    auto object = CopyForMapping(source);
    if (source.box) object.box = local.forward_box(*source.box);
    if (!source.polygons.empty()) {
      for (const auto& ring : source.polygons) {
        auto parts = ClipRing(ring, local, options, budget,
                              object.polygon_partitioned, stop);
        for (auto& part : parts) object.polygons.push_back(std::move(part));
      }
      if (object.polygons.empty()) continue;
      object.box = Bounds(object.polygons);
    } else if (source.oriented_box) {
      if (!local.source_region()) continue;
      const bool uncut = std::all_of(
          source.oriented_box->begin(), source.oriented_box->end(),
          [&](ImagePoint p) { return Inside(p, *local.source_region()); });
      if (!uncut && options.oriented_box_crop == OrientedBoxCrop::discard)
        continue;
      auto parts = ClipRing(*source.oriented_box, local, options, budget,
                            object.polygon_partitioned, stop);
      if (parts.empty()) continue;
      object.box = Bounds(parts);
      if (uncut) {
        for (std::size_t i = 0; i < 4; ++i)
          (*object.oriented_box)[i] = parts[0][i];
      } else {
        object.oriented_box.reset();
        object.polygons = std::move(parts);
        object.oriented_box_to_polygon = true;
      }
    } else if (source.rle && !MapRle(object, steps, extents, options, local,
                                     budget, copied_runs, output_runs, stop))
      continue;
    if (source.box && !object.box) continue;
    if (!BoxKeeps(source, object, local, options) ||
        !MapPose(object, steps, options, local))
      continue;
    if (mapped_owner && record->objects.size() == record->objects.capacity()) {
      const auto maximum =
          std::min(options.limits.max_objects, record->objects.max_size());
      const auto capacity = NextCapacity(record->objects.capacity(), maximum,
                                         "transform.object_capacity");
      auto next_lease =
          ReserveAnnotation(options.resource_budget, capacity,
                            sizeof(VisionObject), "transform.object_bytes");
      record->objects.reserve(capacity);
      mapped_owner->object_storage = std::move(next_lease);
    }
    if (output_runs.active()) {
      KeepOutputLease(mapped_owner->run_leases, mapped_owner->run_lease_storage,
                      std::move(output_runs), options,
                      "transform.rle_lease_capacity",
                      "transform.rle_lease_bytes");
    }
    if (copied_keypoints.active()) {
      KeepOutputLease(
          mapped_owner->keypoint_leases, mapped_owner->keypoint_lease_storage,
          std::move(copied_keypoints), options, "transform.pose_lease_capacity",
          "transform.pose_lease_bytes");
    }
    record->objects.push_back(std::move(object));
  }
  output.annotation = std::move(record);
}

}  // namespace jyppx::liuding::data::image::detail
