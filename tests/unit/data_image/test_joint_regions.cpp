// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/data_image/panoptic.hpp>

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <numeric>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;

VisionSample Sample(std::uint32_t w = 6, std::uint32_t h = 6) {
  VisionSample sample;
  sample.width = w;
  sample.height = h;
  sample.pixels.color = PixelColorModel::gray;
  sample.image = torch::zeros({1, h, w}, torch::kUInt8);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{41, "shape"}});
  return sample;
}

std::shared_ptr<VisionAnnotationRecord> Record(VisionSample& sample) {
  auto record = std::make_shared<VisionAnnotationRecord>();
  record->width = sample.width;
  record->height = sample.height;
  record->image_path = "image.png";
  sample.annotation = record;
  return record;
}

double Area(const std::vector<std::vector<ImagePoint>>& rings) {
  double area = 0;
  for (const auto& ring : rings) {
    double twice = 0;
    for (std::size_t i = 0; i < ring.size(); ++i) {
      const auto a = ring[i], b = ring[(i + 1) % ring.size()];
      twice += a.x * b.y - b.x * a.y;
    }
    area += std::abs(twice) / 2;
  }
  return area;
}

bool Contains(const std::vector<ImagePoint>& ring, ImagePoint p) {
  bool inside = false;
  for (std::size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
    const auto a = ring[i], b = ring[j];
    if ((a.y > p.y) != (b.y > p.y) &&
        p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

TEST(JointRegionTest, ConvexPolygonClipsMapsAndKeepsSourceMetadata) {
  auto sample = Sample();
  auto record = Record(sample);
  record->objects.push_back(
      {.source_id = 77,
       .polygons = {{{0, 0}, {5, 0}, {5, 4}, {0, 4}, {0, 0}}},
       .area = 999,
       .crowd = true});
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::crop,
                        .extent = {3, 3},
                        .left = 2,
                        .top = 1},
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {6, 3}},
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto result = transform_sample(sample, steps);
  ASSERT_EQ(result.annotation->objects.size(), 1U);
  const auto& object = result.annotation->objects[0];
  EXPECT_DOUBLE_EQ(Area(object.polygons), 18);
  EXPECT_DOUBLE_EQ(object.box->x_min, 0);
  EXPECT_DOUBLE_EQ(object.box->x_max, 6);
  EXPECT_DOUBLE_EQ(object.box->y_max, 3);
  EXPECT_EQ(object.source_id, 77);
  EXPECT_EQ(object.area, 999);
  EXPECT_TRUE(object.crowd);
  EXPECT_FALSE(object.polygon_partitioned);
  EXPECT_EQ(record->objects[0].polygons[0].size(), 5U);
}

TEST(JointRegionTest, ConcaveCropPreservesDisconnectedRegionsWithoutBridges) {
  // A U shape: cutting off its base leaves two disjoint vertical bars.
  const std::vector<ImagePoint> u{{0, 0}, {6, 0}, {6, 6}, {4, 6},
                                  {4, 2}, {2, 2}, {2, 6}, {0, 6}};
  for (const bool reversed : {false, true}) {
    auto sample = Sample();
    auto record = Record(sample);
    auto ring = u;
    if (reversed) std::reverse(ring.begin(), ring.end());
    record->objects.push_back({.polygons = {ring}});
    const std::array steps{ImageGeometryStep{
        .kind = ImageGeometryKind::crop, .extent = {6, 3}, .top = 3}};
    const auto result = transform_sample(sample, steps);
    ASSERT_EQ(result.annotation->objects.size(), 1U);
    const auto& object = result.annotation->objects[0];
    EXPECT_TRUE(object.polygon_partitioned);
    EXPECT_NEAR(Area(object.polygons), 12, 1e-12);
    // Nonvertex sampling also checks no overlapping interiors or filled gap.
    for (int y = 0; y < 12; ++y)
      for (int x = 0; x < 24; ++x) {
        const ImagePoint p{(x + 0.317) / 4, (y + 0.193) / 4};
        const auto count =
            std::count_if(object.polygons.begin(), object.polygons.end(),
                          [&](const auto& part) { return Contains(part, p); });
        EXPECT_EQ(count, p.x < 2 || p.x > 4 ? 1 : 0);
      }
  }
}

TEST(JointRegionTest,
     VariedConcaveRingsMatchIndependentPointMembershipAfterSequence) {
  for (int variant = 0; variant < 12; ++variant) {
    auto sample = Sample();
    auto record = Record(sample);
    std::vector<ImagePoint> ring;
    const int vertices = 9 + variant;
    for (int i = 0; i < vertices; ++i) {
      const double angle = 2 * std::numbers::pi * i / vertices;
      const double radius = i % 2 ? 1.2 : 2.9;
      ring.push_back(
          {3 + radius * std::cos(angle), 3 + radius * std::sin(angle)});
    }
    record->objects.push_back({.polygons = {ring}});
    const std::array steps{
        ImageGeometryStep{.kind = ImageGeometryKind::crop,
                          .extent = {3, 4},
                          .left = 2,
                          .top = 1},
        ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {9, 6}},
        ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip},
        ImageGeometryStep{.kind = ImageGeometryKind::vertical_flip},
        ImageGeometryStep{
            .kind = ImageGeometryKind::pad, .left = 1, .top = 2, .right = 2}};
    const auto output = transform_sample(sample, steps);
    ASSERT_EQ(output.annotation->objects.size(), 1U);
    for (int y = 0; y < 23; ++y)
      for (int x = 0; x < 29; ++x) {
        const ImagePoint p{12 * (x + 0.319) / 29, 8 * (y + 0.173) / 23};
        const auto source = output.geometry->inverse_point(p);
        const bool expected = source && Contains(ring, *source);
        const auto& parts = output.annotation->objects[0].polygons;
        const auto count =
            std::count_if(parts.begin(), parts.end(),
                          [&](const auto& part) { return Contains(part, p); });
        EXPECT_EQ(count, expected ? 1 : 0) << variant << ": " << x << ", " << y;
      }
  }
}

TEST(JointRegionTest, RegionThresholdsApplyPerSourceRingAndRespectEquality) {
  auto sample = Sample();
  auto record = Record(sample);
  record->objects.push_back({.polygons = {{{0, 0}, {4, 0}, {4, 4}, {0, 4}},
                                          {{4, 4}, {6, 4}, {6, 6}, {4, 6}}}});
  const std::array steps{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {4, 4}, .left = 2, .top = 2}};
  JointTransformOptions options;
  options.min_region_visibility = 0.25;
  options.min_region_area = 4;
  EXPECT_DOUBLE_EQ(Area(transform_sample(sample, steps, options)
                            .annotation->objects[0]
                            .polygons),
                   8);
  options.min_region_visibility = 0.251;
  EXPECT_DOUBLE_EQ(Area(transform_sample(sample, steps, options)
                            .annotation->objects[0]
                            .polygons),
                   4);
  options.min_region_area = 4.01;
  EXPECT_TRUE(
      transform_sample(sample, steps, options).annotation->objects.empty());
}

TEST(JointRegionTest, UncutObbKeepsCornerOrderButCroppedObbUsesExplicitPolicy) {
  auto sample = Sample();
  auto record = Record(sample);
  record->objects.emplace_back();
  const std::array<ImagePoint, 4> diamond{{{3, 0}, {6, 3}, {3, 6}, {0, 3}}};
  record->objects[0].oriented_box = diamond;
  const std::array flip{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto flipped = transform_sample(sample, flip);
  ASSERT_TRUE(flipped.annotation->objects[0].oriented_box);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ((*flipped.annotation->objects[0].oriented_box)[i].x,
                     6 - diamond[i].x);
    EXPECT_DOUBLE_EQ((*flipped.annotation->objects[0].oriented_box)[i].y,
                     diamond[i].y);
  }
  const std::array crop{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {5, 6}, .left = 1}};
  const auto clipped = transform_sample(sample, crop);
  const auto& object = clipped.annotation->objects[0];
  EXPECT_FALSE(object.oriented_box);
  EXPECT_TRUE(object.oriented_box_to_polygon);
  ASSERT_EQ(object.polygons.size(), 1U);
  EXPECT_EQ(object.polygons[0].size(), 5U);
  EXPECT_DOUBLE_EQ(Area(object.polygons), 17);
  JointTransformOptions discard;
  discard.oriented_box_crop = OrientedBoxCrop::discard;
  EXPECT_TRUE(
      transform_sample(sample, crop, discard).annotation->objects.empty());
}

ImageMaskRle Encode(std::uint32_t w, std::uint32_t h,
                    const std::vector<int>& row_major) {
  ImageMaskRle rle{w, h, {0}};
  int previous = 0;
  for (std::uint32_t x = 0; x < w; ++x)
    for (std::uint32_t y = 0; y < h; ++y) {
      const auto value = row_major[y * w + x];
      if (value != previous) {
        rle.counts.push_back(0);
        previous = value;
      }
      ++rle.counts.back();
    }
  return rle;
}

std::vector<int> Decode(const ImageMaskRle& rle) {
  std::vector<int> pixels(static_cast<std::size_t>(rle.width) * rle.height);
  std::size_t index = 0;
  int value = 0;
  for (auto run : rle.counts) {
    while (run--) {
      pixels[(index % rle.height) * rle.width + index / rle.height] = value;
      ++index;
    }
    value = 1 - value;
  }
  return pixels;
}

TEST(JointRegionTest, RleResizesSequentiallyAndCountsActualPixels) {
  auto sample = Sample(3, 1);
  auto record = Record(sample);
  record->objects.push_back(
      {.source_id = 91, .rle = Encode(3, 1, {0, 1, 0}), .area = 999});
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {2, 1}},
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {3, 1}}};
  // 3->2 samples source indices [0,2]: the foreground disappears. A single
  // composed 3->3 nearest lookup would incorrectly retain the middle pixel.
  EXPECT_TRUE(transform_sample(sample, steps).annotation->objects.empty());
  const std::array up{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {6, 1}}};
  const auto result = transform_sample(sample, up);
  ASSERT_EQ(result.annotation->objects.size(), 1U);
  const auto& object = result.annotation->objects[0];
  EXPECT_EQ(Decode(*object.rle), (std::vector<int>{0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(object.mask_pixel_area, 2U);
  EXPECT_EQ(object.area, 999);
  EXPECT_DOUBLE_EQ(object.box->x_min, 2);
  EXPECT_DOUBLE_EQ(object.box->x_max, 4);
  EXPECT_EQ(record->objects[0].rle->counts,
            (std::vector<std::uint64_t>{1, 1, 1}));
}

TEST(JointRegionTest, RleRasterBudgetPrecedesAllocationAndReleasesOnFailure) {
  auto sample = Sample(3, 1);
  auto record = Record(sample);
  record->objects.push_back({.rle = Encode(3, 1, {0, 1, 0})});
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {6, 1}}};
  JointTransformOptions options;
  ImageResourceBudgetLimits limits;

  limits.max_annotation_bytes = 2;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_sample(sample, steps, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  limits.max_annotation_bytes = 8;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_sample(sample, steps, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak, 3U);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  limits.max_annotation_bytes = 9;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  const auto result = transform_sample(sample, steps, options);
  ASSERT_EQ(result.annotation->objects.size(), 1U);
  EXPECT_EQ(Decode(*result.annotation->objects[0].rle),
            (std::vector<int>{0, 0, 1, 1, 0, 0}));
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak, 9U);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);
}

TEST(JointRegionTest, RleCropPadFlipMatchesIndependentRasterAndRebinds) {
  auto sample = Sample(3, 2);
  auto record = Record(sample);
  record->objects.push_back({.rle = Encode(3, 2, {1, 0, 1, 0, 1, 0})});
  const std::array steps{
      ImageGeometryStep{
          .kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1},
      ImageGeometryStep{.kind = ImageGeometryKind::pad, .left = 1, .bottom = 1},
      ImageGeometryStep{.kind = ImageGeometryKind::vertical_flip},
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto result = transform_sample(sample, steps);
  const auto& object = result.annotation->objects[0];
  EXPECT_EQ(Decode(*object.rle), (std::vector<int>{0, 0, 0, 0, 1, 0, 1, 0, 0}));
  EXPECT_EQ(object.mask_pixel_area, 2U);
  EXPECT_DOUBLE_EQ(object.box->x_min, 0);
  EXPECT_DOUBLE_EQ(object.box->y_min, 1);
  EXPECT_NO_THROW(transform_sample(result, {}));
}

TEST(JointRegionTest,
     MaskVisibilityUsesFractionalSourceCellsAndPixelOutputArea) {
  auto sample = Sample(2, 1);
  auto record = Record(sample);
  record->objects.push_back({.rle = Encode(2, 1, {1, 0})});
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {4, 1}},
      ImageGeometryStep{
          .kind = ImageGeometryKind::crop, .extent = {3, 1}, .left = 1}};
  JointTransformOptions options;
  options.min_region_visibility = 0.5;
  options.min_region_area = 1;
  EXPECT_EQ(transform_sample(sample, steps, options).annotation->objects.size(),
            1U);
  options.min_region_visibility = 0.501;
  EXPECT_TRUE(
      transform_sample(sample, steps, options).annotation->objects.empty());
  options.min_region_visibility = 0;
  options.min_region_area = 1.01;
  EXPECT_TRUE(
      transform_sample(sample, steps, options).annotation->objects.empty());
}

TEST(JointRegionTest, ZeroRunsPreserveParityAndEmptyMasksAreRemoved) {
  auto sample = Sample(3, 1);
  auto record = Record(sample);
  record->objects.push_back({.rle = ImageMaskRle{3, 1, {0, 0, 0, 3}}});
  const auto result = transform_sample(sample, {});
  EXPECT_EQ(Decode(*result.annotation->objects[0].rle),
            (std::vector<int>{1, 1, 1}));
  record->objects[0].rle->counts = {3};
  EXPECT_TRUE(transform_sample(sample, {}).annotation->objects.empty());
}

TEST(JointRegionTest, RejectsMalformedGeometryBeforePublishing) {
  auto sample = Sample();
  auto record = Record(sample);
  record->objects.push_back({.polygons = {{{0, 0}, {6, 6}, {0, 6}, {6, 0}}}});
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  record->objects[0].polygons = {{{0, 0}, {7, 0}, {0, 1}}};
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  record->objects[0].polygons.clear();
  record->objects[0].oriented_box =
      std::array<ImagePoint, 4>{{{0, 0}, {6, 0}, {1, 1}, {0, 6}}};
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  record->objects[0].oriented_box.reset();
  record->objects[0].rle = ImageMaskRle{6, 6, {37}};
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  record->objects[0].rle->counts = {35};
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  record->objects[0].rle->counts = {36};
  record->objects[0].mask_pixel_area = 1;
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  EXPECT_EQ(sample.image.sum().item<int>(), 0);
}

TEST(JointRegionTest, EnforcesIndependentGeometryAndAggregateMaskBudgets) {
  auto sample = Sample(3, 1);
  auto record = Record(sample);
  record->objects.push_back({.rle = Encode(3, 1, {1, 0, 1})});
  record->objects.push_back(record->objects[0]);
  JointTransformOptions options;
  options.limits.max_mask_work_pixels =
      17;  // Each identity RLE costs 3*3 visits.
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  options.limits.max_mask_work_pixels = 18;
  EXPECT_EQ(transform_sample(sample, {}, options).annotation->objects.size(),
            2U);
  options.limits.max_rle_runs =
      7;  // Four runs per instance, including initial 0.
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  sample = Sample();
  record = Record(sample);
  record->objects.push_back(
      {.polygons = {
           {{0, 0}, {6, 0}, {6, 6}, {4, 6}, {4, 2}, {2, 2}, {2, 6}, {0, 6}}}});
  const std::array crop{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {6, 3}, .top = 3}};
  options = {};
  options.limits.max_polygon_points = 7;
  EXPECT_THROW(transform_sample(sample, crop, options), ResourceExhausted);
  options = {};
  options.limits.max_polygon_fragments = 1;
  EXPECT_THROW(transform_sample(sample, crop, options), ResourceExhausted);
  options = {};
  options.limits.max_geometry_checks = 1;
  EXPECT_THROW(transform_sample(sample, crop, options), ResourceExhausted);
}

VisionSample Panoptic() {
  auto sample = Sample(3, 2);
  sample.mask_target.emplace();
  auto& mask = *sample.mask_target;
  mask.width = 3;
  mask.height = 2;
  mask.encoding = MaskEncoding::panoptic_rgb;
  mask.void_id = 0;
  mask.ignore_id = -7;
  mask.ids = torch::tensor({1, 1, 16777215, 0, -7, 16777215}, torch::kInt64)
                 .reshape({2, 3});
  auto record = Record(sample);
  record->panoptic_mask_path = "segments.png";
  record->objects.push_back(
      {.source_id = 1, .box = ImageBox{0, 0, 2, 1}, .area = 2});
  record->objects.push_back({.source_id = 16777215,
                             .box = ImageBox{2, 0, 3, 2},
                             .area = 2,
                             .crowd = true});
  return sample;
}

TEST(PanopticTest, BindsStridedIdsAndPreservesSourceTable) {
  auto sample = Panoptic();
  auto mask = *sample.mask_target;
  mask.ids = torch::stack({mask.ids, mask.ids}, 2).select(2, 0);
  ASSERT_FALSE(mask.ids.is_contiguous());
  const auto bound = bind_panoptic_mask(*sample.annotation, mask, 1);
  EXPECT_TRUE(bound.panoptic_mask_materialized);
  EXPECT_EQ(bound.objects[0].mask_pixel_area, 2U);
  EXPECT_EQ(bound.objects[1].mask_pixel_area, 2U);
  EXPECT_EQ(bound.objects[1].source_id, 16777215);
  EXPECT_TRUE(bound.objects[1].crowd);
  EXPECT_FALSE(sample.annotation->panoptic_mask_materialized);
  EXPECT_FALSE(sample.annotation->objects[0].mask_pixel_area);
}

TEST(PanopticTest, RejectsUnknownDuplicateMissingAndReservedSegmentIds) {
  const auto sample = Panoptic();
  auto record = *sample.annotation;
  auto mask = *sample.mask_target;
  record.objects[1].source_id = 1;
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
  record = *sample.annotation;
  record.objects[1].source_id = 2;
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
  record = *sample.annotation;
  record.objects.push_back({.source_id = 2});
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
  record = *sample.annotation;
  mask.void_id = 1;
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
  mask = *sample.mask_target;
  record.objects[0].source_id = 16777216;
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
}

TEST(PanopticTest,
     SourceVerificationIsExplicitButCurrentStatisticsCannotBeStale) {
  const auto sample = Panoptic();
  auto record = *sample.annotation;
  const auto& mask = *sample.mask_target;
  record.objects[0].area = 99;
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1), SerializationError);
  PanopticBindOptions options;
  options.verify_declared_area = false;
  record.objects[0].box = ImageBox{0, 0, 3, 2};
  EXPECT_THROW(bind_panoptic_mask(record, mask, 1, options),
               SerializationError);
  options.verify_declared_box = false;
  auto bound = bind_panoptic_mask(record, mask, 1, options);
  EXPECT_EQ(bound.objects[0].area, 99);
  EXPECT_EQ(bound.objects[0].mask_pixel_area, 2U);
  EXPECT_DOUBLE_EQ(bound.objects[0].box->x_max, 2);
  bound.objects[0].mask_pixel_area = 1;
  EXPECT_THROW(bind_panoptic_mask(bound, mask, 1, options), SerializationError);
  bound.objects[0].mask_pixel_area = 2;
  bound.objects[0].box->x_max = 3;
  EXPECT_THROW(bind_panoptic_mask(bound, mask, 1, options), SerializationError);
}

TEST(PanopticTest,
     CropPadFlipUpdatesPixelsAndTableAndSupportsAnotherTransform) {
  const auto sample = Panoptic();
  const std::array steps{
      ImageGeometryStep{
          .kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1},
      ImageGeometryStep{.kind = ImageGeometryKind::pad, .left = 1},
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  JointTransformOptions options;
  options.mask_padding_id = 0;
  const auto result = transform_sample(sample, steps, options);
  EXPECT_TRUE(torch::equal(
      result.mask_target->ids,
      torch::tensor({16777215, 1, 0, 16777215, -7, 0}, torch::kInt64)
          .reshape({2, 3})));
  ASSERT_EQ(result.annotation->objects.size(), 2U);
  const auto& first = result.annotation->objects[0];
  EXPECT_EQ(first.mask_pixel_area, 1U);
  EXPECT_EQ(first.area, 2);
  EXPECT_DOUBLE_EQ(first.box->x_min, 1);
  EXPECT_DOUBLE_EQ(first.box->x_max, 2);
  EXPECT_TRUE(result.annotation->panoptic_mask_materialized);
  EXPECT_EQ(result.annotation->panoptic_mask_path, "segments.png");
  EXPECT_NO_THROW(
      bind_panoptic_mask(*result.annotation, *result.mask_target, 1));
  EXPECT_NO_THROW(transform_sample(result, {}));
  EXPECT_EQ(sample.mask_target->ids[0][0].item<int>(), 1);
}

TEST(PanopticTest, FilteringReplacesRemovedIdsWithExplicitVoid) {
  const auto sample = Panoptic();
  const std::array crop{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1}};
  JointTransformOptions options;
  options.min_region_visibility = 0.501;
  const auto result = transform_sample(sample, crop, options);
  ASSERT_EQ(result.annotation->objects.size(), 1U);
  EXPECT_EQ(result.annotation->objects[0].source_id, 16777215);
  EXPECT_TRUE(
      torch::equal(result.mask_target->ids,
                   torch::tensor({0, 16777215, -7, 16777215}, torch::kInt64)
                       .reshape({2, 2})));
  EXPECT_NO_THROW(
      bind_panoptic_mask(*result.annotation, *result.mask_target, 1));
  options.min_region_area = 3;
  const auto empty = transform_sample(sample, crop, options);
  EXPECT_TRUE(empty.annotation->objects.empty());
  EXPECT_EQ((empty.mask_target->ids != 0).sum().item<int>(),
            1);  // Ignore stays -7.
  EXPECT_NO_THROW(bind_panoptic_mask(*empty.annotation, *empty.mask_target, 1));
}

TEST(PanopticTest, PaddingRequiresVoidAndMissingAssociationFails) {
  auto sample = Panoptic();
  const std::array pad{
      ImageGeometryStep{.kind = ImageGeometryKind::pad, .right = 1}};
  JointTransformOptions options;
  options.mask_padding_id = 1;
  EXPECT_THROW(transform_sample(sample, pad, options), InvalidArgument);
  options.mask_padding_id = -7;
  EXPECT_THROW(transform_sample(sample, pad, options), InvalidArgument);
  sample.mask_target.reset();
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
}

TEST(PanopticTest, BoundedBindingAndCancellationRejectBeforePublication) {
  const auto sample = Panoptic();
  PanopticBindOptions options;
  options.max_working_bytes = 255;
  EXPECT_THROW(
      bind_panoptic_mask(*sample.annotation, *sample.mask_target, 1, options),
      ResourceExhausted);
  options = {};
  options.limits.max_pixels = 5;
  EXPECT_THROW(
      bind_panoptic_mask(*sample.annotation, *sample.mask_target, 1, options),
      ResourceExhausted);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(bind_panoptic_mask(*sample.annotation, *sample.mask_target, 1,
                                  {}, stop.get_token()),
               Cancelled);
  JointTransformOptions joint;
  joint.limits.max_mask_work_pixels = 29;  // Identity mask + four scans = 30.
  EXPECT_THROW(transform_sample(sample, {}, joint), ResourceExhausted);
  joint.limits.max_mask_work_pixels = 30;
  EXPECT_NO_THROW(transform_sample(sample, {}, joint));
}

TEST(PanopticTest, JointMappingKeepsOnlyOutputAnnotationCharged) {
  const auto sample = Panoptic();
  JointTransformOptions options;
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  auto output = transform_sample(sample, {}, options);
  const auto live = budget->usage(ImageResourceKind::annotation_bytes);
  EXPECT_GT(live.current, 0U);
  EXPECT_GT(live.peak, live.current);
  auto retained = output;
  output = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current,
            live.current);
  retained = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  ImageResourceBudgetLimits limits;
  limits.max_annotation_bytes = 255;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  const auto& record = *sample.annotation;
  const auto record_bytes = sizeof(VisionAnnotationRecord) +
                            record.objects.size() * sizeof(VisionObject) +
                            record.image_path.size() +
                            record.panoptic_mask_path->size() + 2U;
  limits.max_annotation_bytes = 2 * record_bytes + 255U;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak,
            2 * record_bytes);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);
}

TEST(JointRegionTest, IdentityBindsEveryNewPolicyAndResourceLimit) {
  const auto identity = joint_transform_identity({}, {});
  JointTransformOptions options;
  options.min_region_area = 1;
  EXPECT_NE(joint_transform_identity({}, options), identity);
  options = {};
  options.min_region_visibility = 0.5;
  EXPECT_NE(joint_transform_identity({}, options), identity);
  options = {};
  options.oriented_box_crop = OrientedBoxCrop::discard;
  EXPECT_NE(joint_transform_identity({}, options), identity);
  for (int i = 0; i < 5; ++i) {
    options = {};
    switch (i) {
      case 0:
        --options.limits.max_polygon_points;
        break;
      case 1:
        --options.limits.max_polygon_fragments;
        break;
      case 2:
        --options.limits.max_geometry_checks;
        break;
      case 3:
        --options.limits.max_rle_runs;
        break;
      case 4:
        --options.limits.max_mask_work_pixels;
        break;
    }
    EXPECT_NE(joint_transform_identity({}, options), identity);
  }
}
}  // namespace
