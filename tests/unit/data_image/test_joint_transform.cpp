// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bicubic_reference.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <string>

namespace {
using namespace jyppx::liuding::data::image;
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;

VisionSample Grid(std::uint32_t width = 4, std::uint32_t height = 3) {
  VisionSample sample;
  sample.width = width;
  sample.height = height;
  sample.pixels.color = PixelColorModel::gray;
  sample.image =
      torch::arange(static_cast<std::int64_t>(width) * height, torch::kInt64)
          .reshape({1, height, width})
          .to(torch::kUInt8);
  sample.geometry.emplace(ImageExtent{width, height});
  sample.image_path = "grid.png";
  return sample;
}

void AddMask(VisionSample& sample) {
  sample.mask_target.emplace();
  sample.mask_target->ids =
      sample.image[0].to(torch::kInt64) + 9007199254740993LL;
  sample.mask_target->width = sample.width;
  sample.mask_target->height = sample.height;
  sample.mask_target->ignore_id = -9;
  sample.mask_target->void_id = -10;
}

void AddPose(VisionSample& sample) {
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{
          {.source_id = 41,
           .name = "person",
           .keypoint_names = {"left", "right", "absent"}}});
  auto record = std::make_shared<VisionAnnotationRecord>();
  record->source_image_id = 123;
  record->image_path = sample.image_path;
  record->width = sample.width;
  record->height = sample.height;
  VisionObject object;
  object.source_id = 77;
  object.box = ImageBox{0, 0, 2, 2};
  object.area = 123;
  object.group_id = 9;
  object.crowd = object.ignore = object.difficult = object.truncated = true;
  object.keypoints = {{{1.5, 1}, 2}, {{0, 1}, 1}, {{123, 456}, 0}};
  record->objects.push_back(object);
  sample.annotation = record;
}

const std::array<ImageGeometryStep, 6> kSequence{
    {{.kind = ImageGeometryKind::resize, .extent = {8, 6}},
     {.kind = ImageGeometryKind::crop, .extent = {4, 4}, .left = 2, .top = 1},
     {.kind = ImageGeometryKind::pad, .left = 1, .top = 2, .right = 1},
     {.kind = ImageGeometryKind::horizontal_flip},
     {.kind = ImageGeometryKind::vertical_flip},
     {.kind = ImageGeometryKind::crop, .extent = {6, 6}}}};

JointTransformOptions PoseOptions() {
  JointTransformOptions options;
  options.image_padding = 17;
  options.mask_padding_id = -9;
  options.horizontal_keypoint_permutations = {{1, 0, 2}};
  options.vertical_keypoint_permutations = {{0, 1, 2}};
  return options;
}

TEST(JointTransformTest, GridLabelsBoxesPoseAndTraceUseOneAnalyticGeometry) {
  auto sample = Grid();
  AddMask(sample);
  AddPose(sample);
  const auto result = transform_sample(sample, kSequence, PoseOptions());
  ASSERT_EQ(result.width, 6U);
  ASSERT_EQ(result.height, 6U);
  ASSERT_TRUE(result.mask_target);
  EXPECT_EQ(result.mask_target->ignore_id, -9);
  EXPECT_EQ(result.mask_target->void_id, -10);
  EXPECT_EQ(result.mask_target->width, 6U);
  for (int y = 0; y < 6; ++y)
    for (int x = 0; x < 6; ++x) {
      // Independently undo flips, padding and crop; the initial resize repeats
      // 2x.
      const int rx = 5 - x - 1, ry = 5 - y - 2;
      const bool real = rx >= 0 && rx < 4 && ry >= 0 && ry < 4;
      const int source = real ? ((ry + 1) / 2) * 4 + (rx + 2) / 2 : 0;
      EXPECT_EQ(result.image[0][y][x].item<int>(), real ? source : 17);
      EXPECT_EQ(result.mask_target->ids[y][x].item<std::int64_t>(),
                real ? 9007199254740993LL + source : -9);
    }
  ASSERT_TRUE(result.annotation);
  ASSERT_EQ(result.annotation->objects.size(), 1U);
  const auto& object = result.annotation->objects[0];
  ASSERT_TRUE(object.box);
  EXPECT_DOUBLE_EQ(object.box->x_min, 3);
  EXPECT_DOUBLE_EQ(object.box->x_max, 5);
  EXPECT_DOUBLE_EQ(object.box->y_min, 1);
  EXPECT_DOUBLE_EQ(object.box->y_max, 4);
  EXPECT_EQ(object.keypoints[0].visibility, 0);
  EXPECT_EQ(object.keypoints[1].visibility, 2);
  EXPECT_DOUBLE_EQ(object.keypoints[1].point.x, 4);
  EXPECT_DOUBLE_EQ(object.keypoints[1].point.y, 3);
  EXPECT_EQ(object.keypoints[2].visibility, 0);
  EXPECT_DOUBLE_EQ(object.keypoints[2].point.x, 0);
  EXPECT_EQ(object.source_id, 77);
  EXPECT_EQ(object.category, 0U);
  EXPECT_EQ(object.area, 123);
  EXPECT_EQ(object.group_id, 9);
  EXPECT_TRUE(object.crowd && object.ignore && object.difficult &&
              object.truncated);
  EXPECT_EQ(result.categories, sample.categories);
  EXPECT_EQ(result.annotation->source_image_id, 123);
  EXPECT_EQ(result.annotation->width, 6U);
  ASSERT_TRUE(result.geometry);
  const auto restored = result.geometry->inverse_point({4, 3});
  ASSERT_TRUE(restored);
  EXPECT_DOUBLE_EQ(restored->x, 1.5);
  EXPECT_DOUBLE_EQ(restored->y, 1);
  EXPECT_FALSE(result.geometry->inverse_point({0, 0}));
  EXPECT_EQ(sample.annotation->width, 4U);
  EXPECT_EQ(sample.annotation->objects[0].keypoints[0].visibility, 2);
  EXPECT_EQ(sample.mask_target->ids[0][0].item<std::int64_t>(),
            9007199254740993LL);
}

TEST(JointTransformTest, OutputLeasesOutliveSourceWithoutRetainingItsCharge) {
  auto budget = std::make_shared<ImageResourceBudget>();
  auto sample = Grid();
  sample.image_resource_leases = std::make_shared<ImageResourceLeaseSet>();
  sample.image_resource_leases->push_back(
      budget->reserve(ImageResourceKind::pixels, 12));
  sample.image_resource_leases->push_back(
      budget->reserve(ImageResourceKind::decoded_bytes, 12));
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  auto output = transform_sample(sample, steps);
  EXPECT_NE(output.image_resource_leases, sample.image_resource_leases);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 24U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).peak, 36U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 24U);
  sample = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 12U);
  output = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
}

TEST(JointTransformTest, TargetUsesItsOwnBudgetAndFailedStepRollsBack) {
  auto image_budget = std::make_shared<ImageResourceBudget>();
  ImageResourceBudgetLimits mask_limits;
  mask_limits.max_decoded_bytes = 288;
  auto mask_budget = std::make_shared<ImageResourceBudget>(mask_limits);
  auto sample = Grid();
  AddMask(sample);
  sample.image_resource_leases = std::make_shared<ImageResourceLeaseSet>();
  sample.image_resource_leases->push_back(
      image_budget->reserve(ImageResourceKind::decoded_bytes, 12));
  sample.mask_target->resource_leases =
      std::make_shared<ImageResourceLeaseSet>();
  sample.mask_target->resource_leases->push_back(
      mask_budget->reserve(ImageResourceKind::decoded_bytes, 96));
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  auto output = transform_sample(sample, steps);
  EXPECT_EQ(image_budget->usage(ImageResourceKind::decoded_bytes).current, 24U);
  EXPECT_EQ(mask_budget->usage(ImageResourceKind::decoded_bytes).current, 192U);
  output = {};
  EXPECT_EQ(image_budget->usage(ImageResourceKind::decoded_bytes).current, 12U);
  EXPECT_EQ(mask_budget->usage(ImageResourceKind::decoded_bytes).current, 96U);

  ImageResourceBudgetLimits tight_limits;
  tight_limits.max_decoded_bytes = 24;
  auto tight_budget = std::make_shared<ImageResourceBudget>(tight_limits);
  sample.image_resource_leases = std::make_shared<ImageResourceLeaseSet>();
  sample.image_resource_leases->push_back(
      tight_budget->reserve(ImageResourceKind::decoded_bytes, 12));
  EXPECT_THROW(transform_sample(sample, steps), ResourceExhausted);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::decoded_bytes).current, 12U);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::pixels).current, 0U);
  std::stop_source cancelled;
  cancelled.request_stop();
  EXPECT_THROW(transform_sample(sample, steps, {}, cancelled.get_token()),
               Cancelled);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::decoded_bytes).current, 12U);
  sample = {};
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(mask_budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(image_budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
}

TEST(JointTransformTest, BoxFilteringMeasuresFinalAreaAndOriginalVisibility) {
  auto sample = Grid();
  AddPose(sample);
  auto options = PoseOptions();
  options.min_box_visibility = 0.375;
  options.min_box_area = 6;
  EXPECT_EQ(
      transform_sample(sample, kSequence, options).annotation->objects.size(),
      1U);
  options.min_box_visibility = 0.376;
  EXPECT_TRUE(
      transform_sample(sample, kSequence, options).annotation->objects.empty());
  options.min_box_visibility = 0;
  options.min_box_area = 6.01;
  EXPECT_TRUE(
      transform_sample(sample, kSequence, options).annotation->objects.empty());
}

TEST(JointTransformTest,
     NearestLabelsKeepInt64ExtremesAndExactCenterSelection) {
  auto sample = Grid(3, 1);
  AddMask(sample);
  sample.mask_target->ids =
      torch::tensor({std::numeric_limits<std::int64_t>::max(),
                     std::int64_t{9007199254740993LL},
                     std::numeric_limits<std::int64_t>::min()},
                    torch::kInt64)
          .reshape({1, 3});
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {5, 1}}};
  const auto result = transform_sample(sample, steps);
  const std::array<int, 5> source{0, 0, 1, 2, 2};
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(result.mask_target->ids[0][i].item<std::int64_t>(),
              sample.mask_target->ids[0][source[i]].item<std::int64_t>());
}

TEST(JointTransformTest, BilinearPhotosNeverInterpolateDiscreteTargets) {
  auto sample = Grid(2, 2);
  AddMask(sample);
  sample.image = sample.image.to(torch::kFloat32);
  sample.pixels.dtype = torch::kFloat32;
  sample.pixels.range = PixelValueRange::unbounded;
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bilinear;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {3, 3}}};
  const auto result = transform_sample(sample, steps, options);
  EXPECT_NEAR(result.image[0][1][1].item<double>(), 1.5, 1e-6);
  EXPECT_EQ(result.mask_target->ids[1][1].item<std::int64_t>(),
            9007199254740996LL);
  EXPECT_EQ(result.image.scalar_type(), torch::kFloat32);
}

TEST(JointTransformTest, BilinearDownsamplingUsesAntialiasedTriangleWeights) {
  auto sample = Grid(4, 4);
  sample.image = sample.image.to(torch::kFloat64);
  sample.pixels.dtype = torch::kFloat64;
  sample.pixels.range = PixelValueRange::unbounded;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {2, 2}}};
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bilinear;
  const auto result = transform_sample(sample, steps, options);
  // The left output samples input positions 0/1/2 with weights 3/7,3/7,1/7.
  // The right samples 1/2/3 with weights 1/7,3/7,3/7; separable in x and y.
  EXPECT_NEAR(result.image[0][0][0].item<double>(), 25.0 / 7.0, 1e-12);
  EXPECT_NEAR(result.image[0][0][1].item<double>(), 36.0 / 7.0, 1e-12);
  EXPECT_NEAR(result.image[0][1][0].item<double>(), 69.0 / 7.0, 1e-12);
  EXPECT_NEAR(result.image[0][1][1].item<double>(), 80.0 / 7.0, 1e-12);
}

TEST(JointTransformTest, Uint16PhotoBilinearRoundsWithoutTruncatingHighBits) {
  auto sample = Grid(2, 1);
  sample.image = torch::tensor({0, 65535}, torch::kInt32)
                     .to(torch::kUInt16)
                     .reshape({1, 1, 2});
  sample.pixels.dtype = torch::kUInt16;
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bilinear;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {3, 1}}};
  const auto result = transform_sample(sample, steps, options);
  const auto values = result.image.to(torch::kInt32);
  EXPECT_EQ(values[0][0][0].item<int>(), 0);
  EXPECT_EQ(values[0][0][1].item<int>(), 32768);
  EXPECT_EQ(values[0][0][2].item<int>(), 65535);
}

TEST(JointTransformTest, BicubicMatchesIndependentCubicReference) {
  using namespace bicubic_reference;
  const auto check = [](std::span<const double> source, std::uint32_t width,
                        std::uint32_t height, ImageExtent after,
                        std::span<const double> golden) {
    auto sample = Grid(width, height);
    sample.pixels.dtype = torch::kFloat64;
    sample.pixels.range = PixelValueRange::unbounded;
    sample.image =
        torch::tensor(std::vector<double>(source.begin(), source.end()),
                      torch::kFloat64)
            .reshape({1, height, width});
    JointTransformOptions options;
    options.interpolation = ImageInterpolation::bicubic;
    const std::array steps{
        ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = after}};
    const auto result = transform_sample(sample, steps, options);
    const auto expected =
        torch::tensor(std::vector<double>(golden.begin(), golden.end()),
                      torch::kFloat64)
            .reshape({1, after.height, after.width});
    EXPECT_TRUE(torch::allclose(result.image, expected, 1e-12, 1e-12));
  };
  check(kUpInput, 3, 2, {7, 4}, kUpExpected);
  check(kDownInput, 9, 7, {4, 3}, kDownExpected);
}

TEST(JointTransformTest, BicubicKeepsPairedHwcAndExactMaskIds) {
  using namespace bicubic_reference;
  auto sample = Grid(3, 2);
  AddMask(sample);
  sample.mask_target->ids =
      torch::tensor(
          std::vector<std::int64_t>{
              std::numeric_limits<std::int64_t>::max(), 9007199254740993LL,
              std::numeric_limits<std::int64_t>::min(), -9LL, -10LL, 7LL},
          torch::kInt64)
          .reshape({2, 3});
  const auto original_ids = sample.mask_target->ids.clone();
  sample.image =
      torch::tensor(std::vector<double>(kUpInput.begin(), kUpInput.end()),
                    torch::kFloat64)
          .reshape({1, 2, 3});
  sample.pixels.dtype = torch::kFloat64;
  sample.pixels.range = PixelValueRange::unbounded;
  const auto masked_sample = sample;
  sample.mask_target.reset();
  sample.paired_target = (sample.image * 2 + 3).permute({1, 2, 0});
  sample.paired_target_pixels = sample.pixels;
  sample.paired_target_pixels->layout = ImageLayout::hwc;
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bicubic;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {7, 4}}};
  const auto result = transform_sample(sample, steps, options);
  const auto expected =
      torch::tensor(std::vector<double>(kUpExpected.begin(), kUpExpected.end()),
                    torch::kFloat64)
          .reshape({1, 4, 7});
  ASSERT_TRUE(result.paired_target);
  EXPECT_TRUE(torch::allclose(result.image, expected, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(*result.paired_target,
                              (expected * 2 + 3).permute({1, 2, 0}), 1e-12,
                              1e-12));
  const auto masked_result = transform_sample(masked_sample, steps, options);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 7; ++x)
      EXPECT_EQ(masked_result.mask_target->ids[y][x].item<std::int64_t>(),
                original_ids[(2 * y + 1) * 2 / 8][(2 * x + 1) * 3 / 14]
                    .item<std::int64_t>());
  EXPECT_TRUE(torch::equal(masked_sample.mask_target->ids, original_ids));
  EXPECT_EQ(result.geometry->output_extent().width, 7U);
  EXPECT_EQ(result.geometry->output_extent().height, 4U);
}

TEST(JointTransformTest, BicubicSaturatesIntegersAndRetainsFloatingOvershoot) {
  using namespace bicubic_reference;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {5, 1}}};
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bicubic;
  const auto golden = torch::tensor(std::vector<double>(kStepExpected.begin(),
                                                        kStepExpected.end()),
                                    torch::kFloat64)
                          .reshape({1, 1, 5});
  for (const auto dtype : {torch::kUInt8, torch::kUInt16}) {
    auto sample = Grid(2, 1);
    const auto maximum = dtype == torch::kUInt8 ? 255 : 65535;
    sample.image =
        torch::tensor({0, maximum}, torch::kInt32).to(dtype).reshape({1, 1, 2});
    sample.pixels.dtype = dtype;
    const auto result = transform_sample(sample, steps, options);
    const auto expected =
        (golden * (maximum / 65535.0)).round().clamp(0, maximum);
    EXPECT_TRUE(torch::equal(result.image.to(torch::kInt64),
                             expected.to(torch::kInt64)));
    EXPECT_EQ(result.image.scalar_type(), dtype);
  }
  auto sample = Grid(2, 1);
  sample.image = torch::tensor({0.0F, 1.0F}).reshape({1, 1, 2});
  sample.pixels.dtype = torch::kFloat32;
  sample.pixels.range = PixelValueRange::unit_interval;
  const auto result = transform_sample(sample, steps, options);
  EXPECT_TRUE(torch::allclose(result.image.to(torch::kFloat64), golden / 65535,
                              1e-6, 1e-7));
  EXPECT_LT(result.image.min().item<double>(), 0);
  EXPECT_GT(result.image.max().item<double>(), 1);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unit_interval);
}

TEST(JointTransformTest,
     BicubicAdmitsWorkingAndScratchBudgetsBeforePublishing) {
  const auto sample = Grid(2, 1);
  const auto original = sample.image.clone();
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {5, 1}}};
  JointTransformOptions options;
  options.interpolation = ImageInterpolation::bicubic;
  options.limits.max_working_bytes = 5 * (2 + 32) - 1;
  EXPECT_THROW(transform_sample(sample, steps, options), ResourceExhausted);
  options.limits.max_working_bytes += 1;
  ImageResourceBudgetLimits limits;
  limits.max_codec_scratch_bytes = 5 * 32 - 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_sample(sample, steps, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).peak, 0U);
  limits.max_codec_scratch_bytes += 1;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto result = transform_sample(sample, steps, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).peak, 160U);
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 5U);
  result = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_TRUE(torch::equal(sample.image, original));
}

TEST(JointTransformTest, StridedHwcAndChwPreserveAllSupportedPhotoDtypes) {
  for (const auto dtype :
       {torch::kUInt8, torch::kUInt16, torch::kFloat16, torch::kBFloat16,
        torch::kFloat32, torch::kFloat64}) {
    auto sample = Grid();
    sample.image = torch::arange(36, torch::kInt32)
                       .reshape({3, 3, 4})
                       .to(dtype)
                       .permute({1, 2, 0});
    sample.pixels.dtype = dtype;
    sample.pixels.color = PixelColorModel::rgb;
    sample.pixels.layout = ImageLayout::hwc;
    if (dtype != torch::kUInt8 && dtype != torch::kUInt16)
      sample.pixels.range = PixelValueRange::unbounded;
    const std::array steps{
        ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
    const auto result = transform_sample(sample, steps);
    EXPECT_TRUE(result.image.is_contiguous());
    EXPECT_EQ(result.image.scalar_type(), dtype);
    EXPECT_NO_THROW(validate_pixels(result.image, result.pixels));
    const auto values = result.image.to(torch::kInt32);
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x)
        for (int c = 0; c < 3; ++c)
          EXPECT_EQ(values[y][x][c].item<int>(), c * 12 + y * 4 + 3 - x);
  }
}

TEST(JointTransformTest, AlignedPairedPhotosKeepIndependentPixelPolicies) {
  auto sample = Grid();
  sample.paired_target = sample.image.to(torch::kFloat32) / 255;
  sample.paired_target_pixels = sample.pixels;
  sample.paired_target_pixels->dtype = torch::kFloat32;
  sample.paired_target_pixels->range = PixelValueRange::unit_interval;
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto result = transform_sample(sample, steps);
  EXPECT_TRUE(torch::allclose(*result.paired_target,
                              result.image.to(torch::kFloat32) / 255));
  sample.paired_target = sample.paired_target->slice(2, 0, 2);
  EXPECT_THROW(transform_sample(sample, steps), InvalidArgument);
  sample.paired_target.reset();
  EXPECT_THROW(transform_sample(sample, steps), InvalidArgument);
}

TEST(JointTransformTest, IdentityOwnsStorageAndFurtherTransformsComposeTrace) {
  auto sample = Grid();
  AddMask(sample);
  AddPose(sample);
  auto result = transform_sample(sample, {});
  EXPECT_NE(result.image.data_ptr(), sample.image.data_ptr());
  EXPECT_NE(result.mask_target->ids.data_ptr(),
            sample.mask_target->ids.data_ptr());
  EXPECT_NE(result.annotation, sample.annotation);
  result.image.fill_(99);
  EXPECT_EQ(sample.image[0][0][0].item<int>(), 0);
  const std::array first{
      ImageGeometryStep{.kind = ImageGeometryKind::resize, .extent = {8, 6}}};
  const std::array second{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {4, 4}, .left = 2, .top = 1}};
  result = transform_sample(transform_sample(sample, first), second);
  ASSERT_TRUE(result.geometry->inverse_point({1, 1}));
  EXPECT_DOUBLE_EQ(result.geometry->inverse_point({1, 1})->x, 1.5);
  EXPECT_DOUBLE_EQ(result.geometry->inverse_point({1, 1})->y, 1);
}

TEST(JointTransformTest, MissingFlipPermutationAndMalformedTargetsAreRejected) {
  auto sample = Grid();
  AddPose(sample);
  const std::array flip{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  EXPECT_THROW(transform_sample(sample, flip), InvalidArgument);
  auto options = PoseOptions();
  options.horizontal_keypoint_permutations = {{1, 1, 2}};
  EXPECT_THROW(transform_sample(sample, flip, options), InvalidArgument);
  auto record = std::make_shared<VisionAnnotationRecord>(*sample.annotation);
  sample.annotation = record;
  record->objects[0].polygons = {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
  EXPECT_THROW(transform_sample(sample, {}, PoseOptions()), InvalidArgument);
  record->objects[0].polygons.clear();
  record->objects[0].oriented_box.emplace();
  EXPECT_THROW(transform_sample(sample, {}, PoseOptions()), InvalidArgument);
  record->objects[0].oriented_box.reset();
  record->objects[0].rle.emplace();
  EXPECT_THROW(transform_sample(sample, {}, PoseOptions()), InvalidArgument);
  record->objects[0].rle.reset();
  record->panoptic_mask_path = "segments.png";
  EXPECT_THROW(transform_sample(sample, {}, PoseOptions()), InvalidArgument);
  EXPECT_EQ(sample.image[0][0][0].item<int>(), 0);
}

TEST(JointTransformTest, EmptyPoseAndDiscardedBoxesHaveExplicitFiltering) {
  auto sample = Grid();
  AddPose(sample);
  const std::array crop{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {1, 1}, .left = 1, .top = 1}};
  auto options = PoseOptions();
  auto record = std::make_shared<VisionAnnotationRecord>(*sample.annotation);
  record->objects[0].keypoints = {{{0, 0}, 2}, {{0, 0}, 1}, {{0, 0}, 0}};
  sample.annotation = record;
  EXPECT_EQ(transform_sample(sample, crop, options).annotation->objects.size(),
            1U);
  options.drop_empty_pose = true;
  EXPECT_TRUE(
      transform_sample(sample, crop, options).annotation->objects.empty());
  const std::array outside{ImageGeometryStep{
      .kind = ImageGeometryKind::crop, .extent = {1, 1}, .left = 3, .top = 2}};
  EXPECT_TRUE(transform_sample(sample, outside, PoseOptions())
                  .annotation->objects.empty());
}

TEST(JointTransformTest, RejectsInvalidGeometryShapesPaddingAndAutograd) {
  auto sample = Grid();
  const std::array inactive{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip, .left = 1}};
  EXPECT_THROW(transform_sample(sample, inactive), InvalidArgument);
  const std::array outside{
      ImageGeometryStep{.kind = ImageGeometryKind::crop, .extent = {5, 1}}};
  EXPECT_THROW(transform_sample(sample, outside), InvalidArgument);
  auto options = JointTransformOptions{};
  options.image_padding = 0.5;
  EXPECT_THROW(transform_sample(sample, {}, options), InvalidArgument);
  options = {};
  options.min_box_visibility = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(transform_sample(sample, {}, options), InvalidArgument);
  sample.geometry = GeometryTrace({99, 99});
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  sample = Grid();
  sample.image = sample.image.to(torch::kFloat32).set_requires_grad(true);
  sample.pixels.dtype = torch::kFloat32;
  sample.pixels.range = PixelValueRange::unbounded;
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
}

TEST(JointTransformTest, MaskPaddingIsExplicitAndBudgetsPreflightEveryStage) {
  auto sample = Grid();
  AddMask(sample);
  const std::array pad{
      ImageGeometryStep{.kind = ImageGeometryKind::pad, .left = 1}};
  EXPECT_THROW(transform_sample(sample, pad), InvalidArgument);
  auto options = JointTransformOptions{};
  options.mask_padding_id = std::numeric_limits<std::int64_t>::min();
  const auto result = transform_sample(sample, pad, options);
  EXPECT_EQ(result.mask_target->ids[0][0].item<std::int64_t>(),
            std::numeric_limits<std::int64_t>::min());
  options.limits.max_output_bytes =
      108;  // Original 12*(1+8) bytes fits; padded does not.
  EXPECT_THROW(transform_sample(sample, pad, options), ResourceExhausted);
  options = {};
  options.limits.max_working_bytes = 1;
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  options = {};
  options.limits.max_dimension = 3;
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  std::vector<ImageGeometryStep> many(
      129, {.kind = ImageGeometryKind::horizontal_flip});
  EXPECT_THROW(transform_sample(sample, many), ResourceExhausted);
  auto trace = *sample.geometry;
  for (int i = 0; i < 128; ++i) trace = trace.flipped_horizontal();
  sample.geometry = trace;
  const std::array flip{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  EXPECT_THROW(transform_sample(sample, flip), ResourceExhausted);
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(transform_sample(sample, {}, {}, source.get_token()), Cancelled);
}

TEST(JointTransformTest, SamplerUsesStableSlotsAndNeverConsumesTorchRng) {
  RandomGeometryOptions random{.resize = ImageExtent{8, 6},
                               .crop = ImageExtent{3, 2},
                               .horizontal_flip_probability = 1,
                               .vertical_flip_probability = 0};
  const auto identity = joint_transform_identity(random, {});
  const SampleRandom rng({.dataset_id = "dataset-v1",
                          .sample_id = "image-1",
                          .transform_id = identity});
  torch::manual_seed(7);
  const auto expected = torch::rand({8});
  torch::manual_seed(7);
  const auto steps = sample_geometry({4, 3}, random, rng);
  ASSERT_EQ(steps.size(), 3U);
  EXPECT_EQ(steps[1].left, rng.bounded(6, 0));
  EXPECT_EQ(steps[1].top, rng.bounded(5, 1));
  EXPECT_EQ(steps[2].kind, ImageGeometryKind::horizontal_flip);
  EXPECT_TRUE(torch::equal(expected, torch::rand({8})));
  random.horizontal_flip_probability = 0;
  const auto unchanged_crop = sample_geometry({4, 3}, random, rng);
  EXPECT_EQ(unchanged_crop[1].left, steps[1].left);
  EXPECT_EQ(unchanged_crop[1].top, steps[1].top);
  EXPECT_NE(joint_transform_identity(random, {}), identity);
  random.crop = ImageExtent{99, 1};
  EXPECT_THROW(sample_geometry({4, 3}, random, rng), InvalidArgument);
}

TEST(JointTransformTest, ConfigurationIdentityIncludesLimitsPoseAndSignedZero) {
  RandomGeometryOptions random;
  JointTransformOptions joint;
  const auto expected = joint_transform_identity(random, joint);
  joint.image_padding = -0.0;
  EXPECT_EQ(joint_transform_identity(random, joint), expected);
  joint.mask_padding_id = 0;
  EXPECT_NE(joint_transform_identity(random, joint), expected);
  joint = {};
  joint.limits.max_pixels--;
  EXPECT_NE(joint_transform_identity(random, joint), expected);
  joint = {};
  joint.horizontal_keypoint_permutations = {{1, 0}};
  EXPECT_NE(joint_transform_identity(random, joint), expected);
}

TEST(JointTransformTest, MetadataBudgetsAndMalformedMasksAreRejected) {
  auto sample = Grid();
  AddPose(sample);
  auto options = JointTransformOptions{};
  options.limits.max_keypoints = 2;
  EXPECT_THROW(transform_sample(sample, {}, options), ResourceExhausted);
  sample = Grid();
  AddMask(sample);
  sample.mask_target->ids = sample.mask_target->ids.to(torch::kFloat32);
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  sample = Grid();
  AddMask(sample);
  sample.mask_target->width++;
  EXPECT_THROW(transform_sample(sample, {}), InvalidArgument);
  sample = Grid();
  sample.image = sample.image.to(torch::kFloat32);
  sample.pixels.dtype = torch::kFloat32;
  sample.pixels.range = PixelValueRange::unit_interval;
  options = {};
  options.image_padding = 1.1;
  EXPECT_THROW(transform_sample(sample, {}, options), InvalidArgument);
}

class MemoryState final : public StateWriter, public StateReader {
 public:
  explicit MemoryState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs_[std::string(name)] = {bytes.begin(), bytes.end()};
  }
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto& bytes = blobs_.at(std::string(name));
    if (bytes.size() > max_bytes) throw std::runtime_error("state budget");
    return bytes;
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

std::unique_ptr<DataPipeline> Pipeline(std::size_t workers,
                                       std::string identity = "",
                                       std::string dataset = "dataset-v1") {
  RandomGeometryOptions random{.crop = ImageExtent{3, 2},
                               .horizontal_flip_probability = 0.5};
  JointTransformOptions joint;
  DataPipelineOptions options;
  options.dataset_version = dataset;
  options.dataset_size = 12;
  options.batch_size = 2;
  options.num_workers = workers;
  options.batch_factory_id = identity.empty()
                                 ? joint_transform_identity(random, joint) +
                                       ":rng1:seed42:view0:unique"
                                 : identity;
  return std::make_unique<DataPipeline>(
      std::make_shared<samplers::SequentialSampler>(12), options,
      [random, joint, dataset](std::span<const std::size_t> indices,
                               const DataWorkerContext& context,
                               std::stop_token stop) -> c10::IValue {
        std::vector<Tensor> images;
        for (auto index : indices) {
          const SampleRandom rng(
              {.dataset_id = dataset,
               .sample_id = std::to_string(index),
               .transform_id = joint_transform_identity(random, joint),
               .seed = 42,
               .epoch = context.epoch});
          auto sample = Grid(8, 6);
          images.push_back(
              transform_sample(sample, sample_geometry({8, 6}, random, rng),
                               joint, stop)
                  .image);
        }
        return torch::stack(images);
      });
}

TEST(JointTransformPipelineTest,
     WorkerCountsHaveSameValuesButRestoreTopologyStaysStrict) {
  auto inline_pipeline = Pipeline(0);
  auto single = Pipeline(1);
  auto threaded = Pipeline(3);
  inline_pipeline->set_epoch(9);
  single->set_epoch(9);
  threaded->set_epoch(9);
  while (auto expected = inline_pipeline->next()) {
    const auto one = single->next();
    const auto three = threaded->next();
    ASSERT_TRUE(one && three);
    EXPECT_TRUE(
        torch::equal(expected->value.toTensor(), one->value.toTensor()));
    EXPECT_TRUE(
        torch::equal(expected->value.toTensor(), three->value.toTensor()));
  }
  EXPECT_FALSE(single->next());
  EXPECT_FALSE(threaded->next());
  MemoryState saved(single->describe_state());
  single->save_state(saved);
  EXPECT_THROW(threaded->load_state(saved, RestorePolicy::exact_only),
               CompatibilityError);
}

TEST(JointTransformPipelineTest,
     RestoreReplaysPixelsAndRejectsConfigurationOrSourceDrift) {
  auto source = Pipeline(2);
  source->set_epoch(4);
  ASSERT_TRUE(source->next());
  MemoryState saved(source->describe_state());
  source->save_state(saved);
  auto restored = Pipeline(2);
  restored->load_state(saved, RestorePolicy::exact_only);
  while (auto expected = source->next()) {
    const auto actual = restored->next();
    ASSERT_TRUE(actual);
    EXPECT_TRUE(
        torch::equal(expected->value.toTensor(), actual->value.toTensor()));
  }
  EXPECT_FALSE(restored->next());
  auto changed = Pipeline(2, "different-transform");
  EXPECT_THROW(changed->load_state(saved, RestorePolicy::exact_only),
               CompatibilityError);
  auto data_changed = Pipeline(2, "", "dataset-v2");
  EXPECT_THROW(data_changed->load_state(saved, RestorePolicy::exact_only),
               CompatibilityError);
}
}  // namespace
