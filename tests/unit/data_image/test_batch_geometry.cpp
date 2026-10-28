// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bicubic_reference.hpp"

#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>

#include <array>
#include <gtest/gtest.h>
#include <limits>

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
#include <c10/cuda/CUDAGuard.h>
#endif

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;

VisionSample Photo(std::uint32_t width, std::uint32_t height,
                   std::uint8_t first) {
  VisionSample sample;
  sample.width = width;
  sample.height = height;
  sample.image_path = "fixture.png";
  sample.pixels.color = PixelColorModel::gray;
  sample.image =
      torch::arange(width * height, torch::kUInt8).reshape({1, height, width}) +
      first;
  sample.geometry.emplace(ImageExtent{width, height});
  return sample;
}

DecodedMask Mask(const VisionSample& sample, std::int64_t base) {
  DecodedMask mask;
  mask.ids =
      torch::arange(static_cast<std::int64_t>(sample.width) * sample.height,
                    torch::kInt64)
          .reshape({sample.height, sample.width}) +
      base;
  mask.width = sample.width;
  mask.height = sample.height;
  mask.ignore_id = -8;
  mask.void_id = -9;
  return mask;
}

bool SameSteps(const std::vector<ImageGeometryStep>& a,
               const std::vector<ImageGeometryStep>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].kind != b[i].kind || a[i].extent.width != b[i].extent.width ||
        a[i].extent.height != b[i].extent.height || a[i].left != b[i].left ||
        a[i].top != b[i].top || a[i].right != b[i].right ||
        a[i].bottom != b[i].bottom)
      return false;
  return true;
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
ImageBatch ToCuda(ImageBatch batch) {
  if (batch.layout == ImageBatchLayout::ragged) {
    for (auto& image : batch.image_list) image = image.cuda();
    for (auto& valid : batch.valid_list) valid = valid.cuda();
    for (auto& mask : batch.mask_list) mask = mask.cuda();
    for (auto& paired : batch.paired_image_list) paired = paired.cuda();
  } else {
    batch.images = batch.images.cuda();
    batch.valid = batch.valid.cuda();
    if (batch.masks) batch.masks = batch.masks->cuda();
    if (batch.paired_images) batch.paired_images = batch.paired_images->cuda();
  }
  batch.labels = batch.labels.cuda();
  return batch;
}
#endif

ImageBatch CubicBatch(Dtype dtype, ImageBatchLayout layout) {
  auto sample = Photo(9, 7, 1);
  sample.image =
      torch::tensor(std::vector<double>(bicubic_reference::kDownInput.begin(),
                                        bicubic_reference::kDownInput.end()),
                    torch::kFloat64)
          .reshape({1, 7, 9})
          .to(dtype);
  sample.pixels.dtype = dtype;
  sample.pixels.range = (dtype == torch::kUInt8 || dtype == torch::kUInt16)
                            ? PixelValueRange::unsigned_full_scale
                            : PixelValueRange::unbounded;
  sample.mask_target = Mask(sample, 9007199254740993LL);
  ImageCollateOptions options;
  options.layout = layout;
  if (layout == ImageBatchLayout::pad) {
    options.padded_extent = ImageExtent{11, 8};
    options.image_padding = 99;
    options.mask_padding_id = -9;
  }
  return collate_images(std::array{sample}, std::array{ImageBatchKey{"cubic"}},
                        options);
}

ImageBatchGeometryOptions CubicOptions(ImageBatchLayout layout) {
  ImageBatchGeometryOptions options;
  options.joint.interpolation = ImageInterpolation::bicubic;
  options.joint.mask_padding_id = -9;
  if (layout == ImageBatchLayout::pad)
    options.output_padded_extent = ImageExtent{5, 4};
  return options;
}

double PhotoTolerance(Dtype dtype) {
  if (dtype == torch::kBFloat16) return 1.0 / 128;
  if (dtype == torch::kFloat16) return 1.0 / 1024;
  if (dtype == torch::kFloat32) return 2e-6;
  if (dtype == torch::kFloat64) return 1e-12;
  return 0;
}

TEST(ImageBatchGeometryTest, BicubicCpuBatchMatchesIndependentReference) {
  torch::set_num_threads(1);
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::resize, .extent = {4, 3}},
  }};
  const auto golden =
      torch::tensor(
          std::vector<double>(bicubic_reference::kDownExpected.begin(),
                              bicubic_reference::kDownExpected.end()),
          torch::kFloat64)
          .reshape({1, 3, 4});
  for (const auto dtype :
       {torch::kUInt8, torch::kUInt16, torch::kFloat16, torch::kBFloat16,
        torch::kFloat32, torch::kFloat64}) {
    const auto expected = (dtype == torch::kUInt8 || dtype == torch::kUInt16)
                              ? golden.round()
                              : golden.to(dtype).to(torch::kFloat64);
    for (const auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                              ImageBatchLayout::ragged}) {
      const auto input = CubicBatch(dtype, layout);
      const auto result =
          transform_image_batch_geometry(input, steps, CubicOptions(layout));
      const auto photo = layout == ImageBatchLayout::ragged
                             ? result.image_list[0]
                             : result.images[0].narrow(1, 0, 3).narrow(2, 0, 4);
      EXPECT_TRUE(torch::allclose(photo.to(torch::kFloat64), expected,
                                  PhotoTolerance(dtype),
                                  PhotoTolerance(dtype)));
      EXPECT_EQ(photo.scalar_type(), dtype);
      EXPECT_EQ(result.items[0].valid_extent.width, 4U);
      EXPECT_EQ(result.items[0].valid_extent.height, 3U);
    }
  }
}

TEST(ImageBatchGeometryTest, BicubicHasDistinctStableGeometryIdentity) {
  const RandomGeometryOptions random{.resize = ImageExtent{4, 3}};
  ImageBatchGeometryOptions nearest;
  auto bilinear = nearest;
  bilinear.joint.interpolation = ImageInterpolation::bilinear;
  auto cubic = nearest;
  cubic.joint.interpolation = ImageInterpolation::bicubic;
  const auto identity = image_batch_geometry_identity(random, cubic);
  EXPECT_EQ(identity, image_batch_geometry_identity(random, cubic));
  EXPECT_NE(identity, image_batch_geometry_identity(random, nearest));
  EXPECT_NE(identity, image_batch_geometry_identity(random, bilinear));
  EXPECT_NE(joint_transform_identity(random, cubic.joint),
            joint_transform_identity(random, bilinear.joint));
  cubic.joint.interpolation = static_cast<ImageInterpolation>(255);
  EXPECT_THROW(image_batch_geometry_identity(random, cubic), InvalidArgument);
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchGeometryTest,
     BicubicCudaMatchesCpuAndUsesCallerStreamForAllDtypes) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  torch::set_num_threads(1);
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::resize, .extent = {4, 3}},
  }};
  for (const auto interpolation :
       {ImageInterpolation::bilinear, ImageInterpolation::bicubic})
    for (const auto dtype :
         {torch::kUInt8, torch::kUInt16, torch::kFloat16, torch::kBFloat16,
          torch::kFloat32, torch::kFloat64})
      for (const auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                                ImageBatchLayout::ragged}) {
        auto options = CubicOptions(layout);
        options.joint.interpolation = interpolation;
        const auto cpu = CubicBatch(dtype, layout);
        const auto expected =
            transform_image_batch_geometry(cpu, steps, options);
        const auto cuda = ToCuda(cpu);
        const auto result =
            transform_image_batch_geometry(cuda, steps, options);
        EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
        const auto photo = layout == ImageBatchLayout::ragged
                               ? result.image_list[0]
                               : result.images;
        const auto reference = layout == ImageBatchLayout::ragged
                                   ? expected.image_list[0]
                                   : expected.images;
        const auto mask = layout == ImageBatchLayout::ragged
                              ? result.mask_list[0]
                              : *result.masks;
        const auto reference_mask = layout == ImageBatchLayout::ragged
                                        ? expected.mask_list[0]
                                        : *expected.masks;
        EXPECT_TRUE(photo.device().is_cuda());
        EXPECT_TRUE(mask.device().is_cuda());
        EXPECT_EQ(photo.scalar_type(), dtype);
        EXPECT_TRUE(torch::allclose(
            photo.cpu().to(torch::kFloat64), reference.to(torch::kFloat64),
            PhotoTolerance(dtype), PhotoTolerance(dtype)));
        EXPECT_TRUE(torch::equal(mask.cpu(), reference_mask));
        ImageBatchTransformOptions numeric;
        numeric.image.scale = BatchPixelScale::preserve;
        numeric.image.mean = {3};
        numeric.image.std = {7};
        const auto normalized = transform_image_batch(result, numeric);
        const auto normalized_photo = layout == ImageBatchLayout::ragged
                                          ? normalized.image_list[0]
                                          : normalized.images;
        EXPECT_TRUE(torch::allclose(normalized_photo.cpu(),
                                    (photo.cpu().to(torch::kFloat32) - 3) / 7,
                                    2e-6, 2e-6));
      }
}

TEST(ImageBatchGeometryTest,
     BicubicCudaScratchBudgetIsAdmittedBeforeAllocation) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(2, 1, 1);
  sample.mask_target = Mask(sample, 9007199254740993LL);
  const auto cpu = collate_images(std::array{sample},
                                  std::array{ImageBatchKey{"cubic-budget"}});
  const auto cuda = ToCuda(cpu);
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::resize, .extent = {5, 1}},
  }};
  auto options = CubicOptions(ImageBatchLayout::stack);
  // Five output pixels: image, valid and int64 mask = 50 bytes. Working
  // payload: 5*(2*9+32) + 50 + (5+1)*32 = 492 bytes, excluding backend
  // workspace.
  ImageResourceBudgetLimits limits;
  limits.max_gpu_bytes = 50 + 492 - 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch_geometry(cuda, steps, options),
               ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 50U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
  limits.max_gpu_bytes += 1;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto result = transform_image_batch_geometry(cuda, steps, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 542U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 50U);
  result = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
}
#endif

TEST(ImageBatchGeometryTest, PaddedBatchTransformsValidSamplesAndTargets) {
  auto a = Photo(3, 1, 10);
  auto b = Photo(2, 1, 30);
  a.mask_target = Mask(a, 9007199254740993LL);
  b.mask_target = Mask(b, 9007199254740997LL);
  auto categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 7, .name = "object"}});
  b.categories = categories;
  auto record = std::make_shared<VisionAnnotationRecord>();
  record->image_path = b.image_path;
  record->width = b.width;
  record->height = b.height;
  record->objects.push_back(
      {.source_id = 9, .category = 0, .box = ImageBox{0, 0, 1, 1}});
  b.annotation = record;
  ImageCollateOptions collate;
  collate.layout = ImageBatchLayout::pad;
  collate.image_padding = 250;
  collate.mask_padding_id = -9;
  const auto batch = collate_images(
      std::array{a, b}, std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}},
      collate);

  ImageBatchGeometryOptions options;
  options.joint.mask_padding_id = -9;
  const std::vector<std::vector<ImageGeometryStep>> steps{
      {{.kind = ImageGeometryKind::horizontal_flip}},
      {{.kind = ImageGeometryKind::horizontal_flip}}};
  const auto result = transform_image_batch_geometry(batch, steps, options);

  EXPECT_EQ(result.layout, ImageBatchLayout::pad);
  EXPECT_EQ(result.images.sizes(), torch::IntArrayRef({2, 1, 1, 3}));
  EXPECT_EQ(result.valid.sum().item<int>(), 5);
  EXPECT_EQ(result.images[0][0][0][0].item<int>(), 12);
  EXPECT_EQ(result.images[1][0][0][0].item<int>(), 31);
  EXPECT_EQ(result.images[1][0][0][2].item<int>(), 0);
  ASSERT_TRUE(result.masks);
  EXPECT_EQ((*result.masks)[1][0][0].item<std::int64_t>(), 9007199254740998LL);
  EXPECT_EQ((*result.masks)[1][0][2].item<std::int64_t>(), -9);
  ASSERT_TRUE(result.items[1].annotation);
  ASSERT_EQ(result.items[1].annotation->objects.size(), 1U);
  const auto box = result.items[1].annotation->objects[0].box;
  ASSERT_TRUE(box);
  EXPECT_DOUBLE_EQ(box->x_min, 1);
  EXPECT_DOUBLE_EQ(box->x_max, 2);
  EXPECT_EQ(result.items[1].valid_extent.width, 2U);
  EXPECT_EQ(result.items[1].geometry->output_extent().width, 3U);
  EXPECT_EQ(batch.images[1][0][0][0].item<int>(), 30);
}

TEST(ImageBatchGeometryTest, StackAndRaggedKeepTheirLayouts) {
  for (const auto layout :
       {ImageBatchLayout::stack, ImageBatchLayout::ragged}) {
    auto a = Photo(2, 2, 1);
    auto b = Photo(2, 2, 5);
    ImageCollateOptions collate;
    collate.layout = layout;
    const auto batch = collate_images(
        std::array{a, b}, std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}},
        collate);
    const std::vector<std::vector<ImageGeometryStep>> steps{
        {{.kind = ImageGeometryKind::vertical_flip}},
        {{.kind = ImageGeometryKind::vertical_flip}}};
    const auto result = transform_image_batch_geometry(batch, steps);
    EXPECT_EQ(result.layout, layout);
    if (layout == ImageBatchLayout::ragged) {
      EXPECT_FALSE(result.images.defined());
      EXPECT_EQ(result.image_list[0][0][0][0].item<int>(), 3);
    } else {
      EXPECT_EQ(result.images[0][0][0][0].item<int>(), 3);
    }
  }
}

TEST(ImageBatchGeometryTest, RleAnnotationsUseTheSameDiscreteGeometry) {
  auto sample = Photo(3, 1, 10);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 7, .name = "object"}});
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->objects.push_back({.source_id = 11,
                                 .category = 0,
                                 .rle = ImageMaskRle{3, 1, {1, 2}},
                                 .area = 2,
                                 .mask_pixel_area = 2});
  sample.annotation = annotation;
  const auto batch =
      collate_images(std::array{sample}, std::array{ImageBatchKey{"rle"}});
  const std::vector<std::vector<ImageGeometryStep>> steps{
      {{.kind = ImageGeometryKind::horizontal_flip}}};
  const auto result = transform_image_batch_geometry(batch, steps);
  ASSERT_TRUE(result.items[0].annotation);
  ASSERT_EQ(result.items[0].annotation->objects.size(), 1U);
  const auto& object = result.items[0].annotation->objects[0];
  ASSERT_TRUE(object.rle);
  EXPECT_EQ(object.rle->counts, (std::vector<std::uint64_t>{0, 2, 1}));
  EXPECT_EQ(object.mask_pixel_area, 2U);
  EXPECT_EQ(batch.items[0].annotation->objects[0].rle->counts,
            (std::vector<std::uint64_t>{1, 2}));
}

TEST(ImageBatchGeometryTest, PanopticCropAndRepadKeepMaskAndSegmentBinding) {
  auto sample = Photo(3, 2, 10);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 4, .name = "thing"}});
  DecodedMask mask;
  mask.ids = torch::tensor({1, 1, 16777215, 0, -7, 16777215}, torch::kInt64)
                 .reshape({2, 3});
  mask.encoding = MaskEncoding::panoptic_rgb;
  mask.width = 3;
  mask.height = 2;
  mask.ignore_id = -7;
  mask.void_id = 0;
  sample.mask_target = mask;
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->panoptic_mask_path = "segments.png";
  annotation->objects.push_back(
      {.source_id = 1, .category = 0, .box = ImageBox{0, 0, 2, 1}, .area = 2});
  annotation->objects.push_back({.source_id = 16777215,
                                 .category = 0,
                                 .box = ImageBox{2, 0, 3, 2},
                                 .area = 2,
                                 .crowd = true});
  sample.annotation = annotation;

  ImageCollateOptions collate;
  collate.layout = ImageBatchLayout::pad;
  collate.padded_extent = ImageExtent{4, 3};
  collate.mask_padding_id = 0;
  const auto batch = collate_images(
      std::array{sample}, std::array{ImageBatchKey{"panoptic"}}, collate);
  const std::vector<std::vector<ImageGeometryStep>> steps{
      {{.kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1}}};
  ImageBatchGeometryOptions options;
  options.joint.mask_padding_id = 0;
  options.output_padded_extent = ImageExtent{3, 3};
  const auto result = transform_image_batch_geometry(batch, steps, options);

  EXPECT_EQ(result.images.sizes(), torch::IntArrayRef({1, 1, 3, 3}));
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(torch::equal(
      (*result.masks)[0],
      torch::tensor({1, 16777215, 0, -7, 16777215, 0, 0, 0, 0}, torch::kInt64)
          .reshape({3, 3})));
  ASSERT_TRUE(result.items[0].annotation);
  const auto& transformed = *result.items[0].annotation;
  EXPECT_TRUE(transformed.panoptic_mask_materialized);
  ASSERT_EQ(transformed.objects.size(), 2U);
  EXPECT_EQ(transformed.objects[0].mask_pixel_area, 1U);
  EXPECT_EQ(transformed.objects[1].mask_pixel_area, 2U);
  EXPECT_DOUBLE_EQ(transformed.objects[1].box->x_min, 1);
  EXPECT_DOUBLE_EQ(transformed.objects[1].box->x_max, 2);
  EXPECT_EQ(transformed.width, 3U);
  EXPECT_EQ(transformed.height, 3U);
}

TEST(ImageBatchGeometryTest, PaddedOutputPolicyShapesAndBindsIdentity) {
  const auto batch =
      collate_images(std::array{Photo(2, 1, 1), Photo(1, 1, 5)},
                     std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}},
                     ImageCollateOptions{.layout = ImageBatchLayout::pad});
  const std::vector<std::vector<ImageGeometryStep>> steps(2);
  ImageBatchGeometryOptions options;
  options.output_padded_extent = ImageExtent{5, 3};
  options.output_size_divisor = 4;
  const auto result = transform_image_batch_geometry(batch, steps, options);
  EXPECT_EQ(result.images.sizes(), torch::IntArrayRef({2, 1, 4, 8}));
  EXPECT_EQ(result.valid.sum().item<int>(), 3);

  const auto identity = image_batch_geometry_identity({}, options);
  auto changed = options;
  changed.output_padded_extent = ImageExtent{6, 3};
  EXPECT_NE(image_batch_geometry_identity({}, changed), identity);
  changed = options;
  changed.output_size_divisor = 8;
  EXPECT_NE(image_batch_geometry_identity({}, changed), identity);

  const auto stack_batch =
      collate_images(std::array{Photo(2, 1, 1), Photo(2, 1, 5)},
                     std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}});
  ImageBatchGeometryOptions invalid;
  invalid.output_size_divisor = 2;
  EXPECT_THROW(transform_image_batch_geometry(stack_batch, steps, invalid),
               InvalidArgument);
  EXPECT_THROW(sample_image_batch_geometry(stack_batch, "dataset", {}, invalid),
               InvalidArgument);
  invalid = {};
  invalid.output_padded_extent = ImageExtent{1, 1};
  EXPECT_THROW(transform_image_batch_geometry(batch, steps, invalid),
               InvalidArgument);
  invalid = options;
  invalid.limits.max_pixels = 63;
  EXPECT_THROW(transform_image_batch_geometry(batch, steps, invalid),
               ResourceExhausted);
}

TEST(ImageBatchGeometryTest, SamplingIsStableAndIdentityBindsLimits) {
  auto a = Photo(8, 6, 1);
  auto b = Photo(8, 6, 2);
  const auto batch = collate_images(
      std::array{a, b},
      std::array{ImageBatchKey{"a", 4, 1}, ImageBatchKey{"b", 2, 3}});
  RandomGeometryOptions random{.resize = ImageExtent{10, 8},
                               .crop = ImageExtent{6, 5},
                               .horizontal_flip_probability = 0.5,
                               .vertical_flip_probability = 0.25};
  ImageBatchGeometryOptions options;
  const auto first = sample_image_batch_geometry(batch, "dataset-sha256",
                                                 random, options, 17, 9);
  const auto second = sample_image_batch_geometry(batch, "dataset-sha256",
                                                  random, options, 17, 9);
  ASSERT_EQ(first.size(), second.size());
  EXPECT_TRUE(SameSteps(first[0], second[0]));
  EXPECT_TRUE(SameSteps(first[1], second[1]));
  const auto identity = image_batch_geometry_identity(random, options);
  --options.limits.max_geometry_checks;
  EXPECT_NE(image_batch_geometry_identity(random, options), identity);
  const auto reversed = collate_images(
      std::array{b, a},
      std::array{ImageBatchKey{"b", 2, 3}, ImageBatchKey{"a", 4, 1}});
  const auto reordered = sample_image_batch_geometry(
      reversed, "dataset-sha256", random, ImageBatchGeometryOptions{}, 17, 9);
  EXPECT_TRUE(SameSteps(first[0], reordered[1]));
  EXPECT_TRUE(SameSteps(first[1], reordered[0]));
  EXPECT_NE(first[0].size(), 0U);
}

TEST(ImageBatchGeometryTest, AggregateBudgetAndCancellationDoNotMutateInput) {
  const auto batch =
      collate_images(std::array{Photo(8, 8, 1), Photo(8, 8, 2)},
                     std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}});
  const auto original = batch.images.clone();
  const std::vector<std::vector<ImageGeometryStep>> steps(2);
  ImageBatchGeometryOptions options;
  options.limits.max_output_bytes = 1;
  EXPECT_THROW(transform_image_batch_geometry(batch, steps, options),
               ResourceExhausted);
  options = {};
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      transform_image_batch_geometry(batch, steps, options, stop.get_token()),
      Cancelled);
  EXPECT_TRUE(torch::equal(batch.images, original));
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchGeometryTest, CudaOutputGpuBudgetPrecedesAllocation) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(2, 2, 1);
  sample.mask_target = Mask(sample, 9007199254740993LL);
  const std::vector<std::vector<ImageGeometryStep>> steps(1);
  for (const auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                            ImageBatchLayout::ragged}) {
    ImageCollateOptions collate;
    collate.layout = layout;
    if (layout == ImageBatchLayout::pad) {
      collate.padded_extent = ImageExtent{3, 3};
      collate.mask_padding_id = -9;
    }
    const auto cpu = collate_images(
        std::array{sample}, std::array{ImageBatchKey{"cuda-budget"}}, collate);
    const auto input = ToCuda(cpu);
    const auto output_bytes = layout == ImageBatchLayout::pad ? 90U : 40U;
    ImageBatchGeometryOptions options;
    if (layout == ImageBatchLayout::pad) {
      options.joint.mask_padding_id = -9;
      options.output_padded_extent = ImageExtent{3, 3};
    }
    ImageResourceBudgetLimits limits;
    limits.max_gpu_bytes = output_bytes - 1;
    auto budget = std::make_shared<ImageResourceBudget>(limits);
    options.resource_budget = budget;
    EXPECT_THROW(transform_image_batch_geometry(input, steps, options),
                 ResourceExhausted);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 0U);
    EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);

    const auto maximum_pixels = layout == ImageBatchLayout::pad ? 9U : 4U;
    const auto working_bytes = 2U * maximum_pixels * 9U + output_bytes;
    limits.max_gpu_bytes = output_bytes + working_bytes - 1;
    budget = std::make_shared<ImageResourceBudget>(limits);
    options.resource_budget = budget;
    EXPECT_THROW(transform_image_batch_geometry(input, steps, options),
                 ResourceExhausted);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, output_bytes);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

    limits.max_gpu_bytes = output_bytes + working_bytes;
    budget = std::make_shared<ImageResourceBudget>(limits);
    options.resource_budget = budget;
    auto result = transform_image_batch_geometry(input, steps, options);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current,
              output_bytes);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak,
              output_bytes + working_bytes);
    EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 1U);
    if (layout == ImageBatchLayout::ragged) {
      ASSERT_EQ(result.mask_list.size(), 1U);
      EXPECT_TRUE(torch::equal(result.mask_list[0].cpu(), cpu.mask_list[0]));
    } else {
      ASSERT_TRUE(result.masks);
      EXPECT_TRUE(torch::equal(result.masks->cpu(), *cpu.masks));
    }
    result = {};
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
    EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);

    limits.max_annotation_bytes = 1;
    budget = std::make_shared<ImageResourceBudget>(limits);
    options.resource_budget = budget;
    EXPECT_THROW(transform_image_batch_geometry(input, steps, options),
                 ResourceExhausted);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
  }
}

TEST(ImageBatchGeometryTest, CudaNearestResizeUsesBoundedAxisIndexes) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto before : {ImageExtent{31, 2}, ImageExtent{2, 31}}) {
    auto sample = Photo(before.width, before.height, 1);
    sample.mask_target = Mask(sample, 9007199254740993LL);
    const auto cpu = collate_images(std::array{sample},
                                    std::array{ImageBatchKey{"cuda-nearest"}});
    const auto cuda = ToCuda(cpu);
    const ImageExtent after =
        before.width > before.height ? ImageExtent{2, 29} : ImageExtent{29, 2};
    const std::vector<std::vector<ImageGeometryStep>> steps{{
        {.kind = ImageGeometryKind::resize, .extent = after},
    }};
    const auto expected = transform_image_batch_geometry(cpu, steps);
    const auto output_bytes =
        static_cast<std::uint64_t>(after.width) * after.height * 10U;
    const auto index_bytes =
        static_cast<std::uint64_t>(after.width + after.height) * 32U;
    const auto working_bytes = 62U * 18U + output_bytes + index_bytes;
    ImageResourceBudgetLimits limits;
    limits.max_gpu_bytes = output_bytes + working_bytes - 1;
    auto budget = std::make_shared<ImageResourceBudget>(limits);
    ImageBatchGeometryOptions options;
    options.resource_budget = budget;
    EXPECT_THROW(transform_image_batch_geometry(cuda, steps, options),
                 ResourceExhausted);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, output_bytes);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

    limits.max_gpu_bytes = output_bytes + working_bytes;
    budget = std::make_shared<ImageResourceBudget>(limits);
    options.resource_budget = budget;
    auto result = transform_image_batch_geometry(cuda, steps, options);
    EXPECT_TRUE(torch::equal(result.images.cpu(), expected.images));
    ASSERT_TRUE(result.masks);
    EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current,
              output_bytes);
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak,
              output_bytes + working_bytes);
    result = {};
    EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
  }
}

TEST(ImageBatchGeometryTest, CudaPixelGeometryMatchesCpuAndKeepsMaskIds) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(3, 2, 10);
  sample.mask_target = Mask(sample, 9007199254740993LL);
  ImageCollateOptions collate;
  collate.layout = ImageBatchLayout::pad;
  collate.padded_extent = ImageExtent{4, 3};
  collate.image_padding = 101;
  collate.mask_padding_id = -9;
  const auto cpu = collate_images(
      std::array{sample}, std::array{ImageBatchKey{"cuda-geometry"}}, collate);

  ImageBatchGeometryOptions options;
  options.joint.mask_padding_id = -9;
  options.output_padded_extent = ImageExtent{6, 5};
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1},
      {.kind = ImageGeometryKind::resize, .extent = {3, 3}},
      {.kind = ImageGeometryKind::pad, .left = 1, .top = 1, .right = 1},
      {.kind = ImageGeometryKind::horizontal_flip},
      {.kind = ImageGeometryKind::vertical_flip},
  }};
  const auto expected = transform_image_batch_geometry(cpu, steps, options);

  auto cuda_batch = ToCuda(cpu);
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  const auto result =
      transform_image_batch_geometry(cuda_batch, steps, options);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
  EXPECT_TRUE(result.images.device().is_cuda());
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(result.masks->device().is_cuda());
  EXPECT_TRUE(result.valid.device().is_cuda());
  EXPECT_TRUE(torch::equal(result.images.cpu(), expected.images));
  EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
  EXPECT_TRUE(torch::equal(result.valid.cpu(), expected.valid));
  EXPECT_EQ(result.items[0].valid_extent.width, 5U);
  EXPECT_EQ(result.items[0].valid_extent.height, 4U);
  EXPECT_EQ(result.valid.sum().item<int>(), 20);
  EXPECT_EQ((*result.masks)[0][0][0].item<std::int64_t>(), -9);
  EXPECT_EQ((*result.masks)[0][0][5].item<std::int64_t>(), -9);
  EXPECT_EQ((*result.masks)[0][4][0].item<std::int64_t>(), -9);
  EXPECT_EQ((*result.masks)[0][1][1].item<std::int64_t>(),
            (*expected.masks)[0][1][1].item<std::int64_t>());
  EXPECT_EQ((*result.masks)[0].max().item<std::int64_t>(), 9007199254740998LL);
}

TEST(ImageBatchGeometryTest, CudaRichAnnotationsMatchCpuWithoutPixelReadback) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(3, 2, 1);
  sample.mask_target = Mask(sample, 9007199254740993LL);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 7,
                                   .name = "object",
                                   .keypoint_names = {"left", "right"}}});
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->objects.push_back(
      {.source_id = 11,
       .box = ImageBox{0, 0, 2, 2},
       .keypoints = {{{0.5, 0.5}, 2}, {{1.5, 1.5}, 2}}});
  annotation->objects.push_back(
      {.source_id = 12, .rle = ImageMaskRle{3, 2, {1, 2, 3}}});
  annotation->objects.push_back(
      {.source_id = 13, .polygons = {{{1, 0}, {3, 0}, {3, 2}, {1, 2}}}});
  annotation->objects.push_back({.source_id = 14,
                                 .oriented_box = std::array<ImagePoint, 4>{
                                     ImagePoint{0.5, 0}, ImagePoint{2.5, 0},
                                     ImagePoint{2.5, 2}, ImagePoint{0.5, 2}}});
  sample.annotation = annotation;
  ImageCollateOptions collate;
  collate.layout = ImageBatchLayout::pad;
  collate.padded_extent = ImageExtent{4, 3};
  collate.mask_padding_id = -9;
  const auto cpu = collate_images(
      std::array{sample}, std::array{ImageBatchKey{"cuda-rich"}}, collate);
  const auto cuda_batch = ToCuda(cpu);
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1},
      {.kind = ImageGeometryKind::horizontal_flip},
      {.kind = ImageGeometryKind::resize, .extent = {4, 2}},
  }};
  ImageBatchGeometryOptions options;
  options.joint.mask_padding_id = -9;
  options.joint.horizontal_keypoint_permutations = {{1, 0}};
  options.output_padded_extent = ImageExtent{6, 4};
  const auto expected = transform_image_batch_geometry(cpu, steps, options);
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  const auto result =
      transform_image_batch_geometry(cuda_batch, steps, options);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
  EXPECT_TRUE(result.images.device().is_cuda());
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(result.masks->device().is_cuda());
  EXPECT_TRUE(torch::equal(result.images.cpu(), expected.images));
  EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
  EXPECT_TRUE(torch::equal(result.valid.cpu(), expected.valid));
  ASSERT_TRUE(result.items[0].annotation);
  ASSERT_TRUE(expected.items[0].annotation);
  const auto& actual_record = *result.items[0].annotation;
  const auto& expected_record = *expected.items[0].annotation;
  EXPECT_EQ(actual_record.width, expected_record.width);
  EXPECT_EQ(actual_record.height, expected_record.height);
  ASSERT_EQ(actual_record.objects.size(), expected_record.objects.size());
  ASSERT_EQ(actual_record.objects.size(), 4U);
  for (std::size_t i = 0; i < actual_record.objects.size(); ++i) {
    const auto& actual = actual_record.objects[i];
    const auto& reference = expected_record.objects[i];
    EXPECT_EQ(actual.source_id, reference.source_id);
    EXPECT_EQ(actual.box.has_value(), reference.box.has_value());
    if (actual.box && reference.box) {
      EXPECT_DOUBLE_EQ(actual.box->x_min, reference.box->x_min);
      EXPECT_DOUBLE_EQ(actual.box->y_min, reference.box->y_min);
      EXPECT_DOUBLE_EQ(actual.box->x_max, reference.box->x_max);
      EXPECT_DOUBLE_EQ(actual.box->y_max, reference.box->y_max);
    }
    ASSERT_EQ(actual.polygons.size(), reference.polygons.size());
    for (std::size_t ring = 0; ring < actual.polygons.size(); ++ring) {
      ASSERT_EQ(actual.polygons[ring].size(), reference.polygons[ring].size());
      for (std::size_t point = 0; point < actual.polygons[ring].size();
           ++point) {
        EXPECT_DOUBLE_EQ(actual.polygons[ring][point].x,
                         reference.polygons[ring][point].x);
        EXPECT_DOUBLE_EQ(actual.polygons[ring][point].y,
                         reference.polygons[ring][point].y);
      }
    }
    EXPECT_EQ(actual.oriented_box.has_value(),
              reference.oriented_box.has_value());
    EXPECT_EQ(actual.rle.has_value(), reference.rle.has_value());
    if (actual.rle && reference.rle) {
      EXPECT_EQ(actual.rle->width, reference.rle->width);
      EXPECT_EQ(actual.rle->height, reference.rle->height);
      EXPECT_EQ(actual.rle->counts, reference.rle->counts);
    }
    ASSERT_EQ(actual.keypoints.size(), reference.keypoints.size());
    for (std::size_t point = 0; point < actual.keypoints.size(); ++point) {
      EXPECT_DOUBLE_EQ(actual.keypoints[point].point.x,
                       reference.keypoints[point].point.x);
      EXPECT_DOUBLE_EQ(actual.keypoints[point].point.y,
                       reference.keypoints[point].point.y);
      EXPECT_EQ(actual.keypoints[point].visibility,
                reference.keypoints[point].visibility);
    }
  }
  EXPECT_EQ(cuda_batch.items[0].annotation->width, 4U);
  auto limited = options;
  limited.joint.limits.max_rle_runs = 1;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, limited),
               ResourceExhausted);
  limited = options;
  limited.joint.min_region_area = -1;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, limited),
               InvalidArgument);
}

TEST(ImageBatchGeometryTest, CudaPanopticRequiresExplicitSummaryTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(2, 2, 1);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 7, .name = "object"}});
  DecodedMask mask;
  mask.ids = torch::tensor({1, 1, 0, 0}, torch::kInt64).reshape({2, 2});
  mask.encoding = MaskEncoding::panoptic_rgb;
  mask.width = 2;
  mask.height = 2;
  mask.void_id = 0;
  sample.mask_target = mask;
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->panoptic_mask_path = "segments.png";
  annotation->objects.push_back(
      {.source_id = 1, .box = ImageBox{0, 0, 2, 1}, .area = 2});
  sample.annotation = annotation;
  const auto cpu = collate_images(std::array{sample},
                                  std::array{ImageBatchKey{"cuda-panoptic"}});
  const auto cuda_batch = ToCuda(cpu);
  const std::vector<std::vector<ImageGeometryStep>> steps(1);
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps),
               InvalidArgument);
  ImageBatchGeometryOptions options;
  options.allow_cuda_panoptic_summary_transfer = true;
  EXPECT_NE(image_batch_geometry_identity({}, options),
            image_batch_geometry_identity({}, {}));
  const auto expected = transform_image_batch_geometry(cpu, steps, options);
  const auto result =
      transform_image_batch_geometry(cuda_batch, steps, options);
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
  ASSERT_TRUE(result.items[0].annotation);
  ASSERT_EQ(result.items[0].annotation->objects.size(), 1U);
  EXPECT_EQ(result.items[0].annotation->objects[0].mask_pixel_area, 2U);

  ImageResourceBudgetLimits limits;
  limits.max_gpu_bytes = 872;
  limits.max_annotation_bytes = 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, options),
               ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

  const auto scratch_bytes =
      13U * (sizeof(Tensor) + 2U * sizeof(double)) + sizeof(std::size_t);
  limits.max_annotation_bytes = scratch_bytes;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, options),
               ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).peak,
            scratch_bytes);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

  limits.max_annotation_bytes =
      ImageResourceBudgetLimits{}.max_annotation_bytes;
  limits.max_gpu_bytes = 871;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, options),
               ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 152U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

  limits.max_gpu_bytes = 872;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto bounded = transform_image_batch_geometry(cuda_batch, steps, options);
  ASSERT_TRUE(bounded.masks);
  EXPECT_TRUE(torch::equal(bounded.masks->cpu(), *expected.masks));
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 872U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 40U);
  EXPECT_GT(budget->usage(ImageResourceKind::annotation_bytes).peak,
            budget->usage(ImageResourceKind::annotation_bytes).current);
  EXPECT_GT(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);
  bounded = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);
}

TEST(ImageBatchGeometryTest, CudaPanopticMatchesCpuAndValidatesBinding) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(3, 2, 10);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 4, .name = "thing"}});
  DecodedMask mask;
  mask.ids = torch::tensor({1, 1, 16777215, 0, -7, 16777215}, torch::kInt64)
                 .reshape({2, 3});
  mask.encoding = MaskEncoding::panoptic_rgb;
  mask.width = 3;
  mask.height = 2;
  mask.ignore_id = -7;
  mask.void_id = 0;
  sample.mask_target = mask;
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->panoptic_mask_path = "segments.png";
  annotation->objects.push_back(
      {.source_id = 1, .category = 0, .box = ImageBox{0, 0, 2, 1}, .area = 2});
  annotation->objects.push_back({.source_id = 16777215,
                                 .category = 0,
                                 .box = ImageBox{2, 0, 3, 2},
                                 .area = 2});
  sample.annotation = annotation;
  ImageCollateOptions collate;
  collate.layout = ImageBatchLayout::pad;
  collate.padded_extent = ImageExtent{4, 3};
  collate.mask_padding_id = 0;
  const auto cpu = collate_images(
      std::array{sample}, std::array{ImageBatchKey{"cuda-panoptic-filter"}},
      collate);
  const auto cuda_batch = ToCuda(cpu);
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1},
      {.kind = ImageGeometryKind::resize, .extent = {4, 4}},
      {.kind = ImageGeometryKind::horizontal_flip},
  }};
  ImageBatchGeometryOptions options;
  options.allow_cuda_panoptic_summary_transfer = true;
  options.joint.mask_padding_id = 0;
  options.joint.min_region_visibility = 0.6;
  options.output_padded_extent = ImageExtent{5, 5};
  const auto expected = transform_image_batch_geometry(cpu, steps, options);
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  const auto result =
      transform_image_batch_geometry(cuda_batch, steps, options);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(result.masks->device().is_cuda());
  EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
  EXPECT_GT(budget->usage(ImageResourceKind::gpu_bytes).peak,
            budget->usage(ImageResourceKind::gpu_bytes).current);
  EXPECT_TRUE(torch::equal(cuda_batch.masks->cpu(), *cpu.masks));
  ASSERT_TRUE(result.items[0].annotation);
  ASSERT_TRUE(expected.items[0].annotation);
  const auto& actual = *result.items[0].annotation;
  const auto& reference = *expected.items[0].annotation;
  EXPECT_EQ(actual.width, reference.width);
  EXPECT_EQ(actual.height, reference.height);
  EXPECT_TRUE(actual.panoptic_mask_materialized);
  ASSERT_EQ(actual.objects.size(), reference.objects.size());
  ASSERT_EQ(actual.objects.size(), 1U);
  EXPECT_EQ(actual.objects[0].source_id, reference.objects[0].source_id);
  EXPECT_EQ(actual.objects[0].mask_pixel_area,
            reference.objects[0].mask_pixel_area);
  ASSERT_TRUE(actual.objects[0].box);
  ASSERT_TRUE(reference.objects[0].box);
  EXPECT_DOUBLE_EQ(actual.objects[0].box->x_min,
                   reference.objects[0].box->x_min);
  EXPECT_DOUBLE_EQ(actual.objects[0].box->y_min,
                   reference.objects[0].box->y_min);
  EXPECT_DOUBLE_EQ(actual.objects[0].box->x_max,
                   reference.objects[0].box->x_max);
  EXPECT_DOUBLE_EQ(actual.objects[0].box->y_max,
                   reference.objects[0].box->y_max);

  auto corrupted = cuda_batch;
  corrupted.masks = cuda_batch.masks->clone();
  (*corrupted.masks)[0][0][0] = 9;
  EXPECT_THROW(transform_image_batch_geometry(corrupted, steps, options),
               SerializationError);
  (*corrupted.masks)[0][0][0] = 16777215;
  EXPECT_THROW(transform_image_batch_geometry(corrupted, steps, options),
               SerializationError);
  auto limited = options;
  limited.joint.limits.max_mask_work_pixels = 1;
  EXPECT_THROW(transform_image_batch_geometry(cuda_batch, steps, limited),
               ResourceExhausted);
}

TEST(ImageBatchGeometryTest, CudaPanopticAcceptsEmptySegmentTable) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto sample = Photo(2, 2, 1);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 7, .name = "object"}});
  DecodedMask mask;
  mask.ids = torch::zeros({2, 2}, torch::kInt64);
  mask.encoding = MaskEncoding::panoptic_rgb;
  mask.width = 2;
  mask.height = 2;
  mask.void_id = 0;
  sample.mask_target = mask;
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = sample.image_path;
  annotation->width = sample.width;
  annotation->height = sample.height;
  annotation->panoptic_mask_path = "segments.png";
  sample.annotation = annotation;
  const auto cpu = collate_images(std::array{sample},
                                  std::array{ImageBatchKey{"empty-panoptic"}});
  const std::vector<std::vector<ImageGeometryStep>> steps(1);
  ImageBatchGeometryOptions options;
  options.allow_cuda_panoptic_summary_transfer = true;
  const auto result =
      transform_image_batch_geometry(ToCuda(cpu), steps, options);
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(torch::equal(result.masks->cpu(), *cpu.masks));
  ASSERT_TRUE(result.items[0].annotation);
  EXPECT_TRUE(result.items[0].annotation->panoptic_mask_materialized);
  EXPECT_TRUE(result.items[0].annotation->objects.empty());
}

TEST(ImageBatchGeometryTest, CudaPixelGeometryPreservesStackAndRaggedLayouts) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto layout :
       {ImageBatchLayout::stack, ImageBatchLayout::ragged}) {
    auto first = Photo(2, 2, 1);
    auto second = Photo(2, 2, 5);
    first.mask_target = Mask(first, 100);
    second.mask_target = Mask(second, 200);
    ImageCollateOptions collate;
    collate.layout = layout;
    collate.mask_padding_id = -9;
    const auto cpu = collate_images(std::array{first, second},
                                    std::array{ImageBatchKey{"cuda-layout-a"},
                                               ImageBatchKey{"cuda-layout-b"}},
                                    collate);
    const std::vector<std::vector<ImageGeometryStep>> steps{
        {{.kind = ImageGeometryKind::horizontal_flip}},
        {{.kind = ImageGeometryKind::vertical_flip}}};
    ImageBatchGeometryOptions options;
    options.joint.mask_padding_id = -9;
    const auto expected = transform_image_batch_geometry(cpu, steps, options);
    const auto result =
        transform_image_batch_geometry(ToCuda(cpu), steps, options);
    EXPECT_EQ(result.layout, layout);
    if (layout == ImageBatchLayout::ragged) {
      ASSERT_EQ(result.image_list.size(), 2U);
      ASSERT_EQ(result.mask_list.size(), 2U);
      for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(
            torch::equal(result.image_list[i].cpu(), expected.image_list[i]));
        EXPECT_TRUE(
            torch::equal(result.mask_list[i].cpu(), expected.mask_list[i]));
        EXPECT_TRUE(result.valid_list[i].device().is_cuda());
      }
    } else {
      ASSERT_TRUE(result.masks);
      EXPECT_TRUE(torch::equal(result.images.cpu(), expected.images));
      EXPECT_TRUE(torch::equal(result.masks->cpu(), *expected.masks));
      EXPECT_TRUE(result.valid.device().is_cuda());
    }
  }
}
#endif
}  // namespace
