// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/batch_geometry.hpp>

#include <array>
#include <gtest/gtest.h>

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
  sample.image = torch::arange(width * height, torch::kUInt8)
                     .reshape({1, height, width}) + first;
  sample.geometry.emplace(ImageExtent{width, height});
  return sample;
}

DecodedMask Mask(const VisionSample& sample, std::int64_t base) {
  DecodedMask mask;
  mask.ids = torch::arange(
                 static_cast<std::int64_t>(sample.width) * sample.height,
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
  record->objects.push_back({.source_id = 9,
                             .category = 0,
                             .box = ImageBox{0, 0, 1, 1}});
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
  const auto result =
      transform_image_batch_geometry(batch, steps, options);

  EXPECT_EQ(result.layout, ImageBatchLayout::pad);
  EXPECT_EQ(result.images.sizes(), torch::IntArrayRef({2, 1, 1, 3}));
  EXPECT_EQ(result.valid.sum().item<int>(), 5);
  EXPECT_EQ(result.images[0][0][0][0].item<int>(), 12);
  EXPECT_EQ(result.images[1][0][0][0].item<int>(), 31);
  EXPECT_EQ(result.images[1][0][0][2].item<int>(), 0);
  ASSERT_TRUE(result.masks);
  EXPECT_EQ((*result.masks)[1][0][0].item<std::int64_t>(),
            9007199254740998LL);
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
  annotation->objects.push_back(
      {.source_id = 11,
       .category = 0,
       .rle = ImageMaskRle{3, 1, {1, 2}},
       .area = 2,
       .mask_pixel_area = 2});
  sample.annotation = annotation;
  const auto batch = collate_images(
      std::array{sample}, std::array{ImageBatchKey{"rle"}});
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
      {.source_id = 1,
       .category = 0,
       .box = ImageBox{0, 0, 2, 1},
       .area = 2});
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
  const std::vector<std::vector<ImageGeometryStep>> steps{{
      {.kind = ImageGeometryKind::crop, .extent = {2, 2}, .left = 1}}};
  ImageBatchGeometryOptions options;
  options.joint.mask_padding_id = 0;
  options.output_padded_extent = ImageExtent{3, 3};
  const auto result = transform_image_batch_geometry(batch, steps, options);

  EXPECT_EQ(result.images.sizes(), torch::IntArrayRef({1, 1, 3, 3}));
  ASSERT_TRUE(result.masks);
  EXPECT_TRUE(torch::equal(
      (*result.masks)[0],
      torch::tensor({1, 16777215, 0, -7, 16777215, 0, 0, 0, 0},
                    torch::kInt64)
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
  const auto batch = collate_images(
      std::array{Photo(2, 1, 1), Photo(1, 1, 5)},
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

  const auto stack_batch = collate_images(
      std::array{Photo(2, 1, 1), Photo(2, 1, 5)},
      std::array{ImageBatchKey{"a"}, ImageBatchKey{"b"}});
  ImageBatchGeometryOptions invalid;
  invalid.output_size_divisor = 2;
  EXPECT_THROW(transform_image_batch_geometry(stack_batch, steps, invalid),
               InvalidArgument);
  EXPECT_THROW(
      sample_image_batch_geometry(stack_batch, "dataset", {}, invalid),
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
      std::array{a, b}, std::array{ImageBatchKey{"a", 4, 1},
                                  ImageBatchKey{"b", 2, 3}});
  RandomGeometryOptions random{.resize = ImageExtent{10, 8},
                               .crop = ImageExtent{6, 5},
                               .horizontal_flip_probability = 0.5,
                               .vertical_flip_probability = 0.25};
  ImageBatchGeometryOptions options;
  const auto first = sample_image_batch_geometry(
      batch, "dataset-sha256", random, options, 17, 9);
  const auto second = sample_image_batch_geometry(
      batch, "dataset-sha256", random, options, 17, 9);
  ASSERT_EQ(first.size(), second.size());
  EXPECT_TRUE(SameSteps(first[0], second[0]));
  EXPECT_TRUE(SameSteps(first[1], second[1]));
  const auto identity = image_batch_geometry_identity(random, options);
  --options.limits.max_geometry_checks;
  EXPECT_NE(image_batch_geometry_identity(random, options), identity);
  const auto reversed = collate_images(
      std::array{b, a}, std::array{ImageBatchKey{"b", 2, 3},
                                  ImageBatchKey{"a", 4, 1}});
  const auto reordered = sample_image_batch_geometry(
      reversed, "dataset-sha256", random, ImageBatchGeometryOptions{}, 17, 9);
  EXPECT_TRUE(SameSteps(first[0], reordered[1]));
  EXPECT_TRUE(SameSteps(first[1], reordered[0]));
  EXPECT_NE(first[0].size(), 0U);
}

TEST(ImageBatchGeometryTest, AggregateBudgetAndCancellationDoNotMutateInput) {
  const auto batch = collate_images(
      std::array{Photo(8, 8, 1), Photo(8, 8, 2)},
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
  EXPECT_THROW(transform_image_batch_geometry(batch, steps, options,
                                              stop.get_token()),
               Cancelled);
  EXPECT_TRUE(torch::equal(batch.images, original));
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchGeometryTest, RejectsCudaWithoutImplicitTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  auto batch = collate_images(
      std::array{Photo(2, 2, 1)}, std::array{ImageBatchKey{"a"}});
  batch.images = batch.images.cuda();
  batch.valid = batch.valid.cuda();
  batch.labels = batch.labels.cuda();
  const std::vector<std::vector<ImageGeometryStep>> steps(1);
  EXPECT_THROW(transform_image_batch_geometry(batch, steps), InvalidArgument);
}
#endif
}  // namespace
