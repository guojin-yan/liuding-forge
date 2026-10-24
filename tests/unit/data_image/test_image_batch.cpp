// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <array>
#include <bit>
#include <gtest/gtest.h>
#include <limits>
#include <map>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;
using namespace jyppx::liuding::data::image;

VisionSample Photo(std::uint32_t w = 3, std::uint32_t h = 2, int channels = 3) {
  VisionSample s;
  s.width = w;
  s.height = h;
  s.image_path = "fixture.png";
  s.pixels.color = channels == 1 ? PixelColorModel::gray : PixelColorModel::rgb;
  s.image =
      torch::arange(static_cast<std::int64_t>(channels) * w * h, torch::kInt64)
          .reshape({channels, h, w})
          .to(torch::kUInt8);
  s.geometry.emplace(ImageExtent{w, h});
  return s;
}
std::vector<ImageBatchKey> Keys(std::size_t n) {
  std::vector<ImageBatchKey> keys;
  for (std::size_t i = 0; i < n; ++i)
    keys.push_back({"sample-" + std::to_string(i), i, i + 1});
  return keys;
}
ImageBatch Batch(const std::vector<VisionSample>& samples,
                 ImageCollateOptions options = {}) {
  return collate_images(samples, Keys(samples.size()), options);
}
void Mask(VisionSample& s) {
  s.mask_target.emplace();
  s.mask_target->ids =
      torch::arange(static_cast<std::int64_t>(s.width) * s.height,
                    torch::kInt64)
          .reshape({s.height, s.width}) +
      9007199254740993LL;
  s.mask_target->width = s.width;
  s.mask_target->height = s.height;
  s.mask_target->ignore_id = -7;
  s.mask_target->void_id = -9;
}
std::shared_ptr<VisionAnnotationRecord> Record(VisionSample& s) {
  s.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{
          {.source_id = 71, .name = "object", .keypoint_names = {"one"}}});
  auto r = std::make_shared<VisionAnnotationRecord>();
  r->source_image_id = std::numeric_limits<std::int64_t>::min();
  r->image_path = s.image_path;
  r->width = s.width;
  r->height = s.height;
  s.annotation = r;
  return r;
}
ImageMaskRle Encode(const std::vector<int>& row_major, std::uint32_t w,
                    std::uint32_t h) {
  ImageMaskRle r{w, h, {0}};
  int previous = 0;
  for (std::uint32_t x = 0; x < w; ++x)
    for (std::uint32_t y = 0; y < h; ++y) {
      const int value = row_major[y * w + x];
      if (previous != value) {
        r.counts.push_back(0);
        previous = value;
      }
      ++r.counts.back();
    }
  return r;
}
std::vector<int> Decode(const ImageMaskRle& r) {
  std::vector<int> row_major(static_cast<std::size_t>(r.width) * r.height);
  std::uint64_t index = 0;
  bool value = false;
  for (const auto count : r.counts) {
    for (std::uint64_t i = 0; i < count; ++i, ++index)
      row_major[(index % r.height) * r.width + index / r.height] =
          value ? 1 : 0;
    value = !value;
  }
  return row_major;
}

TEST(ImageBatchTest, StackAcceptsMixedStridedLayoutsAndOwnsFinalStorage) {
  auto a = Photo();
  auto b = a;
  b.image = torch::zeros({2, 6, 3}, torch::kUInt8).slice(1, 0, 6, 2);
  b.image.copy_(a.image.permute({1, 2, 0}));
  b.pixels.layout = ImageLayout::hwc;
  ASSERT_FALSE(b.image.is_contiguous());
  const auto batch = Batch({a, b});
  EXPECT_EQ(batch.images.sizes(), torch::IntArrayRef({2, 3, 2, 3}));
  EXPECT_TRUE(batch.images.is_contiguous());
  EXPECT_FALSE(batch.images.is_pinned());
  EXPECT_TRUE(torch::equal(batch.images[0], batch.images[1]));
  EXPECT_EQ(batch.valid.sum().item<int>(), 12);
  a.image.zero_();
  b.image.zero_();
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
  EXPECT_EQ(batch.items[1].key.id, "sample-1");
  EXPECT_EQ(batch.items[1].key.occurrence, 1U);
  EXPECT_EQ(batch.items[1].key.view, 2U);
}

TEST(ImageBatchTest,
     ClassificationOffsetsAndMissingVersusEmptyAnnotationsRoundTrip) {
  auto a = Photo(), b = Photo(), c = Photo();
  const auto classes = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{3, "cat"}, {9, "dog"}});
  a.label_categories = b.label_categories = c.label_categories = classes;
  a.labels = {1, 0};
  c.labels = {0};
  Record(b);
  auto record = Record(c);
  record->objects.push_back({.source_id = 811,
                             .box = ImageBox{0, 0, 2, 1},
                             .keypoints = {{{1, 1}, 2}},
                             .area = 42,
                             .group_id = -11,
                             .crowd = true,
                             .ignore = true,
                             .difficult = true,
                             .truncated = true});
  auto keys = Keys(3);
  keys[1].occurrence = std::numeric_limits<std::uint64_t>::max();
  keys[2].view = 1ULL << 63U;
  const auto batch = collate_images(std::array{a, b, c}, keys);
  const auto value = image_batch_value(batch);
  validate_batch_value(value, image_batch_schema(batch, torch::kCPU));
  const auto result = image_batch_from_value(value);
  EXPECT_TRUE(
      torch::equal(result.labels, torch::tensor({1, 0, 0}, torch::kInt64)));
  EXPECT_EQ(result.label_offsets, (std::vector<std::int64_t>{0, 2, 2, 3}));
  EXPECT_FALSE(result.items[0].annotation);
  ASSERT_TRUE(result.items[1].annotation);
  EXPECT_TRUE(result.items[1].annotation->objects.empty());
  ASSERT_EQ(result.items[2].annotation->objects.size(), 1U);
  const auto& object = result.items[2].annotation->objects[0];
  EXPECT_EQ(object.source_id, 811);
  EXPECT_EQ(object.group_id, -11);
  EXPECT_TRUE(object.crowd && object.ignore && object.difficult &&
              object.truncated);
  EXPECT_EQ(object.area, 42);
  EXPECT_EQ(object.keypoints[0].visibility, 2);
  EXPECT_EQ(result.items[1].key.occurrence,
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(result.items[2].key.view, 1ULL << 63U);
  EXPECT_EQ(result.items[2].annotation->source_image_id,
            std::numeric_limits<std::int64_t>::min());
  EXPECT_EQ((*result.items[2].label_categories)[1].source_id, 9);
}

TEST(ImageBatchTest, PadUsesExplicitFillValidExtentsAndExistingContentTrace) {
  auto a = Photo(2, 2), b = Photo(3, 1);
  Mask(a);
  Mask(b);
  a.geometry = GeometryTrace({1, 2}).padded(0, 0, 1, 0);
  const auto original = a.mask_target->ids.clone();
  ImageCollateOptions options;
  options.layout = ImageBatchLayout::pad;
  options.size_divisor = 4;
  options.image_padding = 77;
  options.mask_padding_id = -7;
  const auto batch = Batch({a, b}, options);
  EXPECT_EQ(batch.images.sizes(), torch::IntArrayRef({2, 3, 4, 4}));
  EXPECT_EQ(batch.valid.sum().item<int>(), 7);
  EXPECT_TRUE(batch.valid[0][0][1]
                  .item<bool>());  // preexisting content padding remains valid
  EXPECT_FALSE(batch.items[0].geometry->inverse_point({1.5, 1}));
  EXPECT_FALSE(batch.valid[0][0][2].item<bool>());
  EXPECT_EQ(batch.images[1][0][3][3].item<int>(), 77);
  EXPECT_EQ((*batch.masks)[0][3][3].item<std::int64_t>(), -7);
  EXPECT_EQ((*batch.masks)[0][0][0].item<std::int64_t>(), 9007199254740993LL);
  EXPECT_TRUE(torch::equal(original, a.mask_target->ids));
  EXPECT_EQ(batch.items[0].valid_extent.width, 2U);
  EXPECT_EQ(batch.items[0].geometry->output_extent().width, 4U);
}

TEST(ImageBatchTest, RaggedOwnsDifferentSizesAndSeparatePairedPixelSemantics) {
  auto a = Photo(), b = Photo(1, 4);
  for (auto* s : {&a, &b}) {
    s->paired_target = s->image.to(torch::kFloat64).div(255);
    s->paired_target_pixels = s->pixels;
    s->paired_target_pixels->dtype = torch::kFloat64;
    s->paired_target_pixels->range = PixelValueRange::unit_interval;
    s->target_path = "paired.png";
  }
  const auto batch = Batch({a, b}, {.layout = ImageBatchLayout::ragged});
  EXPECT_FALSE(batch.images.defined());
  EXPECT_FALSE(batch.paired_images);
  ASSERT_EQ(batch.image_list.size(), 2U);
  ASSERT_EQ(batch.paired_image_list.size(), 2U);
  EXPECT_EQ(batch.valid_list[1].sizes(), torch::IntArrayRef({4, 1}));
  const auto result = image_batch_from_value(image_batch_value(batch));
  EXPECT_TRUE(torch::equal(result.paired_image_list[1], *b.paired_target));
  b.paired_target->zero_();
  EXPECT_GT(result.paired_image_list[1].sum().item<double>(), 0);
}

TEST(ImageBatchTest, PadRleMatchesIndependentPixelsForAllSmallBinaryMasks) {
  for (unsigned bits = 0; bits < 512; ++bits) {
    auto s = Photo(3, 3, 1);
    auto record = Record(s);
    std::vector<int> pixels(9);
    for (unsigned i = 0; i < 9; ++i) pixels[i] = (bits >> i) & 1U;
    auto runs = Encode(pixels, 3, 3);
    // Additional zero runs retain parity but must not lose or invent pixels.
    runs.counts.insert(runs.counts.begin(), {0, 0});
    record->objects.push_back(
        {.source_id = 17,
         .rle = runs,
         .area = 42,
         .mask_pixel_area = static_cast<std::uint64_t>(std::popcount(bits)),
         .crowd = true});
    const auto result = Batch({s}, {.layout = ImageBatchLayout::pad,
                                    .padded_extent = ImageExtent{5, 4}});
    const auto& object = result.items[0].annotation->objects[0];
    const auto actual = Decode(*object.rle);
    for (unsigned y = 0; y < 4; ++y)
      for (unsigned x = 0; x < 5; ++x)
        ASSERT_EQ(actual[y * 5 + x], x < 3 && y < 3 ? pixels[y * 3 + x] : 0)
            << bits;
    EXPECT_EQ(object.area, 42);
    EXPECT_TRUE(object.crowd);
    EXPECT_EQ(object.mask_pixel_area, std::popcount(bits));
    EXPECT_EQ(record->width, 3U);
    EXPECT_EQ(record->objects[0].rle->height, 3U);
  }
}

TEST(ImageBatchTest, PadRleHasHandDerivedColumnMajorRuns) {
  auto s = Photo(2, 2);
  auto r = Record(s);
  r->objects.push_back(
      {.rle = ImageMaskRle{2, 2, {1, 2, 1}}, .mask_pixel_area = 2});
  const auto batch = Batch({s}, {.layout = ImageBatchLayout::pad,
                                 .padded_extent = ImageExtent{3, 3}});
  EXPECT_EQ(batch.items[0].annotation->objects[0].rle->counts,
            (std::vector<std::uint64_t>{1, 1, 1, 1, 5}));
}

TEST(ImageBatchTest,
     AllRichGeometryAndConversionFlagsSurvivePaddingAndValueRoundTrip) {
  auto s = Photo(3, 3);
  auto r = Record(s);
  r->objects.push_back({.source_id = 8,
                        .polygons = {{{0, 0}, {2, 0}, {1, 2}}},
                        .oriented_box_to_polygon = true,
                        .polygon_partitioned = true});
  r->objects.push_back({.source_id = 9,
                        .oriented_box = std::array<ImagePoint, 4>{
                            {{0, 0}, {2, 0}, {2, 1}, {0, 1}}}});
  const auto batch = Batch({s}, {.layout = ImageBatchLayout::pad,
                                 .padded_extent = ImageExtent{4, 5}});
  const auto result = image_batch_from_value(image_batch_value(batch));
  const auto& record = *result.items[0].annotation;
  EXPECT_EQ(record.width, 4U);
  EXPECT_EQ(record.height, 5U);
  ASSERT_EQ(record.objects.size(), 2U);
  EXPECT_TRUE(record.objects[0].polygon_partitioned &&
              record.objects[0].oriented_box_to_polygon);
  EXPECT_DOUBLE_EQ(record.objects[0].polygons[0][2].y, 2);
  EXPECT_DOUBLE_EQ((*record.objects[1].oriented_box)[2].x, 2);
}

VisionSample Panoptic() {
  auto s = Photo(2, 2);
  Mask(s);
  auto r = Record(s);
  s.mask_target->ids = torch::tensor({{4, 4}, {0, -7}}, torch::kInt64);
  s.mask_target->encoding = MaskEncoding::panoptic_rgb;
  s.mask_target->void_id = 0;
  r->panoptic_mask_path = "segments.png";
  r->objects.push_back(
      {.source_id = 4, .box = ImageBox{0, 0, 2, 1}, .area = 2, .crowd = true});
  return s;
}
TEST(ImageBatchTest, PanopticBindsOnCpuAndPaddingMustUseVoid) {
  auto s = Panoptic();
  ImageCollateOptions options{.layout = ImageBatchLayout::pad,
                              .padded_extent = ImageExtent{3, 3},
                              .mask_padding_id = 0};
  const auto batch = Batch({s}, options);
  const auto result = image_batch_from_value(image_batch_value(batch));
  EXPECT_EQ((*result.masks)[0][0][1].item<std::int64_t>(), 4);
  EXPECT_EQ((*result.masks)[0][1][1].item<std::int64_t>(), -7);
  EXPECT_EQ((*result.masks)[0][2][2].item<std::int64_t>(), 0);
  const auto& record = *result.items[0].annotation;
  EXPECT_TRUE(record.panoptic_mask_materialized);
  EXPECT_EQ(record.objects[0].mask_pixel_area, 2);
  EXPECT_EQ(record.objects[0].area, 2);
  EXPECT_DOUBLE_EQ(record.objects[0].box->y_max, 1);
  EXPECT_FALSE(s.annotation->panoptic_mask_materialized);
  options.mask_padding_id = -7;
  EXPECT_THROW(Batch({s}, options), InvalidArgument);
  s.mask_target->ids[0][0] = 999;
  options.mask_padding_id = 0;
  EXPECT_THROW(Batch({s}, options), SerializationError);
}

TEST(ImageBatchTest, RejectsAmbiguousShapesTargetsFillsLabelsAndCancellation) {
  auto a = Photo(), b = Photo(2, 3);
  EXPECT_THROW(Batch({a, b}), InvalidArgument);
  EXPECT_THROW(Batch({a}, {.padded_extent = ImageExtent{4, 4}}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.layout = ImageBatchLayout::pad,
                           .padded_extent = ImageExtent{2, 2}}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.image_padding = 0.5}), InvalidArgument);
  EXPECT_THROW(
      Batch({a}, {.image_padding = std::numeric_limits<double>::infinity()}),
      InvalidArgument);
  Mask(a);
  EXPECT_THROW(Batch({a, b}, {.layout = ImageBatchLayout::ragged}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.layout = ImageBatchLayout::pad, .size_divisor = 4}),
               InvalidArgument);
  a = Photo();
  a.labels = {0};
  EXPECT_THROW(Batch({a}), InvalidArgument);
  a = Photo();
  a.image = a.image.to(torch::kFloat32).requires_grad_(true);
  a.pixels.dtype = torch::kFloat32;
  a.pixels.range = PixelValueRange::unbounded;
  EXPECT_THROW(Batch({a}), InvalidArgument);
  std::stop_source cancel;
  cancel.request_stop();
  EXPECT_THROW(
      collate_images(std::array{Photo()}, Keys(1), {}, cancel.get_token()),
      Cancelled);
  EXPECT_THROW(collate_images({}, {}), InvalidArgument);
}

TEST(ImageBatchTest,
     RejectsMetadataTensorPixelAndWorkingBudgetBeforePublication) {
  const auto s = Photo();
  for (const int field : {0, 1, 2, 3, 4, 5}) {
    ImageCollateOptions options;
    if (field == 0) options.limits.max_output_bytes = 18;
    if (field == 1) options.limits.max_pixels = 5;
    if (field == 2) options.limits.max_dimension = 2;
    if (field == 3) options.limits.max_metadata_nodes = 32;
    if (field == 4) options.limits.max_metadata_bytes = 100;
    if (field == 5) options.limits.max_working_bytes = 100;
    EXPECT_THROW(Batch({s}, options), ResourceExhausted) << field;
  }
  auto malformed = s;
  auto r = Record(malformed);
  r->objects.push_back({.rle = ImageMaskRle{3, 2, {5}}});
  EXPECT_THROW(Batch({malformed}), InvalidArgument);
  r->objects[0].rle =
      ImageMaskRle{3, 2, {std::numeric_limits<std::uint64_t>::max()}};
  EXPECT_THROW(Batch({malformed}), InvalidArgument);
}

TEST(ImageBatchTest,
     ValueRejectsWrongVersionFieldsArityTensorTypeAndSemantics) {
  const auto batch = Batch({Photo()});
  const auto mutate = [&](const char* key, c10::IValue replacement) {
    auto value = image_batch_value(batch);
    auto dictionary = value.toGenericDict();
    dictionary.insert_or_assign(key, std::move(replacement));
    EXPECT_THROW(image_batch_from_value(value), InvalidArgument) << key;
  };
  mutate("format", 2);
  mutate("layout", 77);
  mutate("extra", 1);
  mutate("images", batch.images.to(torch::kFloat32));
  mutate("valid", torch::ones({1, 2, 3}, torch::kInt64));
  mutate("valid", torch::ones({0, 2, 3}, torch::kBool));
  mutate("pixels", c10::ivalue::Tuple::create({1, 2}));
  mutate("label_offsets", c10::List<std::int64_t>({0, 1}));
  auto bad = batch;
  bad.items[0].valid_extent.width = 4;
  EXPECT_THROW(image_batch_value(bad), InvalidArgument);
  ImageBatchLimits limits;
  limits.max_metadata_nodes = 30;
  EXPECT_THROW(image_batch_from_value(image_batch_value(batch), limits),
               ResourceExhausted);
  limits = {};
  limits.max_output_bytes = 1;
  EXPECT_THROW(image_batch_from_value(image_batch_value(batch), limits),
               ResourceExhausted);
}

TEST(ImageBatchTransformTest,
     PackedNormalizationUsesIndependentAnalyticValues) {
  auto a = Photo(), b = Photo();
  Mask(a);
  Mask(b);
  const auto batch = Batch({a, b});
  ImageBatchTransformOptions options;
  options.image.dtype = torch::kFloat64;
  options.image.swap_red_blue = true;
  options.image.mean = {0.1, 0.2, 0.3};
  options.image.std = {0.25, 0.5, 2};
  const auto result = transform_image_batch(batch, options);
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x) {
          const double encoded = (2 - c) * 6 + y * 3 + x;
          EXPECT_NEAR(
              result.images[n][c][y][x].item<double>(),
              (encoded / 255 - options.image.mean[c]) / options.image.std[c],
              1e-14);
        }
  EXPECT_EQ(result.pixels.color, PixelColorModel::bgr);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unbounded);
  EXPECT_TRUE(torch::equal(*result.masks, *batch.masks));
  EXPECT_EQ(result.masks->data_ptr(), batch.masks->data_ptr());
  EXPECT_EQ(batch.images.scalar_type(), torch::kUInt8);
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
  EXPECT_NE(result.images.data_ptr(), batch.images.data_ptr());
}

TEST(ImageBatchTransformTest, Uint16ScaleAndRaggedPairedPoliciesAreExplicit) {
  auto a = Photo(2, 1, 1), b = Photo(1, 3, 1);
  for (auto* s : {&a, &b}) {
    s->image = torch::full({1, s->height, s->width}, 65535, torch::kUInt16);
    s->pixels.dtype = torch::kUInt16;
    s->paired_target =
        torch::full({1, s->height, s->width}, 127, torch::kUInt8);
    s->paired_target_pixels = s->pixels;
    s->paired_target_pixels->dtype = torch::kUInt8;
  }
  const auto batch = Batch({a, b}, {.layout = ImageBatchLayout::ragged});
  ImageBatchTransformOptions options;
  options.paired = BatchPixelTransform{.dtype = torch::kFloat64,
                                       .scale = BatchPixelScale::preserve};
  const auto result = transform_image_batch(batch, options);
  EXPECT_EQ(result.image_list[0][0][0][0].item<float>(), 1);
  EXPECT_EQ(result.paired_image_list[1][0][2][0].item<double>(), 127);
  EXPECT_EQ(result.paired_pixels->range, PixelValueRange::unbounded);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unit_interval);
  const auto main_only = transform_image_batch(batch);
  EXPECT_EQ(main_only.paired_image_list[0].data_ptr(),
            batch.paired_image_list[0].data_ptr());
}

TEST(ImageBatchTransformTest,
     LowPrecisionCastsUseFloatComputeAndUnitInputIsNotRescaled) {
  auto s = Photo(2, 1, 1);
  s.image = torch::tensor({0.25, 0.75}, torch::kFloat32).reshape({1, 1, 2});
  s.pixels.dtype = torch::kFloat32;
  s.pixels.range = PixelValueRange::unit_interval;
  const auto batch = Batch({s});
  for (const auto dtype :
       {torch::kFloat16, torch::kBFloat16, torch::kFloat32, torch::kFloat64}) {
    ImageBatchTransformOptions options;
    options.image.dtype = dtype;
    options.image.mean = {0.5};
    options.image.std = {0.25};
    const auto result = transform_image_batch(batch, options);
    EXPECT_EQ(result.images.scalar_type(), dtype);
    EXPECT_EQ(result.images[0][0][0][0].item<double>(), -1);
    EXPECT_EQ(result.images[0][0][0][1].item<double>(), 1);
  }
}

TEST(ImageBatchTransformTest,
     AlphaAssociationIsPreservedAndNormalizationIsExplicitlyRejected) {
  auto s = Photo(1, 1);
  s.image = torch::tensor({16, 32, 64, 128}, torch::kUInt8).reshape({4, 1, 1});
  s.pixels.color = PixelColorModel::rgba;
  s.pixels.alpha = PixelAlphaMode::premultiplied;
  const auto batch = Batch({s});
  ImageBatchTransformOptions o;
  o.image.dtype = torch::kFloat64;
  o.image.swap_red_blue = true;
  const auto result = transform_image_batch(batch, o);
  EXPECT_EQ(result.pixels.color, PixelColorModel::bgra);
  EXPECT_EQ(result.pixels.alpha, PixelAlphaMode::premultiplied);
  EXPECT_DOUBLE_EQ(result.images[0][0][0][0].item<double>(), 64.0 / 255);
  EXPECT_DOUBLE_EQ(result.images[0][3][0][0].item<double>(), 128.0 / 255);
  o.image.mean = {0, 0, 0, 0};
  o.image.std = {1, 1, 1, 1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
}

TEST(ImageBatchTransformTest, PackedPairedPhotosUseIndependentNumericPolicies) {
  auto s = Photo(2, 1, 1);
  s.paired_target = torch::full({1, 1, 2}, 0.75, torch::kFloat64);
  s.paired_target_pixels = s.pixels;
  s.paired_target_pixels->dtype = torch::kFloat64;
  s.paired_target_pixels->range = PixelValueRange::unit_interval;
  const auto batch = Batch({s, s});
  ImageBatchTransformOptions o;
  o.paired = BatchPixelTransform{
      .dtype = torch::kFloat64, .mean = {0.5}, .std = {0.25}};
  const auto result = transform_image_batch(batch, o);
  EXPECT_DOUBLE_EQ((*result.paired_images)[1][0][0][1].item<double>(), 1);
  EXPECT_DOUBLE_EQ((*batch.paired_images)[1][0][0][1].item<double>(), 0.75);
  EXPECT_NE(result.paired_images->data_ptr(), batch.paired_images->data_ptr());
  EXPECT_EQ(result.pixels.range, PixelValueRange::unit_interval);
  EXPECT_EQ(result.paired_pixels->range, PixelValueRange::unbounded);
}

TEST(ImageBatchTransformTest,
     RejectsInvalidParametersAndBudgetsWithoutMutatingInput) {
  const auto batch = Batch({Photo()});
  ImageBatchTransformOptions o;
  o.image.mean = {0};
  o.image.std = {1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o.image.mean = {0, 0, 0};
  o.image.std = {1, 0, 1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.image.dtype = torch::kInt64;
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.paired.emplace();
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.limits.max_output_bytes = 24;
  EXPECT_THROW(transform_image_batch(batch, o), ResourceExhausted);
  o = {};
  o.limits.max_working_bytes = 1200;
  EXPECT_THROW(transform_image_batch(batch, o), ResourceExhausted);
  auto normalized = transform_image_batch(
      batch, {.image = {.scale = BatchPixelScale::preserve}});
  EXPECT_THROW(transform_image_batch(normalized), InvalidArgument);
  std::stop_source cancel;
  cancel.request_stop();
  EXPECT_THROW(transform_image_batch(batch, {}, cancel.get_token()), Cancelled);
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
}

TEST(ImageBatchTest, IdentityBindsAllPoliciesAndSchemaBindsPixelMeaning) {
  ImageCollateOptions c;
  const auto original = image_batch_identity(c);
  // Independently encoded with Python hashlib/struct, without C++ helpers.
  EXPECT_EQ(original,
            "a101f266d838ae1d48e5b6a4048c0336d8b1d7880a5dd292ba6fd1c03a044ffc");
  c.image_padding = -0.0;
  EXPECT_EQ(image_batch_identity(c), original);
  c.limits.max_geometry_checks--;
  EXPECT_NE(image_batch_identity(c), original);
  c = {};
  c.mask_padding_id = 0;
  EXPECT_NE(image_batch_identity(c), original);
  const auto numeric = image_batch_identity({}, ImageBatchTransformOptions{});
  EXPECT_NE(original, numeric);
  ImageBatchTransformOptions n;
  n.image.mean = {0, 0, 0};
  n.image.std = {1, 1, 1};
  const auto norm = image_batch_identity({}, n);
  n.image.std[2] = 2;
  EXPECT_NE(image_batch_identity({}, n), norm);
  n = {};
  n.limits.max_metadata_nodes--;
  EXPECT_NE(image_batch_identity({}, n), numeric);
  auto a = Batch({Photo()}), b = a;
  b.pixels.color = PixelColorModel::bgr;
  EXPECT_NE(canonical_batch_schema(image_batch_schema(a, torch::kCPU)),
            canonical_batch_schema(image_batch_schema(b, torch::kCPU)));
  b = a;
  b.layout = ImageBatchLayout::pad;
  EXPECT_NE(canonical_batch_schema(image_batch_schema(a, torch::kCPU)),
            canonical_batch_schema(image_batch_schema(b, torch::kCPU)));
}

TEST(ImageBatchCUDATest, FinalBatchPinTransferNumericAndRaggedMetadata) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                            ImageBatchLayout::ragged}) {
    auto a = Photo(), b = Photo();
    Mask(a);
    Mask(b);
    ImageCollateOptions collate;
    collate.layout = layout;
    if (layout == ImageBatchLayout::pad) {
      collate.size_divisor = 4;
      collate.mask_padding_id = -7;
    }
    const auto batch = Batch({a, b}, collate);
    BatchMemoryOptions memory{.pin_memory = true,
                              .target_device = Device(torch::kCUDA, 0),
                              .non_blocking = true};
    const auto prepared = prepare_image_batch(batch, memory);
    const auto& pinned = layout == ImageBatchLayout::ragged
                             ? prepared.image_list[0]
                             : prepared.images;
    EXPECT_TRUE(pinned.device().is_cpu());
    EXPECT_TRUE(pinned.is_pinned());
    EXPECT_EQ(prepared.memory.pinned_tensor_count,
              layout == ImageBatchLayout::ragged ? 6U : 3U);
    const auto gpu = transfer_image_batch(prepared, memory);
    ImageBatchTransformOptions numeric;
    numeric.image.mean = {0.1, 0.2, 0.3};
    numeric.image.std = {0.25, 0.5, 2};
    const auto converted = transform_image_batch(gpu, numeric);
    EXPECT_FALSE(converted.memory.staged);
    const auto round_trip =
        image_batch_from_value(image_batch_value(converted));
    const auto cpu = transform_image_batch(batch, numeric);
    const auto& actual = layout == ImageBatchLayout::ragged
                             ? round_trip.image_list[0]
                             : round_trip.images;
    const auto& expected =
        layout == ImageBatchLayout::ragged ? cpu.image_list[0] : cpu.images;
    EXPECT_TRUE(actual.device().is_cuda());
    EXPECT_TRUE(torch::allclose(actual.cpu(), expected, 1e-6, 1e-7));
    const auto& mask =
        layout == ImageBatchLayout::ragged ? gpu.mask_list[0] : *gpu.masks;
    const auto& original =
        layout == ImageBatchLayout::ragged ? batch.mask_list[0] : *batch.masks;
    EXPECT_TRUE(torch::equal(mask.cpu(), original));
    EXPECT_EQ(round_trip.items[0].key.id, "sample-0");
  }
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchCUDATest, NumericAndTransferUseCallerCurrentNondefaultStream) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto batch = Batch({Photo(), Photo()});
  BatchMemoryOptions memory{.pin_memory = true,
                            .target_device = Device(torch::kCUDA, 0),
                            .non_blocking = true};
  const auto pinned = prepare_image_batch(batch, memory);
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  auto transferred = transfer_image_batch(pinned, memory);
  transferred.images.fill_(255);  // Ordered producer operation on this stream.
  ImageBatchTransformOptions numeric;
  numeric.image.mean = {0.5, 0.5, 0.5};
  numeric.image.std = {0.25, 0.25, 0.25};
  const auto result = transform_image_batch(transferred, numeric);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
  EXPECT_EQ(result.images.sum().item<float>(), 72);  // synchronize only in test
}
#endif

std::unique_ptr<DataPipeline> Pipeline(std::size_t workers, bool fail = false) {
  DataPipelineOptions options;
  options.dataset_size = 5;
  options.dataset_version = "image-batch-fixture/v1";
  options.batch_factory_id = "image-batch-fixture/stack-v1";
  options.batch_size = 2;
  options.num_workers = workers;
  options.schema = image_batch_schema(Batch({Photo()}), torch::kCPU);
  return std::make_unique<DataPipeline>(
      std::make_shared<samplers::SequentialSampler>(5), options,
      [fail](std::span<const std::size_t> indices,
             const DataWorkerContext& context, std::stop_token stop) {
        if (fail && context.sequence == 1)
          throw InvalidArgument("injected image collator failure");
        std::vector<VisionSample> samples;
        std::vector<ImageBatchKey> keys;
        for (const auto index : indices) {
          samples.push_back(Photo());
          samples.back().image.fill_(static_cast<int>(index));
          keys.push_back({std::to_string(index), context.epoch, 0});
        }
        return image_batch_value(collate_images(samples, keys, {}, stop));
      });
}

class SavedState final : public StateReader, public StateWriter {
 public:
  explicit SavedState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& b = blobs_.at(std::string(name));
    if (b.size() > maximum) throw std::runtime_error("state size");
    return b;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs_[std::string(name)] = {bytes.begin(), bytes.end()};
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

TEST(ImageBatchPipelineTest, FullSchemaStableKeysTailAndSameTopologyRestore) {
  auto inline_pipeline = Pipeline(0), workers = Pipeline(2);
  inline_pipeline->set_epoch(8);
  workers->set_epoch(8);
  ASSERT_TRUE(workers->next());
  ASSERT_TRUE(inline_pipeline->next());
  SavedState state(workers->describe_state());
  workers->save_state(state);
  auto resumed = Pipeline(2);
  resumed->load_state(state, RestorePolicy::exact_only);
  while (const auto expected = inline_pipeline->next()) {
    const auto actual = resumed->next();
    ASSERT_TRUE(actual);
    const auto a = image_batch_from_value(actual->value),
               b = image_batch_from_value(expected->value);
    EXPECT_TRUE(torch::equal(a.images, b.images));
    ASSERT_EQ(a.items.size(), b.items.size());
    EXPECT_EQ(a.items[0].key.id, b.items[0].key.id);
    EXPECT_EQ(a.items[0].key.occurrence, 8U);
  }
  EXPECT_FALSE(resumed->next());
  EXPECT_EQ(resumed->state().next_cursor, 5U);
  EXPECT_THROW(inline_pipeline->load_state(state, RestorePolicy::exact_only),
               CompatibilityError);
}

TEST(ImageBatchPipelineTest,
     FactoryFailurePreservesCommittedCursorAndCanReplay) {
  for (const auto workers : {0U, 2U}) {
    auto pipeline = Pipeline(workers, true);
    pipeline->set_epoch(8);
    ASSERT_TRUE(pipeline->next());
    EXPECT_THROW(pipeline->next(), InvalidArgument);
    EXPECT_EQ(pipeline->state().next_cursor, 2U);
    SavedState state(pipeline->describe_state());
    pipeline->save_state(state);
    auto resumed = Pipeline(workers);
    resumed->load_state(state, RestorePolicy::exact_only);
    const auto next = resumed->next();
    ASSERT_TRUE(next);
    EXPECT_EQ(image_batch_from_value(next->value).items[0].key.id, "2");
  }
}
}  // namespace
