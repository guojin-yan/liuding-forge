// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/visualization/render.hpp>

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <stop_token>
#include <vector>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;
using namespace jyppx::liuding::visualization;

PixelSpec RgbSpec() {
  PixelSpec spec;
  spec.layout = ImageLayout::chw;
  spec.color = PixelColorModel::rgb;
  spec.alpha = PixelAlphaMode::none;
  spec.dtype = torch::kUInt8;
  spec.range = PixelValueRange::unsigned_full_scale;
  return spec;
}

VisionAnnotationRecord Record() {
  VisionAnnotationRecord record;
  record.width = 5;
  record.height = 4;
  record.objects.push_back({.category = 0,
                            .box = ImageBox{1, 1, 4, 3},
                            .polygons = {{{1, 1}, {4, 1}, {2.5, 3}}},
                            .keypoints = {{{{2, 2}, 2}, {{3, 2}, 2}}}});
  return record;
}

TEST(RenderTest, ProducesIndependentRgbCanvasAndDrawsGeometry) {
  auto image = torch::zeros({3, 4, 5}, torch::kUInt8);
  image[0][2][2] = 127;
  const auto input_copy = image.clone();
  const std::array<VisionCategory, 1> categories{{VisionCategory{
      .source_id = 1, .name = "object", .skeleton = {{{0, 1}}}}}};

  const auto result =
      render_annotations(image, RgbSpec(), Record(), categories);
  ASSERT_EQ(result.pixels.sizes(), torch::IntArrayRef({4, 5, 3}));
  EXPECT_TRUE(result.pixels.device().is_cpu());
  EXPECT_EQ(result.pixels.scalar_type(), torch::kUInt8);
  EXPECT_EQ(result.spec.layout, ImageLayout::hwc);
  EXPECT_TRUE(torch::equal(image, input_copy));
  EXPECT_EQ(result.pixels[1][1][0].item<std::uint8_t>(), 230U);
  EXPECT_EQ(result.pixels[1][1][1].item<std::uint8_t>(), 25U);
  EXPECT_EQ(result.pixels[1][1][2].item<std::uint8_t>(), 75U);
}

TEST(RenderTest, PreservesDeclaredLayoutAndCompositesAlpha) {
  auto bgra_spec = RgbSpec();
  bgra_spec.layout = ImageLayout::hwc;
  bgra_spec.color = PixelColorModel::bgra;
  bgra_spec.alpha = PixelAlphaMode::straight;
  auto bgra = torch::zeros({1, 1, 4}, torch::kUInt8);
  bgra[0][0][2] = 200;
  bgra[0][0][3] = 128;
  RenderOptions options;
  options.background = {10, 20, 30};
  const VisionAnnotationRecord one_pixel{.width = 1, .height = 1};
  const auto rendered_bgra =
      render_annotations(bgra, bgra_spec, one_pixel, {}, {}, {}, options);
  EXPECT_EQ(rendered_bgra.pixels[0][0][0].item<std::uint8_t>(), 105U);
  EXPECT_EQ(rendered_bgra.pixels[0][0][1].item<std::uint8_t>(), 10U);
  EXPECT_EQ(rendered_bgra.pixels[0][0][2].item<std::uint8_t>(), 15U);

  auto gray_alpha_spec = RgbSpec();
  gray_alpha_spec.layout = ImageLayout::hwc;
  gray_alpha_spec.color = PixelColorModel::gray_alpha;
  gray_alpha_spec.alpha = PixelAlphaMode::straight;
  auto gray_alpha = torch::zeros({1, 1, 2}, torch::kUInt8);
  gray_alpha[0][0][0] = 128;
  gray_alpha[0][0][1] = 128;
  const auto rendered_gray = render_annotations(gray_alpha, gray_alpha_spec,
                                                one_pixel, {}, {}, {}, options);
  EXPECT_EQ(rendered_gray.pixels[0][0][0].item<std::uint8_t>(), 69U);
  EXPECT_EQ(rendered_gray.pixels[0][0][1].item<std::uint8_t>(), 74U);
  EXPECT_EQ(rendered_gray.pixels[0][0][2].item<std::uint8_t>(), 79U);
}

TEST(RenderTest, ReadsNoncontiguousChwPixelsUsingTheirStrides) {
  const auto hwc = torch::arange(12, torch::kUInt8).reshape({2, 2, 3});
  const auto chw = hwc.permute({2, 0, 1});
  auto spec = RgbSpec();
  spec.layout = ImageLayout::chw;
  const auto rendered = render_annotations(
      chw, spec, VisionAnnotationRecord{.width = 2, .height = 2});
  EXPECT_TRUE(torch::equal(rendered.pixels, hwc));
}

TEST(RenderTest, BlendsMaskAndHeatmapWithoutChangingSource) {
  const auto image = torch::full({3, 2, 3}, 100, torch::kUInt8);
  const auto mask = torch::tensor({{0, 1, 0}, {0, 0, 0}}, torch::kInt64);
  const auto heatmap = torch::zeros({2, 3}, torch::kFloat32);
  heatmap[0][0] = 1.0F;
  auto options = RenderOptions{};
  options.mask_alpha = 255;
  options.heatmap_alpha = 255;
  options.draw_boxes = false;
  options.draw_polygons = false;
  options.draw_keypoints = false;
  options.mask_ignore_id = 0;
  const auto result = render_annotations(
      image, RgbSpec(), VisionAnnotationRecord{.width = 3, .height = 2}, {},
      mask, heatmap, options);
  EXPECT_EQ(result.pixels[0][0][0].item<std::uint8_t>(), 255U);
  EXPECT_EQ(result.pixels[0][0][1].item<std::uint8_t>(), 0U);
  EXPECT_EQ(result.pixels[0][0][2].item<std::uint8_t>(), 0U);
  EXPECT_NE(result.pixels[0][1][0].item<std::uint8_t>(), 100U);
  EXPECT_TRUE(torch::equal(image, torch::full({3, 2, 3}, 100, torch::kUInt8)));
}

TEST(RenderTest, RendersColumnMajorInstanceRle) {
  VisionAnnotationRecord record{.width = 2, .height = 2};
  record.objects.push_back(
      {.category = 0, .rle = ImageMaskRle{2, 2, {0, 2, 2}}});
  auto options = RenderOptions{};
  options.draw_boxes = false;
  options.draw_polygons = false;
  options.draw_keypoints = false;
  options.mask_alpha = 255;
  const auto result =
      render_annotations(torch::zeros({3, 2, 2}, torch::kUInt8), RgbSpec(),
                         record, {}, {}, {}, options);
  EXPECT_EQ(result.pixels[0][0][0].item<std::uint8_t>(), 230U);
  EXPECT_EQ(result.pixels[1][0][0].item<std::uint8_t>(), 230U);
  EXPECT_EQ(result.pixels[0][1][0].item<std::uint8_t>(), 0U);
  EXPECT_EQ(result.pixels[1][1][0].item<std::uint8_t>(), 0U);
}

TEST(RenderTest, RejectsNonCpuAndInvalidHeatmapContracts) {
  auto image = torch::zeros({3, 2, 2}, torch::kUInt8);
  auto record = VisionAnnotationRecord{.width = 2, .height = 2};
  auto heatmap = torch::zeros({2, 3}, torch::kFloat32);
  EXPECT_THROW(render_annotations(image, RgbSpec(), record, {}, {}, heatmap),
               InvalidArgument);
  heatmap = torch::zeros({2, 2}, torch::kFloat32);
  heatmap[0][0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(render_annotations(image, RgbSpec(), record, {}, {}, heatmap),
               InvalidArgument);
  if (torch::cuda::is_available()) {
    const auto cuda_image = image.to(torch::kCUDA);
    EXPECT_THROW(render_annotations(cuda_image, RgbSpec(), record),
                 InvalidArgument);
  }
}

TEST(RenderTest, BoundsBaseAndOverlayPixelWork) {
  const auto image = torch::zeros({3, 2, 2}, torch::kUInt8);
  const VisionAnnotationRecord record{.width = 2, .height = 2};
  auto options = RenderOptions{};
  options.max_raster_ops = 3;
  EXPECT_THROW(
      render_annotations(image, RgbSpec(), record, {}, {}, {}, options),
      ResourceExhausted);

  const auto mask = torch::zeros({2, 2}, torch::kInt64);
  options.max_raster_ops = 7;
  options.mask_alpha = 0;
  EXPECT_THROW(
      render_annotations(image, RgbSpec(), record, {}, mask, {}, options),
      ResourceExhausted);
}

TEST(RenderTest, CountsGeneratedBoxVerticesAgainstPointBudget) {
  const auto image = torch::zeros({3, 2, 2}, torch::kUInt8);
  auto record = VisionAnnotationRecord{.width = 2, .height = 2};
  record.objects.push_back({.category = 0, .box = ImageBox{0, 0, 2, 2}});
  auto options = RenderOptions{};
  options.max_points = 3;
  EXPECT_THROW(
      render_annotations(image, RgbSpec(), record, {}, {}, {}, options),
      ResourceExhausted);
}

TEST(RenderTest, HonorsCancellationAndRasterBudget) {
  auto image = torch::zeros({3, 32, 32}, torch::kUInt8);
  auto record = VisionAnnotationRecord{.width = 32, .height = 32};
  record.objects.push_back({.category = 0, .box = ImageBox{0, 0, 32, 32}});
  auto options = RenderOptions{};
  options.max_raster_ops = 2;
  EXPECT_THROW(
      render_annotations(image, RgbSpec(), record, {}, {}, {}, options),
      ResourceExhausted);
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(
      render_annotations(image, RgbSpec(),
                         VisionAnnotationRecord{.width = 32, .height = 32}, {},
                         {}, {}, {}, source.get_token()),
      Cancelled);
}

}  // namespace
