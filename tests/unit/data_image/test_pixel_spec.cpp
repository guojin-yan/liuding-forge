// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/pixel_spec.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <gtest/gtest.h>

namespace {
using namespace jyppx::liuding::data::image;
using jyppx::liuding::InvalidArgument;

TEST(PixelSpecTest, AcceptsExplicitChwAndStridedHwcSemantics) {
  const auto chw = torch::zeros({3, 5, 7}, torch::kUInt8);
  PixelSpec spec;
  EXPECT_EQ(spec.channels(), 3U);
  EXPECT_NO_THROW(validate_pixels(chw, spec));
  spec.layout = ImageLayout::hwc;
  spec.color = PixelColorModel::bgr;
  const auto hwc = chw.permute({1, 2, 0});
  ASSERT_FALSE(hwc.is_contiguous());
  EXPECT_NO_THROW(validate_pixels(hwc, spec));
  EXPECT_THROW(validate_pixels(chw, spec), InvalidArgument);
}

TEST(PixelSpecTest, KeepsSixteenBitIntensityAndExplicitFloatRangeDistinct) {
  PixelSpec spec;
  spec.color = PixelColorModel::gray;
  spec.dtype = torch::kUInt16;
  EXPECT_NO_THROW(
      validate_pixels(torch::zeros({1, 2, 3}, torch::kUInt16), spec));
  spec.dtype = torch::kFloat32;
  EXPECT_THROW(validate_pixels(torch::zeros({1, 2, 3}), spec), InvalidArgument);
  spec.range = PixelValueRange::unit_interval;
  EXPECT_NO_THROW(validate_pixels(torch::zeros({1, 2, 3}), spec));
  spec.range = PixelValueRange::unbounded;
  EXPECT_NO_THROW(validate_pixels(torch::full({1, 2, 3}, -12.5), spec));
  spec.dtype = torch::kUInt8;
  EXPECT_THROW(validate_pixels(torch::zeros({1, 2, 3}, torch::kUInt8), spec),
               InvalidArgument);
}

TEST(PixelSpecTest, RequiresAlphaAssociationToMatchChannelMeaning) {
  PixelSpec spec;
  spec.color = PixelColorModel::rgba;
  const auto pixels = torch::zeros({4, 2, 3}, torch::kUInt8);
  EXPECT_THROW(validate_pixels(pixels, spec), InvalidArgument);
  spec.alpha = PixelAlphaMode::straight;
  EXPECT_NO_THROW(validate_pixels(pixels, spec));
  spec.alpha = PixelAlphaMode::premultiplied;
  EXPECT_NO_THROW(validate_pixels(pixels, spec));
  spec.color = PixelColorModel::rgb;
  EXPECT_THROW(validate_pixels(pixels.narrow(0, 0, 3), spec), InvalidArgument);
  spec.color = PixelColorModel::gray_alpha;
  EXPECT_EQ(spec.channels(), 2U);
  EXPECT_NO_THROW(validate_pixels(pixels.narrow(0, 0, 2), spec));
  spec.color = PixelColorModel::bgra;
  EXPECT_EQ(spec.channels(), 4U);
  EXPECT_NO_THROW(validate_pixels(pixels, spec));
}

TEST(PixelSpecTest, RejectsUndefinedEmptyBatchMismatchedAndUnknownValues) {
  PixelSpec spec;
  const auto valid = torch::zeros({3, 2, 4}, torch::kUInt8);
  EXPECT_THROW(validate_pixels({}, spec), InvalidArgument);
  EXPECT_THROW(validate_pixels(torch::zeros({3, 0, 4}, torch::kUInt8), spec),
               InvalidArgument);
  EXPECT_THROW(validate_pixels(valid.unsqueeze(0), spec), InvalidArgument);
  EXPECT_THROW(validate_pixels(valid.to(torch::kFloat32), spec),
               InvalidArgument);
  spec.color = static_cast<PixelColorModel>(255);
  EXPECT_THROW(validate_pixels(valid, spec), InvalidArgument);
  spec = {};
  spec.layout = static_cast<ImageLayout>(255);
  EXPECT_THROW(validate_pixels(valid, spec), InvalidArgument);
  spec = {};
  spec.alpha = static_cast<PixelAlphaMode>(255);
  EXPECT_THROW(validate_pixels(valid, spec), InvalidArgument);
  spec = {};
  spec.range = static_cast<PixelValueRange>(255);
  EXPECT_THROW(validate_pixels(valid, spec), InvalidArgument);
}

TEST(PixelSpecCUDATest, ValidatesDeclaredDevicePixelsWithoutMovingStorage) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "CUDA runtime is unavailable";
  const auto pixels = torch::zeros(
      {3, 2, 4},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  const auto* identity = pixels.unsafeGetTensorImpl();
  PixelSpec spec;
  spec.dtype = torch::kFloat32;
  spec.range = PixelValueRange::unit_interval;
  EXPECT_NO_THROW(validate_pixels(pixels, spec));
  EXPECT_EQ(pixels.unsafeGetTensorImpl(), identity);
  EXPECT_TRUE(pixels.is_cuda());
  spec.layout = ImageLayout::hwc;
  EXPECT_NO_THROW(validate_pixels(pixels.permute({1, 2, 0}), spec));
}

}  // namespace
