// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/geometry_trace.hpp>

#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace jyppx::liuding::data::image;
using jyppx::liuding::InvalidArgument;
using jyppx::liuding::ResourceExhausted;

void ExpectBox(const std::optional<ImageBox>& actual, ImageBox expected) {
  ASSERT_TRUE(actual);
  EXPECT_NEAR(actual->x_min, expected.x_min, 1e-9);
  EXPECT_NEAR(actual->y_min, expected.y_min, 1e-9);
  EXPECT_NEAR(actual->x_max, expected.x_max, 1e-9);
  EXPECT_NEAR(actual->y_max, expected.y_max, 1e-9);
}

TEST(GeometryTraceTest,
     IdentityUsesContinuousBoundariesAndRejectsOutsidePoints) {
  const GeometryTrace trace({100, 60});
  EXPECT_TRUE(trace.retains_full_source());
  EXPECT_EQ(trace.source_extent().width, 100U);
  EXPECT_EQ(trace.output_extent().height, 60U);
  ExpectBox(trace.output_region(), {0, 0, 100, 60});
  ASSERT_TRUE(trace.forward_point({100, 60}));
  EXPECT_DOUBLE_EQ(trace.forward_point({100, 60})->x, 100);
  EXPECT_FALSE(trace.forward_point({-0.1, 30}));
  EXPECT_FALSE(trace.inverse_point({30, 60.1}));
  ExpectBox(trace.forward_box({-10, -10, 120, 80}), {0, 0, 100, 60});
  EXPECT_FALSE(trace.inverse_box({110, 20, 130, 30}));
}

TEST(GeometryTraceTest, LetterboxHasExactInverseOnlyForRealContent) {
  const auto trace =
      GeometryTrace({640, 480}).resized({320, 240}).padded(0, 40, 0, 40);
  EXPECT_EQ(trace.output_extent().width, 320U);
  EXPECT_EQ(trace.output_extent().height, 320U);
  EXPECT_TRUE(trace.retains_full_source());
  ExpectBox(trace.output_region(), {0, 40, 320, 280});
  ExpectBox(trace.forward_box({100, 80, 300, 280}), {50, 80, 150, 180});
  ExpectBox(trace.inverse_box({50, 80, 150, 180}), {100, 80, 300, 280});
  EXPECT_FALSE(trace.inverse_point({10, 20}));
  EXPECT_FALSE(trace.inverse_box({0, 0, 320, 30}));
  ExpectBox(trace.inverse_box({0, 0, 320, 320}), {0, 0, 640, 480});
}

TEST(GeometryTraceTest, CropAndFlipsNeverReconstructDiscardedSourceContent) {
  const auto trace = GeometryTrace({100, 80})
                         .cropped(20, 10, {50, 40})
                         .flipped_horizontal()
                         .flipped_vertical();
  EXPECT_FALSE(trace.retains_full_source());
  ExpectBox(trace.source_region(), {20, 10, 70, 50});
  ExpectBox(trace.output_region(), {0, 0, 50, 40});
  const auto point = trace.forward_point({30, 25});
  ASSERT_TRUE(point);
  EXPECT_DOUBLE_EQ(point->x, 40);
  EXPECT_DOUBLE_EQ(point->y, 25);
  const auto restored = trace.inverse_point(*point);
  ASSERT_TRUE(restored);
  EXPECT_DOUBLE_EQ(restored->x, 30);
  EXPECT_DOUBLE_EQ(restored->y, 25);
  EXPECT_FALSE(trace.forward_point({10, 25}));
  ExpectBox(trace.forward_box({10, 5, 40, 30}), {30, 20, 50, 40});
  ExpectBox(trace.inverse_box({30, 20, 50, 40}), {20, 10, 40, 30});
}

TEST(GeometryTraceTest, CroppingOnlyPaddingProducesNoInventedImageContent) {
  const auto trace = GeometryTrace({10, 10})
                         .padded(20, 20, 20, 20)
                         .cropped(0, 0, {10, 10})
                         .resized({40, 30})
                         .flipped_horizontal()
                         .padded(2, 3, 4, 5);
  EXPECT_FALSE(trace.source_region());
  EXPECT_FALSE(trace.output_region());
  EXPECT_FALSE(trace.retains_full_source());
  EXPECT_FALSE(trace.forward_point({5, 5}));
  EXPECT_FALSE(trace.inverse_point({5, 5}));
  EXPECT_FALSE(trace.forward_box({0, 0, 10, 10}));
  EXPECT_FALSE(trace.inverse_box({0, 0, 10, 10}));
}

TEST(GeometryTraceTest, ClippedContentSurvivesFurtherPaddingAndCropping) {
  const auto trace = GeometryTrace({10, 10})
                         .padded(5, 5, 5, 5)
                         .cropped(0, 0, {10, 20})
                         .flipped_horizontal()
                         .padded(3, 2, 4, 5)
                         .cropped(3, 2, {10, 20});
  ExpectBox(trace.source_region(), {0, 0, 5, 10});
  ExpectBox(trace.output_region(), {0, 5, 5, 15});
  EXPECT_FALSE(trace.inverse_point({7, 7}));
  ExpectBox(trace.inverse_box({0, 0, 10, 20}), {0, 0, 5, 10});
}

TEST(GeometryTraceTest, ComposedNonSquareGeometryMatchesIndependentPointGrid) {
  // Resize by (3/2,2), crop [15,12,135,132], pad (7,9,11,13), then flip.
  const auto trace = GeometryTrace({100, 80})
                         .resized({150, 160})
                         .cropped(15, 12, {120, 120})
                         .padded(7, 9, 11, 13)
                         .flipped_horizontal()
                         .flipped_vertical();
  for (int y = 0; y <= 80; ++y) {
    for (int x = 0; x <= 100; ++x) {
      const auto mapped =
          trace.forward_point({static_cast<double>(x), static_cast<double>(y)});
      const bool survives = x >= 10 && x <= 90 && y >= 6 && y <= 66;
      ASSERT_EQ(mapped.has_value(), survives) << x << "," << y;
      if (!mapped) continue;
      EXPECT_NEAR(mapped->x, 146.0 - 1.5 * x, 1e-9);
      EXPECT_NEAR(mapped->y, 145.0 - 2.0 * y, 1e-9);
      const auto restored = trace.inverse_point(*mapped);
      ASSERT_TRUE(restored);
      EXPECT_NEAR(restored->x, x, 1e-9);
      EXPECT_NEAR(restored->y, y, 1e-9);
    }
  }
  const auto matrix = trace.matrix();
  EXPECT_DOUBLE_EQ(matrix[0], -1.5);
  EXPECT_DOUBLE_EQ(matrix[2], 146);
  EXPECT_DOUBLE_EQ(matrix[4], -2);
  EXPECT_DOUBLE_EQ(matrix[5], 145);
  EXPECT_DOUBLE_EQ(matrix[8], 1);
}

TEST(GeometryTraceTest,
     InvalidOperationsPreserveOriginalTraceAndCheckOverflow) {
  EXPECT_THROW(GeometryTrace({0, 1}), InvalidArgument);
  const GeometryTrace trace({100, 60});
  const auto original = trace.matrix();
  EXPECT_THROW(trace.resized({0, 1}), InvalidArgument);
  EXPECT_THROW(trace.cropped(90, 0, {20, 10}), InvalidArgument);
  EXPECT_THROW(
      trace.cropped(std::numeric_limits<std::uint32_t>::max(), 0, {1, 1}),
      InvalidArgument);
  EXPECT_THROW(trace.padded(std::numeric_limits<std::uint32_t>::max(), 0, 0, 0),
               InvalidArgument);
  EXPECT_THROW(
      trace.forward_point({std::numeric_limits<double>::infinity(), 0}),
      InvalidArgument);
  EXPECT_THROW(
      trace.inverse_point({0, std::numeric_limits<double>::quiet_NaN()}),
      InvalidArgument);
  EXPECT_THROW(trace.forward_box({0, 0, 0, 1}), InvalidArgument);
  EXPECT_THROW(trace.inverse_box({2, 2, 1, 1}), InvalidArgument);
  EXPECT_EQ(trace.matrix(), original);
  EXPECT_TRUE(trace.retains_full_source());
}

TEST(GeometryTraceTest,
     CompositionIsBoundedWithoutAllocatingAnOperationHistory) {
  GeometryTrace trace({17, 11});
  for (int i = 0; i < 128; ++i) trace = trace.flipped_horizontal();
  EXPECT_TRUE(trace.retains_full_source());
  ExpectBox(trace.output_region(), {0, 0, 17, 11});
  EXPECT_THROW(trace.flipped_vertical(), ResourceExhausted);
  EXPECT_TRUE(trace.inverse_point({8, 5}));
}

TEST(GeometryTraceTest,
     ResizingAndCroppingTheFullCanvasDoesNotInventContentLoss) {
  auto trace = GeometryTrace({101, 83}).padded(7, 9, 11, 13);
  for (int i = 0; i < 20; ++i) {
    trace = trace.resized({157, 113})
                .cropped(0, 0, {157, 113})
                .flipped_horizontal()
                .flipped_vertical();
    EXPECT_TRUE(trace.retains_full_source());
    const auto mapped = trace.forward_point({37, 41});
    ASSERT_TRUE(mapped);
    const auto restored = trace.inverse_point(*mapped);
    ASSERT_TRUE(restored);
    EXPECT_NEAR(restored->x, 37, 1e-9);
    EXPECT_NEAR(restored->y, 41, 1e-9);
  }
}

}  // namespace
