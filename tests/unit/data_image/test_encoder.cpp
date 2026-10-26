// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/data_image/image_encoder.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data::image;

PixelSpec RgbSpec(ImageLayout layout = ImageLayout::hwc) {
  PixelSpec spec;
  spec.layout = layout;
  spec.color = PixelColorModel::rgb;
  spec.alpha = PixelAlphaMode::none;
  spec.dtype = torch::kUInt8;
  spec.range = PixelValueRange::unsigned_full_scale;
  return spec;
}

TEST(ImageEncoderTest, PngIsLosslessForHwcAndChwRgb) {
  const auto hwc = torch::arange(24, torch::kUInt8).reshape({2, 4, 3});
  const auto encoded = encode_image(hwc, RgbSpec());
  EXPECT_EQ(encoded.format, ImageFormat::png);
  EXPECT_EQ(detect_image_format(encoded.bytes), ImageFormat::png);
  EXPECT_EQ(encoded.width, 4U);
  EXPECT_EQ(encoded.height, 2U);

  ImageDecodeOptions decode;
  decode.format = ImageFormat::png;
  decode.channels = ImageChannels::rgb;
  decode.layout = ImageLayout::hwc;
  const auto decoded = decode_image(encoded.bytes, decode);
  EXPECT_TRUE(torch::equal(decoded.tensor, hwc));

  const auto chw = hwc.permute({2, 0, 1});
  const auto encoded_chw = encode_image(chw, RgbSpec(ImageLayout::chw));
  const auto decoded_chw = decode_image(encoded_chw.bytes, decode);
  EXPECT_TRUE(torch::equal(decoded_chw.tensor, hwc));
}

TEST(ImageEncoderTest, PngPreservesAlphaAndJpegRejectsIt) {
  auto rgba = torch::arange(16, torch::kUInt8).reshape({2, 2, 4});
  auto spec = RgbSpec();
  spec.color = PixelColorModel::rgba;
  spec.alpha = PixelAlphaMode::straight;
  const auto encoded = encode_image(rgba, spec);
  ImageDecodeOptions decode;
  decode.format = ImageFormat::png;
  decode.channels = ImageChannels::rgba;
  decode.layout = ImageLayout::hwc;
  EXPECT_TRUE(torch::equal(decode_image(encoded.bytes, decode).tensor, rgba));

  ImageEncodeOptions jpeg;
  jpeg.format = ImageEncodeFormat::jpeg;
  EXPECT_THROW(encode_image(rgba, spec, jpeg), InvalidArgument);
}

TEST(ImageEncoderTest, JpegRoundTripsDimensionsAndCanBeWrittenToFile) {
  auto image = torch::zeros({3, 4, 5}, torch::kUInt8);
  image[0].fill_(255);
  ImageEncodeOptions options;
  options.format = ImageEncodeFormat::jpeg;
  options.jpeg_quality = 85;
  const auto encoded = encode_image(image, RgbSpec(ImageLayout::chw), options);
  EXPECT_EQ(encoded.format, ImageFormat::jpeg);
  EXPECT_EQ(detect_image_format(encoded.bytes), ImageFormat::jpeg);

  ImageDecodeOptions decode;
  decode.format = ImageFormat::jpeg;
  decode.channels = ImageChannels::rgb;
  decode.layout = ImageLayout::hwc;
  const auto decoded = decode_image(encoded.bytes, decode);
  EXPECT_EQ(decoded.tensor.sizes(), torch::IntArrayRef({4, 5, 3}));

  const auto unique = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto path = std::filesystem::temp_directory_path() /
                    ("liuding-image-encoder-" + unique + ".jpg");
  ASSERT_NO_THROW(
      write_image_file(path, image, RgbSpec(ImageLayout::chw), options));
  EXPECT_TRUE(std::filesystem::exists(path));
  EXPECT_EQ(read_image_file(path, decode).tensor.sizes(),
            torch::IntArrayRef({4, 5, 3}));
  std::error_code error;
  std::filesystem::remove(path, error);
}

TEST(ImageEncoderTest, BoundsInvalidTypesAndCancellationAreExplicit) {
  const auto image = torch::zeros({3, 4, 5}, torch::kUInt8);
  auto limited = ImageEncodeOptions{};
  limited.limits.max_output_bytes = 8;
  EXPECT_THROW(encode_image(image, RgbSpec(ImageLayout::chw), limited),
               ResourceExhausted);
  limited = {};
  limited.limits.max_working_bytes = 1;
  EXPECT_THROW(encode_image(image, RgbSpec(ImageLayout::chw), limited),
               ResourceExhausted);

  auto float_spec = RgbSpec();
  float_spec.dtype = torch::kFloat32;
  EXPECT_THROW(encode_image(image.to(torch::kFloat32), float_spec),
               InvalidArgument);

  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      encode_image(image, RgbSpec(ImageLayout::chw), {}, stop.get_token()),
      Cancelled);
}

TEST(ImageResourceBudgetTest, LeasesTrackPeakAndRejectAggregateOverflow) {
  ImageResourceBudgetLimits limits;
  limits.max_encoded_bytes = 8;
  auto budget = std::make_shared<ImageResourceBudget>(limits);

  auto first = budget->reserve(ImageResourceKind::encoded_bytes, 6);
  EXPECT_TRUE(first.active());
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 6U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).peak, 6U);
  EXPECT_THROW(budget->reserve(ImageResourceKind::encoded_bytes, 3),
               ResourceExhausted);

  first.release();
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);
  auto second = budget->reserve(ImageResourceKind::encoded_bytes, 8);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).peak, 8U);
}

TEST(ImageResourceBudgetTest, WaitingAdmissionWakesOnReleaseAndCancellation) {
  ImageResourceBudgetLimits limits;
  limits.max_pipeline_inflight_batches = 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  auto first = budget->reserve(ImageResourceKind::pipeline_inflight_batches, 1);
  std::atomic<bool> acquired = false;
  std::jthread waiter([&](std::stop_token stop) {
    auto second = budget->reserve_wait(
        ImageResourceKind::pipeline_inflight_batches, 1, stop);
    acquired.store(second.active());
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(acquired.load());
  first.release();
  waiter.join();
  EXPECT_TRUE(acquired.load());
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).peak,
            1U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            0U);

  first = budget->reserve(ImageResourceKind::pipeline_inflight_batches, 1);
  std::atomic<bool> cancelled = false;
  std::jthread blocked([&](std::stop_token stop) {
    try {
      static_cast<void>(budget->reserve_wait(
          ImageResourceKind::pipeline_inflight_batches, 1, stop));
    } catch (const Cancelled&) {
      cancelled.store(true);
    }
  });
  blocked.request_stop();
  blocked.join();
  EXPECT_TRUE(cancelled.load());
  EXPECT_THROW(
      budget->reserve_wait(ImageResourceKind::pipeline_inflight_batches, 2),
      ResourceExhausted);
}

TEST(ImageEncoderTest, AggregateBudgetRetainsEncodedLeaseUntilConsumption) {
  const auto image = torch::zeros({3, 4, 5}, torch::kUInt8);
  const auto reference = encode_image(image, RgbSpec(ImageLayout::chw));

  ImageResourceBudgetLimits limits;
  limits.max_encoded_bytes = reference.bytes.size();
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageEncodeOptions options;
  options.resource_budget = budget;
  auto encoded = encode_image(image, RgbSpec(ImageLayout::chw), options);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current,
            encoded.bytes.size());
  EXPECT_THROW(encode_image(image, RgbSpec(ImageLayout::chw), options),
               ResourceExhausted);
  encoded.output_lease.release();
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageEncoderTest, RejectsCudaWithoutImplicitTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto image = torch::zeros({3, 2, 2}, torch::kUInt8).cuda();
  EXPECT_THROW(encode_image(image, RgbSpec(ImageLayout::chw)), InvalidArgument);
}
#endif

}  // namespace
