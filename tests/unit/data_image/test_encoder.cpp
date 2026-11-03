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
#include <limits>
#include <memory>
#include <thread>
#include <utility>

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

TEST(ImageEncoderTest, BgrAndBgraPngPreserveColorAlphaAndCallerStorage) {
  for (const auto channels : {3, 4}) {
    for (const auto layout : {ImageLayout::hwc, ImageLayout::chw}) {
      SCOPED_TRACE(channels);
      SCOPED_TRACE(static_cast<int>(layout));
      auto storage = torch::arange(2 * 6 * channels, torch::kUInt8)
                         .reshape({2, 6, channels});
      const auto original = storage.clone();
      auto bgr = storage.slice(1, 0, 6, 2);
      ASSERT_FALSE(bgr.is_contiguous());
      auto expected = bgr.clone();
      expected.select(2, 0).copy_(bgr.select(2, 2));
      expected.select(2, 2).copy_(bgr.select(2, 0));
      auto spec = RgbSpec(layout);
      spec.color = channels == 3 ? PixelColorModel::bgr : PixelColorModel::bgra;
      spec.alpha =
          channels == 3 ? PixelAlphaMode::none : PixelAlphaMode::straight;
      const auto input =
          layout == ImageLayout::hwc ? bgr : bgr.permute({2, 0, 1});
      const auto strides = input.strides().vec();
      const auto encoded = encode_image(input, spec);
      ImageDecodeOptions decode;
      decode.layout = ImageLayout::hwc;
      const auto result = decode_image(encoded.bytes, decode);
      EXPECT_TRUE(torch::equal(result.tensor, expected));
      EXPECT_EQ(result.channels, channels);
      EXPECT_EQ(result.pixels.color,
                channels == 3 ? PixelColorModel::rgb : PixelColorModel::rgba);
      EXPECT_TRUE(torch::equal(storage, original));
      EXPECT_EQ(input.strides().vec(), strides);
    }
  }
}

TEST(ImageEncoderTest, BgrJpegMatchesExplicitRgbForBothLayoutsAndOwnedResults) {
  const auto rgb = torch::arange(12 * 10 * 3, torch::kInt64)
                       .remainder(251)
                       .to(torch::kUInt8)
                       .reshape({12, 10, 3});
  auto bgr = rgb.clone();
  bgr.select(2, 0).copy_(rgb.select(2, 2));
  bgr.select(2, 2).copy_(rgb.select(2, 0));
  ImageEncodeOptions options;
  options.format = ImageEncodeFormat::jpeg;
  options.jpeg_quality = 93;
  const auto expected = encode_image(rgb, RgbSpec(), options);
  for (const auto layout : {ImageLayout::hwc, ImageLayout::chw}) {
    auto spec = RgbSpec(layout);
    spec.color = PixelColorModel::bgr;
    const auto input =
        layout == ImageLayout::hwc ? bgr : bgr.permute({2, 0, 1});
    const auto actual = encode_image(input, spec, options);
    EXPECT_EQ(actual.bytes, expected.bytes);
    EXPECT_TRUE(torch::equal(bgr.select(2, 0), rgb.select(2, 2)));
    EXPECT_TRUE(torch::equal(bgr.select(2, 2), rgb.select(2, 0)));
  }
}

TEST(ImageEncoderTest, ColorConversionFailureReleasesEverySharedReservation) {
  const auto image = torch::arange(24, torch::kUInt8).reshape({2, 4, 3});
  const auto original = image.clone();
  auto spec = RgbSpec();
  spec.color = PixelColorModel::bgr;
  ImageResourceBudgetLimits limits;
  limits.max_codec_scratch_bytes = 47;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageEncodeOptions options;
  options.resource_budget = budget;
  EXPECT_THROW(encode_image(image, spec, options), ResourceExhausted);
  for (const auto kind :
       {ImageResourceKind::decoded_bytes, ImageResourceKind::pixels,
        ImageResourceKind::codec_scratch_bytes,
        ImageResourceKind::encoded_bytes})
    EXPECT_EQ(budget->usage(kind).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).peak, 0U);
  EXPECT_TRUE(torch::equal(image, original));

  options.resource_budget.reset();
  options.limits.max_working_bytes = 47;
  EXPECT_THROW(encode_image(image, spec, options), ResourceExhausted);
  options.limits.max_working_bytes = 48;
  const auto encoded = encode_image(image, spec, options);
  EXPECT_FALSE(encoded.bytes.empty());
  EXPECT_TRUE(torch::equal(image, original));
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

TEST(ImageResourceBudgetTest, ByteAndBatchAdmissionIsAtomicAndCancellable) {
  ImageResourceBudgetLimits limits;
  limits.max_pipeline_inflight_batches = 2;
  limits.max_pipeline_inflight_bytes = 64;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  const std::array requests{
      ImageResourceRequest{ImageResourceKind::pipeline_inflight_bytes, 64},
      ImageResourceRequest{ImageResourceKind::pipeline_inflight_batches, 1}};
  auto occupied =
      budget->reserve(ImageResourceKind::pipeline_inflight_bytes, 64);
  std::atomic<bool> acquired = false;
  std::jthread waiter([&](std::stop_token stop) {
    auto leases = budget->reserve_wait_all(requests, stop);
    acquired.store(leases.size() == 2U);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(acquired.load());
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            0U);
  occupied.release();
  waiter.join();
  EXPECT_TRUE(acquired.load());
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).peak,
            64U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).current,
            0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            0U);

  occupied = budget->reserve(ImageResourceKind::pipeline_inflight_bytes, 64);
  std::atomic<bool> cancelled = false;
  std::jthread blocked([&](std::stop_token stop) {
    try {
      static_cast<void>(budget->reserve_wait_all(requests, stop));
    } catch (const Cancelled&) {
      cancelled.store(true);
    }
  });
  blocked.request_stop();
  blocked.join();
  EXPECT_TRUE(cancelled.load());
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            0U);
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_THROW(budget->reserve_wait_all(requests, stopped.get_token()),
               Cancelled);
  const std::array oversized{
      ImageResourceRequest{ImageResourceKind::pipeline_inflight_bytes, 33},
      ImageResourceRequest{ImageResourceKind::pipeline_inflight_bytes, 33}};
  EXPECT_THROW(budget->reserve_wait_all(oversized), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).current,
            64U);
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

TEST(ImageResourceBudgetTest, EveryCategoryEnforcesItsOwnCapAndDiagnostics) {
  const ImageResourceBudgetLimits limits{3, 3, 3, 3, 3, 3, 3,
                                         3, 3, 3, 3, 3, 3, 3};
  ImageResourceBudget budget(limits);
  constexpr std::array kinds{ImageResourceKind::encoded_bytes,
                             ImageResourceKind::decoded_bytes,
                             ImageResourceKind::pixels,
                             ImageResourceKind::codec_scratch_bytes,
                             ImageResourceKind::annotation_bytes,
                             ImageResourceKind::index_bytes,
                             ImageResourceKind::pinned_bytes,
                             ImageResourceKind::gpu_bytes,
                             ImageResourceKind::staged_batches,
                             ImageResourceKind::render_queue,
                             ImageResourceKind::encoded_queue_bytes,
                             ImageResourceKind::disk_spool_bytes,
                             ImageResourceKind::pipeline_inflight_batches,
                             ImageResourceKind::pipeline_inflight_bytes};
  constexpr std::array names{"encoded_bytes",
                             "decoded_bytes",
                             "pixels",
                             "codec_scratch_bytes",
                             "annotation_bytes",
                             "index_bytes",
                             "pinned_bytes",
                             "gpu_bytes",
                             "staged_batches",
                             "render_queue",
                             "encoded_queue_bytes",
                             "disk_spool_bytes",
                             "pipeline_inflight_batches",
                             "pipeline_inflight_bytes"};
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    SCOPED_TRACE(names[i]);
    EXPECT_EQ(budget.limit(kinds[i]), 3U);
    auto first = budget.reserve(kinds[i], 2);
    auto second = budget.reserve_wait(kinds[i], 1);
    const auto rejected = [&](auto&& reserve, const char* prefix,
                              const char* actual) {
      try {
        reserve();
        FAIL() << "accepted a reservation beyond the category cap";
      } catch (const ResourceExhausted& error) {
        EXPECT_EQ(error.details().component, "data.image.resource_budget");
        EXPECT_EQ(error.details().operation, std::string(prefix) + names[i]);
        EXPECT_EQ(error.details().expected, "3");
        EXPECT_EQ(error.details().actual, actual);
      }
      EXPECT_EQ(budget.usage(kinds[i]).current, 3U);
      EXPECT_EQ(budget.usage(kinds[i]).peak, 3U);
    };
    rejected([&] { static_cast<void>(budget.reserve(kinds[i], 1)); },
             "reserve.", "3+1");
    rejected([&] { static_cast<void>(budget.reserve_wait(kinds[i], 4)); },
             "reserve_wait.", "4");
    const std::array requests{ImageResourceRequest{kinds[i], 4}};
    rejected([&] { static_cast<void>(budget.reserve_wait_all(requests)); },
             "reserve_wait_all.", "4");
    first.release();
    second.release();
    EXPECT_EQ(budget.usage(kinds[i]).current, 0U);
    EXPECT_EQ(budget.usage(kinds[i]).peak, 3U);
  }
}

TEST(ImageResourceBudgetTest, InvalidKindsAndOverflowLeaveAllUsageUnchanged) {
  ImageResourceBudget budget;
  const auto invalid = static_cast<ImageResourceKind>(255);
  EXPECT_EQ(budget.limit(invalid), 0U);
  EXPECT_EQ(budget.usage(invalid).current, 0U);
  EXPECT_EQ(budget.usage(invalid).peak, 0U);
  EXPECT_EQ(budget.usage(invalid).limit, 0U);
  EXPECT_FALSE(budget.reserve(invalid, 0).active());
  std::stop_source stopped;
  stopped.request_stop();
  EXPECT_FALSE(budget.reserve_wait(invalid, 0, stopped.get_token()).active());
  EXPECT_THROW(budget.reserve(invalid, 1), InvalidArgument);
  EXPECT_THROW(budget.reserve_wait(invalid, 1), InvalidArgument);
  auto held = budget.reserve(ImageResourceKind::decoded_bytes, 7);
  const std::array invalid_requests{
      ImageResourceRequest{ImageResourceKind::decoded_bytes, 1},
      ImageResourceRequest{invalid, 0}};
  EXPECT_THROW(budget.reserve_wait_all(invalid_requests), InvalidArgument);
  const std::array overflow{
      ImageResourceRequest{ImageResourceKind::decoded_bytes,
                           std::numeric_limits<std::uint64_t>::max()},
      ImageResourceRequest{ImageResourceKind::decoded_bytes, 1}};
  EXPECT_THROW(budget.reserve_wait_all(overflow), ResourceExhausted);
  EXPECT_EQ(budget.usage(ImageResourceKind::decoded_bytes).current, 7U);
  EXPECT_EQ(budget.usage(ImageResourceKind::decoded_bytes).peak, 7U);
}

TEST(ImageResourceBudgetTest, RelatedLeasesSurviveOwnerAndMoveAssignment) {
  ImageResourceLease inactive;
  EXPECT_THROW(inactive.reserve_related(ImageResourceKind::encoded_bytes, 1),
               InvalidArgument);
  ImageResourceLease retained;
  {
    ImageResourceBudgetLimits limits;
    limits.max_encoded_bytes = 8;
    ImageResourceBudget budget(limits);
    auto first = budget.reserve(ImageResourceKind::encoded_bytes, 3);
    auto second = budget.reserve(ImageResourceKind::encoded_bytes, 2);
    second = std::move(first);
    EXPECT_FALSE(first.active());
    EXPECT_EQ(budget.usage(ImageResourceKind::encoded_bytes).current, 3U);
    ImageResourceLease moved(std::move(second));
    EXPECT_FALSE(second.active());
    retained = std::move(moved);
    EXPECT_FALSE(moved.active());
  }
  auto related = retained.reserve_related(ImageResourceKind::encoded_bytes, 5);
  EXPECT_THROW(retained.reserve_related(ImageResourceKind::encoded_bytes, 1),
               ResourceExhausted);
  retained.release();
  retained.release();
  EXPECT_FALSE(retained.active());
  auto remaining = related.reserve_related(ImageResourceKind::encoded_bytes, 3);
  EXPECT_TRUE(remaining.active());
  EXPECT_THROW(retained.reserve_related(ImageResourceKind::encoded_bytes, 0),
               InvalidArgument);
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageEncoderTest, RejectsCudaWithoutImplicitTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto image = torch::zeros({3, 2, 2}, torch::kUInt8).cuda();
  EXPECT_THROW(encode_image(image, RgbSpec(ImageLayout::chw)), InvalidArgument);
}
#endif

}  // namespace
