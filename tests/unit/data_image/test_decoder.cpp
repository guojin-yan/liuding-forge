// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stop_token>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
namespace fs = std::filesystem;

std::vector<std::byte> ReadFixture(const char* name) {
  const auto path = fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / name;
  std::ifstream stream(path, std::ios::binary);
  EXPECT_TRUE(stream.good()) << path.string();
  const std::vector<char> raw((std::istreambuf_iterator<char>(stream)), {});
  std::vector<std::byte> result(raw.size());
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

TEST(ImageDecoderTest, DetectsCommonMagicAndPublishesChwPng) {
  const auto bytes = ReadFixture("rgb.png");
  EXPECT_EQ(detect_image_format(bytes), ImageFormat::png);
  const auto result = decode_image(bytes);
  EXPECT_EQ(result.format, ImageFormat::png);
  EXPECT_EQ(result.layout, ImageLayout::chw);
  EXPECT_EQ(result.width, 2U);
  EXPECT_EQ(result.height, 1U);
  EXPECT_EQ(result.channels, 3U);
  EXPECT_FALSE(result.source_is_16bit);
  EXPECT_EQ(result.tensor.scalar_type(), torch::kUInt8);
  ASSERT_EQ(result.tensor.dim(), 3);
  EXPECT_EQ(result.tensor.size(0), 3);
  EXPECT_EQ(result.tensor.size(1), 1);
  EXPECT_EQ(result.tensor.size(2), 2);
  EXPECT_EQ(result.tensor[0][0][0].item<int>(), 255);
  EXPECT_EQ(result.tensor[1][0][1].item<int>(), 255);
}

TEST(ImageDecoderTest, ConvertsJpegToRgbChannelsLast) {
  const auto bytes = ReadFixture("rgb.jpg");
  ImageDecodeOptions options;
  options.channels = ImageChannels::rgb;
  options.layout = ImageLayout::hwc;
  const auto result = decode_image(bytes, options);
  EXPECT_EQ(result.format, ImageFormat::jpeg);
  EXPECT_EQ(result.layout, ImageLayout::hwc);
  EXPECT_EQ(result.channels, 3U);
  EXPECT_EQ(result.tensor.scalar_type(), torch::kUInt8);
  ASSERT_EQ(result.tensor.sizes(), torch::IntArrayRef({1, 2, 3}));
  EXPECT_GT(result.tensor[0][0][0].item<int>(), 180);
  EXPECT_GT(result.tensor[0][1][1].item<int>(), 150);
}

TEST(ImageDecoderTest, ConvertsBmpAndPgmToRequestedChannels) {
  auto options = ImageDecodeOptions{};
  options.channels = ImageChannels::rgba;
  const auto bmp = decode_image(ReadFixture("gray.bmp"), options);
  EXPECT_EQ(bmp.format, ImageFormat::bmp);
  EXPECT_EQ(bmp.channels, 4U);
  EXPECT_EQ(bmp.tensor.size(0), 4);
  const auto pgm = decode_image(ReadFixture("gray.pgm"));
  EXPECT_EQ(pgm.format, ImageFormat::pnm);
  EXPECT_EQ(pgm.channels, 1U);
}

TEST(ImageDecoderTest, PreservesSixteenBitPngAndCanNormalizeToFloat) {
  const auto bytes = ReadFixture("gray16.png");
  const auto preserved = decode_image(bytes);
  EXPECT_EQ(preserved.format, ImageFormat::png);
  EXPECT_TRUE(preserved.source_is_16bit);
  EXPECT_EQ(preserved.tensor.scalar_type(), torch::kUInt16);
  EXPECT_EQ(preserved.tensor[0][0][0].item<int>(), 0);
  EXPECT_EQ(preserved.tensor[0][0][1].item<int>(), 65535);

  ImageDecodeOptions options;
  options.dtype = ImageDtype::float32_normalized;
  const auto normalized = decode_image(bytes, options);
  EXPECT_EQ(normalized.tensor.scalar_type(), torch::kFloat32);
  EXPECT_FLOAT_EQ(normalized.tensor[0][0][1].item<float>(), 1.0F);
}

TEST(ImageDecoderTest, FileReaderUsesSameBoundedPathEntryPoint) {
  const auto path =
      fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / "rgb.png";
  const auto result = read_image_file(path);
  EXPECT_EQ(result.width, 2U);
  EXPECT_EQ(result.height, 1U);
}

TEST(ImageDecoderTest, RejectsCorruptInputLimitsAndCancellation) {
  const std::vector<std::byte> corrupt{std::byte{0x01}, std::byte{0x02}};
  EXPECT_THROW(decode_image(corrupt), jyppx::liuding::SerializationError);

  auto options = ImageDecodeOptions{};
  options.limits.max_input_bytes = 1;
  EXPECT_THROW(decode_image(ReadFixture("rgb.png"), options),
               jyppx::liuding::ResourceExhausted);

  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(decode_image(ReadFixture("rgb.png"), {}, source.get_token()),
               jyppx::liuding::Cancelled);
}

TEST(ImageDecoderTest, ExplicitFormatMismatchIsRejected) {
  auto options = ImageDecodeOptions{};
  options.format = ImageFormat::jpeg;
  EXPECT_THROW(decode_image(ReadFixture("rgb.png"), options),
               jyppx::liuding::SerializationError);
}

}  // namespace
