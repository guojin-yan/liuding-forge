// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
namespace fs = std::filesystem;

TEST(ImageDecoderTest, FormatNamesRemainStableForEveryPublicValue) {
  constexpr std::array formats{ImageFormat::auto_detect, ImageFormat::jpeg,
                               ImageFormat::png,         ImageFormat::bmp,
                               ImageFormat::gif,         ImageFormat::tga,
                               ImageFormat::pnm,         ImageFormat::hdr,
                               ImageFormat::pic,         ImageFormat::unknown};
  constexpr std::array names{"auto", "jpeg", "png", "bmp", "gif",
                             "tga",  "pnm",  "hdr", "pic", "unknown"};
  for (std::size_t i = 0; i < formats.size(); ++i)
    EXPECT_EQ(image_format_name(formats[i]), names[i]);
  EXPECT_EQ(image_format_name(static_cast<ImageFormat>(255)), "unknown");
}

std::vector<std::byte> ReadFixture(const char* name) {
  const auto path =
      fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / name;
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
  EXPECT_EQ(result.pixels.color, PixelColorModel::rgb);
  EXPECT_EQ(result.pixels.layout, ImageLayout::chw);
  EXPECT_EQ(result.pixels.dtype, torch::kUInt8);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unsigned_full_scale);
  EXPECT_EQ(result.pixels.alpha, PixelAlphaMode::none);
  EXPECT_NO_THROW(validate_pixels(result.tensor, result.pixels));
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
  EXPECT_EQ(result.pixels.layout, ImageLayout::hwc);
  EXPECT_EQ(result.pixels.color, PixelColorModel::rgb);
  EXPECT_NO_THROW(validate_pixels(result.tensor, result.pixels));
}

TEST(ImageDecoderTest, ConvertsBmpAndPgmToRequestedChannels) {
  auto options = ImageDecodeOptions{};
  options.channels = ImageChannels::rgba;
  const auto bmp = decode_image(ReadFixture("gray.bmp"), options);
  EXPECT_EQ(bmp.format, ImageFormat::bmp);
  EXPECT_EQ(bmp.channels, 4U);
  EXPECT_EQ(bmp.tensor.size(0), 4);
  EXPECT_EQ(bmp.pixels.color, PixelColorModel::rgba);
  EXPECT_EQ(bmp.pixels.alpha, PixelAlphaMode::straight);
  const auto pgm = decode_image(ReadFixture("gray.pgm"));
  EXPECT_EQ(pgm.format, ImageFormat::pnm);
  EXPECT_EQ(pgm.channels, 1U);
  EXPECT_EQ(pgm.pixels.color, PixelColorModel::gray);
  EXPECT_EQ(pgm.pixels.alpha, PixelAlphaMode::none);
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
  EXPECT_EQ(preserved.pixels.dtype, torch::kUInt16);
  EXPECT_EQ(normalized.pixels.dtype, torch::kFloat32);
  EXPECT_EQ(normalized.pixels.range, PixelValueRange::unit_interval);
  EXPECT_NO_THROW(validate_pixels(normalized.tensor, normalized.pixels));
}

TEST(ImageDecoderTest, NormalizationOwnsPixelsForBothDepthsAndLayouts) {
  for (const auto* name : {"rgb.png", "gray16.png"}) {
    const auto bytes = ReadFixture(name);
    for (const auto layout : {ImageLayout::hwc, ImageLayout::chw}) {
      ImageDecodeOptions options;
      options.layout = layout;
      const auto preserved = decode_image(bytes, options);
      options.dtype = ImageDtype::float32_normalized;
      auto normalized = decode_image(bytes, options);
      const auto divisor = preserved.source_is_16bit ? 65535.0 : 255.0;
      EXPECT_TRUE(torch::allclose(
          normalized.tensor, preserved.tensor.to(torch::kFloat32) / divisor));
      EXPECT_TRUE(normalized.tensor.is_contiguous());
      EXPECT_NE(normalized.tensor.data_ptr(), preserved.tensor.data_ptr());
      EXPECT_NO_THROW(validate_pixels(normalized.tensor, normalized.pixels));
    }
  }
}

TEST(ImageDecoderTest, RejectsUnknownLayoutAndScalarPoliciesBeforeDecode) {
  const auto bytes = ReadFixture("rgb.png");
  ImageDecodeOptions options;
  options.layout = static_cast<ImageLayout>(255);
  EXPECT_THROW(decode_image(bytes, options), jyppx::liuding::InvalidArgument);
  options = {};
  options.dtype = static_cast<ImageDtype>(255);
  EXPECT_THROW(decode_image(bytes, options), jyppx::liuding::InvalidArgument);
}

TEST(ImageDecoderTest, RejectsAppleCgbiInsteadOfMislabelingPremultipliedBgra) {
  const auto bytes = ReadFixture("cgbi.png");
  EXPECT_EQ(detect_image_format(bytes), ImageFormat::png);
  try {
    static_cast<void>(decode_image(bytes));
    FAIL() << "CgBI pixels must not be published with ordinary RGB semantics";
  } catch (const jyppx::liuding::SerializationError& error) {
    EXPECT_EQ(error.details().operation, "decode.png.header");
    EXPECT_NE(error.details().actual.find("CgBI"), std::string::npos);
  }
}

TEST(ImageDecoderTest, FileReaderUsesSameBoundedPathEntryPoint) {
  const auto path =
      fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / "rgb.png";
  const auto result = read_image_file(path);
  EXPECT_EQ(result.width, 2U);
  EXPECT_EQ(result.height, 1U);
}

TEST(ImageDecoderTest, AggregateBudgetTracksReadDecodeAndConcurrentResults) {
  const auto path =
      fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / "rgb.png";
  const auto bytes = ReadFixture("rgb.png");
  ImageResourceBudgetLimits limits;
  limits.max_encoded_bytes = bytes.size() - 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageDecodeOptions options;
  options.resource_budget = budget;
  EXPECT_THROW(read_image_file(path, options),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);

  limits.max_encoded_bytes = bytes.size();
  limits.max_decoded_bytes = 6;
  limits.max_pixels = 2;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto decoded = read_image_file(path, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).peak, bytes.size());
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);
  EXPECT_GT(budget->usage(ImageResourceKind::codec_scratch_bytes).peak, 12U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 6U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 2U);
  auto concurrent = std::async(std::launch::async,
                               [&] { return decode_image(bytes, options); });
  EXPECT_THROW(concurrent.get(), jyppx::liuding::ResourceExhausted);

  auto copy = decoded;
  decoded = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 6U);
  copy = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);

  limits.max_codec_scratch_bytes = 1;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(decode_image(bytes, options), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
}

TEST(ImageDecoderTest,
     PrivateCodecAllocationRejectsBeforePublishingAndRecovers) {
  const auto bytes = ReadFixture("rgb.png");
  ImageResourceBudgetLimits limits;
  // The six-pixel-byte output and its known copies fit; stb's private PNG
  // workspace does not. The original estimated-only accounting accepted this.
  limits.max_codec_scratch_bytes = 12;
  ImageDecodeOptions options;
  options.resource_budget = std::make_shared<ImageResourceBudget>(limits);
  EXPECT_THROW(decode_image(bytes, options), jyppx::liuding::ResourceExhausted);
  for (const auto kind :
       {ImageResourceKind::codec_scratch_bytes,
        ImageResourceKind::decoded_bytes, ImageResourceKind::pixels})
    EXPECT_EQ(options.resource_budget->usage(kind).current, 0U);
  EXPECT_LE(
      options.resource_budget->usage(ImageResourceKind::codec_scratch_bytes)
          .peak,
      12U);
  // A failed scope must not poison an unbudgeted call on the same thread.
  EXPECT_TRUE(
      torch::equal(decode_image(bytes).tensor, decode_image(bytes).tensor));
}

TEST(ImageDecoderTest,
     CodecBudgetIncludesMetadataAndPreservesPixelsAtBoundary) {
  for (const auto* name : {"rgb.png", "rgb.jpg"}) {
    SCOPED_TRACE(name);
    const auto bytes = ReadFixture(name);
    const auto reference = decode_image(bytes);
    ImageDecodeOptions options;
    options.resource_budget = std::make_shared<ImageResourceBudget>();
    EXPECT_TRUE(
        torch::equal(reference.tensor, decode_image(bytes, options).tensor));
    const auto peak =
        options.resource_budget->usage(ImageResourceKind::codec_scratch_bytes)
            .peak;
    ASSERT_GT(peak, 12U);
    EXPECT_EQ(
        options.resource_budget->usage(ImageResourceKind::codec_scratch_bytes)
            .current,
        0U);
    ImageResourceBudgetLimits limits;
    limits.max_codec_scratch_bytes = peak;
    options.resource_budget = std::make_shared<ImageResourceBudget>(limits);
    EXPECT_TRUE(
        torch::equal(reference.tensor, decode_image(bytes, options).tensor));
    limits.max_codec_scratch_bytes = peak - 1;
    options.resource_budget = std::make_shared<ImageResourceBudget>(limits);
    EXPECT_THROW(decode_image(bytes, options),
                 jyppx::liuding::ResourceExhausted);
    for (const auto kind :
         {ImageResourceKind::codec_scratch_bytes,
          ImageResourceKind::decoded_bytes, ImageResourceKind::pixels})
      EXPECT_EQ(options.resource_budget->usage(kind).current, 0U);
  }
}

TEST(ImageDecoderTest, CodecScopesKeepConcurrentThreadBudgetsIndependent) {
  const auto bytes = ReadFixture("rgb.png");
  ImageResourceBudgetLimits limits;
  limits.max_codec_scratch_bytes = 12;
  auto small = std::make_shared<ImageResourceBudget>(limits);
  auto large = std::make_shared<ImageResourceBudget>();
  const auto decode = [&](const std::shared_ptr<ImageResourceBudget>& budget) {
    ImageDecodeOptions options;
    options.resource_budget = budget;
    return decode_image(bytes, options);
  };
  auto rejected = std::async(std::launch::async, [&] { return decode(small); });
  auto accepted = std::async(std::launch::async, [&] { return decode(large); });
  EXPECT_THROW(rejected.get(), jyppx::liuding::ResourceExhausted);
  auto result = accepted.get();
  EXPECT_TRUE(torch::equal(result.tensor, decode_image(bytes).tensor));
  EXPECT_EQ(small->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);
  EXPECT_EQ(large->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);
  EXPECT_EQ(small->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(large->usage(ImageResourceKind::decoded_bytes).current, 6U);
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

std::vector<std::byte> JpegFrame(std::uint8_t first_sampling = 0x11,
                                 std::uint8_t second_sampling = 0x11) {
  // SOI and an 8x8 baseline SOF with three components; no entropy payload.
  const std::array<unsigned char, 21> raw{0xff,
                                          0xd8,
                                          0xff,
                                          0xc0,
                                          0x00,
                                          0x11,
                                          0x08,
                                          0x00,
                                          0x08,
                                          0x00,
                                          0x08,
                                          0x03,
                                          0x01,
                                          first_sampling,
                                          0x00,
                                          0x02,
                                          second_sampling,
                                          0x00,
                                          0x03,
                                          0x11,
                                          0x00};
  std::vector<std::byte> result(raw.size());
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<std::byte> PnmSixteenBit() {
  constexpr std::string_view header = "P5\n2 1\n65535\n";
  std::vector<std::byte> result(header.size());
  std::memcpy(result.data(), header.data(), header.size());
  result.insert(result.end(), {std::byte{0x00}, std::byte{0x00},
                               std::byte{0xff}, std::byte{0xff}});
  return result;
}

void ExpectPixelDecodeFailure(const std::vector<std::byte>& bytes,
                              std::string_view reason) {
  ImageDecodeOptions options;
  options.resource_budget = std::make_shared<ImageResourceBudget>();
  try {
    static_cast<void>(decode_image(bytes, options));
    FAIL() << "Malformed pixels must not be published";
  } catch (const jyppx::liuding::SerializationError& error) {
    EXPECT_EQ(error.details().operation, "decode.pixels");
    EXPECT_NE(error.details().actual.find(reason), std::string::npos)
        << error.details().actual;
  }
  for (const auto kind :
       {ImageResourceKind::codec_scratch_bytes,
        ImageResourceKind::decoded_bytes, ImageResourceKind::pixels})
    EXPECT_EQ(options.resource_budget->usage(kind).current, 0U);
  // Failure must release the codec scope and allow a valid call on this thread.
  {
    const auto recovered = decode_image(ReadFixture("rgb.png"), options);
    EXPECT_EQ(recovered.tensor[0][0][0].item<int>(), 255);
  }
  for (const auto kind :
       {ImageResourceKind::codec_scratch_bytes,
        ImageResourceKind::decoded_bytes, ImageResourceKind::pixels})
    EXPECT_EQ(options.resource_budget->usage(kind).current, 0U);
}

TEST(ImageDecoderTest, RejectsFractionalJpegSamplingWithResourceRecovery) {
  // JPEG permits factors in [1,4], but these 3:2 ratios cannot be resampled.
  for (const auto horizontal : {true, false}) {
    SCOPED_TRACE(horizontal ? "horizontal" : "vertical");
    auto bytes = horizontal ? JpegFrame(0x31, 0x21) : JpegFrame(0x13, 0x12);
    bytes.insert(bytes.end(), {std::byte{0xff}, std::byte{0xd9}});
    ExpectPixelDecodeFailure(bytes, horizontal ? "bad H" : "bad V");
  }
}

TEST(ImageDecoderTest,
     RejectsOversizedJpegHuffmanSizeListWithResourceRecovery) {
  auto bytes = JpegFrame();
  // DHT: length 2+1+16+257, DC table 0, 255 one-bit and 2 two-bit symbols.
  // Reject the table's symbol count before constructing the Huffman codes.
  bytes.insert(bytes.end(), {std::byte{0xff}, std::byte{0xc4}, std::byte{0x01},
                             std::byte{0x14}, std::byte{0x00}, std::byte{0xff},
                             std::byte{0x02}});
  bytes.insert(bytes.end(), 14 + 257, std::byte{0});
  bytes.insert(bytes.end(), {std::byte{0xff}, std::byte{0xd9}});
  ExpectPixelDecodeFailure(bytes, "bad DHT header");
}

TEST(ImageDecoderTest, RejectsJpegWithoutScanWithResourceRecovery) {
  auto bytes = JpegFrame();
  bytes.insert(bytes.end(), {std::byte{0xff}, std::byte{0xd9}});
  ExpectPixelDecodeFailure(bytes, "no scan");
}

TEST(ImageDecoderTest, PreservesSixteenBitPnmWhenExpandingChannels) {
  ImageDecodeOptions options;
  options.channels = ImageChannels::rgba;
  options.resource_budget = std::make_shared<ImageResourceBudget>();
  {
    const auto result = decode_image(PnmSixteenBit(), options);
    EXPECT_TRUE(result.source_is_16bit);
    EXPECT_EQ(result.tensor.scalar_type(), torch::kUInt16);
    ASSERT_EQ(result.tensor.sizes(), torch::IntArrayRef({4, 1, 2}));
    for (int channel = 0; channel < 4; ++channel) {
      EXPECT_EQ(result.tensor[channel][0][0].item<int>(),
                channel == 3 ? 65535 : 0);
      EXPECT_EQ(result.tensor[channel][0][1].item<int>(), 65535);
    }
    EXPECT_EQ(options.resource_budget->usage(ImageResourceKind::pixels).current,
              2U);
  }
  for (const auto kind :
       {ImageResourceKind::codec_scratch_bytes,
        ImageResourceKind::decoded_bytes, ImageResourceKind::pixels})
    EXPECT_EQ(options.resource_budget->usage(kind).current, 0U);
}

TEST(ImageDecoderTest, RejectsTruncatedSixteenBitPnmWithResourceRecovery) {
  auto bytes = PnmSixteenBit();
  bytes.pop_back();
  ExpectPixelDecodeFailure(bytes, "bad PNM");
}

TEST(ImageDecoderTest, RejectsHdrBeforeEnteringCodecWorkspace) {
  constexpr std::string_view header =
      "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n";
  std::vector<std::byte> bytes(header.size());
  std::memcpy(bytes.data(), header.data(), header.size());
  // A zero-length RLE packet cannot advance a scanline decoder.
  bytes.insert(bytes.end(), {std::byte{2}, std::byte{2}, std::byte{0},
                             std::byte{8}, std::byte{0}});
  ImageDecodeOptions options;
  options.resource_budget = std::make_shared<ImageResourceBudget>();
  try {
    static_cast<void>(decode_image(bytes, options));
    FAIL() << "HDR has no supported scalar contract";
  } catch (const jyppx::liuding::SerializationError& error) {
    EXPECT_EQ(error.details().operation, "decode.format");
  }
  EXPECT_EQ(
      options.resource_budget->usage(ImageResourceKind::codec_scratch_bytes)
          .peak,
      0U);
}

}  // namespace
