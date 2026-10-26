// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/mask_decoder.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
using jyppx::liuding::InvalidArgument;
using jyppx::liuding::ResourceExhausted;
using jyppx::liuding::SerializationError;

std::filesystem::path FixturePath(std::string_view name) {
  return std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" /
         "masks" / name;
}

std::vector<std::byte> Read(std::string_view name) {
  std::ifstream input(FixturePath(name), std::ios::binary);
  if (!input)
    throw std::runtime_error("missing mask fixture: " + std::string(name));
  const std::vector<char> raw((std::istreambuf_iterator<char>(input)), {});
  std::vector<std::byte> bytes(raw.size());
  std::memcpy(bytes.data(), raw.data(), raw.size());
  return bytes;
}

void CheckLabels(const DecodedMask& mask, unsigned depth, bool interlaced,
                 std::uint32_t width = 9, std::uint32_t height = 7) {
  ASSERT_EQ(mask.width, width);
  ASSERT_EQ(mask.height, height);
  EXPECT_EQ(mask.source_bit_depth, depth);
  EXPECT_EQ(mask.source_interlaced, interlaced);
  EXPECT_EQ(mask.ids.scalar_type(), torch::kInt64);
  EXPECT_TRUE(mask.ids.device().is_cpu());
  EXPECT_TRUE(mask.ids.is_contiguous());
  EXPECT_FALSE(mask.ids.requires_grad());
  ASSERT_EQ(mask.ids.sizes(), torch::IntArrayRef({height, width}));
  constexpr std::array<std::int64_t, 9> k16{0,     1,     255,   256, 1024,
                                            32768, 65534, 65535, 42};
  const auto values = mask.ids.accessor<std::int64_t, 2>();
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const auto expected =
          depth == 16 ? k16[(x + y) % 9] : (x * 3 + y * 5) % (1U << depth);
      EXPECT_EQ(values[y][x], expected) << "x=" << x << ", y=" << y;
    }
  }
}

class GrayMaskTest
    : public ::testing::TestWithParam<std::pair<unsigned, bool>> {};

TEST_P(GrayMaskTest, PreservesEveryDiscreteValueAcrossAllPngFilters) {
  const auto [depth, interlaced] = GetParam();
  const auto name =
      "gray" + std::to_string(depth) + (interlaced ? "-adam7.png" : ".png");
  MaskDecodeOptions options;
  options.ignore_id = 255;
  options.void_id = 0;
  const auto mask = decode_mask(Read(name), options);
  CheckLabels(mask, depth, interlaced);
  EXPECT_EQ(mask.encoding, MaskEncoding::class_index);
  EXPECT_EQ(mask.ignore_id, 255);
  EXPECT_EQ(mask.void_id, 0);
  EXPECT_TRUE(mask.palette.empty());
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionAndInterlace, GrayMaskTest,
    ::testing::Values(std::pair{1U, false}, std::pair{2U, false},
                      std::pair{4U, false}, std::pair{8U, false},
                      std::pair{16U, false}, std::pair{1U, true},
                      std::pair{2U, true}, std::pair{4U, true},
                      std::pair{8U, true}, std::pair{16U, true}));

class PaletteMaskTest
    : public ::testing::TestWithParam<std::pair<unsigned, bool>> {};

TEST_P(PaletteMaskTest,
       RetainsIndicesDespiteDuplicateDisplayColorsAndTransparency) {
  const auto [depth, interlaced] = GetParam();
  const auto name =
      "palette" + std::to_string(depth) + (interlaced ? "-adam7.png" : ".png");
  MaskDecodeOptions options;
  options.encoding = MaskEncoding::palette_index;
  options.ignore_id = -100;
  const auto mask = decode_mask(Read(name), options);
  CheckLabels(mask, depth, interlaced);
  ASSERT_EQ(mask.palette.size(), 1U << depth);
  EXPECT_EQ(mask.palette[0].rgb, mask.palette[1].rgb);
  for (std::size_t i = 0; i < mask.palette.size(); ++i)
    EXPECT_EQ(mask.palette[i].alpha, i);
  EXPECT_EQ(mask.ids[0][0].item<std::int64_t>(), 0);
  EXPECT_EQ(mask.ignore_id, -100);
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionAndInterlace, PaletteMaskTest,
    ::testing::Values(std::pair{1U, false}, std::pair{2U, false},
                      std::pair{4U, false}, std::pair{8U, false},
                      std::pair{1U, true}, std::pair{2U, true},
                      std::pair{4U, true}, std::pair{8U, true}));

constexpr std::array<std::array<std::uint8_t, 3>, 9> kRgb{{{0, 0, 0},
                                                           {255, 0, 0},
                                                           {0, 255, 0},
                                                           {0, 0, 255},
                                                           {17, 23, 31},
                                                           {255, 255, 255},
                                                           {1, 2, 3},
                                                           {17, 23, 31},
                                                           {0, 0, 0}}};
constexpr std::int64_t kLargeId = (std::int64_t{1} << 54) + 3;
constexpr std::array<std::int64_t, 9> kIds{0,  7,    42,       65535, kLargeId,
                                           -9, -100, kLargeId, 0};

MaskDecodeOptions RgbOptions() {
  MaskDecodeOptions options;
  options.encoding = MaskEncoding::rgb_mapping;
  options.ignore_id = -100;
  options.void_id = -9;
  for (std::size_t i = 0; i < 7; ++i)
    options.colors.push_back({kRgb[i], kIds[i]});
  return options;
}

TEST(MaskDecoderTest,
     MapsRgbWithoutFloatConversionOrImplicitUnknownColorFallback) {
  for (const auto name : {"rgb.png", "rgb-adam7.png"}) {
    auto options = RgbOptions();
    const auto mask = decode_mask(Read(name), options);
    for (std::size_t y = 0; y < 7; ++y)
      for (std::size_t x = 0; x < 9; ++x)
        EXPECT_EQ(
            mask.ids[static_cast<std::int64_t>(y)][static_cast<std::int64_t>(x)]
                .item<std::int64_t>(),
            kIds[(x + y) % 9]);
    options.colors.erase(options.colors.begin() + 4);
    EXPECT_THROW(decode_mask(Read(name), options), SerializationError);
    options.unknown_color_is_ignore = true;
    const auto fallback = decode_mask(Read(name), options);
    EXPECT_EQ(fallback.ids[0][4].item<std::int64_t>(), -100);
    EXPECT_EQ(fallback.ids[0][1].item<std::int64_t>(), 7);
  }
}

TEST(MaskDecoderTest, PreservesAll24PanopticIdBitsInCocoRgbOrder) {
  MaskDecodeOptions options;
  options.encoding = MaskEncoding::panoptic_rgb;
  options.void_id = 0;
  for (const auto name : {"rgb.png", "rgb-adam7.png"}) {
    const auto mask = decode_mask(Read(name), options);
    const auto values = mask.ids.accessor<std::int64_t, 2>();
    for (std::size_t y = 0; y < 7; ++y)
      for (std::size_t x = 0; x < 9; ++x) {
        const auto& rgb = kRgb[(x + y) % 9];
        const auto expected =
            static_cast<std::int64_t>(rgb[0]) + 256 * rgb[1] + 65536 * rgb[2];
        EXPECT_EQ(values[y][x], expected);
      }
    EXPECT_EQ(mask.ids[0][5].item<std::int64_t>(), 16777215);
    EXPECT_EQ(mask.void_id, 0);
  }
}

TEST(MaskDecoderTest, HandlesAdam7PassesWithEmptyRowsAndColumns) {
  for (const auto [width, height] :
       {std::pair{1U, 1U}, {1U, 9U}, {9U, 1U}, {2U, 3U}}) {
    const auto mask = decode_mask(Read("tiny-" + std::to_string(width) + "x" +
                                       std::to_string(height) + ".png"));
    CheckLabels(mask, 4, true, width, height);
  }
}

TEST(MaskDecoderTest, AcceptsSplitAndEmptyConsecutiveIdatChunks) {
  CheckLabels(decode_mask(Read("split-idat.png")), 8, false);
}

TEST(MaskDecoderTest,
     PublishesIndependentOwningStorageAndSupportsConcurrentReaders) {
  auto bytes = Read("gray16.png");
  const auto first = decode_mask(bytes);
  std::fill(bytes.begin(), bytes.end(), std::byte{0});
  CheckLabels(first, 16, false);
  std::array<std::future<DecodedMask>, 2> reads;
  for (auto& job : reads)
    job = std::async(std::launch::async,
                     [] { return decode_mask(Read("gray16-adam7.png")); });
  for (auto& job : reads) CheckLabels(job.get(), 16, true);
}

TEST(MaskDecoderTest, FileEntryPointSharesLabelSemanticsAndReportsIoFailure) {
  CheckLabels(read_mask_file(FixturePath("gray16.png")), 16, false);
  EXPECT_THROW(read_mask_file(FixturePath("absent.png")),
               jyppx::liuding::IoError);
}

TEST(MaskDecoderTest, AggregateBudgetTracksWorkingMemoryAndRetainedLabels) {
  const auto bytes = Read("gray8.png");
  ImageResourceBudgetLimits limits;
  limits.max_encoded_bytes = bytes.size();
  limits.max_decoded_bytes = 9U * 7U * sizeof(std::int64_t);
  limits.max_pixels = 9U * 7U;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  MaskDecodeOptions options;
  options.resource_budget = budget;
  auto mask = read_mask_file(FixturePath("gray8.png"), options);
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).peak, bytes.size());
  EXPECT_EQ(budget->usage(ImageResourceKind::encoded_bytes).current, 0U);
  EXPECT_GT(budget->usage(ImageResourceKind::codec_scratch_bytes).peak, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current,
            limits.max_decoded_bytes);
  auto concurrent = std::async(std::launch::async,
                               [&] { return decode_mask(bytes, options); });
  EXPECT_THROW(concurrent.get(), ResourceExhausted);
  mask = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
  EXPECT_THROW(decode_mask(Read("bad-filter.png"), options),
               SerializationError);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::codec_scratch_bytes).current, 0U);

  limits.max_codec_scratch_bytes = 1;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(decode_mask(bytes, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
}

TEST(MaskDecoderTest, ChecksAllInputOutputRasterChunkAndMappingBudgets) {
  const auto bytes = Read("gray8.png");
  std::vector<MaskDecodeOptions> cases(6);
  cases[0].limits.max_input_bytes = bytes.size() - 1;
  cases[1].limits.max_dimension = 8;
  cases[2].limits.max_pixels = 62;
  cases[3].limits.max_output_bytes = 63 * sizeof(std::int64_t) - 1;
  cases[4].max_working_bytes = 1;
  cases[5].max_chunks = 2;
  for (const auto& options : cases)
    EXPECT_THROW(decode_mask(bytes, options), ResourceExhausted);
  auto working = MaskDecodeOptions{};
  working.max_working_bytes =
      70;  // Exactly the filtered raster, with no IDAT allowance.
  EXPECT_THROW(decode_mask(bytes, working), ResourceExhausted);
  auto colors = RgbOptions();
  colors.max_color_entries = 6;
  EXPECT_THROW(decode_mask(Read("rgb.png"), colors), ResourceExhausted);
  EXPECT_THROW(read_mask_file(FixturePath("gray8.png"), cases[0]),
               ResourceExhausted);
}

TEST(MaskDecoderTest, RejectsInvalidOrAmbiguousOptions) {
  const auto bytes = Read("gray8.png");
  std::vector<MaskDecodeOptions> cases(9);
  cases[0].limits.max_input_bytes = 0;
  cases[1].limits.max_dimension = 0;
  cases[2].limits.max_pixels = 0;
  cases[3].limits.max_output_bytes = 0;
  cases[4].max_working_bytes = 0;
  cases[5].max_chunks = 0;
  cases[6].max_color_entries = 0;
  cases[7].encoding = static_cast<MaskEncoding>(255);
  cases[8].unknown_color_is_ignore = true;
  for (const auto& options : cases)
    EXPECT_THROW(decode_mask(bytes, options), InvalidArgument);
  auto rgb = RgbOptions();
  rgb.colors.push_back(rgb.colors.front());
  EXPECT_THROW(decode_mask(Read("rgb.png"), rgb), InvalidArgument);
  rgb = RgbOptions();
  rgb.colors.front().id = -77;
  EXPECT_THROW(decode_mask(Read("rgb.png"), rgb), InvalidArgument);
  rgb = RgbOptions();
  rgb.unknown_color_is_ignore = true;
  rgb.ignore_id.reset();
  EXPECT_THROW(decode_mask(Read("rgb.png"), rgb), InvalidArgument);
  rgb = {};
  rgb.encoding = MaskEncoding::rgb_mapping;
  EXPECT_THROW(decode_mask(Read("rgb.png"), rgb), InvalidArgument);
  rgb = RgbOptions();
  rgb.encoding = MaskEncoding::class_index;
  EXPECT_THROW(decode_mask(bytes, rgb), InvalidArgument);
}

TEST(MaskDecoderTest, RejectsLossyWrongConventionAlphaAndInvalidPngStructures) {
  const auto rgb = Read("rgb.png");
  EXPECT_THROW(decode_mask(rgb), SerializationError);
  auto options = RgbOptions();
  EXPECT_THROW(decode_mask(Read("gray8.png"), options), SerializationError);
  EXPECT_THROW(decode_mask(Read("rgba.png"), options), SerializationError);
  for (const auto name : {"bad-filter.png", "bad-adler.png", "short-raster.png",
                          "apng.png", "critical.png", "gray-trns.png"})
    EXPECT_THROW(decode_mask(Read(name)), SerializationError) << name;
  options = {};
  options.encoding = MaskEncoding::palette_index;
  EXPECT_THROW(decode_mask(Read("bad-palette-index.png"), options),
               SerializationError);
  const std::array<std::byte, 4> jpeg{std::byte{0xff}, std::byte{0xd8},
                                      std::byte{0xff}, std::byte{0}};
  EXPECT_THROW(decode_mask(jpeg), SerializationError);
}

TEST(MaskDecoderTest, RejectsEveryTruncatedPrefixAndCorruptChunkCrc) {
  const auto bytes = Read("gray8.png");
  for (std::size_t size = 0; size < bytes.size(); ++size)
    EXPECT_THROW(decode_mask(std::span(bytes).first(size)), SerializationError)
        << size;
  auto corrupt = bytes;
  corrupt[29] ^= std::byte{1};  // IHDR CRC, independently of valid dimensions.
  EXPECT_THROW(decode_mask(corrupt), SerializationError);
  corrupt = bytes;
  corrupt.push_back(std::byte{0});
  EXPECT_THROW(decode_mask(corrupt), SerializationError);
}

TEST(MaskDecoderTest, CancellationWinsBeforeDecodeOrFilesystemAccess) {
  std::stop_source cancellation;
  cancellation.request_stop();
  EXPECT_THROW(decode_mask({}, {}, cancellation.get_token()),
               jyppx::liuding::Cancelled);
  EXPECT_THROW(
      read_mask_file(FixturePath("absent.png"), {}, cancellation.get_token()),
      jyppx::liuding::Cancelled);
}

}  // namespace
