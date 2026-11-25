// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <torch/cuda.h>

#include <cstring>
#include <fstream>
#include <future>
#include <gtest/gtest.h>

namespace {
using namespace jyppx::liuding;
namespace im = data::image;
std::vector<std::byte> Fixture(std::string_view name) {
  const auto path = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                    "data_image" / "jpeg_reference" / name;
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("missing JPEG reference fixture");
  const std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
  std::vector<std::byte> bytes(raw.size());
  std::memcpy(bytes.data(), raw.data(), raw.size());
  return bytes;
}
im::ImageDecodeOptions Options() {
  im::ImageDecodeOptions o;
  o.jpeg_backend = im::JpegDecodeBackend::libjpeg_turbo;
  o.channels = im::ImageChannels::rgb;
  o.layout = im::ImageLayout::hwc;
  return o;
}
bool Available() {
  return im::jpeg_decode_backend_available(
      im::JpegDecodeBackend::libjpeg_turbo);
}
TEST(TurboJpegDecoder, AvailabilityIsExplicitAndUnavailableRequestsFail) {
  EXPECT_TRUE(im::jpeg_decode_backend_available(im::JpegDecodeBackend::stb));
  EXPECT_FALSE(im::jpeg_decode_backend_available(
      static_cast<im::JpegDecodeBackend>(255)));
#ifdef JYPPX_LIUDING_TEST_TURBOJPEG
  EXPECT_TRUE(Available());
#else
  EXPECT_FALSE(Available());
  EXPECT_THROW(im::decode_image(Fixture("baseline.jpg"), Options()),
               InvalidArgument);
#endif
  auto o = Options();
  o.jpeg_backend = static_cast<im::JpegDecodeBackend>(255);
  EXPECT_THROW(im::decode_image(Fixture("baseline.jpg"), o), InvalidArgument);
}
TEST(TurboJpegDecoder, BaselineProgressiveAndGrayMatchIndependentPillowBytes) {
  if (!Available()) {
    EXPECT_THROW(im::decode_image(Fixture("baseline.jpg"), Options()),
                 InvalidArgument);
    return;
  }
  for (const auto* name : {"baseline", "progressive", "gray"}) {
    SCOPED_TRACE(name);
    const auto encoded = Fixture(std::string(name) + ".jpg");
    const auto expected = Fixture(std::string(name) + ".rgb");
    auto o = Options();
    const auto result = im::decode_image(encoded, o);
    ASSERT_EQ(result.tensor.sizes(), torch::IntArrayRef({13, 17, 3}));
    ASSERT_EQ(expected.size(), result.tensor.numel());
    EXPECT_EQ(std::memcmp(result.tensor.const_data_ptr(), expected.data(),
                          expected.size()),
              0);
    EXPECT_EQ(result.format, im::ImageFormat::jpeg);
    EXPECT_EQ(result.pixels.color, im::PixelColorModel::rgb);
    EXPECT_TRUE(result.tensor.is_contiguous());
    o.layout = im::ImageLayout::chw;
    o.dtype = im::ImageDtype::float32_normalized;
    const auto chw = im::decode_image(encoded, o);
    EXPECT_TRUE(torch::equal(
        chw.tensor,
        result.tensor.permute({2, 0, 1}).to(torch::kFloat32) / 255));
    EXPECT_EQ(chw.pixels.range, im::PixelValueRange::unit_interval);
    o.dtype = im::ImageDtype::preserve;
    o.channels = im::ImageChannels::rgba;
    const auto rgba = im::decode_image(encoded, o);
    EXPECT_TRUE(torch::equal(rgba.tensor.slice(0, 0, 3),
                             result.tensor.permute({2, 0, 1})));
    EXPECT_TRUE((rgba.tensor[3] == 255).all().item<bool>());
    EXPECT_EQ(rgba.pixels.alpha, im::PixelAlphaMode::straight);
  }
  auto o = Options();
  o.channels = im::ImageChannels::source;
  auto gray = im::decode_image(Fixture("gray.jpg"), o);
  EXPECT_EQ(gray.channels, 1U);
  EXPECT_EQ(gray.pixels.color, im::PixelColorModel::gray);
  const auto expected = Fixture("gray.gray");
  EXPECT_EQ(std::memcmp(gray.tensor.const_data_ptr(), expected.data(),
                        expected.size()),
            0);
}
TEST(TurboJpegDecoder, RejectsUnsupportedTargetsMalformedAndResourceBounds) {
  if (!Available()) {
    EXPECT_THROW(im::decode_image(Fixture("baseline.jpg"), Options()),
                 InvalidArgument);
    return;
  }
  const auto encoded = Fixture("baseline.jpg");
  auto o = Options();
  o.limits.max_input_bytes = 1;
  EXPECT_THROW(im::decode_image(encoded, o), ResourceExhausted);
  o = Options();
  o.limits.max_pixels = 200;
  EXPECT_THROW(im::decode_image(encoded, o), ResourceExhausted);
  o = Options();
  o.limits.max_dimension = 16;
  EXPECT_THROW(im::decode_image(encoded, o), ResourceExhausted);
  o = Options();
  o.limits.max_output_bytes = 200;
  EXPECT_THROW(im::decode_image(encoded, o), ResourceExhausted);
  o = Options();
  o.max_working_bytes = 1;
  EXPECT_THROW(im::decode_image(encoded, o), ResourceExhausted);
  o = Options();
  o.resource_budget = std::make_shared<im::ImageResourceBudget>();
  EXPECT_THROW(im::decode_image(encoded, o), InvalidArgument);
  o = Options();
  o.channels = im::ImageChannels::gray_alpha;
  EXPECT_THROW(im::decode_image(encoded, o), InvalidArgument);
  o = Options();
  o.channels = static_cast<im::ImageChannels>(255);
  EXPECT_THROW(im::decode_image(encoded, o), InvalidArgument);
  EXPECT_THROW(im::decode_image(Fixture("cmyk.jpg"), Options()),
               SerializationError);
  auto short_jpeg = encoded;
  short_jpeg.resize(encoded.size() / 2);
  EXPECT_THROW(im::decode_image(short_jpeg, Options()), SerializationError);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(im::decode_image(encoded, Options(), stop.get_token()),
               Cancelled);
  EXPECT_TRUE(torch::equal(im::decode_image(encoded, Options()).tensor,
                           im::decode_image(encoded, Options()).tensor));
}
TEST(TurboJpegDecoder, ConcurrentInstancesAndPinnedOutputPreservePixels) {
  if (!Available()) {
    EXPECT_THROW(im::decode_image(Fixture("baseline.jpg"), Options()),
                 InvalidArgument);
    return;
  }
  const auto encoded = Fixture("progressive.jpg");
  auto a = std::async(std::launch::async,
                      [&] { return im::decode_image(encoded, Options()); });
  auto b = std::async(std::launch::async,
                      [&] { return im::decode_image(encoded, Options()); });
  const auto first = a.get(), second = b.get();
  EXPECT_TRUE(torch::equal(first.tensor, second.tensor));
  auto o = Options();
  o.pin_memory = true;
  if (torch::cuda::is_available()) {
    const auto pinned = im::decode_image(encoded, o);
    EXPECT_TRUE(pinned.tensor.is_pinned());
    EXPECT_TRUE(torch::equal(pinned.tensor, first.tensor));
    o.layout = im::ImageLayout::chw;
    const auto chw = im::decode_image(encoded, o);
    EXPECT_TRUE(chw.tensor.is_pinned());
    EXPECT_TRUE(torch::equal(chw.tensor, first.tensor.permute({2, 0, 1})));
  } else
    EXPECT_THROW(im::decode_image(encoded, o), DeviceError);
}
}  // namespace
