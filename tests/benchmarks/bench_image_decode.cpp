// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "resource_support.hpp"

#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace bench = jyppx::liuding::benchmarks;
namespace image = jyppx::liuding::data::image;
constexpr std::size_t kWidth = 2048, kHeight = 2048, kChannels = 3;
constexpr std::uint64_t kIterations = 8;

unsigned char Pixel(std::size_t x, std::size_t y, std::size_t channel) {
  return static_cast<unsigned char>((x * 13U + y * 17U + channel * 71U) & 255U);
}

std::vector<std::byte> Input() {
  constexpr std::string_view header = "P6\n2048 2048\n255\n";
  std::vector<std::byte> bytes(header.size() + kWidth * kHeight * kChannels);
  std::memcpy(bytes.data(), header.data(), header.size());
  auto* raster = reinterpret_cast<unsigned char*>(bytes.data() + header.size());
  for (std::size_t y = 0; y < kHeight; ++y)
    for (std::size_t x = 0; x < kWidth; ++x)
      for (std::size_t c = 0; c < kChannels; ++c)
        raster[(y * kWidth + x) * kChannels + c] = Pixel(x, y, c);
  return bytes;
}

void VerifyPixels(const jyppx::liuding::Tensor& tensor, bool chw,
                  bool floating) {
  const auto count = kWidth * kHeight * kChannels;
  const auto* floats = floating ? tensor.data_ptr<float>() : nullptr;
  const auto* integers = floating ? nullptr : tensor.data_ptr<std::uint8_t>();
  for (std::size_t i = 0; i < count; ++i) {
    const auto pixel = chw ? i % (kWidth * kHeight) : i / kChannels;
    const auto channel = chw ? i / (kWidth * kHeight) : i % kChannels;
    const auto expected = Pixel(pixel % kWidth, pixel / kWidth, channel);
    if (floating)
      bench::require(std::abs(floats[i] -
                              static_cast<float>(expected) / 255.0F) <= 1.0e-7F,
                     "normalized pixel mismatch");
    else
      bench::require(integers[i] == expected, "integer pixel mismatch");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = bench::arguments(argc, argv);
    bench::require(args.device == "cpu", "decoded Tensor output must be CPU");
    const bool chw = args.profile != "image-decode-hwc-u8";
    const bool pinned = args.profile == "image-decode-chw-f32-pinned";
    const bool floating = chw;
    bench::require(args.profile == "image-decode-chw-f32" ||
                       args.profile == "image-decode-hwc-u8" || pinned,
                   "unknown image decode profile");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    if (pinned)
      bench::require(torch::cuda::is_available(),
                     "pinned decoding requires an available CUDA runtime");
    const auto bytes = Input();
    auto sample =
        bench::report(args,
                      {{"fixture", "pnm-rgb-pattern-v1"},
                       {"encoded_sha256", jyppx::liuding::sha256_hex(bytes)},
                       {"width", kWidth},
                       {"height", kHeight},
                       {"channels", kChannels},
                       {"layout", chw ? "chw" : "hwc"},
                       {"dtype", floating ? "float32_normalized" : "uint8"},
                       {"pinned", pinned},
                       {"iterations", kIterations}},
                      kIterations);
    image::ImageDecodeOptions options;
    options.layout = chw ? image::ImageLayout::chw : image::ImageLayout::hwc;
    options.dtype = floating ? image::ImageDtype::float32_normalized
                             : image::ImageDtype::preserve;
    options.pin_memory = pinned;
    options.resource_budget = std::make_shared<image::ImageResourceBudget>();
    double elapsed = 0;
    std::string digest;
    for (std::uint64_t iteration = 0; iteration < kIterations; ++iteration) {
      const auto start = bench::Clock::now();
      auto decoded = image::decode_image(bytes, options);
      elapsed +=
          std::chrono::duration<double>(bench::Clock::now() - start).count();
      bench::require(decoded.width == kWidth && decoded.height == kHeight &&
                         decoded.channels == kChannels &&
                         decoded.tensor.is_cpu() &&
                         decoded.tensor.is_contiguous() &&
                         decoded.tensor.is_pinned() == pinned,
                     "decoded output metadata or storage mismatch");
      if (iteration == 0) VerifyPixels(decoded.tensor, chw, floating);
      const auto payload = std::span(
          static_cast<const std::byte*>(decoded.tensor.const_data_ptr()),
          decoded.tensor.nbytes());
      const auto observed = jyppx::liuding::sha256_hex(payload);
      if (iteration == 0) digest = observed;
      bench::require(observed == digest,
                     "decoded content changed across iterations");
    }
    for (const auto kind : {image::ImageResourceKind::decoded_bytes,
                            image::ImageResourceKind::pinned_bytes,
                            image::ImageResourceKind::pixels,
                            image::ImageResourceKind::codec_scratch_bytes})
      bench::require(options.resource_budget->usage(kind).current == 0,
                     "decode reservation leaked");
    sample["validation"]["output_sha256"] = digest;
    sample["validation"]["independent_pixel_check"] =
        "all first-image elements";
    sample["validation"]["training_updates"] = 0;
    sample["metrics"]["decode_wall_seconds"] =
        bench::metric(elapsed, "seconds");
    sample["metrics"]["images_per_second"] = bench::metric(
        static_cast<double>(kIterations) / elapsed, "images/s", "higher");
    sample["metrics"]["codec_scratch_peak_bytes"] = bench::metric(
        static_cast<double>(
            options.resource_budget
                ->usage(image::ImageResourceKind::codec_scratch_bytes)
                .peak),
        "bytes");
    sample["execution"]["intraop_threads"] = torch::get_num_threads();
    sample["execution"]["interop_threads"] = torch::get_num_interop_threads();
    sample["execution"]["timing_scope"] =
        "decode call; fixture generation and content verification excluded";
    bench::finish(sample, kIterations);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
