// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "turbojpeg_decoder.hpp"

#ifdef JYPPX_LIUDING_HAS_TURBOJPEG
#include <algorithm>
#include <climits>
#include <limits>
#include <memory>
#include <turbojpeg.h>
#endif

namespace jyppx::liuding::data::image {
bool jpeg_decode_backend_available(JpegDecodeBackend backend) noexcept {
  if (backend == JpegDecodeBackend::stb) return true;
#ifdef JYPPX_LIUDING_HAS_TURBOJPEG
  return backend == JpegDecodeBackend::libjpeg_turbo;
#else
  return false;
#endif
}
DecodedImage codec_detail::decode_turbojpeg(std::span<const std::byte> bytes,
                                            const ImageDecodeOptions& o,
                                            std::stop_token stop) try {
#ifndef JYPPX_LIUDING_HAS_TURBOJPEG
  throw InvalidArgument("libjpeg-turbo decoding was not enabled at build time");
#else
  const auto check_stop = [&] {
    if (stop.stop_requested()) throw Cancelled("JPEG decode cancelled");
  };
  check_stop();
  if (o.resource_budget)
    throw InvalidArgument(
        "libjpeg-turbo does not support aggregate codec heap accounting");
  if (o.channels == ImageChannels::gray_alpha)
    throw InvalidArgument("libjpeg-turbo does not support gray-alpha output");
  constexpr std::uint64_t mib = 1U << 20U;
  const auto codec_quota = [](std::uint64_t bytes) {
    return static_cast<int>(
        (std::min)({bytes / (1U << 20U),
                    static_cast<std::uint64_t>(LONG_MAX) / (1U << 20U),
                    static_cast<std::uint64_t>(INT_MAX)}));
  };
  if (o.max_working_bytes < mib)
    throw ResourceExhausted("libjpeg-turbo needs at least one MiB codec quota");
  std::unique_ptr<void, decltype(&tj3Destroy)> decoder(
      tj3Init(TJINIT_DECOMPRESS), tj3Destroy);
  if (!decoder)
    throw ResourceExhausted("Cannot allocate libjpeg-turbo decoder");
  const auto set = [&](int parameter, int value) {
    if (tj3Set(decoder.get(), parameter, value))
      throw InvalidArgument(tj3GetErrorStr(decoder.get()));
  };
  set(TJPARAM_STOPONWARNING, 1);
  set(TJPARAM_FASTDCT, 0);
  set(TJPARAM_FASTUPSAMPLE, 0);
  set(TJPARAM_SCANLIMIT, 500);
  set(TJPARAM_MAXPIXELS, static_cast<int>((std::min)(o.limits.max_pixels,
                                                     std::uint64_t{INT_MAX})));
  set(TJPARAM_MAXMEMORY, codec_quota(o.max_working_bytes));
  if (tj3DecompressHeader(decoder.get(),
                          reinterpret_cast<const unsigned char*>(bytes.data()),
                          bytes.size()))
    throw SerializationError(tj3GetErrorStr(decoder.get()));
  const auto width = tj3Get(decoder.get(), TJPARAM_JPEGWIDTH),
             height = tj3Get(decoder.get(), TJPARAM_JPEGHEIGHT),
             color = tj3Get(decoder.get(), TJPARAM_COLORSPACE);
  if (width <= 0 || height <= 0 ||
      static_cast<std::uint32_t>(width) > o.limits.max_dimension ||
      static_cast<std::uint32_t>(height) > o.limits.max_dimension)
    throw ResourceExhausted("JPEG dimension limit exceeded");
  if ((color != TJCS_GRAY && color != TJCS_RGB && color != TJCS_YCbCr) ||
      tj3Get(decoder.get(), TJPARAM_PRECISION) != 8 ||
      tj3Get(decoder.get(), TJPARAM_LOSSLESS))
    throw SerializationError(
        "libjpeg-turbo profile requires lossy 8-bit gray/RGB JPEG");
  const auto channels =
      o.channels == ImageChannels::gray ||
              (o.channels == ImageChannels::source && color == TJCS_GRAY)
          ? 1
      : o.channels == ImageChannels::rgba ? 4
                                          : 3;
  const auto pixels = std::uint64_t(width) * height;
  const auto bytes_per_pixel =
      channels *
      (o.dtype == ImageDtype::float32_normalized ? sizeof(float) : 1);
  if (pixels > o.limits.max_pixels ||
      pixels > o.limits.max_output_bytes / bytes_per_pixel)
    throw ResourceExhausted("JPEG pixel/output limit exceeded");
  const bool direct =
      o.layout == ImageLayout::hwc && o.dtype == ImageDtype::preserve;
  const auto staging = direct ? 0 : pixels * channels;
  if (staging > o.max_working_bytes || o.max_working_bytes - staging < mib)
    throw ResourceExhausted("JPEG staging/codec working limit exceeded");
  set(TJPARAM_MAXMEMORY, codec_quota(o.max_working_bytes - staging));
  check_stop();
  if (o.pin_memory && !torch::cuda::is_available())
    throw DeviceError("Pinned JPEG memory requires an accelerator runtime");
  auto hwc = torch::empty({height, width, channels},
                          torch::TensorOptions()
                              .dtype(torch::kUInt8)
                              .device(torch::kCPU)
                              .pinned_memory(direct && o.pin_memory));
  const auto format = channels == 1   ? TJPF_GRAY
                      : channels == 4 ? TJPF_RGBA
                                      : TJPF_RGB;
  if (tj3Decompress8(
          decoder.get(), reinterpret_cast<const unsigned char*>(bytes.data()),
          bytes.size(), hwc.mutable_data_ptr<std::uint8_t>(), 0, format))
    throw SerializationError(tj3GetErrorStr(decoder.get()));
  check_stop();
  auto tensor = hwc;
  if (!direct) {
    auto source = o.layout == ImageLayout::chw ? hwc.permute({2, 0, 1}) : hwc;
    const auto dtype = o.dtype == ImageDtype::float32_normalized
                           ? torch::kFloat32
                           : torch::kUInt8;
    tensor =
        torch::empty(source.sizes(),
                     source.options().dtype(dtype).pinned_memory(o.pin_memory));
    tensor.copy_(source);
    if (o.dtype == ImageDtype::float32_normalized) tensor.div_(255.);
  }
  DecodedImage result;
  result.tensor = std::move(tensor);
  result.format = ImageFormat::jpeg;
  result.layout = o.layout;
  result.width = width;
  result.height = height;
  result.channels = channels;
  result.pixels.dtype = result.tensor.scalar_type();
  result.pixels.layout = o.layout;
  result.pixels.color = channels == 1   ? PixelColorModel::gray
                        : channels == 4 ? PixelColorModel::rgba
                                        : PixelColorModel::rgb;
  result.pixels.alpha =
      channels == 4 ? PixelAlphaMode::straight : PixelAlphaMode::none;
  result.pixels.range = o.dtype == ImageDtype::float32_normalized
                            ? PixelValueRange::unit_interval
                            : PixelValueRange::unsigned_full_scale;
  validate_pixels(result.tensor, result.pixels);
  check_stop();
  return result;
#endif
} catch (const c10::Error& error) {
  detail::rethrow_backend_error(
      error, detail::BackendErrorContext{
                 .component = "data.image.decoder",
                 .operation = "decode.turbojpeg.tensor",
                 .failure_kind = detail::BackendFailureKind::device,
                 .hint = "Check the selected LibTorch allocator and dtype"});
} catch (const std::bad_alloc&) {
  throw ResourceExhausted("Cannot allocate JPEG decode storage");
}
}  // namespace jyppx::liuding::data::image
