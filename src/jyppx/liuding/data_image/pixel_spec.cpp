// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/pixel_spec.hpp>
#include <jyppx/liuding/foundation/error.hpp>

namespace jyppx::liuding::data::image {
namespace {

[[noreturn]] void Invalid(const char* operation, const char* expected) {
  throw InvalidArgument("Pixel semantics do not match the image",
                        ErrorDetails{.component = "data.image.pixels",
                                     .operation = operation,
                                     .expected = expected});
}

}  // namespace

std::uint8_t PixelSpec::channels() const {
  switch (color) {
    case PixelColorModel::gray:
      return 1;
    case PixelColorModel::gray_alpha:
      return 2;
    case PixelColorModel::rgb:
    case PixelColorModel::bgr:
      return 3;
    case PixelColorModel::rgba:
    case PixelColorModel::bgra:
      return 4;
  }
  Invalid("pixels.color", "a known color model");
}

void validate_pixels(const Tensor& pixels, const PixelSpec& spec) {
  const auto channel_count = spec.channels();
  if (spec.layout != ImageLayout::chw && spec.layout != ImageLayout::hwc)
    Invalid("pixels.layout", "CHW or HWC");
  if (spec.alpha != PixelAlphaMode::none &&
      spec.alpha != PixelAlphaMode::straight &&
      spec.alpha != PixelAlphaMode::premultiplied)
    Invalid("pixels.alpha", "a known alpha association");
  const bool has_alpha = spec.color == PixelColorModel::gray_alpha ||
                         spec.color == PixelColorModel::rgba ||
                         spec.color == PixelColorModel::bgra;
  if (has_alpha != (spec.alpha != PixelAlphaMode::none))
    Invalid("pixels.alpha", "alpha association matching the color model");
  const bool floating =
      spec.dtype == torch::kFloat16 || spec.dtype == torch::kBFloat16 ||
      spec.dtype == torch::kFloat32 || spec.dtype == torch::kFloat64;
  switch (spec.range) {
    case PixelValueRange::unsigned_full_scale:
      if (spec.dtype != torch::kUInt8 && spec.dtype != torch::kUInt16)
        Invalid("pixels.range", "uint8 or uint16 for unsigned full scale");
      break;
    case PixelValueRange::unit_interval:
    case PixelValueRange::unbounded:
      if (!floating)
        Invalid("pixels.range", "floating-point storage for this range");
      break;
    default:
      Invalid("pixels.range", "a known value range");
  }
  if (!pixels.defined() || pixels.dim() != 3 ||
      pixels.scalar_type() != spec.dtype ||
      (!pixels.device().is_cpu() && !pixels.device().is_cuda()) ||
      pixels.layout() != torch::kStrided)
    Invalid("pixels.tensor",
            "a strided CPU/CUDA rank-three tensor of the declared dtype");
  const auto channel_axis = spec.layout == ImageLayout::chw ? 0 : 2;
  if (pixels.size(channel_axis) != channel_count || pixels.size(0) <= 0 ||
      pixels.size(1) <= 0 || pixels.size(2) <= 0)
    Invalid("pixels.shape",
            "positive dimensions and the declared channel count");
}

}  // namespace jyppx::liuding::data::image
