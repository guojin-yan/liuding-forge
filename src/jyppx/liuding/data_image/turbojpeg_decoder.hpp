// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_DATA_IMAGE_TURBOJPEG_DECODER_HPP
#define JYPPX_LIUDING_DATA_IMAGE_TURBOJPEG_DECODER_HPP
#include <jyppx/liuding/data_image/image_decoder.hpp>
namespace jyppx::liuding::data::image::codec_detail {
DecodedImage decode_turbojpeg(std::span<const std::byte> bytes,
                              const ImageDecodeOptions& options,
                              std::stop_token stop);
}
#endif
